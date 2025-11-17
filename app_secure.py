# app_secure.py
import os
import time
from typing import Optional, Dict, Any

import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from cachetools import TTLCache, cached

app = FastAPI(title="Private Stock Fetcher for Opal (Single User)")

# --- Security (OpenAPI shows Authorize button) ---
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header_value: Optional[str] = Security(api_key_header)) -> APIKey:
    expected = os.getenv("PRIVATE_API_KEY")
    if (not expected) or (api_key_header_value != expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API key.",
        )
    return api_key_header_value

# --- Cache: reduce repeated calls (per instance) ---
# TTL in seconds (adjust as needed). Increase to e.g. 600 or 900 for heavier caching.
cache = TTLCache(maxsize=2000, ttl=600)

# Helper: robust yfinance fetch with retry/backoff
def fetch_history_with_retries(ticker: str, period: str = "1y", interval: str = "1d", max_retries: int = 3, backoff_base: float = 1.0) -> pd.DataFrame:
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, interval=interval, auto_adjust=False)
            if hist is None or hist.empty:
                raise ValueError("No history returned for ticker")
            hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
            hist.index = pd.to_datetime(hist.index)
            return hist
        except Exception as e:
            last_exception = e
            # inspect common indicators of rate-limiting
            txt = str(e).lower()
            if "too many requests" in txt or "429" in txt or "rate limit" in txt:
                # exponential backoff
                time.sleep(backoff_base * (2 ** (attempt - 1)))
                continue
            # transient network errors: backoff and retry
            time.sleep(backoff_base * (2 ** (attempt - 1)))
    # after retries, raise more informative error
    msg = f"Data fetch error: {str(last_exception)}"
    # if rate-limited, return a 429
    if last_exception and ("too many" in str(last_exception).lower() or "429" in str(last_exception)):
        raise HTTPException(status_code=429, detail="Data fetch error: Too Many Requests. Rate limited. Try after a while.")
    raise HTTPException(status_code=502, detail=msg)


def compute_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["Close"]
    # RSI 14
    try:
        rsi14 = RSIIndicator(close, window=14).rsi().iloc[-1]
    except Exception:
        rsi14 = None
    # MACD
    try:
        macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd = {
            "macd": macd_ind.macd().iloc[-1],
            "signal": macd_ind.macd_signal().iloc[-1],
            "hist": macd_ind.macd_diff().iloc[-1],
        }
    except Exception:
        macd = {"macd": None, "signal": None, "hist": None}
    # SMA 50/200
    try:
        sma50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    except Exception:
        sma50 = None
    try:
        sma200 = SMAIndicator(close, window=200).sma_indicator().iloc[-1]
    except Exception:
        sma200 = None

    return {
        "rsi_14": None if pd.isna(rsi14) else float(rsi14),
        "macd": {k: (None if pd.isna(v) else float(v)) for k, v in macd.items()},
        "sma_50": None if sma50 is None or pd.isna(sma50) else float(sma50),
        "sma_200": None if sma200 is None or pd.isna(sma200) else float(sma200),
    }

@cached(cache)
def prepare_payload(ticker: str) -> Dict[str, Any]:
    hist = fetch_history_with_retries(ticker, period="1y", interval="1d", max_retries=3)
    last_close = float(hist['Close'].iloc[-1])
    prev_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else last_close
    tech = compute_technical_indicators(hist)
    info = yf.Ticker(ticker).info
    pe = info.get('trailingPE') or info.get('forwardPE')
    eps = info.get('trailingEps') or info.get('epsTrailingTwelveMonths')
    market_cap = info.get('marketCap')
    pct_1d = ((last_close - prev_close) / prev_close) * 100 if prev_close else 0.0
    payload = {
        "ticker": ticker,
        "name": info.get('longName') or info.get('shortName'),
        "price": last_close,
        "prev_close": prev_close,
        "pct_1d": round(pct_1d, 4),
        "eps_trailing": float(eps) if eps else None,
        "pe_trailing": float(pe) if pe else None,
        "market_cap": int(market_cap) if market_cap else None,
        "rsi_14": tech.get("rsi_14"),
        "macd": tech.get("macd"),
        "sma_50": tech.get("sma_50"),
        "sma_200": tech.get("sma_200"),
        "ohlc_last_5": hist.tail(5).reset_index().to_dict(orient='records')
    }
    return payload

@app.get("/opal_payload", summary="Get Opal-ready payload for a ticker", tags=["opal"])
async def opal_payload(ticker: str, shares: float = 0.0, cost: float = 0.0, api_key: APIKey = Depends(get_api_key)):
    """
    Query parameters:
    - ticker: the stock ticker (e.g., AAPL or RELIANCE.NS). Prefer ticker symbol (not full company name).
    - shares: user's holding
    - cost: user's cost per share
    """
    # Validate / normalize ticker - recommend symbol; do not accept human company names
    if not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise HTTPException(status_code=400, detail="Invalid ticker parameter.")
    ticker = ticker.strip()

    # Prepare payload (cached)
    payload = prepare_payload(ticker.upper())

    eps = payload.get("eps_trailing")
    industry_pe = 44.6
    fair_value = round(eps * industry_pe, 2) if eps else None

    result = {
        "ticker": payload["ticker"],
        "name": payload.get("name"),
        "price": payload.get("price"),
        "prev_close": payload.get("prev_close"),
        "eps_trailing": payload.get("eps_trailing"),
        "pe_trailing": payload.get("pe_trailing"),
        "industry_pe": industry_pe,
        "fair_value_estimate": fair_value,
        "rsi_14": payload.get("rsi_14"),
        "macd": payload.get("macd"),
        "sma_50": payload.get("sma_50"),
        "sma_200": payload.get("sma_200"),
        "ohlc_last_5": payload.get("ohlc_last_5"),
        "user_holding": {"shares": shares, "cost": cost}
    }
    return {"status": "ok", "data": result}
