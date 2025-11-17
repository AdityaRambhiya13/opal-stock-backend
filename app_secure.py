# app_secure.py
import os
from typing import Optional, Dict, Any

import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

# ----------------
# CONFIG
# ----------------
API_KEY_NAME = "x-api-key"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY", None)

# Security dependency (adds OpenAPI security scheme -> "Authorize" button)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header_value: Optional[str] = Security(api_key_header)):
    if PRIVATE_API_KEY is None:
        # If env var not set, treat as unauthorized (safer)
        raise HTTPException(status_code=401, detail="Server misconfiguration: PRIVATE_API_KEY not set.")
    if api_key_header_value == PRIVATE_API_KEY:
        return api_key_header_value
    raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key.")

app = FastAPI(
    title="Private Stock Fetcher for Opal",
    description="Returns live price, PnL and technical indicators for any ticker supported by yfinance.",
)

# ---------------------------
# Technical indicators helper
# ---------------------------
def compute_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["Close"]

    # RSI 14
    try:
        rsi14 = RSIIndicator(close, window=14).rsi().iloc[-1]
    except Exception:
        rsi14 = None

    # MACD (12,26,9)
    try:
        macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd = {
            "macd": macd_ind.macd().iloc[-1],
            "signal": macd_ind.macd_signal().iloc[-1],
            "hist": macd_ind.macd_diff().iloc[-1]
        }
    except Exception:
        macd = {"macd": None, "signal": None, "hist": None}

    # SMA 50 / 200
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
        "macd": {
            "macd": None if pd.isna(macd["macd"]) else float(macd["macd"]),
            "signal": None if pd.isna(macd["signal"]) else float(macd["signal"]),
            "hist": None if pd.isna(macd["hist"]) else float(macd["hist"]),
        },
        "sma_50": None if sma50 is None or pd.isna(sma50) else float(sma50),
        "sma_200": None if sma200 is None or pd.isna(sma200) else float(sma200)
    }

# ---------------------------
# Endpoint (protected)
# ---------------------------
@app.get("/opal_payload", summary="Get Opal-ready stock payload", tags=["opal"])
async def opal_payload(ticker: str, shares: float = 0.0, cost: float = 0.0, api_key: str = Depends(get_api_key)):
    """
    Returns live price, PnL and technical indicators for the requested ticker.
    Protected by x-api-key header (use the Authorize button in /docs to set it).
    """
    ticker = ticker.strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Missing ticker parameter.")

    # fetch 1y daily history
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y", interval="1d", auto_adjust=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch error: {e}")

    if hist is None or hist.empty:
        raise HTTPException(status_code=404, detail="Invalid ticker or no data found.")

    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
    hist.index = pd.to_datetime(hist.index)

    tech = compute_technical_indicators(hist)
    last_close = float(hist['Close'].iloc[-1])
    prev_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else last_close
    investment_value = shares * last_close
    pnl = (last_close - cost) * shares

    info = {}
    try:
        info = tk.info
    except Exception:
        info = {}

    response = {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName"),
        "shares": float(shares),
        "buy_cost": float(cost),
        "current_price": float(last_close),
        "prev_close": float(prev_close),
        "investment_value": float(investment_value),
        "pnl": float(pnl),
        "technicals": tech,
        "ohlc_last_5": hist.tail(5).reset_index().to_dict(orient="records")
    }

    return JSONResponse({"status": "ok", "data": response})
