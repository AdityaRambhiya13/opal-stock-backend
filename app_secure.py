import os
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

app = FastAPI()

# ---------------------------
# TECHNICAL INDICATOR LOGIC
# ---------------------------
def compute_technical_indicators(df):
    close = df["Close"]

    # RSI 14
    try:
        rsi14 = RSIIndicator(close, window=14).rsi().iloc[-1]
    except:
        rsi14 = None

    # MACD (12, 26, 9)
    try:
        macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd = {
            "macd": macd_ind.macd().iloc[-1],
            "signal": macd_ind.macd_signal().iloc[-1],
            "hist": macd_ind.macd_diff().iloc[-1]
        }
    except:
        macd = {"macd": None, "signal": None, "hist": None}

    # SMA 50
    try:
        sma50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    except:
        sma50 = None

    # SMA 200
    try:
        sma200 = SMAIndicator(close, window=200).sma_indicator().iloc[-1]
    except:
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
# API ENDPOINT
# ---------------------------
@app.get("/opal_payload")
async def generate_payload(request: Request, ticker: str, shares: float, cost: float):
    # API key validation
    private_key = os.getenv("PRIVATE_API_KEY")
    incoming_key = request.headers.get("x-api-key")

    if not private_key or incoming_key != private_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key.")

    # Download stock data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if hist.empty:
        raise HTTPException(status_code=404, detail="Invalid ticker or no data found.")

    tech = compute_technical_indicators(hist)
    current_price = hist["Close"].iloc[-1]
    investment_value = shares * current_price
    pnl = (current_price - cost) * shares

    return {
        "ticker": ticker,
        "shares": shares,
        "buy_cost": cost,
        "current_price": float(current_price),
        "investment_value": float(investment_value),
        "pnl": float(pnl),
        "technicals": tech
    }
