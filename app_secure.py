import os
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from ta.momentum import RSIIndicator
from ta.trend import MACD

app = FastAPI()

# Load API key from environment variable
API_KEY = os.getenv("PRIVATE_API_KEY")

def verify_api_key(request: Request):
    """Check API key"""
    client_key = request.headers.get("x-api-key")
    if not client_key or client_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key.")

@app.get("/")
def root():
    return {"message": "Backend is running!"}

@app.get("/opal_payload")
def stock_analysis(ticker: str, shares: float, cost: float, request: Request):
    """Main stock analysis endpoint"""
    verify_api_key(request)

    data = yf.download(ticker, period="1y", progress=False)
    if data.empty:
        raise HTTPException(status_code=404, detail="Ticker not found")

    close = data["Close"]

    # RSI
    rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]

    # MACD
    macd_ind = MACD(close=close)
    macd_val = macd_ind.macd().iloc[-1]
    signal_val = macd_ind.macd_signal().iloc[-1]
    hist_val = macd_ind.macd_diff().iloc[-1]

    # SMA
    sma_50 = close.rolling(50).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1]

    # Latest price
    last_price = float(close.iloc[-1])
    investment_value = shares * last_price
    pnl = investment_value - (shares * cost)

    return {
        "ticker": ticker.upper(),
        "shares": shares,
        "buy_cost": cost,
        "current_price": last_price,
        "investment_value": investment_value,
        "pnl": pnl,
        "technicals": {
            "rsi_14": float(rsi),
            "macd": {
                "macd": float(macd_val),
                "signal": float(signal_val),
                "hist": float(hist_val)
            },
            "sma_50": float(sma_50),
            "sma_200": float(sma_200)
        }
    }

