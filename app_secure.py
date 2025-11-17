import os
import yfinance as yf
import pandas_ta as ta
from fastapi import FastAPI, Request, HTTPException
from fastapi.security.api_key import APIKeyHeader
from fastapi.openapi.models import APIKey, APIKeyIn, SecurityScheme
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------------------------------
# SECURITY SCHEME (makes Authorize button appear in Swagger UI)
# ----------------------------------------------------------------------------
api_key_scheme = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")

def verify_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    if api_key != PRIVATE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key.")
    return True

@app.get("/opal_payload")
def opal_payload(request: Request, ticker: str, shares: float, cost: float):

    verify_api_key(request)

    data = yf.Ticker(ticker).history(period="1y")
    if data.empty:
        raise HTTPException(status_code=404, detail="Ticker not found")

    data["rsi_14"] = ta.rsi(data["Close"], length=14)
    macd = ta.macd(data["Close"])
    data["sma_50"] = ta.sma(data["Close"], length=50)
    data["sma_200"] = ta.sma(data["Close"], length=200)

    current_price = float(data["Close"].iloc[-1])
    investment_value = shares * current_price
    pnl = investment_value - (shares * cost)

    return {
        "ticker": ticker,
        "shares": shares,
        "buy_cost": cost,
        "current_price": current_price,
        "investment_value": investment_value,
        "pnl": pnl,
        "technicals": {
            "rsi_14": float(data["rsi_14"].iloc[-1]),
            "macd": {
                "macd": float(macd["MACD_12_26_9"].iloc[-1]),
                "signal": float(macd["MACDs_12_26_9"].iloc[-1]),
                "hist": float(macd["MACDh_12_26_9"].iloc[-1]),
            },
            "sma_50": float(data["sma_50"].iloc[-1]),
            "sma_200": float(data["sma_200"].iloc[-1]),
        }
    }
