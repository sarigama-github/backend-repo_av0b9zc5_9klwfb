import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import math
import statistics
import requests

# Database helpers
from database import db, create_document, get_documents
from schemas import StrategyConfig, Signal, IndicatorSnapshot

app = FastAPI(title="Binary Trading Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Utility: Technical Indicators ---------

def ema(values: List[float], period: int) -> List[float]:
    if period <= 1 or len(values) == 0:
        return values[:]
    k = 2 / (period + 1)
    ema_vals = []
    prev = values[0]
    for v in values:
        prev = (v * k) + (prev * (1 - k))
        ema_vals.append(prev)
    return ema_vals

def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return [50.0] * len(values)
    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    rsis = [50.0] * len(values)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rs = (avg_gain / avg_loss) if avg_loss != 0 else float('inf')
    rsis[period] = 100 - (100 / (1 + rs))
    for i in range(period + 1, len(values)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float('inf')
        rsis[i] = 100 - (100 / (1 + rs))
    # Fill initial part
    for i in range(period):
        rsis[i] = 50.0
    return rsis

# --------- External Market Data (public binance) ---------

BINANCE_BASE = "https://api.binance.com"

class Candle(BaseModel):
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


def fetch_klines(symbol: str, interval: str, limit: int = 100) -> List[Candle]:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Market data error: {r.text}")
    data = r.json()
    candles: List[Candle] = []
    for c in data:
        candles.append(Candle(
            open_time=c[0],
            open=float(c[1]),
            high=float(c[2]),
            low=float(c[3]),
            close=float(c[4]),
            volume=float(c[5]),
            close_time=c[6]
        ))
    return candles

# --------- Strategy Logic (Binary signal) ---------

class EvaluateResponse(BaseModel):
    symbol: str
    interval: str
    side: str
    confidence: float
    reason: str
    indicators: IndicatorSnapshot


def evaluate_strategy(symbol: str, interval: str, ema_period: int, rsi_period: int, volume_multiplier: float) -> EvaluateResponse:
    candles = fetch_klines(symbol, interval, limit=max(ema_period, rsi_period) + 50)
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    ema_vals = ema(closes, ema_period)
    rsi_vals = rsi(closes, rsi_period)

    last_close = closes[-1]
    last_ema = ema_vals[-1]
    last_rsi = rsi_vals[-1]

    avg_vol = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else statistics.mean(volumes)
    last_vol = volumes[-1]
    vol_spike = last_vol > avg_vol * volume_multiplier

    side = "HOLD"
    reasons = []
    score = 0

    if last_close > last_ema:
        score += 0.35
        reasons.append("Price above EMA")
    else:
        reasons.append("Price below EMA")

    if last_rsi < 30:
        score += 0.35
        reasons.append("RSI oversold")
    elif last_rsi > 70:
        reasons.append("RSI overbought")
        score -= 0.15

    if vol_spike:
        score += 0.3
        reasons.append("Volume spike")
    else:
        reasons.append("Normal volume")

    if score >= 0.6:
        side = "BUY"
    elif score <= 0.1:
        side = "SELL"
    else:
        side = "HOLD"

    confidence = max(0.0, min(1.0, score))

    snapshot = IndicatorSnapshot(
        close=last_close,
        ema=last_ema,
        rsi=last_rsi,
        volume=last_vol,
        avg_volume=avg_vol,
    )

    return EvaluateResponse(
        symbol=symbol.upper(),
        interval=interval,
        side=side,
        confidence=confidence,
        reason=", ".join(reasons),
        indicators=snapshot,
    )

# --------- Routes ---------

@app.get("/")
def root():
    return {"message": "Binary Trading Signal API running"}

@app.get("/api/signal", response_model=EvaluateResponse)
def get_signal(
    symbol: str = Query(..., description="Market symbol, e.g., BTCUSDT"),
    interval: str = Query("5m", description="Interval, e.g., 1m,3m,5m,15m"),
    ema_period: int = Query(20, ge=2),
    rsi_period: int = Query(14, ge=2),
    volume_multiplier: float = Query(1.5, ge=0.1)
):
    """Compute a BUY/SELL/HOLD signal using EMA+RSI+Volume."""
    return evaluate_strategy(symbol, interval, ema_period, rsi_period, volume_multiplier)

class SaveSignalRequest(BaseModel):
    symbol: str
    interval: str
    side: str
    confidence: float
    reason: str
    strategy_name: str
    indicators: IndicatorSnapshot

@app.post("/api/signal/save")
def save_signal(req: SaveSignalRequest):
    doc = Signal(
        symbol=req.symbol.upper(),
        interval=req.interval,
        side=req.side,
        confidence=max(0.0, min(1.0, req.confidence)),
        at=datetime.utcnow(),
        strategy_name=req.strategy_name,
        reason=req.reason,
        indicators=req.indicators,
    ).model_dump()
    create_document("signal", doc)
    return {"ok": True}

@app.get("/api/signal/history")
def signal_history(symbol: Optional[str] = None, limit: int = 50):
    filt = {"symbol": symbol.upper()} if symbol else {}
    docs = get_documents("signal", filt, limit)
    return docs

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
