import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
from datetime import datetime, timezone
import math
import statistics
import random
import time
import requests

# Database helpers
from database import db, create_document, get_documents
from schemas import StrategyConfig, Signal, IndicatorSnapshot, FutureSignal

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


# --------- SMC/ICT/VSA Heuristics (lightweight approximations) ---------

def true_range(c: Candle) -> float:
    return c.high - c.low


def detect_bos(candles: List[Candle]) -> Optional[str]:
    if len(candles) < 5:
        return None
    highs = [c.high for c in candles[-6:-1]]
    lows = [c.low for c in candles[-6:-1]]
    last = candles[-1]
    if last.close > max(highs):
        return "BOS Up"
    if last.close < min(lows):
        return "BOS Down"
    return None


def detect_liquidity_sweep(candles: List[Candle]) -> Optional[str]:
    if len(candles) < 3:
        return None
    prev = candles[-2]
    last = candles[-1]
    # Long wick vs body suggests sweep
    body = abs(last.close - last.open)
    upper_wick = last.high - max(last.close, last.open)
    lower_wick = min(last.close, last.open) - last.low
    if upper_wick > body * 1.5 and last.close < prev.close:
        return "Liquidity sweep (buy stops)"
    if lower_wick > body * 1.5 and last.close > prev.close:
        return "Liquidity sweep (sell stops)"
    return None


def detect_order_block(candles: List[Candle]) -> Optional[str]:
    if len(candles) < 5:
        return None
    # Use last strong opposite candle range as OB hint
    last = candles[-1]
    for c in reversed(candles[-6:-1]):
        body = abs(c.close - c.open)
        rng = c.high - c.low
        if body / (rng + 1e-9) > 0.6:
            if c.close < c.open:
                return "Bullish OB below"
            else:
                return "Bearish OB above"
    return None


def vsa_read(candles: List[Candle]) -> Tuple[str, bool]:
    vols = [c.volume for c in candles]
    avg_vol = statistics.mean(vols[-20:]) if len(vols) >= 20 else statistics.mean(vols)
    last_vol = vols[-1]
    spread = true_range(candles[-1])
    avg_spread = statistics.mean([true_range(c) for c in candles[-20:]]) if len(candles) >= 20 else statistics.mean([true_range(c) for c in candles])
    spike = last_vol > avg_vol * 1.6 and spread > avg_spread * 1.2
    label = "No VSA anomaly"
    if spike:
        label = "Volume spread spike"
    return label, spike


def big_player_footprint(candles: List[Candle]) -> bool:
    last = candles[-1]
    rng = true_range(last)
    avg_rng = statistics.mean([true_range(c) for c in candles[-20:]]) if len(candles) >= 20 else statistics.mean([true_range(c) for c in candles])
    return rng > avg_rng * 1.8


def session_strength(now: Optional[datetime] = None) -> Tuple[str, float]:
    now = now or datetime.now(timezone.utc)
    hour = now.hour
    # Approx sessions in UTC
    # London 7-16, New York 12-21
    if 7 <= hour < 16:
        return "London", 1.15
    if 12 <= hour < 21:
        return "New York", 1.10
    return "Asia", 0.95


# --------- Core Binary Strategy (base EMA/RSI/Vol) ---------

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


# --------- Higher-level SMC/ICT/VSA Signal ---------

class AdvancedSignal(BaseModel):
    symbol: str
    interval: str
    signal: str
    reason: str
    confidence: int


SMC_LABELS = [
    "Break of Structure",
    "Liquidity sweep",
    "Order block",
    "Volume spread anomaly",
    "Big player activity",
]


def evaluate_smc_signal(symbol: str, interval: str) -> AdvancedSignal:
    # If symbol not supported on Binance, return deterministic pseudo signal
    use_binance = symbol.upper().endswith("USDT")
    if use_binance:
        candles = fetch_klines(symbol, interval, limit=120)
    else:
        # Pseudo-random walk based on symbol hash for demo purposes
        seed = abs(hash((symbol.upper(), interval, int(time.time() // 60)))) % (2**32)
        random.seed(seed)
        base = 100 + random.random() * 50
        candles = []
        t = int(time.time() // 60) * 60000
        for i in range(120):
            o = base + math.sin(i / 7) * 0.3 + random.uniform(-0.2, 0.2)
            c = o + random.uniform(-0.5, 0.5)
            h = max(o, c) + random.random() * 0.3
            l = min(o, c) - random.random() * 0.3
            v = 100 + random.random() * 50
            candles.append(Candle(open_time=t + i * 60000, open=o, high=h, low=l, close=c, volume=v, close_time=t + i * 60000 + 59000))

    bos = detect_bos(candles)
    sweep = detect_liquidity_sweep(candles)
    ob = detect_order_block(candles)
    vsa_label, vsa_spike = vsa_read(candles)
    big = big_player_footprint(candles)

    last = candles[-1]
    prev = candles[-2]
    direction_up = last.close > prev.close

    session_name, session_boost = session_strength()

    score = 50
    reasons: List[str] = []

    if bos == "BOS Up":
        score += 12
        reasons.append("Break of Structure")
    elif bos == "BOS Down":
        score -= 12
        reasons.append("Break of Structure")

    if sweep:
        reasons.append("Liquidity hunt")
        score += 8 if direction_up else -8

    if ob:
        reasons.append("Order block")
        score += 6 if ("Bullish" in ob) else -6

    if vsa_spike:
        reasons.append("Volume spread anomaly")
        score += 7 if direction_up else -7

    if big:
        reasons.append("Big player activity")
        score += 9 if direction_up else -9

    # Session filter
    if session_name == "London":
        score *= 1.05
    elif session_name == "New York":
        score *= 1.03

    score = max(0, min(100, int(round(score))))
    sig = "CALL" if score >= 55 else ("PUT" if score <= 45 else "HOLD")

    reason_str = ", ".join(reasons[:3]) or "Neutral conditions"
    return AdvancedSignal(symbol=symbol.upper(), interval=interval, signal=sig, reason=reason_str, confidence=score)


# --------- Routes ---------

@app.get("/")
def root():
    return {"message": "Binary Trading Signal API running"}


@app.get("/api/signal")
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


# Advanced SMC/ICT/VSA endpoints

@app.get("/api/smc-signal")
def smc_signal(symbol: str = Query(...), interval: str = Query("1m")):
    """Generate an advanced signal using SMC/ICT/VSA-inspired heuristics."""
    return evaluate_smc_signal(symbol, interval)


@app.get("/api/future-signals")
def future_signals(symbol: str = Query(...), interval: str = Query("1m")):
    base = evaluate_smc_signal(symbol, interval)
    seeds = [1, 2, 3, 4, 5]
    out: List[FutureSignal] = []
    for i in seeds:
        # deterministic tweak
        rnd = (abs(hash((symbol.upper(), interval, i))) % 100) / 100.0
        side = base.signal if rnd > 0.5 else ("PUT" if base.signal == "CALL" else "CALL")
        conf = max(40, min(95, int(base.confidence + (rnd - 0.5) * 30)))
        reason = "Liquidity sweep expected" if i % 2 == 0 else "Volume imbalance"
        out.append(FutureSignal(symbol=symbol.upper(), interval=interval, side=side, reason=reason, confidence=conf/100.0, eta_minutes=i))
    return out[:5]


@app.get("/api/pairs")
def pairs():
    pairs = [
        {"symbol": "EURUSD", "label": "EUR/USD"},
        {"symbol": "GBPJPY", "label": "GBP/JPY"},
        {"symbol": "AUDCAD", "label": "AUD/CAD"},
        {"symbol": "EURJPY", "label": "EUR/JPY"},
        {"symbol": "XAUUSD", "label": "Gold (XAU)"},
        {"symbol": "BTCUSDT", "label": "BTC/USDT"},
    ]
    out = []
    now = datetime.now(timezone.utc)
    sess, boost = session_strength(now)
    for p in pairs:
        sym = p["symbol"]
        # If we can fetch, use actual trend from last two closes on Binance (only for USDT)
        trend = "Neutral"
        vol = 0.5
        liq = 0.5
        try:
            if sym.endswith("USDT"):
                cs = fetch_klines(sym, "1m", 20)
                if cs[-1].close > cs[-2].close:
                    trend = "Bullish"
                elif cs[-1].close < cs[-2].close:
                    trend = "Bearish"
                vols = [c.volume for c in cs]
                vol = max(0.0, min(1.0, (vols[-1] / (statistics.mean(vols) + 1e-9))))
                liq = max(0.1, min(1.0, statistics.mean([true_range(c) for c in cs]) / (cs[-1].close * 0.01)))
            else:
                # Heuristic values
                seed = abs(hash((sym, int(time.time() // 120)))) % 1000
                trend = ["Bullish", "Bearish", "Neutral"][seed % 3]
                vol = (seed % 100) / 100
                liq = (seed % 70) / 70
        except Exception:
            pass
        out.append({
            "symbol": sym,
            "label": p["label"],
            "current_trend": trend,
            "volatility": round(vol, 2),
            "session": sess,
            "session_strength": round(boost, 2),
            "liquidity": round(liq, 2),
        })
    return out


@app.get("/api/klines")
def klines(symbol: str = Query(...), interval: str = Query("1m"), limit: int = Query(100, ge=10, le=500)):
    cs = fetch_klines(symbol, interval, limit)
    return [{
        "t": c.open_time,
        "o": c.open,
        "h": c.high,
        "l": c.low,
        "c": c.close,
        "v": c.volume,
    } for c in cs]


@app.get("/api/time/next_close")
def next_close(interval: str = Query("1m")):
    # Approximate next candle close (align to minute multiples)
    now = datetime.now(timezone.utc)
    seconds = 60
    if interval.endswith('m'):
        seconds = int(interval[:-1]) * 60
    elif interval.endswith('h'):
        seconds = int(interval[:-1]) * 3600
    epoch = int(now.timestamp())
    remaining = seconds - (epoch % seconds)
    return {"seconds_remaining": remaining}


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
