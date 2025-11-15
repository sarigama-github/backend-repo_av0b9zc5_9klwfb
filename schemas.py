"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

# Example schemas (you can keep or remove later)

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Trading-related schemas

Side = Literal["BUY", "SELL", "HOLD"]

class StrategyConfig(BaseModel):
    """User-defined strategy configuration"""
    name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Market symbol, e.g., BTCUSDT")
    interval: str = Field(..., description="Candle interval, e.g., 1m, 5m, 15m")
    ema_period: int = Field(20, ge=2, description="EMA period")
    rsi_period: int = Field(14, ge=2, description="RSI period")
    volume_multiplier: float = Field(1.5, ge=0.1, description="Volume spike multiplier vs. average")
    enabled: bool = Field(True)

class IndicatorSnapshot(BaseModel):
    close: float
    ema: float
    rsi: float
    volume: float
    avg_volume: float

class Signal(BaseModel):
    """Signals collection: store strategy-generated binary buy/sell/hold"""
    symbol: str
    interval: str
    side: Side
    confidence: float = Field(..., ge=0, le=1)
    at: datetime = Field(default_factory=datetime.utcnow)
    strategy_name: str
    reason: str
    indicators: IndicatorSnapshot

# Add your own schemas below as needed
