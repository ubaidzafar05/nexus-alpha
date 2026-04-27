"""Typed event contracts for ingestion and feature pipelines."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from nexus_alpha.schema_types import ExchangeName


class EventKind(str, Enum):
    TICK = "tick"
    OHLCV = "ohlcv"
    FEATURE_SNAPSHOT = "feature_snapshot"


class TickEventPayload(BaseModel):
    """Hot-path market tick payload."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=3)
    exchange: ExchangeName
    timestamp: datetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    last_price: float = Field(gt=0)
    volume_24h: float = Field(ge=0)
    bid_size: float = Field(ge=0, default=0.0)
    ask_size: float = Field(ge=0, default=0.0)


class OHLCVEventPayload(BaseModel):
    """Normalized candlestick payload."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=3)
    exchange: ExchangeName
    timeframe: str = Field(min_length=2)
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    trades: int = Field(ge=0, default=0)


class FeatureSnapshotPayload(BaseModel):
    """Feature materialization payload."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=3)
    exchange: ExchangeName
    timestamp: datetime
    features: dict[str, float]
    source: str = Field(min_length=2, default="realtime")


class IngestionEvent(BaseModel):
    """Envelope around hot-path events for bus transport."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=8)
    kind: EventKind
    produced_at: datetime = Field(default_factory=datetime.utcnow)
    partition_key: str = Field(min_length=1)
    payload: TickEventPayload | OHLCVEventPayload | FeatureSnapshotPayload

