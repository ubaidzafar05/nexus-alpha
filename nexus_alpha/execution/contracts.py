"""Execution interface contracts for routing and OMS auditability."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from nexus_alpha.types import ExchangeName, OrderSide, OrderType


class OrderRequest(BaseModel):
    """Normalized order request for execution services."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=8)
    symbol: str = Field(min_length=3)
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0)
    limit_price: float | None = Field(default=None, gt=0)
    created_at: datetime


class RouteDecision(BaseModel):
    """Execution venue routing decision with traceable rationale."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=8)
    primary_exchange: ExchangeName
    allocations: dict[ExchangeName, float]
    estimated_cost_bps: float
    reasoning: str = Field(min_length=3)
    decided_at: datetime


class ExecutionReport(BaseModel):
    """Post-trade execution report suitable for audit and benchmarking."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=8)
    symbol: str = Field(min_length=3)
    side: OrderSide
    requested_qty: float = Field(gt=0)
    filled_qty: float = Field(ge=0)
    avg_fill_price: float = Field(gt=0)
    arrival_price: float = Field(gt=0)
    implementation_shortfall_bps: float
    completed_at: datetime
