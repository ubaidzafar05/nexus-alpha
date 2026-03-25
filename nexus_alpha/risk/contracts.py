"""Risk decision contracts for pre-trade and deployment gates."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskAction(str, Enum):
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"


class RiskDecision(BaseModel):
    """Risk firewall output consumed by execution layer."""

    model_config = ConfigDict(extra="forbid")

    action: RiskAction
    reason_codes: list[str] = Field(default_factory=list)
    requested_size: float = Field(ge=0.0)
    approved_size: float = Field(ge=0.0)
    reduction_factor: float = Field(ge=0.0, le=1.0)
    timestamp: datetime


class DeploymentGateResult(BaseModel):
    """Adversarial gate pass/fail summary for release progression."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    reason_codes: list[str] = Field(default_factory=list)
    scenarios_total: int = Field(ge=0)
    scenarios_failed: int = Field(ge=0)
    worst_drawdown: float
    generated_at: datetime
