"""Signal contracts for strategy/agent lifecycle promotion flow."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SignalCandidate(BaseModel):
    """Raw candidate signal before validation gates."""

    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(min_length=6)
    source: str = Field(min_length=2)
    symbol: str = Field(min_length=3)
    direction: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    features_used: list[str] = Field(default_factory=list)


class ValidatedSignal(BaseModel):
    """Candidate after causal/risk quality checks."""

    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(min_length=6)
    source: str = Field(min_length=2)
    symbol: str = Field(min_length=3)
    direction: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    causal_validated: bool
    causal_effect: float
    validation_score: float = Field(ge=0.0, le=1.0)
    rejected_reasons: list[str] = Field(default_factory=list)
    timestamp: datetime
