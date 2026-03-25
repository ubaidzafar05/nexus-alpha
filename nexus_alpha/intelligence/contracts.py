"""Standardized contracts for Phase 1 intelligence outputs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from nexus_alpha.types import MarketRegime


class PredictionBand(BaseModel):
    """Distribution summary for a prediction target."""

    model_config = ConfigDict(extra="forbid")

    p02: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p98: float


class UncertaintyMetrics(BaseModel):
    """Uncertainty telemetry associated with prediction."""

    model_config = ConfigDict(extra="forbid")

    epistemic: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class CausalVerdict(BaseModel):
    """Causal validation summary required for promotion gates."""

    model_config = ConfigDict(extra="forbid")

    is_causal: bool
    effect_size: float
    p_value: float = Field(ge=0.0, le=1.0)
    information_coefficient: float
    granger_p_value: float = Field(ge=0.0, le=1.0)


class ExplainabilitySummary(BaseModel):
    """Top drivers for human/audit interpretation."""

    model_config = ConfigDict(extra="forbid")

    top_drivers: list[tuple[str, float]]


class IntelligenceOutput(BaseModel):
    """Unified output schema for prediction + regime + causality."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=3)
    timestamp: datetime
    prediction: PredictionBand
    uncertainty: UncertaintyMetrics
    regime: MarketRegime
    regime_confidence: float = Field(ge=0.0, le=1.0)
    causal: CausalVerdict
    explainability: ExplainabilitySummary
