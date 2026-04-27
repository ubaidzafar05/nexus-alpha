"""
Archetype Definitions — Specialized trading personalities for Nexus-Alpha.
Each archetype has unique conviction thresholds, risk tolerances, and regime biases.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List
from nexus_alpha.schema_types import FusedSignal, MarketRegime, OrderSide
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

class ArchetypeType(str, Enum):
    SNIPER = "sniper"
    TACTICAL = "tactical"
    SCOUT = "scout"

@dataclass
class ArchetypeConfig:
    min_confidence: float
    max_exposure_pct: float  # Percentage of partitioned NAV to risk
    stop_loss_mult: float    # ATR multiplier
    take_profit_mult: float  # ATR multiplier
    regime_allowance: List[MarketRegime]

class BaseArchetype:
    def __init__(self, atype: ArchetypeType, config: ArchetypeConfig):
        self.atype = atype
        self.config = config

    def judge_signal(self, signal: FusedSignal, regime: MarketRegime) -> Dict[str, Any]:
        """
        Evaluate a signal based on the archetype's personality.
        Returns a dict with 'approved', 'confidence_score', and 'reasoning'.
        """
        if regime not in self.config.regime_allowance:
            return {
                "approved": False,
                "confidence_score": 0.0,
                "reasoning": f"Regime {regime} not in allowance list for {self.atype}"
            }

        if signal.confidence < self.config.min_confidence:
            return {
                "approved": False,
                "confidence_score": signal.confidence,
                "reasoning": f"Confidence {signal.confidence:.3f} below threshold {self.config.min_confidence}"
            }

        return {
            "approved": True,
            "confidence_score": signal.confidence,
            "reasoning": f"{self.atype} approved signal conviction"
        }

class SniperArchetype(BaseArchetype):
    def __init__(self):
        config = ArchetypeConfig(
            min_confidence=0.45,
            max_exposure_pct=0.20,
            stop_loss_mult=2.0,
            take_profit_mult=6.0,
            regime_allowance=[MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR, MarketRegime.CRISIS]
        )
        super().__init__(ArchetypeType.SNIPER, config)

    def judge_signal(self, signal: FusedSignal, regime: MarketRegime) -> Dict[str, Any]:
        # Sniper extra logic: Direction MUST align with the regime
        base_judgement = super().judge_signal(signal, regime)
        if not base_judgement["approved"]:
            return base_judgement

        if regime == MarketRegime.TRENDING_BULL and signal.direction < 0:
            return {"approved": False, "confidence_score": 0.0, "reasoning": "Sniper refuses to short in Bull Trend"}
        if regime == MarketRegime.TRENDING_BEAR and signal.direction > 0:
            return {"approved": False, "confidence_score": 0.0, "reasoning": "Sniper refuses to long in Bear Trend"}

        return base_judgement

class TacticalArchetype(BaseArchetype):
    def __init__(self):
        config = ArchetypeConfig(
            min_confidence=0.32,
            max_exposure_pct=0.50,
            stop_loss_mult=3.0,
            take_profit_mult=4.0,
            regime_allowance=[
                MarketRegime.TRENDING_BULL, 
                MarketRegime.TRENDING_BEAR, 
                MarketRegime.MEAN_REVERTING, 
                MarketRegime.LOW_VOLATILITY,
                MarketRegime.HIGH_VOLATILITY
            ]
        )
        super().__init__(ArchetypeType.TACTICAL, config)

class ScoutArchetype(BaseArchetype):
    def __init__(self):
        config = ArchetypeConfig(
            min_confidence=0.20,
            max_exposure_pct=0.05, # Very small size for exploration
            stop_loss_mult=4.0,    # Wide stops for noise
            take_profit_mult=2.0,    # Lower TP targets
            regime_allowance=[
                MarketRegime.MEAN_REVERTING, 
                MarketRegime.LOW_VOLATILITY, 
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.UNKNOWN
            ]
        )
        super().__init__(ArchetypeType.SCOUT, config)

def get_all_archetypes() -> List[BaseArchetype]:
    return [SniperArchetype(), TacticalArchetype(), ScoutArchetype()]
