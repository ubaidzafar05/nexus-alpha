"""
Fusion Optimization Agents — Wrappers for ensemble microstructure competition.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from nexus_alpha.agents.tournament import BaseAgent
from nexus_alpha.signals.signal_engine import SignalFusionEngine
from nexus_alpha.signals.correlation import LeadLagCorrelationTracker
from nexus_alpha.schema_types import Signal, FusedSignal
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

class FusionEnsembleAgent(BaseAgent):
    """
    Tournament-compatible agent that wraps the SignalFusionEngine.
    Allows competing weights for VPIN/OFI signals to find current market alpha.
    """

    def __init__(
        self, 
        agent_id: str | None = None,
        weight_overrides: dict[str, float] | None = None,
        symbol: str = "BTCUSDT",
        leader_id: str | None = None,
        cluster_id: str | None = None
    ):
        super().__init__(
            agent_id=agent_id, 
            agent_type="fusion-v7" if leader_id else "fusion-v6",
            cluster_id=cluster_id
        )
        self.symbol = symbol
        self.leader_id = leader_id
        self.engine = SignalFusionEngine()
        self.engine.register_defaults()
        
        # V7: Initialize cross-asset correlation tracker if a leader is assigned
        self.correlation_tracker: Optional[LeadLagCorrelationTracker] = None
        if leader_id:
            self.correlation_tracker = LeadLagCorrelationTracker(
                leader_symbol=leader_id,
                follower_symbol=symbol
            )
        
        # Apply weight overrides for ensemble diversity
        if weight_overrides:
            for regime in self.engine.regime_weights:
                for sig_name, weight in weight_overrides.items():
                    if sig_name in self.engine.regime_weights[regime]:
                        self.engine.regime_weights[regime][sig_name] = weight
        
        self.weight_overrides = weight_overrides or {}
        logger.info("fusion_agent_initialized", agent_id=self.agent_id, overrides=self.weight_overrides)

    def generate_signal(self, features: dict[str, np.ndarray]) -> Signal | None:
        """Execute core fusion logic using agent-specific weights."""
        # Convert feature dict back to DataFrame for the SignalFusionEngine
        # We assume the features provided contain a 'close' column at minimum
        try:
            # Reconstruct DataFrame from dictionary components
            df = pd.DataFrame({k: np.asarray(v).flatten() for k, v in features.items()})
            if df.empty or "close" not in df.columns:
                return None

            # Execute fusion
            fused: FusedSignal = self.engine.fuse(df, symbol=self.symbol)

            # V7: Incorporate Leadership Alpha
            leadership_score = 0.0
            if self.correlation_tracker:
                stats = self.correlation_tracker.get_correlation_stats()
                leadership_score = stats["leadership_score"]
                
                # Apply leadership boost if leader and follower are in sync (alpha > 0.6)
                if leadership_score > 0.6:
                    fused.confidence = min(fused.confidence * 1.25, 1.0)

            # Convert FusedSignal (v6 internal) to Signal (Tournament standard)
            return Signal(
                signal_id=uuid.uuid4().hex[:12],
                source=self.agent_id,
                symbol=self.symbol,
                direction=fused.direction,
                confidence=fused.confidence,
                timestamp=datetime.utcnow(),
                timeframe="adaptive",
                metadata={
                    "overrides": self.weight_overrides,
                    "guardian_prob": fused.metadata.get("guardian_probability", 0.0),
                    "regime": fused.metadata.get("regime", "unknown"),
                    "leadership_score": leadership_score,
                    "leader_id": self.leader_id,
                    "lineage_depth": self.lineage_depth,
                    "ancestor_id": self.ancestor_id
                }
            )
        except Exception as e:
            import traceback
            logger.error("fusion_agent_signal_failed", agent_id=self.agent_id, error=str(e), traceback=traceback.format_exc())
            return None

    def update(self, market_data: dict) -> None:
        """Update internal state/engine from market ticks."""
        if "tick" in market_data:
            tick = market_data["tick"]
            symbol = tick.get("symbol")
            
            # Update core engine if it's our own symbol
            if symbol == self.symbol:
                self.engine.on_market_tick(tick)
                
            # Update correlation tracker if it's relevant
            if self.correlation_tracker and symbol in [self.symbol, self.leader_id]:
                # We need VPIN and OFI from the engine to update correlation
                # For the leader, we use a temporary engine or the tracker computes its own.
                vpin = self.engine.get_microstructure_stats()["vpin_max"] if symbol == self.symbol else 0.5
                ofi = self.engine.get_microstructure_stats()["ofi_max"] if symbol == self.symbol else 0.0
                self.correlation_tracker.update(symbol, vpin, ofi)

    def get_genome(self) -> dict[str, Any]:
        """Return the current regime-weight overrides as the agent's DNA."""
        # We focus on a single regime for mutation simplicity, or expose all
        # For Phase 15, we expose the primary weights used across all regimes
        return copy.deepcopy(self.weight_overrides)

    def set_genome(self, genome: dict[str, Any]) -> None:
        """Update weights and re-initialize engine with mutated DNA."""
        self.weight_overrides = genome
        # Update engine weights
        for regime in self.engine.regime_weights:
            for sig_name, weight in genome.items():
                if sig_name in self.engine.regime_weights[regime]:
                    self.engine.regime_weights[regime][sig_name] = weight
        logger.info("fusion_agent_genome_updated", agent_id=self.agent_id, new_dna=genome)
