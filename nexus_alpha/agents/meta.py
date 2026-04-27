"""
Meta-Orchestration — Self-calibration and hyper-parameter evolution.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from nexus_alpha.log_config import get_logger
from nexus_alpha.schema_types import AgentPerformance

logger = get_logger(__name__)

class MetaOrchestrator:
    """
    Final Horizon Agent: Monitors the system's calibration and self-tunes parameters.
    Adjusts TournamentConfig based on regime shifts and realized PnL.
    """
    
    def __init__(self, orchestrator: Any, personality_weights: Dict[str, float] | None = None):
        self.orchestrator = orchestrator # TournamentOrchestrator
        self.personality_weights = personality_weights or {}
        self.calibration_errors: List[float] = []
        self._last_meta_evolution = datetime.utcnow()
        
        # Stability Anchors (Safe Bounds)
        self.bounds = {
            "cull_bottom_pct": (0.05, 0.40),
            "z_threshold": (2.0, 5.0)
        }
        
        logger.info("meta_orchestrator_sentience_initialized")

    def calibrate(self, current_regime: str) -> None:
        """
        Main meta-calibration loop. Executed after tournament evaluations.
        Adjusts global hyper-parameters based on regime-specific performance.
        """
        now = datetime.utcnow()
        if (now - self._last_meta_evolution).total_seconds() < 3600: # Every hour max
            return
            
        performance = self.orchestrator.evaluate_all()
        if not performance:
            return
            
        # 1. System-wide calibration: Total PnL vs volatility
        avg_sharpe = np.mean([p.sharpe_ratio for p in performance.values()])
        avg_drawdown = np.mean([p.max_drawdown for p in performance.values()])
        
        # 2. Regime-Adaptive Policy Drift
        config = self.orchestrator.config
        drift_reason = "stability_maintenance"
        
        if current_regime in ["crisis", "trending_down"]:
            # Crisis Mode: Tighten culling and increase risk sensitivity
            drift_reason = "crisis_tightening"
            config.cull_bottom_pct = min(config.cull_bottom_pct * 1.1, self.bounds["cull_bottom_pct"][1])
            # We would also signal risk agents here to decrease thresholds
        
        elif current_regime in ["trending_up"]:
            # Bullish Momentum: Allow more experimentation
            drift_reason = "momentum_alpha_expansion"
            config.cull_bottom_pct = max(config.cull_bottom_pct * 0.9, self.bounds["cull_bottom_pct"][0])
            
        # 3. Apply Anchor Sanity Checks
        config.cull_bottom_pct = float(np.clip(
            config.cull_bottom_pct, 
            self.bounds["cull_bottom_pct"][0], 
            self.bounds["cull_bottom_pct"][1]
        ))
        
        self._last_meta_evolution = now
        logger.info(
            "meta_calibration_drift_applied",
            reason=drift_reason,
            regime=current_regime,
            new_cull_pct=f"{config.cull_bottom_pct:.3f}",
            avg_sharpe=f"{avg_sharpe:.3f}"
        )
        
        self.calibrate_personalities(current_regime)

    def calibrate_personalities(self, regime: str) -> None:
        """Shift power weights between Sniper, Tactical, and Scout archetypes."""
        if not self.personality_weights:
            return

        total_nav = sum(self.personality_weights.values())
        if total_nav <= 0: return

        # Target allocations based on regime
        targets = {
            "sniper": 0.20,
            "tactical": 0.50,
            "scout": 0.30
        }

        if regime in ["crisis", "high_volatility"]:
            targets = {"sniper": 0.70, "tactical": 0.30, "scout": 0.00}
        elif regime in ["strong_trend"]:
            targets = {"sniper": 0.30, "tactical": 0.50, "scout": 0.20}
        elif regime == "range_bound":
            targets = {"sniper": 0.10, "tactical": 0.40, "scout": 0.50}

        # Apply Smooth Rebalance
        for key in targets:
            if key in self.personality_weights:
                current = self.personality_weights[key]
                target = targets[key] * total_nav
                # Move 5% of the distance toward target each calibration
                self.personality_weights[key] = current + (target - current) * 0.05
        
        logger.info("meta_personality_rebalance", weights={k: f"${v:,.2f}" for k, v in self.personality_weights.items()})

    def compute_calibration_error(self, ensemble_confidence: float, realized_return: float) -> float:
        """Track how well ensemble confidence maps to actual price movement magnitude."""
        if ensemble_confidence <= 0:
            return 0.0
        # Ideal: High confidence -> High return magnitude
        error = abs(ensemble_confidence - min(abs(realized_return) * 100, 1.0))
        self.calibration_errors.append(error)
        return error
