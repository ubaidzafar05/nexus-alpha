"""
Tournament Engine — Model competition and automated promotion.

Manages multiple model 'candidates' and promotes the most profitable
to 'Champion' status based on walk-forward performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus_alpha.learning.walk_forward import WalkForwardSummary, run_walk_forward
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

TOURNAMENT_DIR = Path("data/tournament")


@dataclass
class Candidate:
    id: str
    symbol: str
    timeframe: str
    params: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    is_champion: bool = False
    created_at: str = ""


class TournamentEngine:
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.registry_path = TOURNAMENT_DIR / f"registry_{symbol.replace('/', '_')}.json"
        self._load_registry()

    def _load_registry(self):
        TOURNAMENT_DIR.mkdir(parents=True, exist_ok=True)
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                self.candidates = [Candidate(**c) for c in data]
        else:
            self.candidates = []

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump([c.__dict__ for c in self.candidates], f, indent=2)

    def register_candidate(self, params: dict[str, Any]) -> str:
        """Register a new candidate model for the tournament."""
        cid = f"cand_{len(self.candidates):03d}"
        candidate = Candidate(
            id=cid,
            symbol=self.symbol,
            timeframe=self.timeframe,
            params=params,
            created_at=pd.Timestamp.now().isoformat()
        )
        self.candidates.append(candidate)
        self._save_registry()
        logger.info("candidate_registered", id=cid)
        return cid

    def evaluate_candidates(self) -> dict[str, WalkForwardSummary]:
        """Run walk-forward evaluation for all candidates."""
        results = {}
        for cand in self.candidates:
            logger.info("evaluating_candidate", id=cand.id)
            summary = run_walk_forward(
                symbol=cand.symbol,
                timeframe=cand.timeframe,
                **cand.params
            )
            cand.metrics = {
                "net_return_pct": summary.total_net_return_pct,
                "direction_accuracy": summary.avg_direction_accuracy,
                "coverage": summary.traded_coverage,
                "mae": summary.avg_mae,
            }
            results[cand.id] = summary
            
        self._save_registry()
        return results

    def promote_new_champion(self) -> str | None:
        """
        Determine if any Challenger should be promoted to Champion.
        Criteria: High Sortino Ratio + High Direction Accuracy + > 5% coverage.
        """
        if not self.candidates:
            return None
            
        best_cand = None
        best_score = -999.0
        
        for cand in self.candidates:
            # Health check: must have sufficient data
            metrics = cand.metrics
            if metrics.get("coverage", 0) < 0.05:
                continue
            
            # Risk-adjusted return proxy: Sortino-style score
            # (net_return / abs(mae)) * accuracy
            # Encourages high return, low error, and high directional hit rate
            net_ret = metrics.get("net_return_pct", 0)
            accuracy = metrics.get("direction_accuracy", 0.5)
            mae = metrics.get("mae", 1.0)
            
            score = (net_ret / (mae + 1e-6)) * accuracy
            
            if score > best_score:
                best_score = score
                best_cand = cand
                
        if best_cand and best_score > 0: # Only promote if score is positive
            # Demote current champion
            for cand in self.candidates:
                cand.is_champion = False
                
            best_cand.is_champion = True
            self._save_registry()
            logger.info("new_champion_promoted", id=best_cand.id, score=f"{best_score:.4f}")
            return best_cand.id
            
        return None


if __name__ == "__main__":
    import pandas as pd # For Timestamp
    engine = TournamentEngine()
    # Mock registration
    cid = engine.register_candidate({"n_estimators": 500, "max_depth": 5})
    engine.evaluate_candidates()
    engine.promote_new_champion()
