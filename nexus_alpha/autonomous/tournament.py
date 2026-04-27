"""
Tournament Engine for NEXUS-ULTRA (v6).
Orchestrates 'Champion vs. Challenger' competitive selection using multi-axis T-Scoring.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

class TournamentEngine:
    """
    Evaluates model candidates and manages project-wide Champion status.
    Uses multi-metric T-Score to ensure structural stability and profitability.
    """
    
    def __init__(self, registry_path: str = "data/tournament/registry.json"):
        self.registry_path = Path(registry_path)
        self.weights = {
            "net_return": 0.40,
            "accuracy": 0.20,
            "profit_factor": 0.20,
            "stability": 0.20
        }
        self.promotion_buffer = 0.10 # Challenger must be 10% better in T-Score

    def compute_tscore(self, metrics: Dict[str, Any]) -> float:
        """
        Calculates the Tournament Score (T-Score) based on multi-axis metrics.
        Metrics normalized to 0.0 - 1.0 range where possible.
        """
        # 1. Net Return Score (Normalized against a 50% target for 1.0 score)
        ret = metrics.get("net_return_pct", 0.0)
        ret_score = min(max(ret / 50.0, -1.0), 1.0)
        
        # 2. Accuracy Score (Relative to 0.5 baseline)
        acc = metrics.get("accuracy", 0.5)
        acc_score = (acc - 0.5) * 4.0 # 0.5 -> 0.0, 0.75 -> 1.0
        acc_score = min(max(acc_score, -1.0), 1.0)
        
        # 3. Profit Factor Score (Gain/Loss ratio)
        # If PF is missing, we use a proxy from return/mae
        pf = metrics.get("profit_factor", 1.0)
        pf_score = (pf - 1.0) / 2.0 # 1.0 -> 0.0, 3.0 -> 1.0
        pf_score = min(max(pf_score, -1.0), 1.0)
        
        # 4. Stability Score (Based on worst window performance)
        # Worse the worst window, lower the stability
        worst_window = metrics.get("worst_window_net_pct", 0.0)
        stability_score = 1.0 + (min(worst_window, 0.0) / 20.0) # -20% worst window -> 0.0 score
        stability_score = min(max(stability_score, -1.0), 1.0)
        
        t_score = (
            ret_score * self.weights["net_return"] +
            acc_score * self.weights["accuracy"] +
            pf_score * self.weights["profit_factor"] +
            stability_score * self.weights["stability"]
        )
        
        return float(t_score)

    def run_tournament(self) -> Optional[Dict[str, Any]]:
        """
        Executes the tournament selection logic. 
        Returns the newly promoted Champion or None if no change.
        """
        if not self.registry_path.exists():
            logger.warning("tournament_registry_missing", path=str(self.registry_path))
            return None
            
        try:
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
        except Exception:
            logger.exception("tournament_registry_read_failed")
            return None

        champion = registry.get("champion")
        candidates = registry.get("candidates", [])
        
        if not champion or not candidates:
            logger.info("tournament_skipped_insufficient_competitors")
            return None
            
        champ_metrics = champion.get("metrics", {})
        champion["t_score"] = self.compute_tscore(champ_metrics)
        
        best_challenger = None
        max_challenger_score = -999.0
        
        for cand in candidates:
            cand_metrics = cand.get("metrics", {})
            cand["t_score"] = self.compute_tscore(cand_metrics)
            if cand["t_score"] > max_challenger_score:
                max_challenger_score = cand["t_score"]
                best_challenger = cand
                
        if not best_challenger:
            return None
            
        logger.info(
            "tournament_standings",
            champ_id=champion["id"],
            champ_score=f"{champion['t_score']:.4f}",
            best_challenger_id=best_challenger["id"],
            best_challenger_score=f"{best_challenger['t_score']:.4f}"
        )

        # Promotion threshold check
        if best_challenger["t_score"] > champion["t_score"] + self.promotion_buffer:
            logger.info(
                "tournament_winner_found",
                winner=best_challenger["id"],
                improvement=f"{best_challenger['t_score'] - champion['t_score']:.4f}"
            )
            
            # Archive old champion
            if "past_champions" not in registry:
                registry["past_champions"] = []
            registry["past_champions"].append(champion)
            
            # Promote new champion
            registry["champion"] = best_challenger
            # Remove from candidates
            registry["candidates"] = [c for c in candidates if c["id"] != best_challenger["id"]]
            
            # Persist update
            with open(self.registry_path, "w") as f:
                json.dump(registry, f, indent=4)
                
            return best_challenger
            
        return None
