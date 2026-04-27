"""
Signal Optimizer — Hyperparameter optimization using Optuna.

Finds the most profitable SignalFusionEngine weights and ML parameters
by maximizing walk-forward performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import optuna
except ImportError:
    optuna = None

from nexus_alpha.learning.historical_data import load_ohlcv
from nexus_alpha.learning.walk_forward import run_walk_forward_df
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

OPTIMIZATION_RESULTS_DIR = Path("data/optimization")


class SignalOptimizer:
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        n_trials: int = 50,
    ):
        if optuna is None:
            raise ImportError("Optuna is required for SignalOptimizer. Install with 'pip install optuna'")
            
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.df = load_ohlcv(symbol, timeframe)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        Returns the Net Return (or Sortino) for a set of hyperparameters.
        """
        # 1. Suggest parameters
        n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100)
        max_depth = trial.suggest_int("max_depth", 3, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        min_confidence = trial.suggest_float("min_confidence", 0.1, 0.4)

        try:
            # 2. Run walk-forward evaluation
            summary = run_walk_forward_df(
                self.df,
                symbol=self.symbol,
                timeframe=self.timeframe,
                train_bars=1500,
                test_bars=250,
                step_bars=250,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_confidence=min_confidence,
            )

            # 3. Return metric to maximize
            # Use Net Return as the primary goal, but could be Sharpe/Sortino
            score = summary.total_net_return_pct
            
            # Penalize very low coverage (don't want a bot that trades once a year)
            if summary.traded_coverage < 0.05:
                score -= 5.0
                
            return float(score)

        except Exception as e:
            logger.warning("optimization_trial_failed", error=str(e))
            return -999.0

    def run(self) -> dict[str, Any]:
        """Run the optimization study."""
        OPTIMIZATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        study_name = f"nexus_opt_{self.symbol.replace('/', '_')}_{self.timeframe}"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=f"sqlite:///{OPTIMIZATION_RESULTS_DIR / 'optuna.db'}",
            load_if_exists=True,
        )
        
        logger.info("optimization_started", symbol=self.symbol, n_trials=self.n_trials)
        study.optimize(self.objective, n_trials=self.n_trials)
        
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info("optimization_complete", best_value=best_value, best_params=best_params)
        
        # Save results
        results_path = OPTIMIZATION_RESULTS_DIR / f"best_params_{self.symbol.replace('/', '_')}.json"
        with open(results_path, "w") as f:
            json.dump({
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "best_value": best_value,
                "best_params": best_params,
                "optimized_at": pd.Timestamp.now().isoformat(),
            }, f, indent=2)
            
        return best_params


if __name__ == "__main__":
    # Example usage:
    # python -m nexus_alpha.learning.signal_optimizer
    import asyncio
    from nexus_alpha.config import load_config
    
    optimizer = SignalOptimizer(n_trials=10)
    best = optimizer.run()
    print(f"Optimal Parameters: {best}")
