from __future__ import annotations

import numpy as np
import pandas as pd

from nexus_alpha.agents.lifecycle import StrategyAgentLifecycle
from nexus_alpha.strategy.evolution import TERMINALS


def _build_feature_frame(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({name: rng.normal(0, 1, n) for name in TERMINALS})
    frame["returns_1h"] = frame["close"].diff().fillna(0.0)
    frame["returns_4h"] = frame["close"].diff(4).fillna(0.0)
    frame["returns_1d"] = frame["close"].diff(24).fillna(0.0)
    return frame


def test_strategy_agent_lifecycle_evolve_step_and_promote() -> None:
    frame = _build_feature_frame()
    forward = pd.Series(frame["returns_1h"].shift(-1).fillna(0.0), name="forward")

    lifecycle = StrategyAgentLifecycle()
    lifecycle.bootstrap()
    registered = lifecycle.evolve_and_register(frame, forward_returns=forward, n_top=3)
    assert isinstance(registered, list)
    assert len(lifecycle.tournament.agents) >= 1

    features = {col: frame[col].values[-120:] for col in frame.columns}
    step = lifecycle.run_step(features=features, market_price=65000.0)
    assert step.generated_signals >= step.validated_signals
    if step.combined_signal is not None:
        assert step.combined_signal.symbol == "BTCUSDT"

    promoted = lifecycle.promotion_candidates(min_calmar=-10.0, min_trades=0)
    assert isinstance(promoted, list)
