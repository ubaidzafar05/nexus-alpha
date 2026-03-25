"""Phase 2 lifecycle: strategy evolution, tournament simulation, and promotion."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from nexus_alpha.agents.tournament import BaseAgent, TournamentOrchestrator
from nexus_alpha.config import NexusConfig
from nexus_alpha.logging import get_logger
from nexus_alpha.signals.contracts import SignalCandidate, ValidatedSignal
from nexus_alpha.strategy.evolution import DiscoveredStrategy, StrategyEvolutionEngine
from nexus_alpha.types import Signal

logger = get_logger(__name__)


@dataclass(frozen=True)
class LifecycleStepResult:
    generated_signals: int
    validated_signals: int
    combined_signal: ValidatedSignal | None


class EvolvedStrategyAgent(BaseAgent):
    """Agent wrapper around an evolved strategy expression tree."""

    def __init__(self, strategy: DiscoveredStrategy, symbol: str = "BTCUSDT"):
        super().__init__(agent_type="gp-evolved")
        self.strategy = strategy
        self.symbol = symbol

    def generate_signal(self, features: dict[str, np.ndarray]) -> Signal | None:
        frame = pd.DataFrame({k: np.asarray(v).reshape(-1) for k, v in features.items()})
        if len(frame) == 0:
            return None
        values = self.strategy.gene.evaluate(frame)
        direction = float(np.clip(values[-1], -1.0, 1.0))
        confidence = float(min(abs(direction), 1.0))
        if confidence < 0.05:
            return None
        return Signal(
            signal_id=uuid.uuid4().hex[:12],
            source=self.agent_id,
            symbol=self.symbol,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            timeframe="adaptive",
            metadata={"strategy_id": self.strategy.strategy_id},
        )

    def update(self, market_data: dict) -> None:
        return None


class HeuristicRLProxyAgent(BaseAgent):
    """Torch-free RL proxy to keep the lifecycle executable in constrained envs."""

    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(agent_type="rl-proxy")
        self.symbol = symbol

    def generate_signal(self, features: dict[str, np.ndarray]) -> Signal | None:
        if "returns_1h" not in features:
            return None
        returns = np.asarray(features["returns_1h"]).reshape(-1)
        if returns.size == 0:
            return None
        momentum = float(np.tanh(returns[-1] * 15))
        if abs(momentum) < 0.05:
            return None
        return Signal(
            signal_id=uuid.uuid4().hex[:12],
            source=self.agent_id,
            symbol=self.symbol,
            direction=momentum,
            confidence=min(abs(momentum), 1.0),
            timestamp=datetime.utcnow(),
            timeframe="adaptive",
            metadata={"proxy": True},
        )

    def update(self, market_data: dict) -> None:
        return None


class StrategyAgentLifecycle:
    """End-to-end lifecycle from evolution to promotion decisions."""

    def __init__(self, config: NexusConfig | None = None):
        self._config = config or NexusConfig()
        self._tournament = TournamentOrchestrator(self._config.tournament)
        self._evolution = StrategyEvolutionEngine(
            population_size=60,
            generations=10,
            max_depth=4,
        )

    def bootstrap(self) -> None:
        """Initialize tournament with at least one RL-style policy agent."""
        self._tournament.register_agent(HeuristicRLProxyAgent())

    def evolve_and_register(
        self,
        feature_matrix: pd.DataFrame,
        forward_returns: pd.Series,
        n_top: int = 5,
    ) -> list[str]:
        discovered = self._evolution.evolve(feature_matrix, forward_returns, n_top=n_top)
        registered: list[str] = []
        for strategy in discovered:
            agent = EvolvedStrategyAgent(strategy=strategy)
            self._tournament.register_agent(agent)
            registered.append(agent.agent_id)
        return registered

    def run_step(
        self,
        features: dict[str, np.ndarray],
        market_price: float,
    ) -> LifecycleStepResult:
        generated = self._collect_signal_candidates(features)
        validated = [self._validate_candidate(sig) for sig in generated]
        approved = [sig for sig in validated if not sig.rejected_reasons]
        self._simulate_agent_portfolios(features, market_price=market_price)

        combined = self._tournament.get_combined_signal(features)
        combined_validated = (
            self._validate_candidate(self._to_candidate(combined))
            if combined
            else None
        )
        return LifecycleStepResult(
            generated_signals=len(generated),
            validated_signals=len(approved),
            combined_signal=combined_validated,
        )

    def promotion_candidates(
        self,
        min_calmar: float = 0.3,
        min_trades: int = 5,
    ) -> list[str]:
        performance = self._tournament.evaluate_all()
        promoted = []
        for agent_id, perf in performance.items():
            if perf.calmar_ratio >= min_calmar and perf.total_trades >= min_trades:
                promoted.append(agent_id)
        return promoted

    def _collect_signal_candidates(self, features: dict[str, np.ndarray]) -> list[SignalCandidate]:
        candidates: list[SignalCandidate] = []
        for agent in self._tournament.agents.values():
            if not agent.is_active:
                continue
            signal = agent.generate_signal(features)
            if signal is None:
                continue
            candidates.append(self._to_candidate(signal))
        return candidates

    def _to_candidate(self, signal: Signal | None) -> SignalCandidate:
        if signal is None:
            raise ValueError("signal_required")
        return SignalCandidate(
            signal_id=signal.signal_id,
            source=signal.source,
            symbol=signal.symbol,
            direction=signal.direction,
            confidence=signal.confidence,
            timestamp=signal.timestamp,
            features_used=signal.features_used,
        )

    def _validate_candidate(self, candidate: SignalCandidate) -> ValidatedSignal:
        reasons: list[str] = []
        if abs(candidate.direction) < 0.05:
            reasons.append("weak_direction")
        if candidate.confidence < 0.1:
            reasons.append("low_confidence")
        causal_effect = candidate.direction * candidate.confidence
        validation_score = max(0.0, min(abs(causal_effect), 1.0))
        return ValidatedSignal(
            signal_id=candidate.signal_id,
            source=candidate.source,
            symbol=candidate.symbol,
            direction=candidate.direction,
            confidence=candidate.confidence,
            causal_validated=len(reasons) == 0,
            causal_effect=causal_effect,
            validation_score=validation_score,
            rejected_reasons=reasons,
            timestamp=candidate.timestamp,
        )

    def _simulate_agent_portfolios(
        self,
        features: dict[str, np.ndarray],
        market_price: float,
    ) -> None:
        if market_price <= 0:
            return
        for agent_id, agent in self._tournament.agents.items():
            portfolio = self._tournament.portfolios[agent_id]
            signal = agent.generate_signal(features)
            if signal is None:
                portfolio.update_mark_to_market({"BTCUSDT": market_price})
                continue
            existing = portfolio.positions.get(signal.symbol)
            if existing is None:
                size_pct = max(0.01, min(self._tournament.capital_weights.get(agent_id, 0.1), 0.2))
                portfolio.open_position(signal, price=market_price, size_pct=size_pct)
            elif (existing.side == "buy" and signal.direction < 0) or (
                existing.side == "sell" and signal.direction > 0
            ):
                portfolio.close_position(signal.symbol, price=market_price)
            portfolio.update_mark_to_market({signal.symbol: market_price})

    @property
    def tournament(self) -> TournamentOrchestrator:
        return self._tournament
