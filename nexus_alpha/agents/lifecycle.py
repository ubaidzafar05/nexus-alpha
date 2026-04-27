"""Phase 2 lifecycle: strategy evolution, tournament simulation, and promotion."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from nexus_alpha.agents.tournament import BaseAgent, TournamentOrchestrator
from nexus_alpha.config import NexusConfig
from nexus_alpha.log_config import get_logger
from nexus_alpha.signals.contracts import SignalCandidate, ValidatedSignal
from nexus_alpha.strategy.evolution import DiscoveredStrategy, StrategyEvolutionEngine
from nexus_alpha.agents.evolution import EvolutionaryKernel, RecursiveSpawner
from nexus_alpha.schema_types import Signal

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
        self._kernel = EvolutionaryKernel(mutation_rate=0.05)
        self._spawner = RecursiveSpawner(self._kernel, max_agents=20)

    def bootstrap(self) -> None:
        """Initialize tournament with a diverse population of microstructure agents."""
        from nexus_alpha.agents.optimizer import FusionEnsembleAgent
        from nexus_alpha.agents.risk import TailHedgeAgent
        
        # 1. Baseline Champion (Standard Weights)
        self._tournament.register_agent(FusionEnsembleAgent(agent_id="fusion-champion"))
        
        # 2. VPIN Specialist (Biased towards Informed Trading detection)
        self._tournament.register_agent(
            FusionEnsembleAgent(
                agent_id="v6-vpin-aggro",
                weight_overrides={"vpin": 3.0, "vpin_advanced": 3.0, "ofi_l2": 1.0}
            )
        )
        
        # 3. Order Flow Specialist (Biased towards immediate imbalance)
        self._tournament.register_agent(
            FusionEnsembleAgent(
                agent_id="v6-ofi-aggro",
                weight_overrides={"ofi_l2": 3.5, "vpin": 1.0}
            )
        )
        
        # 4. Balanced Micro-Alpha
        self._tournament.register_agent(
            FusionEnsembleAgent(
                agent_id="v6-micro-fusion",
                weight_overrides={"vpin": 1.5, "ofi_l2": 1.5, "rsi_7": 0.5}
            )
        )
        
        # 5. Volatility / Mean-Reversion Hybrid
        self._tournament.register_agent(
            FusionEnsembleAgent(
                agent_id="v6-vol-alpha",
                symbol="BTCUSDT",
                cluster_id="layer1",
                weight_overrides={"bollinger_low": 2.0, "bollinger_high": 2.0, "vpin": 2.0}
            )
        )
        
        # Keep the proxy for safety/benchmarking
        self._tournament.register_agent(HeuristicRLProxyAgent())

        # ── V7 ULTRA: Leadership Agents (Phase 11) ────────────────────────
        # ETHUSDT following BTCUSDT
        self._tournament.register_agent(
            FusionEnsembleAgent(
                agent_id="v7-eth-btc-leader",
                symbol="ETHUSDT",
                leader_id="BTCUSDT",
                cluster_id="layer1",
                weight_overrides={"vpin": 2.0, "ofi_l2": 2.0}
            )
        )
        
        # SOLUSDT following BTCUSDT
        self._tournament.register_agent(
            FusionEnsembleAgent(
                agent_id="v7-sol-btc-leader",
                symbol="SOLUSDT",
                leader_id="BTCUSDT",
                cluster_id="layer1",
                weight_overrides={"vpin": 2.0, "ofi_l2": 2.0}
            )
        )

    def evolve_swarm(self) -> int:
        """
        Trigger the Evolutionary Kernel to refine the agent swarm DNA.
        """
        performance = self._tournament.evaluate_all()
        agents = list(self._tournament.agents.values())
        mutations = self._kernel.evolve_swarm(agents, performance)
        
        if mutations > 0:
            logger.info("swarm_evolution_complete", mutations=mutations)
        return mutations

    def bootstrap_regime_variants(self) -> int:
        """
        Trigger the Recursive Spawner to expand the ensemble for a new regime.
        Clones the top-performing agent.
        """
        active_agents = len(self._tournament.agents)
        if active_agents >= self._spawner.max_agents:
            logger.warning("bootstrap_ignored", reason="population_cap_reached")
            return 0
            
        performance = self._tournament.evaluate_all()
        if not performance:
            return 0
            
        # Identify the current Champion
        champion_id = max(performance.items(), key=lambda x: x[1].sharpe_ratio)[0]
        champion = self._tournament.agents[champion_id]
        
        # Spawn 2 variants
        new_variants = self._spawner.bootstrap_variations(champion, count=2)
        
        for variant in new_variants:
            self._tournament.register_agent(variant)
            
        if new_variants:
            self._tournament.rebalance_capital()
            logger.info("recursive_bootstrap_completed", n_spawned=len(new_variants))
            
        return len(new_variants)

    def symmetrize_cluster(self, cluster_id: str) -> bool:
        """
        Autonomously bootstrap a Hedge agent to neutralize directional crowding.
        V9 Phase 19 logic.
        """
        performance = self._tournament.evaluate_all()
        if not performance:
            return False
            
        # Identify the Champion within the cluster to use as base for the hedge
        cluster_agents = [aid for aid, a in self._tournament.agents.items() if a.cluster_id == cluster_id]
        if not cluster_agents:
            return False
            
        # Get top performer from cluster
        champion_id = max(cluster_agents, key=lambda aid: performance.get(aid, 0.0).sharpe_ratio)
        champion = self._tournament.agents[champion_id]
        
        hedge = self._spawner.spawn_hedge_agent(champion, cluster_id)
        if hedge:
            self._tournament.register_agent(hedge)
            logger.info("autonomic_symmetrization_triggered", cluster=cluster_id, hedge_id=hedge.agent_id)
            return True
        return False

        # ── V7 ULTRA: Risk Guardians (Phase 12) ──────────────────────────
        # Global Tail Protector
        self._tournament.register_agent(
            TailHedgeAgent(agent_id="guardian-v7-main", z_threshold=3.5)
        )

    def update(self, market_data: dict) -> None:
        """Propagate market updates to the tournament ensemble."""
        self._tournament.update_agents(market_data)

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
