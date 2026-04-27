"""
Agent Tournament — Living tournament of competing autonomous agents.

Each agent independently manages a paper-traded sub-portfolio.
Capital allocation is determined by rolling tournament performance.
The market naturally selects the best strategies.

Tournament rules:
- Allocates virtual capital to agents (starts equal)
- Measures Calmar ratio over rolling 60-day window
- Kills bottom 20% performers monthly
- Spawns new candidate agents via strategy evolution
- Graduated live capital: paper → 0.5% NAV → 5% → 20%
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np

from nexus_alpha.config import TournamentConfig
from nexus_alpha.log_config import get_logger
from nexus_alpha.schema_types import AgentPerformance, Portfolio, Signal

logger = get_logger(__name__)


# ─── Base Agent Interface ─────────────────────────────────────────────────────


class BaseAgent(ABC):
    """Abstract base for all tournament agents."""

    def __init__(self, agent_id: str | None = None, agent_type: str = "base", cluster_id: str | None = None):
        self.agent_id = agent_id or f"{agent_type}-{uuid.uuid4().hex[:8]}"
        self.agent_type = agent_type
        self.created_at = datetime.utcnow()
        self.is_active = True
        
        # V8: Swarm Genealogy (Phase 17)
        self.lineage_depth = 0
        self.ancestor_id: str = self.agent_id
        
        # V9: Portfolio Symmetry (Phase 19)
        self.cluster_id: str | None = cluster_id
        
        self.metadata: dict[str, Any] = {}

    @abstractmethod
    def generate_signal(self, features: dict[str, np.ndarray]) -> Signal | None:
        """Generate a trading signal from current features."""
        ...

    @abstractmethod
    def update(self, market_data: dict) -> None:
        """Update internal state with new market data."""
        ...

    def get_genome(self) -> dict[str, Any]:
        """Return a dictionary of mutable hyper-parameters (Agent DNA)."""
        return {}

    def set_genome(self, genome: dict[str, Any]) -> None:
        """Update internal hyper-parameters from a new genome."""
        pass


# ─── Paper Trading Tracker ────────────────────────────────────────────────────


@dataclass
class PaperTrade:
    """A simulated trade for an agent."""
    trade_id: str
    agent_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float = 0.0
    is_open: bool = True


class PaperPortfolio:
    """Track a paper-traded portfolio for one agent."""

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, PaperTrade] = {}
        self.closed_trades: list[PaperTrade] = []
        self.nav_history: list[tuple[datetime, float]] = [(datetime.utcnow(), initial_capital)]
        self.peak_nav = initial_capital

    @property
    def nav(self) -> float:
        open_pnl = sum(t.pnl for t in self.positions.values())
        return self.cash + open_pnl

    @property
    def max_drawdown(self) -> float:
        if self.peak_nav <= 0:
            return 0.0
        return (self.peak_nav - self.nav) / self.peak_nav

    def record_nav(self) -> None:
        current_nav = self.nav
        self.nav_history.append((datetime.utcnow(), current_nav))
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav

    def open_position(self, signal: Signal, price: float, size_pct: float = 0.05) -> PaperTrade | None:
        """Open a paper position based on signal."""
        if signal.symbol in self.positions:
            return None  # Already have a position

        notional = self.cash * size_pct
        quantity = notional / price if price > 0 else 0
        if quantity <= 0:
            return None

        trade = PaperTrade(
            trade_id=uuid.uuid4().hex[:12],
            agent_id=signal.source,
            symbol=signal.symbol,
            side="buy" if signal.direction > 0 else "sell",
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.utcnow(),
        )
        self.positions[signal.symbol] = trade
        self.cash -= notional
        return trade

    def close_position(self, symbol: str, price: float) -> PaperTrade | None:
        """Close a paper position."""
        if symbol not in self.positions:
            return None

        trade = self.positions.pop(symbol)
        if trade.side == "buy":
            trade.pnl = (price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - price) * trade.quantity

        trade.exit_price = price
        trade.exit_time = datetime.utcnow()
        trade.is_open = False
        self.cash += trade.entry_price * trade.quantity + trade.pnl
        self.closed_trades.append(trade)
        self.record_nav()
        return trade

    def update_mark_to_market(self, prices: dict[str, float]) -> None:
        """Update unrealized PnL for all open positions."""
        for symbol, trade in self.positions.items():
            if symbol in prices:
                price = prices[symbol]
                if trade.side == "buy":
                    trade.pnl = (price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - price) * trade.quantity
        self.record_nav()


# ─── Performance Calculator ──────────────────────────────────────────────────


def compute_agent_performance(
    agent_id: str,
    portfolio: PaperPortfolio,
    window_days: int = 60,
) -> AgentPerformance:
    """Compute rolling performance metrics for a tournament agent."""
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    nav_series = [(t, v) for t, v in portfolio.nav_history if t >= cutoff]

    if len(nav_series) < 2:
        return AgentPerformance(
            agent_id=agent_id,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_win_loss_ratio=0.0,
            total_trades=0,
            pnl=0.0,
            evaluation_window_days=window_days,
        )

    navs = np.array([v for _, v in nav_series])
    returns = np.diff(navs) / navs[:-1]
    returns = returns[np.isfinite(returns)]

    # Sharpe (annualized, assume daily)
    if len(returns) > 1 and np.std(returns) > 1e-10:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(navs)
    drawdowns = (peak - navs) / (peak + 1e-10)
    max_dd = float(np.max(drawdowns))

    # Calmar ratio
    annual_return = (navs[-1] / navs[0]) ** (252 / max(len(navs), 1)) - 1 if navs[0] > 0 else 0
    calmar = annual_return / max_dd if max_dd > 0 else 0.0

    # Trade statistics
    recent_trades = [t for t in portfolio.closed_trades if t.exit_time and t.exit_time >= cutoff]
    wins = [t for t in recent_trades if t.pnl > 0]
    losses = [t for t in recent_trades if t.pnl <= 0]
    win_rate = len(wins) / len(recent_trades) if recent_trades else 0.0
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
    avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return AgentPerformance(
        agent_id=agent_id,
        sharpe_ratio=float(sharpe),
        calmar_ratio=float(calmar),
        max_drawdown=float(max_dd),
        win_rate=float(win_rate),
        avg_win_loss_ratio=float(win_loss_ratio),
        total_trades=len(recent_trades),
        pnl=float(sum(t.pnl for t in recent_trades)),
        evaluation_window_days=window_days,
    )


# ─── Tournament Orchestrator ─────────────────────────────────────────────────


class TournamentOrchestrator:
    """
    Manages the living tournament of competing agents.
    Allocates capital based on rolling performance.
    Culls bottom performers and spawns new candidates.
    """

    def __init__(self, config: TournamentConfig | None = None):
        self.config = config or TournamentConfig()
        self.agents: dict[str, BaseAgent] = {}
        self.portfolios: dict[str, PaperPortfolio] = {}
        self.capital_weights: dict[str, float] = {}
        self.performance_history: dict[str, list[AgentPerformance]] = defaultdict(list)
        self._last_cull = datetime.utcnow()
        self._last_signals: dict[str, Signal] = {} # V9: Cache for cluster delta optimization

        logger.info("tournament_initialized", config=self.config.model_dump())

    def register_agent(self, agent: BaseAgent, initial_capital: float = 100_000.0) -> None:
        """Register a new agent in the tournament."""
        self.agents[agent.agent_id] = agent
        self.portfolios[agent.agent_id] = PaperPortfolio(initial_capital=initial_capital)
        self.capital_weights[agent.agent_id] = 1.0 / max(len(self.agents), 1)

        logger.info(
            "agent_registered",
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            total_agents=len(self.agents),
        )

    def update_agents(self, market_data: dict) -> None:
        """Broadcast market updates (ticks/OHLCV) to all competing agents."""
        for agent in self.agents.values():
            if agent.is_active:
                try:
                    agent.update(market_data)
                except Exception as e:
                    logger.warning("agent_update_failed", agent_id=agent.agent_id, error=str(e))

    def evaluate_all(self) -> dict[str, AgentPerformance]:
        """Evaluate all agents and return performance metrics."""
        results = {}
        for agent_id, portfolio in self.portfolios.items():
            perf = compute_agent_performance(
                agent_id=agent_id,
                portfolio=portfolio,
                window_days=self.config.rolling_window_days,
            )
            results[agent_id] = perf
            self.performance_history[agent_id].append(perf)

        return results

    def rebalance_capital(self, min_total_trades: int = 5) -> dict[str, float]:
        """
        Reallocate capital weights based on Calmar ratio ranking.
        Top performers get more capital, bottom performers get less.
        """
        performance = self.evaluate_all()
        if not performance:
            return self.capital_weights
            
        # V6 ULTRA: Warm-up guard — don't rebalance if trade history is too sparse
        total_recorded_trades = sum(p.total_trades for p in performance.values())
        if total_recorded_trades < min_total_trades:
            return self.capital_weights

        # Rank by Calmar ratio
        ranked = sorted(performance.items(), key=lambda x: x[1].calmar_ratio, reverse=True)

        # Exponential weighting: top agents get exponentially more
        n = len(ranked)
        raw_weights = np.exp(np.linspace(1, 0, n))
        normalized = raw_weights / raw_weights.sum()

        self.capital_weights = {
            agent_id: float(weight)
            for (agent_id, _), weight in zip(ranked, normalized)
        }

        logger.info(
            "capital_rebalanced",
            weights={k: f"{v:.3f}" for k, v in list(self.capital_weights.items())[:5]},
        )

        return self.capital_weights

    def get_cluster_delta(self, cluster_id: str, features: dict[str, np.ndarray]) -> float:
        """
        Calculate the capital-weighted net direction (Delta) for a correlation cluster.
        V9 (Phase 19): Used for Crowding Detection and Symmetrization.
        """
        cluster_agents = [
            (self.capital_weights.get(aid, 0.0), a) 
            for aid, a in self.agents.items() 
            if a.is_active and a.cluster_id == cluster_id
        ]
        
        if not cluster_agents:
            return 0.0
            
        total_weight = sum(w for w, _ in cluster_agents)
        if total_weight <= 0:
            return 0.0
            
        net_delta = 0.0
        for weight, agent in cluster_agents:
            try:
                # V9 Phase 19 Performance Optimization:
                # Use the cached signal from the most recent get_combined_signal pass
                # instead of re-triggering inference for every agent in the cluster.
                sig = self._last_signals.get(agent.agent_id)
                if not sig:
                    sig = agent.generate_signal(features)
                    
                if sig:
                    net_delta += (weight / total_weight) * sig.direction
            except Exception:
                continue
                
        return net_delta

    def save_swarm_state(self, path: Path | str = "data/tournament/swarm_registry.json") -> bool:
        """
        Serialize current swarm state for dashboard visualization and hot-reloads.
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            swarm_data = []
            for agent_id, agent in self.agents.items():
                portfolio = self.portfolios.get(agent_id)
                perf = compute_agent_performance(agent_id, portfolio) if portfolio else None
                
                swarm_data.append({
                    "agent_id": agent_id,
                    "agent_type": agent.agent_type,
                    "lineage_depth": getattr(agent, "lineage_depth", 0),
                    "ancestor_id": getattr(agent, "ancestor_id", agent_id),
                    "cluster_id": getattr(agent, "cluster_id", "unassigned"),
                    "is_active": agent.is_active,
                    "capital_weight": round(self.capital_weights.get(agent_id, 0.0), 4),
                    "metrics": {
                        "sharpe": round(perf.sharpe_ratio, 2) if perf else 0.0,
                        "pnl": round(perf.pnl, 2) if perf else 0.0,
                    } if perf else {},
                    "is_hedge": agent.metadata.get("hedge", False) if hasattr(agent, "metadata") else False
                })
                
            with open(path, "w") as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_agents": len(self.agents),
                    "swarm": swarm_data
                }, f, indent=4)
                
            return True
        except Exception:
            logger.exception("swarm_state_save_failed")
            return False

    def cull_and_spawn(self, spawn_callback=None) -> tuple[list[str], list[str]]:
        """
        Kill bottom N% performers and optionally spawn new candidates.
        Only runs if enough time has passed since last cull.
        """
        now = datetime.utcnow()
        days_since_cull = (now - self._last_cull).days
        if days_since_cull < self.config.cull_frequency_days:
            return [], []

        performance = self.evaluate_all()
        if len(performance) <= self.config.min_agents:
            return [], []

        # Rank by Calmar
        ranked = sorted(performance.items(), key=lambda x: x[1].calmar_ratio)

        # Kill bottom N%
        n_cull = max(1, int(len(ranked) * self.config.cull_bottom_pct))
        culled_ids = []
        for i in range(n_cull):
            agent_id = ranked[i][0]
            if len(self.agents) - len(culled_ids) <= self.config.min_agents:
                break
            self.agents[agent_id].is_active = False
            del self.agents[agent_id]
            del self.portfolios[agent_id]
            del self.capital_weights[agent_id]
            culled_ids.append(agent_id)

        # Spawn new agents if callback provided
        spawned_ids = []
        if spawn_callback:
            new_agents = spawn_callback(n_cull)
            for agent in new_agents:
                self.register_agent(agent)
                spawned_ids.append(agent.agent_id)

        self._last_cull = now
        self.rebalance_capital()

        logger.info(
            "tournament_cull_complete",
            culled=culled_ids,
            spawned=spawned_ids,
            remaining=len(self.agents),
        )

        return culled_ids, spawned_ids

    def get_combined_signal(self, features: dict[str, np.ndarray]) -> Signal | None:
        """
        Get capital-weighted combined signal from all active agents.
        V7: Incorporates Guardian risk-scaling to protect against tail events.
        """
        alpha_signals: list[tuple[float, Signal]] = []
        risk_signals: list[Signal] = []

        self._last_signals.clear() # Clear cache for new tick
        for agent_id, agent in self.agents.items():
            if not agent.is_active:
                continue
            try:
                signal = agent.generate_signal(features)
                if signal is not None:
                    self._last_signals[agent_id] = signal # Cache for cluster-delta optimization
                    if agent.agent_type == "risk-guardian":
                        risk_signals.append(signal)
                    else:
                        weight = self.capital_weights.get(agent_id, 0.0)
                        alpha_signals.append((weight, signal))
            except Exception:
                logger.exception("agent_signal_error", agent_id=agent_id)

        if not alpha_signals:
            return None

        # Capital-weighted average direction and confidence
        total_weight = sum(w for w, _ in alpha_signals)
        if total_weight <= 0:
            return None

        weighted_direction = sum(w * s.direction for w, s in alpha_signals) / total_weight
        weighted_confidence = sum(w * s.confidence for w, s in alpha_signals) / total_weight

        # Phase 17 Genealogy stats
        depths = [agent.lineage_depth for agent_id, agent in self.agents.items() if agent.is_active]
        avg_depth = sum(depths) / len(depths) if depths else 0.0

        risk_scaling_factor = 1.0
        max_stress = 0.0
        if risk_signals:
            # Multi-axis Review: Added default to prevent ValueError if no direction==0.0 signals exist
            neutral_risk_signals = [s.confidence for s in risk_signals if s.direction == 0.0]
            if neutral_risk_signals:
                max_stress = max(neutral_risk_signals)
                # Non-linear scaling: as stress approaches 1.0, confidence drops to 0.0 rapidly
                risk_scaling_factor = max(0.0, 1.0 - (max_stress ** 1.5))
                weighted_confidence *= risk_scaling_factor

        # Use symbol from highest-weight signal
        best_signal = max(alpha_signals, key=lambda x: x[0])[1]

        return Signal(
            signal_id=uuid.uuid4().hex[:12],
            source="tournament_ensemble",
            symbol=best_signal.symbol,
            direction=weighted_direction,
            confidence=weighted_confidence,
            timestamp=datetime.utcnow(),
            timeframe=best_signal.timeframe,
            metadata={
                "n_agents": len(alpha_signals),
                "agent_weights": {s.source: w for w, s in alpha_signals},
                "risk_stress": max_stress,
                "risk_multiplier": risk_scaling_factor,
                "avg_lineage_depth": round(avg_depth, 2)
            },
        )
