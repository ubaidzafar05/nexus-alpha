"""
Autonomous Strategy Discovery Engine — Genetic Programming.

Uses genetic programming to discover novel trading strategies
from a library of mathematical primitives.

This is NOT brute-force indicator testing.
It evolves complex, nonlinear combinations of features
that no human would think to design.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


# ─── Primitive Operations ────────────────────────────────────────────────────


def protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Division that returns 0 when denominator is near-zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-10, a / b, 0.0)
    return result


def protected_log(a: np.ndarray) -> np.ndarray:
    """Log that handles non-positive values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(a > 1e-10, np.log(a), 0.0)
    return result


def protected_sqrt(a: np.ndarray) -> np.ndarray:
    """Square root that handles negative values."""
    return np.sqrt(np.abs(a))


def ts_mean(a: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling mean."""
    result = pd.Series(a).rolling(window, min_periods=1).mean().values
    return result


def ts_std(a: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling standard deviation."""
    result = pd.Series(a).rolling(window, min_periods=2).std().fillna(0).values
    return result


def ts_rank(a: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling percentile rank."""
    s = pd.Series(a)
    result = s.rolling(window, min_periods=1).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
    ).fillna(0.5).values
    return result


def ts_delta(a: np.ndarray, period: int = 1) -> np.ndarray:
    """Change over period."""
    result = np.zeros_like(a)
    result[period:] = a[period:] - a[:-period]
    return result


def ts_max(a: np.ndarray, window: int = 20) -> np.ndarray:
    return pd.Series(a).rolling(window, min_periods=1).max().values


def ts_min(a: np.ndarray, window: int = 20) -> np.ndarray:
    return pd.Series(a).rolling(window, min_periods=1).min().values


def ts_corr(a: np.ndarray, b: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling correlation."""
    return pd.Series(a).rolling(window, min_periods=5).corr(pd.Series(b)).fillna(0).values


def sign_signal(a: np.ndarray) -> np.ndarray:
    """Signal sign: -1, 0, +1."""
    return np.sign(a)


def clip_signal(a: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1]."""
    return np.clip(a, -1, 1)


def zscore(a: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling z-score normalization."""
    s = pd.Series(a)
    mean = s.rolling(window, min_periods=5).mean()
    std = s.rolling(window, min_periods=5).std()
    return ((s - mean) / (std + 1e-10)).fillna(0).values


# ─── Strategy Gene Representation ────────────────────────────────────────────


@dataclass
class StrategyGene:
    """
    A node in the expression tree that represents a trading strategy.
    The tree is evaluated bottom-up to produce a signal series.
    """
    operation: str
    children: list[StrategyGene] = field(default_factory=list)
    terminal_name: str | None = None  # Only for leaf nodes
    window: int = 20  # For time-series operations
    strategy_id: str = ""

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate this gene tree given input data."""
        if self.terminal_name is not None:
            # Leaf node: return the named column
            if self.terminal_name in data.columns:
                return data[self.terminal_name].values.astype(float)
            return np.zeros(len(data))

        # Evaluate children first
        child_vals = [c.evaluate(data) for c in self.children]

        # Apply operation
        return self._apply_op(child_vals)

    def _apply_op(self, children: list[np.ndarray]) -> np.ndarray:
        ops: dict[str, Callable] = {
            "add": lambda: children[0] + children[1],
            "sub": lambda: children[0] - children[1],
            "mul": lambda: children[0] * children[1],
            "div": lambda: protected_div(children[0], children[1]),
            "neg": lambda: -children[0],
            "abs": lambda: np.abs(children[0]),
            "log": lambda: protected_log(children[0]),
            "sqrt": lambda: protected_sqrt(children[0]),
            "sign": lambda: sign_signal(children[0]),
            "clip": lambda: clip_signal(children[0]),
            "ts_mean": lambda: ts_mean(children[0], self.window),
            "ts_std": lambda: ts_std(children[0], self.window),
            "ts_rank": lambda: ts_rank(children[0], self.window),
            "ts_delta": lambda: ts_delta(children[0], max(1, self.window // 5)),
            "ts_max": lambda: ts_max(children[0], self.window),
            "ts_min": lambda: ts_min(children[0], self.window),
            "ts_corr": lambda: ts_corr(children[0], children[1], self.window),
            "zscore": lambda: zscore(children[0], self.window),
        }
        if self.operation in ops:
            return ops[self.operation]()
        return children[0] if children else np.array([0.0])

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        if not self.children:
            return 1
        return 1 + sum(c.size() for c in self.children)


# ─── Strategy Discovery Result ───────────────────────────────────────────────


@dataclass
class DiscoveredStrategy:
    """A strategy found by the genetic programming engine."""
    strategy_id: str
    gene: StrategyGene
    fitness: float
    information_coefficient: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    complexity: int  # Tree size
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    causal_validated: bool = False


# ─── Strategy Evolution Engine ────────────────────────────────────────────────


# Available terminals (input features)
TERMINALS = [
    "open", "high", "low", "close", "volume", "vwap",
    "returns_1h", "returns_4h", "returns_1d",
    "volatility_1h", "volatility_4h", "volatility_1d",
    "rsi_14", "rsi_7", "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_mid",
    "atr_14", "obv", "cvd",
    "funding_rate", "open_interest", "long_short_ratio",
    "btc_dominance", "total_market_cap_change",
]

# Operations grouped by arity
UNARY_OPS = [
    "neg",
    "abs",
    "log",
    "sqrt",
    "sign",
    "clip",
    "ts_mean",
    "ts_std",
    "ts_rank",
    "ts_delta",
    "ts_max",
    "ts_min",
    "zscore",
]
BINARY_OPS = ["add", "sub", "mul", "div", "ts_corr"]


class StrategyEvolutionEngine:
    """
    Uses genetic programming to discover novel trading strategies.

    Pipeline:
    1. Generate random population of expression trees
    2. Evaluate fitness (risk-adjusted IC with turnover penalty)
    3. Select parents via tournament selection
    4. Apply crossover and mutation
    5. Repeat for N generations
    6. Validate top strategies causally before promotion
    """

    def __init__(
        self,
        population_size: int = 200,
        generations: int = 50,
        max_depth: int = 6,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        tournament_k: int = 5,
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self._rng = np.random.default_rng(42)

    def _random_tree(self, depth: int = 0) -> StrategyGene:
        """Generate a random expression tree."""
        if depth >= self.max_depth or (depth > 1 and self._rng.random() < 0.3):
            # Terminal node
            terminal = self._rng.choice(TERMINALS)
            return StrategyGene(operation="terminal", terminal_name=terminal)

        # Choose an operation
        if self._rng.random() < 0.3:
            op = self._rng.choice(BINARY_OPS)
            children = [self._random_tree(depth + 1), self._random_tree(depth + 1)]
        else:
            op = self._rng.choice(UNARY_OPS)
            children = [self._random_tree(depth + 1)]

        window = int(self._rng.choice([5, 10, 20, 40, 60]))
        return StrategyGene(operation=op, children=children, window=window)

    def _fitness(
        self,
        gene: StrategyGene,
        data: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> float:
        """
        Multi-objective fitness function.
        Optimizes for: IC (predictive power), Sharpe, low turnover, low complexity.
        """
        try:
            signal = gene.evaluate(data)
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

            if np.std(signal) < 1e-10:
                return -1.0  # Degenerate signal

            # Align lengths
            n = min(len(signal), len(forward_returns))
            signal = signal[:n]
            fwd = forward_returns.values[:n]

            # 1. Information Coefficient (Spearman rank correlation)
            ic, _ = stats.spearmanr(signal, fwd)
            if np.isnan(ic):
                return -1.0

            # 2. Signal-weighted returns → approximate Sharpe
            signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
            strat_returns = signal_norm[:-1] * fwd[1:]
            sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-10) * np.sqrt(252)

            # 3. Turnover penalty
            turnover = np.mean(np.abs(np.diff(signal_norm)))

            # 4. Complexity penalty (prefer simpler strategies)
            complexity = gene.size()
            complexity_penalty = 0.01 * complexity

            # Combined fitness (higher is better)
            fitness = (
                0.4 * abs(ic)
                + 0.3 * max(sharpe, 0) / 10  # Normalize
                - 0.15 * turnover
                - 0.15 * complexity_penalty
            )

            return float(fitness)
        except Exception:
            return -1.0

    def _tournament_select(self, population: list[tuple[StrategyGene, float]]) -> StrategyGene:
        """Tournament selection: pick k random individuals, return best."""
        indices = self._rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = max(indices, key=lambda i: population[i][1])
        return population[best_idx][0]

    def _crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """Subtree crossover: replace a random subtree in p1 with one from p2."""
        import copy
        child = copy.deepcopy(parent1)
        donor = copy.deepcopy(parent2)

        # Find a random node in child to replace
        child_nodes = self._collect_nodes(child)
        donor_nodes = self._collect_nodes(donor)

        if len(child_nodes) <= 1 or not donor_nodes:
            return child

        # Replace a random internal node
        replace_idx = int(self._rng.integers(1, len(child_nodes)))
        donor_idx = int(self._rng.integers(0, len(donor_nodes)))

        target = child_nodes[replace_idx]
        source = donor_nodes[donor_idx]

        target.operation = source.operation
        target.children = source.children
        target.terminal_name = source.terminal_name
        target.window = source.window

        return child

    def _mutate(self, gene: StrategyGene) -> StrategyGene:
        """Point mutation: change one node's operation or terminal."""
        import copy
        mutant = copy.deepcopy(gene)
        nodes = self._collect_nodes(mutant)

        if not nodes:
            return mutant

        # Pick a random node
        idx = int(self._rng.integers(0, len(nodes)))
        node = nodes[idx]

        if node.terminal_name is not None:
            # Change terminal
            node.terminal_name = str(self._rng.choice(TERMINALS))
        else:
            # Change window or operation
            if self._rng.random() < 0.5:
                node.window = int(self._rng.choice([5, 10, 20, 40, 60]))
            else:
                if len(node.children) == 1:
                    node.operation = str(self._rng.choice(UNARY_OPS))
                elif len(node.children) == 2:
                    node.operation = str(self._rng.choice(BINARY_OPS))

        return mutant

    def _collect_nodes(self, gene: StrategyGene) -> list[StrategyGene]:
        """Collect all nodes in the tree (BFS)."""
        nodes = [gene]
        queue = [gene]
        while queue:
            current = queue.pop(0)
            for child in current.children:
                nodes.append(child)
                queue.append(child)
        return nodes

    def evolve(
        self,
        data: pd.DataFrame,
        forward_returns: pd.Series,
        n_top: int = 10,
    ) -> list[DiscoveredStrategy]:
        """
        Run the full evolutionary strategy discovery pipeline.

        Args:
            data: Feature DataFrame (columns = terminal names)
            forward_returns: Forward returns to predict
            n_top: Number of top strategies to return

        Returns:
            List of DiscoveredStrategy objects, sorted by fitness
        """
        logger.info(
            "evolution_starting",
            population=self.population_size,
            generations=self.generations,
            features=len(data.columns),
            samples=len(data),
        )

        # Initialize population
        population: list[tuple[StrategyGene, float]] = []
        for _ in range(self.population_size):
            gene = self._random_tree()
            fitness = self._fitness(gene, data, forward_returns)
            population.append((gene, fitness))

        # Evolution loop
        for gen in range(self.generations):
            new_population = []

            # Elitism: keep top 10%
            population.sort(key=lambda x: x[1], reverse=True)
            elite_count = max(1, self.population_size // 10)
            new_population.extend(population[:elite_count])

            while len(new_population) < self.population_size:
                if self._rng.random() < self.crossover_rate:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2)
                else:
                    child = self._tournament_select(population)

                if self._rng.random() < self.mutation_rate:
                    child = self._mutate(child)

                # Enforce depth limit
                if child.depth() <= self.max_depth:
                    fitness = self._fitness(child, data, forward_returns)
                    new_population.append((child, fitness))

            population = new_population

            if gen % 10 == 0:
                best_fitness = max(f for _, f in population)
                avg_fitness = np.mean([f for _, f in population])
                logger.info(
                    "evolution_generation",
                    generation=gen,
                    best_fitness=f"{best_fitness:.4f}",
                    avg_fitness=f"{avg_fitness:.4f}",
                )

        # Extract top strategies
        population.sort(key=lambda x: x[1], reverse=True)
        strategies = []

        for i, (gene, fitness) in enumerate(population[:n_top]):
            if fitness <= 0:
                continue

            signal = gene.evaluate(data)
            signal = np.nan_to_num(signal, nan=0.0)
            n = min(len(signal), len(forward_returns))
            ic, _ = stats.spearmanr(signal[:n], forward_returns.values[:n])

            # Compute approximate Sharpe and drawdown
            signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
            strat_returns = signal_norm[:n-1] * forward_returns.values[1:n]
            if len(strat_returns) > 0:
                sharpe = (
                    np.mean(strat_returns)
                    / (np.std(strat_returns) + 1e-10)
                    * np.sqrt(252)
                )
            else:
                sharpe = 0
            cumulative = np.cumsum(strat_returns)
            peak = np.maximum.accumulate(cumulative) if len(cumulative) > 0 else np.array([0])
            dd = peak - cumulative
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0
            turnover = float(np.mean(np.abs(np.diff(signal_norm)))) if len(signal_norm) > 1 else 0

            strategies.append(DiscoveredStrategy(
                strategy_id=f"gp-evolved-{i:03d}",
                gene=gene,
                fitness=fitness,
                information_coefficient=float(ic) if not np.isnan(ic) else 0.0,
                sharpe_ratio=float(sharpe),
                max_drawdown=max_dd,
                turnover=turnover,
                complexity=gene.size(),
            ))

        logger.info(
            "evolution_complete",
            strategies_found=len(strategies),
            best_fitness=f"{strategies[0].fitness:.4f}" if strategies else "none",
        )

        return strategies
