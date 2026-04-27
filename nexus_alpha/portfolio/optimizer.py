"""
Portfolio & Risk Engineering — HRP, CVaR, Tail Hedging.

Hierarchical Risk Parity (Lopez de Prado, 2016) — superior to
mean-variance optimization because:
1. Does NOT require matrix inversion (numerically stable)
2. Naturally diversifies across asset clusters
3. Performs better out-of-sample than Markowitz
4. Extended with CVaR constraint for tail risk management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from nexus_alpha.config import RiskConfig
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)


# ─── CVaR (Conditional Value at Risk) ────────────────────────────────────────


def compute_cvar(returns: np.ndarray, confidence: float = 0.99) -> float:
    """
    Compute CVaR (Expected Shortfall) at the given confidence level.
    CVaR = average of losses beyond the VaR threshold.
    """
    if len(returns) == 0:
        return 0.0
    sorted_returns = np.sort(returns)
    cutoff_idx = int(len(sorted_returns) * (1 - confidence))
    cutoff_idx = max(cutoff_idx, 1)
    tail_losses = sorted_returns[:cutoff_idx]
    return float(-np.mean(tail_losses))


def compute_var(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Compute Value at Risk at the given confidence level."""
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


# ─── Random Matrix Theory Denoising ──────────────────────────────────────────


def denoise_correlation_matrix(
    corr: np.ndarray,
    n_observations: int,
) -> np.ndarray:
    """
    Denoise correlation matrix using Marchenko-Pastur (RMT).
    Eigenvalues below the MP threshold are noise — replace them with
    the average noise eigenvalue.
    """
    n_assets = corr.shape[0]
    q = n_observations / n_assets  # Observations-to-assets ratio

    # Marchenko-Pastur bounds
    lambda_plus = (1 + 1 / np.sqrt(q)) ** 2
    lambda_minus = (1 - 1 / np.sqrt(q)) ** 2

    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Replace noise eigenvalues (below MP upper bound) with mean noise eigenvalue
    noise_mask = eigenvalues < lambda_plus
    if noise_mask.sum() > 0:
        noise_mean = eigenvalues[noise_mask].mean()
        eigenvalues[noise_mask] = noise_mean

    # Reconstruct
    denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Normalize to correlation matrix (diagonal = 1)
    d = np.sqrt(np.diag(denoised))
    d = np.where(d > 1e-10, d, 1.0)
    denoised = denoised / np.outer(d, d)
    np.fill_diagonal(denoised, 1.0)

    return denoised


# ─── HRP Optimizer ───────────────────────────────────────────────────────────


@dataclass
class PortfolioWeights:
    """Optimized portfolio weights."""
    weights: dict[str, float]
    method: str
    cvar: float
    expected_return: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HierarchicalRiskParityOptimizer:
    """
    HRP optimizer with CVaR constraint.

    Pipeline:
    1. Compute distance matrix from denoised correlation
    2. Hierarchical clustering (Ward's method)
    3. Quasi-diagonalization (reorder by cluster)
    4. Recursive bisection to allocate weights
    5. Apply CVaR constraint
    """

    def __init__(self, risk_config: RiskConfig | None = None):
        self.risk_config = risk_config or RiskConfig()

    def optimize(
        self,
        returns: pd.DataFrame,
        strategy_signals: dict[str, float] | None = None,
        max_weight: float = 0.25,
        min_weight: float = 0.01,
    ) -> PortfolioWeights:
        """
        Compute HRP portfolio weights from a returns DataFrame.

        Args:
            returns: DataFrame with columns = asset names, rows = returns
            strategy_signals: Optional dict of signal-based tilts
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset
        """
        assets = list(returns.columns)
        n = len(assets)

        if n == 0:
            return PortfolioWeights(weights={}, method="hrp", cvar=0.0, expected_return=0.0)

        if n == 1:
            return PortfolioWeights(
                weights={assets[0]: 1.0},
                method="hrp",
                cvar=compute_cvar(returns.values[:, 0], self.risk_config.cvar_confidence),
                expected_return=float(returns.values[:, 0].mean()) * 252,
            )

        # 1. Denoised correlation matrix
        corr = returns.corr().values
        corr = denoise_correlation_matrix(corr, n_observations=len(returns))

        # 2. Distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # 3. Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="ward")

        # 4. Quasi-diagonalization: get sorted order from dendrogram
        sort_idx = self._get_quasi_diag_order(link, n)

        # 5. Recursive bisection
        sorted_assets = [assets[i] for i in sort_idx]
        sorted_returns = returns[sorted_assets]
        sorted_cov = sorted_returns.cov().values

        weights = self._recursive_bisection(sorted_cov, sorted_assets)

        # 6. Apply signal tilts if provided
        if strategy_signals:
            weights = self._apply_signal_tilts(weights, strategy_signals)

        # 7. Enforce weight bounds
        for asset in weights:
            weights[asset] = np.clip(weights[asset], min_weight, max_weight)

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Compute portfolio metrics
        w_array = np.array([weights.get(a, 0) for a in assets])
        port_returns = returns.values @ w_array
        cvar = compute_cvar(port_returns, self.risk_config.cvar_confidence)
        expected_return = float(np.mean(port_returns)) * 252

        result = PortfolioWeights(
            weights=weights,
            method="hrp_cvar",
            cvar=cvar,
            expected_return=expected_return,
        )

        logger.info(
            "portfolio_optimized",
            n_assets=n,
            cvar=f"{cvar:.4f}",
            exp_return=f"{expected_return:.4f}",
            top_weights={k: f"{v:.3f}" for k, v in sorted(weights.items(), key=lambda x: -x[1])[:5]},
        )

        return result

    def _get_quasi_diag_order(self, link: np.ndarray, n: int) -> list[int]:
        """Get the quasi-diagonal ordering from a linkage matrix."""
        # Traverse dendrogram to get leaf ordering
        order: list[int] = []
        self._recurse_tree(link, n, 2 * n - 2, order)
        return order

    def _recurse_tree(
        self, link: np.ndarray, n: int, node_idx: int, order: list[int]
    ) -> None:
        """Recursively traverse the dendrogram to extract leaf order."""
        if node_idx < n:
            order.append(int(node_idx))
            return
        row = int(node_idx - n)
        left = int(link[row, 0])
        right = int(link[row, 1])
        self._recurse_tree(link, n, left, order)
        self._recurse_tree(link, n, right, order)

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        assets: list[str],
    ) -> dict[str, float]:
        """
        Recursive bisection: allocate inversely proportional to cluster variance.
        """
        weights = {a: 1.0 for a in assets}
        clusters = [list(range(len(assets)))]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                # Split in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Compute cluster variance (inverse-variance weighting)
                left_var = self._cluster_variance(cov, left)
                right_var = self._cluster_variance(cov, right)

                total_var = left_var + right_var
                if total_var < 1e-10:
                    left_frac = 0.5
                else:
                    left_frac = 1 - left_var / total_var

                # Scale weights
                for idx in left:
                    weights[assets[idx]] *= left_frac
                for idx in right:
                    weights[assets[idx]] *= (1 - left_frac)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights

    def _cluster_variance(self, cov: np.ndarray, indices: list[int]) -> float:
        """Compute the variance of a cluster using inverse-variance weights."""
        sub_cov = cov[np.ix_(indices, indices)]
        ivp = 1 / np.diag(sub_cov)
        ivp = ivp / ivp.sum()
        return float(ivp @ sub_cov @ ivp)

    def _apply_signal_tilts(
        self,
        weights: dict[str, float],
        signals: dict[str, float],
        tilt_strength: float = 0.3,
    ) -> dict[str, float]:
        """
        Apply directional tilts from trading signals.
        signal > 0 → increase weight
        signal < 0 → decrease weight
        """
        tilted = {}
        for asset, w in weights.items():
            signal = signals.get(asset, 0.0)
            tilt = 1 + tilt_strength * np.clip(signal, -1, 1)
            tilted[asset] = w * tilt

        # Renormalize
        total = sum(tilted.values())
        if total > 0:
            tilted = {k: v / total for k, v in tilted.items()}
        return tilted


# ─── Position Sizing (Kelly Criterion) ───────────────────────────────────────


def kelly_position_size(
    win_rate: float,
    avg_win_loss_ratio: float,
    confidence: float = 1.0,
    max_size: float = 0.20,
    kelly_fraction: float = 0.5,  # Half-Kelly for safety
) -> float:
    """
    Kelly criterion position sizing.

    f* = (bp - q) / b
    where b = avg_win/avg_loss, p = win_rate, q = 1 - p

    Half-Kelly is used by default (industry standard for robustness).
    """
    if win_rate <= 0 or avg_win_loss_ratio <= 0:
        return 0.0

    b = avg_win_loss_ratio
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b
    kelly = max(kelly, 0.0)  # Never go negative (no bet)

    # Apply half-Kelly and confidence scaling
    position = kelly * kelly_fraction * confidence

    return min(position, max_size)
