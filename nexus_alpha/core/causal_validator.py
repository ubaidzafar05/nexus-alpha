"""
Causal Signal Validator — Uses Pearl's do-calculus to validate signals.

Every proposed signal must pass causal validation before entering
the live ensemble. Correlation without causation = overfitting.

Key question: "Does signal X CAUSE forward returns Y,
or does some third variable Z cause both?"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CausalValidationResult:
    """Result of causal validation for a signal."""
    signal_name: str
    is_causal: bool
    causal_effect: float
    p_value: float
    confidence_interval: tuple[float, float]
    refutation_passed: bool
    granger_f_stat: float
    granger_p_value: float
    information_coefficient: float
    ic_t_stat: float
    timestamp: datetime


class CausalSignalValidator:
    """
    Validates whether a signal has a genuine causal relationship with forward returns.

    Multi-stage validation:
    1. Information Coefficient (IC) — does the signal rank assets correctly?
    2. Granger Causality — does the signal's past predict future returns?
    3. Placebo test — does a random permutation destroy the effect?
    4. Stability test — is the effect consistent across time windows?
    """

    def __init__(
        self,
        min_ic: float = 0.02,
        min_ic_t_stat: float = 2.0,
        granger_max_lag: int = 5,
        granger_alpha: float = 0.05,
        n_placebo_trials: int = 100,
        min_stability_ratio: float = 0.6,
    ):
        self.min_ic = min_ic
        self.min_ic_t_stat = min_ic_t_stat
        self.granger_max_lag = granger_max_lag
        self.granger_alpha = granger_alpha
        self.n_placebo_trials = n_placebo_trials
        self.min_stability_ratio = min_stability_ratio

    def validate_signal(
        self,
        signal_series: pd.Series,
        forward_returns: pd.Series,
        confounders: pd.DataFrame | None = None,
    ) -> CausalValidationResult:
        """
        Full causal validation pipeline for a signal.

        Args:
            signal_series: The trading signal values (aligned by index)
            forward_returns: The forward returns to predict
            confounders: Optional DataFrame of potential confounding variables
        """
        signal_name = signal_series.name or "unnamed_signal"

        # Align data and drop NaN
        combined = pd.concat([signal_series.rename("signal"), forward_returns.rename("returns")], axis=1).dropna()
        if len(combined) < 50:
            logger.warning("insufficient_data_for_causal_validation", signal=signal_name, n=len(combined))
            return CausalValidationResult(
                signal_name=str(signal_name),
                is_causal=False,
                causal_effect=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                refutation_passed=False,
                granger_f_stat=0.0,
                granger_p_value=1.0,
                information_coefficient=0.0,
                ic_t_stat=0.0,
                timestamp=datetime.utcnow(),
            )

        signal = combined["signal"].values
        returns = combined["returns"].values

        # Stage 1: Information Coefficient (rank correlation)
        ic, ic_p = stats.spearmanr(signal, returns)
        ic_t_stat = ic * np.sqrt((len(signal) - 2) / (1 - ic ** 2 + 1e-10))

        # Stage 2: Granger Causality Test
        granger_f, granger_p = self._granger_causality(signal, returns)

        # Stage 3: Placebo Test (permutation test)
        placebo_passed = self._placebo_test(signal, returns, observed_ic=abs(ic))

        # Stage 4: Stability Test (rolling IC consistency)
        stability_ratio = self._stability_test(signal, returns)

        # Stage 5: Causal effect estimation via OLS with confounders
        causal_effect, effect_p, ci = self._estimate_causal_effect(signal, returns, confounders, combined.index)

        # Final verdict: must pass ALL stages
        is_causal = (
            abs(ic) >= self.min_ic
            and abs(ic_t_stat) >= self.min_ic_t_stat
            and granger_p < self.granger_alpha
            and placebo_passed
            and stability_ratio >= self.min_stability_ratio
        )

        result = CausalValidationResult(
            signal_name=str(signal_name),
            is_causal=is_causal,
            causal_effect=causal_effect,
            p_value=effect_p,
            confidence_interval=ci,
            refutation_passed=placebo_passed,
            granger_f_stat=granger_f,
            granger_p_value=granger_p,
            information_coefficient=ic,
            ic_t_stat=ic_t_stat,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            "causal_validation_complete",
            signal=signal_name,
            is_causal=is_causal,
            ic=f"{ic:.4f}",
            granger_p=f"{granger_p:.4f}",
            placebo=placebo_passed,
            stability=f"{stability_ratio:.2f}",
        )

        return result

    def _granger_causality(self, signal: np.ndarray, returns: np.ndarray) -> tuple[float, float]:
        """
        Simplified Granger causality: does lagged signal improve prediction of returns
        beyond just using lagged returns?
        Uses F-test comparing restricted vs unrestricted regression.
        """
        n = len(signal)
        max_lag = min(self.granger_max_lag, n // 10)
        if max_lag < 1 or n < 20:
            return 0.0, 1.0

        # Build lagged matrices
        y = returns[max_lag:]
        n_obs = len(y)

        # Restricted: only lagged returns
        X_restricted = np.column_stack([returns[max_lag - i - 1: n - i - 1] for i in range(max_lag)])
        X_restricted = np.column_stack([np.ones(n_obs), X_restricted])

        # Unrestricted: lagged returns + lagged signal
        X_signal = np.column_stack([signal[max_lag - i - 1: n - i - 1] for i in range(max_lag)])
        X_unrestricted = np.column_stack([X_restricted, X_signal])

        # OLS for both models
        try:
            _, rss_r, _, _ = np.linalg.lstsq(X_restricted, y, rcond=None)
            _, rss_u, _, _ = np.linalg.lstsq(X_unrestricted, y, rcond=None)

            rss_r = float(rss_r[0]) if len(rss_r) > 0 else np.sum((y - X_restricted @ np.linalg.lstsq(X_restricted, y, rcond=None)[0]) ** 2)
            rss_u = float(rss_u[0]) if len(rss_u) > 0 else np.sum((y - X_unrestricted @ np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]) ** 2)

            df_diff = max_lag
            df_resid = n_obs - X_unrestricted.shape[1]

            if rss_u <= 0 or df_resid <= 0:
                return 0.0, 1.0

            f_stat = ((rss_r - rss_u) / df_diff) / (rss_u / df_resid)
            p_value = 1 - stats.f.cdf(f_stat, df_diff, df_resid)
            return float(f_stat), float(p_value)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 1.0

    def _placebo_test(self, signal: np.ndarray, returns: np.ndarray, observed_ic: float) -> bool:
        """
        Permutation test: shuffle the signal and check if observed IC
        is significantly better than random ICs.
        If observed IC falls outside 95th percentile of placebo distribution → pass.
        """
        placebo_ics = []
        rng = np.random.default_rng(42)
        for _ in range(self.n_placebo_trials):
            shuffled = rng.permutation(signal)
            ic_perm, _ = stats.spearmanr(shuffled, returns)
            placebo_ics.append(abs(ic_perm))

        percentile_95 = np.percentile(placebo_ics, 95)
        return observed_ic > percentile_95

    def _stability_test(self, signal: np.ndarray, returns: np.ndarray, n_windows: int = 5) -> float:
        """
        Check IC sign consistency across rolling windows.
        Returns the fraction of windows where IC has consistent sign.
        """
        window_size = len(signal) // n_windows
        if window_size < 20:
            return 0.0

        ic_signs = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            s_window = signal[start:end]
            r_window = returns[start:end]
            ic, _ = stats.spearmanr(s_window, r_window)
            ic_signs.append(np.sign(ic))

        if not ic_signs:
            return 0.0

        # Majority sign
        majority_sign = np.sign(np.sum(ic_signs))
        consistent = sum(1 for s in ic_signs if s == majority_sign)
        return consistent / len(ic_signs)

    def _estimate_causal_effect(
        self,
        signal: np.ndarray,
        returns: np.ndarray,
        confounders: pd.DataFrame | None,
        index: pd.Index,
    ) -> tuple[float, float, tuple[float, float]]:
        """
        OLS regression of returns on signal, controlling for confounders.
        Returns (coefficient, p-value, 95% CI).
        """
        n = len(signal)
        X = np.column_stack([np.ones(n), signal])

        if confounders is not None:
            conf_aligned = confounders.loc[index].values
            if conf_aligned.shape[0] == n:
                X = np.column_stack([X, conf_aligned])

        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, returns, rcond=None)
            y_hat = X @ beta
            rss = np.sum((returns - y_hat) ** 2)
            mse = rss / (n - X.shape[1])
            cov = mse * np.linalg.inv(X.T @ X)

            se_beta1 = np.sqrt(cov[1, 1])
            t_stat = beta[1] / (se_beta1 + 1e-10)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - X.shape[1]))
            ci = (beta[1] - 1.96 * se_beta1, beta[1] + 1.96 * se_beta1)

            return float(beta[1]), float(p_value), ci
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 1.0, (0.0, 0.0)
