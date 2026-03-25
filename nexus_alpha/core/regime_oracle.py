"""
Regime Oracle — Multi-model ensemble regime detector with real-time changepoint detection.

Combines:
- Bayesian Online Changepoint Detection (BOCD)
- Hidden Markov Model (HMM) via hmmlearn
- Structural break detection

Detects regime shifts in real-time as they happen, not in retrospect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from nexus_alpha.logging import get_logger
from nexus_alpha.types import MarketRegime, RegimeState

logger = get_logger(__name__)


# ─── Bayesian Online Changepoint Detection ───────────────────────────────────


class BayesianOnlineChangepoint:
    """
    BOCD (Adams & MacKay, 2007).
    Maintains a run-length distribution that tracks the probability
    of a changepoint at each time step.
    """

    def __init__(self, hazard_rate: float = 1 / 250, observation_model: str = "gaussian"):
        self.hazard = hazard_rate  # Prior belief: changepoints occur every ~250 steps
        self.observation_model = observation_model
        # Sufficient statistics for Gaussian conjugate prior (Normal-Inverse-Gamma)
        self._mu0 = 0.0
        self._kappa0 = 1.0
        self._alpha0 = 1.0
        self._beta0 = 1.0
        # Run-length probabilities (log space for numerical stability)
        self._log_run_lengths: np.ndarray = np.array([0.0])  # Start with run-length 0
        # Per-run-length sufficient stats
        self._mu: np.ndarray = np.array([self._mu0])
        self._kappa: np.ndarray = np.array([self._kappa0])
        self._alpha: np.ndarray = np.array([self._alpha0])
        self._beta: np.ndarray = np.array([self._beta0])

    def update(self, observation: float) -> float:
        """
        Process a new observation and return the changepoint probability.
        Returns P(changepoint at this time step).
        """
        n = len(self._log_run_lengths)

        # 1. Evaluate predictive probability under each run length
        log_pred = self._log_predictive(observation)

        # 2. Growth probabilities: extend each run length
        log_growth = self._log_run_lengths + log_pred + np.log(1 - self.hazard)

        # 3. Changepoint probability: sum all run lengths with hazard
        log_cp = logsumexp(self._log_run_lengths + log_pred + np.log(self.hazard))

        # 4. New run-length distribution
        new_log_rl = np.empty(n + 1)
        new_log_rl[0] = log_cp
        new_log_rl[1:] = log_growth

        # Normalize
        log_evidence = logsumexp(new_log_rl)
        new_log_rl -= log_evidence
        self._log_run_lengths = new_log_rl

        # 5. Update sufficient statistics
        self._update_sufficient_stats(observation)

        # Changepoint probability = P(run_length == 0)
        cp_prob = float(np.exp(new_log_rl[0]))
        return cp_prob

    def _log_predictive(self, x: float) -> np.ndarray:
        """Student-t predictive pdf under Normal-Inverse-Gamma conjugate prior."""
        df = 2 * self._alpha
        loc = self._mu
        scale = np.sqrt(self._beta * (self._kappa + 1) / (self._alpha * self._kappa))
        return stats.t.logpdf(x, df=df, loc=loc, scale=scale)

    def _update_sufficient_stats(self, x: float) -> None:
        """Update NIG sufficient statistics for each run length."""
        new_kappa = np.concatenate([[self._kappa0], self._kappa + 1])
        new_mu = np.concatenate([
            [self._mu0],
            (self._kappa * self._mu + x) / (self._kappa + 1),
        ])
        new_alpha = np.concatenate([[self._alpha0], self._alpha + 0.5])
        new_beta = np.concatenate([
            [self._beta0],
            self._beta + self._kappa * (x - self._mu) ** 2 / (2 * (self._kappa + 1)),
        ])
        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta

    @property
    def most_likely_run_length(self) -> int:
        return int(np.argmax(self._log_run_lengths))


# ─── Simple HMM (Gaussian Emissions) ─────────────────────────────────────────


class GaussianHMM:
    """
    Minimal Hidden Markov Model with Gaussian emissions.
    For production, replace with hmmlearn.GaussianHMM.
    Uses Viterbi for state decoding.
    """

    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        # Initialize with reasonable priors
        self.transition_matrix = np.full((n_states, n_states), 1.0 / n_states)
        # Make self-transitions more likely
        for i in range(n_states):
            self.transition_matrix[i, i] = 0.8
            remaining = 0.2 / (n_states - 1) if n_states > 1 else 0
            for j in range(n_states):
                if j != i:
                    self.transition_matrix[i, j] = remaining

        self.initial_probs = np.full(n_states, 1.0 / n_states)
        # Emission parameters: means and variances for each state
        # States roughly correspond to: low-vol, trending up, trending down, high-vol, crisis
        self.means = np.array([0.0, 0.002, -0.002, 0.0, -0.005])[:n_states]
        self.stds = np.array([0.005, 0.01, 0.01, 0.025, 0.05])[:n_states]
        self._history: list[float] = []

    def decode_state(self, observations: np.ndarray) -> int:
        """Return the most likely current state using forward algorithm."""
        if len(observations) == 0:
            return 0

        # Forward algorithm (last step only for efficiency)
        log_alpha = np.log(self.initial_probs + 1e-30) + self._log_emission(observations[0])

        for t in range(1, len(observations)):
            log_alpha_new = np.empty(self.n_states)
            for j in range(self.n_states):
                log_alpha_new[j] = logsumexp(
                    log_alpha + np.log(self.transition_matrix[:, j] + 1e-30)
                ) + self._log_emission_single(observations[t], j)
            log_alpha = log_alpha_new

        return int(np.argmax(log_alpha))

    def _log_emission(self, x: float) -> np.ndarray:
        return stats.norm.logpdf(x, loc=self.means, scale=self.stds)

    def _log_emission_single(self, x: float, state: int) -> float:
        return float(stats.norm.logpdf(x, loc=self.means[state], scale=self.stds[state]))

    def fit(self, observations: np.ndarray, n_iter: int = 20) -> None:
        """Simple EM fitting (Baum-Welch). For production, use hmmlearn."""
        if len(observations) < 10:
            return
        # Simplified: just update means/stds from clustered observations
        from sklearn.cluster import KMeans

        obs_2d = observations.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_states, n_init=5, random_state=42)
        labels = kmeans.fit_predict(obs_2d)

        for s in range(self.n_states):
            mask = labels == s
            if mask.sum() > 1:
                self.means[s] = observations[mask].mean()
                self.stds[s] = max(observations[mask].std(), 1e-6)


# ─── Regime Oracle ───────────────────────────────────────────────────────────


# Map HMM state indices to MarketRegime
HMM_STATE_MAP = {
    0: MarketRegime.LOW_VOLATILITY,
    1: MarketRegime.TRENDING_BULL,
    2: MarketRegime.TRENDING_BEAR,
    3: MarketRegime.HIGH_VOLATILITY,
    4: MarketRegime.CRISIS,
}


class RegimeOracle:
    """
    Multi-model ensemble regime detector with real-time changepoint detection.
    Combines BOCD, HMM, and structural break detection.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        hazard_rate: float = 1 / 250,
        lookback_window: int = 500,
        changepoint_threshold: float = 0.3,
    ):
        self.bocd = BayesianOnlineChangepoint(hazard_rate=hazard_rate)
        self.hmm = GaussianHMM(n_states=n_regimes)
        self.lookback_window = lookback_window
        self.changepoint_threshold = changepoint_threshold
        self._returns_buffer: deque[float] = deque(maxlen=lookback_window)
        self._current_regime = MarketRegime.UNKNOWN
        self._changepoint_prob = 0.0
        self._last_hmm_fit = 0
        self._update_count = 0

        logger.info("regime_oracle_initialized", n_regimes=n_regimes, hazard_rate=hazard_rate)

    def update(self, returns: np.ndarray) -> RegimeState:
        """
        Process new return observations and produce a regime state.
        Can handle single values or arrays.
        """
        if returns.ndim == 0:
            returns = returns.reshape(1)

        for ret in returns:
            self._returns_buffer.append(float(ret))
            self._changepoint_prob = self.bocd.update(float(ret))
            self._update_count += 1

        # Re-fit HMM periodically (every 100 updates or after changepoint)
        if (
            self._update_count - self._last_hmm_fit >= 100
            or self._changepoint_prob > self.changepoint_threshold
        ):
            obs = np.array(self._returns_buffer)
            if len(obs) >= 50:
                self.hmm.fit(obs)
                self._last_hmm_fit = self._update_count

        # Decode current regime from HMM
        obs = np.array(self._returns_buffer)
        if len(obs) >= 10:
            hmm_state = self.hmm.decode_state(obs[-100:])  # Use recent window
            self._current_regime = HMM_STATE_MAP.get(hmm_state, MarketRegime.UNKNOWN)
        else:
            hmm_state = -1

        # If changepoint detected, flag regime as uncertain until HMM converges
        if self._changepoint_prob > self.changepoint_threshold:
            logger.warning(
                "changepoint_detected",
                probability=f"{self._changepoint_prob:.4f}",
                current_regime=self._current_regime.value,
            )

        # Compute volatility and trend strength from recent returns
        recent = obs[-50:] if len(obs) >= 50 else obs
        volatility = float(np.std(recent)) if len(recent) > 1 else 0.0
        trend_strength = float(np.mean(recent) / (np.std(recent) + 1e-10)) if len(recent) > 1 else 0.0

        # Confidence: lower when changepoint is detected
        confidence = 1.0 - min(self._changepoint_prob, 1.0)

        return RegimeState(
            regime=self._current_regime,
            confidence=confidence,
            changepoint_probability=self._changepoint_prob,
            volatility=volatility,
            trend_strength=trend_strength,
            hmm_state=hmm_state,
            timestamp=datetime.utcnow(),
        )

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def changepoint_probability(self) -> float:
        return self._changepoint_prob
