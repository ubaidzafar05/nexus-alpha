"""
Signal Intelligence — Complete signal taxonomy and fusion engine.

40+ signal sources across 8 categories:
A: Market Microstructure
B: Technical / Statistical
C: Statistical Arbitrage
D: ML Prediction
E: Options Flow
F: On-Chain
G: Sentiment & Alt-Data
H: Macro / Cross-Asset
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from nexus_alpha.signals.base import BaseSignalGenerator, SignalCategory
from nexus_alpha.learning.guardian import GuardianAI
from nexus_alpha.learning.trade_logger import TradeLogger
from nexus_alpha.learning.causality import CausalSignalValidator
from nexus_alpha.signals.microstructure_l2 import OFISignal, TickVPINSignal
from nexus_alpha.learning.regime_detector import RegimeDetector

from nexus_alpha.log_config import get_logger
from nexus_alpha.schema_types import Signal, FusedSignal
from nexus_alpha.signals.alpha_microstructure import OrderFlowAlpha
from nexus_alpha.signals.alpha_sentiment import SentimentEngine
from nexus_alpha.risk.position_sizer import ATRPositionSizer
from nexus_alpha.intelligence.microstructure import L2MicrostructureEngine

logger = get_logger(__name__)

# IC threshold below which a signal is considered noise and zeroed out during fusion.
# Rationale: rolling |IC| < 0.02 on crypto bars is indistinguishable from coin-flip and
# feeds false confidence into the weighted sum. Coin-flip accuracy with fees = ruin.
IC_GATE_MIN_ABS = 0.02
# Default portfolio NAV used when fuse() is exercised without a live portfolio reference
# (e.g. in tests or isolated agent sandboxes). Real trading always overrides this via
# SignalFusionEngine.set_portfolio_nav() before fuse() is called.
DEFAULT_PORTFOLIO_NAV_USD = 10_000.0


# Moved BaseSignalGenerator and SignalCategory to nexus_alpha.signals.base

HYSTERESIS_THRESHOLD = 0.15
FLIP_MIN_INTERVAL = timedelta(minutes=15)


# ─── Microstructure Signals ──────────────────────────────────────────────────


class OrderBookImbalance(BaseSignalGenerator):
    """A1: (bid_depth - ask_depth) / total_depth at multiple levels."""

    def __init__(self):
        super().__init__("order_book_imbalance", SignalCategory.MICROSTRUCTURE)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        bid = data.get("bid_depth", pd.Series(0, index=data.index))
        ask = data.get("ask_depth", pd.Series(0, index=data.index))
        total = bid + ask
        return ((bid - ask) / total.replace(0, np.nan)).fillna(0)


class VPIN(BaseSignalGenerator):
    """A5: Volume-Synchronized Probability of Informed Trading."""

    def __init__(self, bucket_size: int = 50, n_buckets: int = 50):
        super().__init__("vpin", SignalCategory.MICROSTRUCTURE)
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "volume" not in data.columns or "close" not in data.columns:
            return pd.Series(0.0, index=data.index)

        prices = data["close"]
        volumes = data["volume"]
        returns = prices.pct_change().fillna(0)

        # Classify volume as buy/sell using tick rule
        buy_vol = volumes * (returns > 0).astype(float) + volumes * 0.5 * (returns == 0).astype(float)
        sell_vol = volumes - buy_vol

        # Rolling VPIN
        window = self.n_buckets
        buy_sum = buy_vol.rolling(window, min_periods=1).sum()
        sell_sum = sell_vol.rolling(window, min_periods=1).sum()
        total = buy_sum + sell_sum
        vpin = (np.abs(buy_sum - sell_sum) / total.replace(0, np.nan))
        return vpin.fillna(0)


class KyleLambda(BaseSignalGenerator):
    """A2: Price impact coefficient from trades.

    Estimates lambda from |return| = lambda * |signed_volume| over a rolling
    window using the closed-form slope Cov(|r|, |v|) / Var(|v|). This is the
    OLS regression-through-origin estimator and captures the price-impact
    coefficient without depending on an intercept being identifiable.
    """

    def __init__(self, window: int = 100, min_periods: int = 20):
        super().__init__("kyle_lambda", SignalCategory.MICROSTRUCTURE)
        self.window = window
        self.min_periods = min_periods

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns or "volume" not in data.columns:
            return pd.Series(0.0, index=data.index)

        returns = data["close"].pct_change().fillna(0)
        volume = data["volume"].astype(float)
        signed_vol = (volume * np.sign(returns)).abs()
        abs_ret = returns.abs()

        # Rolling regression-through-origin slope: beta = E[xy] / E[x^2]
        # This is stable, vectorized, and equivalent to the OLS slope when the
        # relationship passes through the origin (no trades => no impact).
        xy = (abs_ret * signed_vol).rolling(self.window, min_periods=self.min_periods).mean()
        xx = (signed_vol * signed_vol).rolling(self.window, min_periods=self.min_periods).mean()

        lam = xy / xx.replace(0, np.nan)
        return lam.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ─── Technical Signals ───────────────────────────────────────────────────────


class RSISignal(BaseSignalGenerator):
    """B: Multi-timeframe RSI."""

    def __init__(self, period: int = 14):
        super().__init__(f"rsi_{period}", SignalCategory.TECHNICAL)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            return pd.Series(50.0, index=data.index)

        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)


class MACDSignal(BaseSignalGenerator):
    """B: MACD line, signal, histogram."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("macd", SignalCategory.TECHNICAL)
        self.fast, self.slow, self.signal_period = fast, slow, signal

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            return pd.Series(0.0, index=data.index)

        close = data["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd - signal_line
        return histogram  # Histogram is the actionable signal


class BollingerBandSignal(BaseSignalGenerator):
    """B: Position within Bollinger Bands [-1, 1]."""

    def __init__(self, window: int = 20, n_std: float = 2.0):
        super().__init__("bollinger_position", SignalCategory.TECHNICAL)
        self.window, self.n_std = window, n_std

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns:
            return pd.Series(0.0, index=data.index)

        close = data["close"]
        mid = close.rolling(self.window, min_periods=1).mean()
        std = close.rolling(self.window, min_periods=2).std().fillna(1)
        upper = mid + self.n_std * std
        lower = mid - self.n_std * std
        band_width = upper - lower
        position = ((close - lower) / band_width.replace(0, np.nan) * 2 - 1).fillna(0)
        return position.clip(-1, 1)


class ATRSignal(BaseSignalGenerator):
    """B: ATR-normalized momentum."""

    def __init__(self, period: int = 14):
        super().__init__("atr", SignalCategory.TECHNICAL)
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if not all(c in data.columns for c in ["high", "low", "close"]):
            return pd.Series(0.0, index=data.index)

        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.period, min_periods=1).mean()
        return atr / close  # Normalized ATR


class OBVSignal(BaseSignalGenerator):
    """B5: On-Balance Volume."""

    def __init__(self, slope_window: int = 20):
        super().__init__("obv", SignalCategory.TECHNICAL)
        self.slope_window = slope_window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns or "volume" not in data.columns:
            return pd.Series(0.0, index=data.index)

        direction = np.sign(data["close"].diff().fillna(0))
        obv = (direction * data["volume"]).cumsum()
        # Return slope of OBV as signal
        obv_slope = obv.diff(self.slope_window) / self.slope_window
        return obv_slope.fillna(0)


# ─── Statistical Arbitrage Signals ──────────────────────────────────────────


class StatisticalArbitrage(BaseSignalGenerator):
    """
    C: Pair trading / Cointegration signal.
    Models the spread between the current symbol and a lead/correlated asset
    using a rolling hedge ratio (beta = Cov(y, x) / Var(x)) and z-scores the
    residual. If the lead series is not present the generator returns zeros —
    it never silently falls back to an uninformative proxy.
    """

    def __init__(self, lead_symbol: str = "BTC", window: int = 100, min_periods: int = 40):
        super().__init__(f"stat_arb_{lead_symbol}", SignalCategory.STATISTICAL_ARB)
        self.lead_symbol = lead_symbol
        self.window = window
        self.min_periods = min_periods

    def compute(self, data: pd.DataFrame) -> pd.Series:
        lead_col = f"lead_{self.lead_symbol}_close"
        if "close" not in data.columns or lead_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        close = pd.to_numeric(data["close"], errors="coerce")
        lead = pd.to_numeric(data[lead_col], errors="coerce")
        if (close <= 0).any() or (lead <= 0).any():
            return pd.Series(0.0, index=data.index)

        y = np.log(close)
        x = np.log(lead)

        # Rolling hedge ratio via vectorized covariance / variance.
        mean_x = x.rolling(self.window, min_periods=self.min_periods).mean()
        mean_y = y.rolling(self.window, min_periods=self.min_periods).mean()
        cov_xy = (x * y).rolling(self.window, min_periods=self.min_periods).mean() - mean_x * mean_y
        var_x = (x * x).rolling(self.window, min_periods=self.min_periods).mean() - mean_x * mean_x
        beta = cov_xy / var_x.replace(0, np.nan)

        # Spread and its z-score, with variance floor to avoid blow-ups on tight ranges.
        spread = y - beta * x
        spread_mean = spread.rolling(self.window, min_periods=self.min_periods).mean()
        spread_std = spread.rolling(self.window, min_periods=self.min_periods).std()
        z = (spread - spread_mean) / spread_std.replace(0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Clip to [-3, 3]: anything beyond is almost certainly a data glitch, not alpha.
        z = z.clip(-3.0, 3.0) / 3.0
        # Negative z (price low vs lead) -> long; positive z (rich vs lead) -> short.
        return -z


# ─── ML Prediction Signals ───────────────────────────────────────────────────


class MLPrediction(BaseSignalGenerator):
    """
    D: ML-based price direction prediction.

    Loads the pre-trained LightweightPredictor checkpoint for the symbol+timeframe
    found in ``data.attrs`` (``symbol``, ``timeframe``) and returns a per-row
    signal series in ``[-1, 1]``. Checkpoints live under
    ``data/checkpoints/lightweight_{BASE}_{QUOTE}_{TIMEFRAME}.pkl``.

    If the checkpoint is missing, ``build_features`` fails, or the predictor
    has no fitted model, ``compute`` returns a zero series. The engine must
    fail safe — an ML stub that silently emits momentum is what caused the
    causal validator to think it had a real signal to prune.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        default_timeframe: str = "1h",
    ):
        super().__init__("ml_prediction", SignalCategory.ML_PREDICTION)
        self.model_dir = Path(model_dir) if model_dir else Path("data/checkpoints")
        self.default_timeframe = default_timeframe
        # Cache predictors keyed by (symbol, timeframe) so we only pay the pickle
        # cost once per symbol across all fuse() calls.
        self._predictors: dict[tuple[str, str], Any] = {}
        self._missing: set[tuple[str, str]] = set()

    @staticmethod
    def _symbol_key(symbol: str) -> str:
        if "/" in symbol:
            return symbol.replace("/", "_")
        if symbol.endswith("USDT") and len(symbol) > 4:
            return f"{symbol[:-4]}_USDT"
        return symbol

    def _get_predictor(self, symbol: str, timeframe: str) -> Any | None:
        key = (symbol, timeframe)
        if key in self._predictors:
            return self._predictors[key]
        if key in self._missing:
            return None

        try:
            from nexus_alpha.learning.offline_trainer import LightweightPredictor
        except Exception as err:
            logger.warning("ml_prediction_import_failed", error=str(err))
            self._missing.add(key)
            return None

        ckpt = self.model_dir / f"lightweight_{self._symbol_key(symbol)}_{timeframe}.pkl"
        predictor = LightweightPredictor(target_horizon="target_1h")
        if not predictor.load(ckpt):
            self._missing.add(key)
            return None
        self._predictors[key] = predictor
        return predictor

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns or len(data) < 60:
            return pd.Series(0.0, index=data.index)

        symbol = data.attrs.get("symbol") if hasattr(data, "attrs") else None
        if not symbol:
            return pd.Series(0.0, index=data.index)
        timeframe = data.attrs.get("timeframe", self.default_timeframe)

        predictor = self._get_predictor(str(symbol), str(timeframe))
        if predictor is None:
            return pd.Series(0.0, index=data.index)

        try:
            from nexus_alpha.learning.historical_data import build_features

            features = build_features(data)
            if features.empty:
                return pd.Series(0.0, index=data.index)

            feature_cols = [c for c in features.columns if not c.startswith("target_")]
            # Keep only the columns the model was trained on, in the exact order.
            trained = getattr(predictor, "feature_names", None) or []
            if trained:
                missing = [c for c in trained if c not in feature_cols]
                if missing:
                    # Schema drift — refuse to fabricate zeros silently for missing
                    # features, but do not crash the fuse pipeline.
                    logger.warning(
                        "ml_prediction_feature_mismatch",
                        symbol=symbol,
                        missing=missing[:5],
                        missing_count=len(missing),
                    )
                    return pd.Series(0.0, index=data.index)
                X = features[trained].values.astype(np.float32)
            else:
                X = features[feature_cols].values.astype(np.float32)

            signals_arr, _confidences = predictor.predict_batch(X)
        except Exception as err:
            logger.warning("ml_prediction_runtime_error", symbol=symbol, error=str(err))
            return pd.Series(0.0, index=data.index)

        out = pd.Series(0.0, index=data.index, dtype=float)
        # build_features drops the first `lookback` rows; align back by tail index.
        n = min(len(signals_arr), len(out))
        if n > 0:
            out.iloc[-n:] = np.clip(signals_arr[-n:], -1.0, 1.0)
        return out


class SentimentSignal(BaseSignalGenerator):
    """
    G: Sentiment & Alt-Data.
    Wraps the HybridSentimentPipeline to provide real-time sentiment scores.
    """

    def __init__(self):
        super().__init__("sentiment", SignalCategory.SENTIMENT)
        # In a real environment, we'd inject the pipeline instance
        self._current_score = 0.0

    def update_score(self, score: float):
        """Update the internal score from the async ingestion pipeline."""
        self._current_score = score

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # Return a series of the current score repeated
        # If no token is provided, we return neutral sentiment to prevent hung requests
        from nexus_alpha.config import LLMConfig
        if not LLMConfig().hf_token:
            return pd.Series(0.0, index=data.index)
            
        return pd.Series(self._current_score, index=data.index)


# ─── Signal Fusion Engine ────────────────────────────────────────────────────





class SignalFusionEngine:
    """
    Combines signals from all categories using weighted aggregation.
    Weights are determined by rolling IC (information coefficient).
    """

    def __init__(self):
        self.generators: list[BaseSignalGenerator] = []
        self.signal_weights: dict[str, float] = {}
        
        # Intelligence Node Caching (Phase 4 Optimization)
        self.obi_alpha = OrderFlowAlpha()
        self.sentiment_engine = SentimentEngine()
        self.risk_sizer = ATRPositionSizer(risk_per_trade_usd=100.0)
        
        # V4 ULTRA: Regime-Aware Weight Matrix
        # Mapping: regime -> {signal_name: weight}
        self.regime_weights: dict[str, dict[str, float]] = {}
        # Mapping: regime -> {signal_name: [ic_history]}
        self._regime_ic_history: dict[str, dict[str, list[float]]] = {}
        
        self.regime_detector = RegimeDetector()
        
        # Signal Hysteresis (Phase 5 Hardening)
        self._last_directions: dict[str, float] = {}
        self._last_signal_times: dict[str, datetime] = {}
        self.min_flip_interval = timedelta(hours=1)
        
        # Portfolio & Guardian Context (V3 Evolution)
        self.guardian = GuardianAI()
        self.trade_logger = TradeLogger()
        self.max_portfolio_heat = 5000.0 # Max $5k at risk across all positions
        self.causal_validator = CausalSignalValidator()
        # Advanced VPIN/OFI (V6 Evolution)
        self.microstructure = L2MicrostructureEngine(symbols=[])
        self._live_microstructure_data: dict[str, dict[str, float]] = {}

        # Live portfolio NAV used for ATR position sizing. Must be set by the
        # orchestrator via set_portfolio_nav() each cycle; the fallback default
        # exists only so standalone tests and sandboxed agents don't crash.
        self._portfolio_nav_usd: float = DEFAULT_PORTFOLIO_NAV_USD

    def set_portfolio_nav(self, nav_usd: float) -> None:
        """Update the NAV used for ATR risk sizing inside fuse().

        The orchestrator calls this every cycle after NAV recalculation so the
        sizer uses live capital instead of a hardcoded bootstrap value.
        """
        if nav_usd and nav_usd > 0:
            self._portfolio_nav_usd = float(nav_usd)

    def register_generator(self, generator: BaseSignalGenerator, weight: float = 1.0) -> None:
        self.generators.append(generator)
        # Initialize default weights for all expected regimes
        for r in ["trending_up", "trending_down", "sideways", "panicked", "unknown"]:
            if r not in self.regime_weights:
                self.regime_weights[r] = {}
            if r not in self._regime_ic_history:
                self._regime_ic_history[r] = {}
            self.regime_weights[r][generator.name] = weight
            self._regime_ic_history[r][generator.name] = []

    def register_defaults(self) -> None:
        """Register the standard set of signal generators."""
        defaults = [
            (OrderBookImbalance(), 1.0),
            (VPIN(), 1.2),
            (KyleLambda(), 0.8),
            (RSISignal(14), 1.0),
            (RSISignal(7), 0.8),
            (MACDSignal(), 1.0),
            (BollingerBandSignal(), 0.9),
            (ATRSignal(), 0.7),
            (OBVSignal(), 0.8),
            (OFISignal(), 1.2),
            (TickVPINSignal(), 1.5),
            (StatisticalArbitrage("BTC"), 1.5),
            (MLPrediction(), 1.2),
            (SentimentSignal(), 1.0),
        ]
        for gen, weight in defaults:
            self.register_generator(gen, weight)

    def get_microstructure_stats(self) -> dict[str, float]:
        """Return the maximum VPIN and OFI across all tracked symbols for telemetry."""
        vpins = [d.get("vpin", 0.0) for d in self._live_microstructure_data.values()]
        ofis = [abs(d.get("ofi", 0.0)) for d in self._live_microstructure_data.values()]
        
        return {
            "vpin_max": float(np.max(vpins)) if vpins else 0.0,
            "ofi_max": float(np.max(ofis)) if ofis else 0.0
        }

    def compute_all(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Compute all registered signals."""
        results = {}
        for gen in self.generators:
            try:
                results[gen.name] = gen.compute(data)
            except Exception:
                logger.exception("signal_computation_error", signal=gen.name)
                results[gen.name] = pd.Series(0.0, index=data.index)

        # Inject real-time microstructure signals if available
        symbol_key = data.attrs.get("symbol")
        if symbol_key in self._live_microstructure_data:
            live = self._live_microstructure_data[symbol_key]
            if "tick_vpin" in results:
                results["tick_vpin"].iloc[-1] = live.get("vpin", 0.0)
            if "ofi_l2" in results:
                results["ofi_l2"].iloc[-1] = live.get("ofi", 0.0)

        return results

    def on_market_tick(self, tick: dict[str, Any]) -> None:
        """Update live microstructure metrics from tick stream."""
        symbol = tick.get("symbol")
        if not symbol:
            return

        # Ensure engine is tracking this symbol
        if symbol not in self.microstructure.symbols:
            self.microstructure.symbols.append(symbol)
            from nexus_alpha.intelligence.microstructure import TickVPINCalculator, OFIEngine
            self.microstructure.vpin_calculators[symbol] = TickVPINCalculator(symbol, bucket_size=50.0)
            self.microstructure.ofi_engines[symbol] = OFIEngine(symbol)

        if symbol not in self._live_microstructure_data:
            self._live_microstructure_data[symbol] = {"vpin": 0.0, "ofi": 0.0}

        schema = tick.get("schema")
        if schema == "trade_v1":
            vpin = self.microstructure.process_trade(
                symbol, tick["price"], tick["amount"], tick["side"]
            )
            self._live_microstructure_data[symbol]["vpin"] = vpin
        elif schema == "orderbook_v1":
            ofi = self.microstructure.process_orderbook(
                symbol, tick["bid_top"], tick["bid_volume_10"], tick["ask_top"], tick["ask_volume_10"]
            )
            self._live_microstructure_data[symbol]["ofi"] = ofi

    def fuse(
        self,
        data: pd.DataFrame,
        symbol: str,
        forward_returns: pd.Series | None = None,
    ) -> FusedSignal:
        """
        Compute all signals and fuse them into a single directional signal.

        Key guarantees vs. earlier implementations:
        - Never mutates the caller's DataFrame. All augmentation happens on a
          local copy so downstream pipeline stages see pristine input.
        - Causal validation does NOT inject a forward-shifted 'future_return'
          into the live decision path. Look-ahead leaked into production is
          how backtests look profitable while paper trading bleeds.
        - Single z-score normalization pass per signal, shared between the
          category consensus step and the weighted sum.
        - Hysteresis is applied exactly once at the end (the earlier double
          block was stacking a 0.2 confidence dampener with a direction
          override on every flip).
        - Signals whose rolling Information Coefficient stays below
          ``IC_GATE_MIN_ABS`` are zero-weighted: a signal that is not
          predictive of forward returns is noise being sold as alpha.
        - ATR sizing reads NAV from ``set_portfolio_nav()`` instead of a
          $10k constant disconnected from the real portfolio.
        """
        # Work on an augmented copy — never touch the caller's frame.
        data_in = data
        data = data.copy()
        try:
            data.attrs.update(dict(getattr(data_in, "attrs", {}) or {}))
        except Exception:
            pass

        signals = self.compute_all(data)

        # 1. Regime Detection (Hardening Layer)
        if not self.regime_detector.is_fitted:
            self.regime_detector.fit(data.tail(2000))

        regime = self.regime_detector.predict_current(data)
        regime_multiplier = self.regime_detector.get_regime_multiplier(regime)

        # 2. Update regime-specific IC weights if we have labeled forward returns.
        # This is the ONLY place forward returns enter the engine — we accept
        # them from the caller explicitly rather than fabricating them in-band.
        if forward_returns is not None:
            self._update_regime_ic_weights(signals, forward_returns, regime)

        current_regime_weights = self.regime_weights.get(
            regime, self.regime_weights.get("unknown", {})
        )

        # ── Single normalization + IC-gate pass ──────────────────────────
        # We compute the per-signal last-bar z-score once and reuse it for
        # both the category consensus check and the weighted sum. Signals
        # whose recent IC (learned by _update_regime_ic_weights) is below
        # the gate threshold are forced to weight 0 so they stop contaminating
        # direction with noise.
        normalized_last: dict[str, float] = {}
        effective_weight: dict[str, float] = {}
        ic_history = self._regime_ic_history.get(regime, {})

        for gen in self.generators:
            name = gen.name
            series = signals.get(name)
            if series is None or len(series) == 0:
                normalized_last[name] = 0.0
                effective_weight[name] = 0.0
                continue

            values = series.values
            std = float(np.std(values))
            last_val = float(series.iloc[-1])
            if not np.isfinite(last_val):
                last_val = 0.0
            if std > 1e-10 and len(values) > 10:
                z = (last_val - float(np.mean(values))) / std
                norm = float(np.clip(z, -3.0, 3.0) / 3.0)
            else:
                norm = 0.0
            normalized_last[name] = norm

            weight = float(current_regime_weights.get(name, 1.0))
            if not np.isfinite(weight):
                weight = 1.0

            # IC gate: if the rolling |IC| is below the threshold the signal is
            # not predictive — zero its weight rather than let it push direction.
            recent_ic = ic_history.get(name, [])
            if recent_ic:
                ic_window = recent_ic[-20:]
                weights_ic = np.exp(np.linspace(-1, 0, len(ic_window)))
                avg_abs_ic = float(np.average(np.abs(ic_window), weights=weights_ic))
                if avg_abs_ic < IC_GATE_MIN_ABS:
                    weight = 0.0
            effective_weight[name] = weight

        # 3. Category Consensus (sign agreement across categories)
        category_scores: dict[SignalCategory, list[float]] = {}
        for gen in self.generators:
            contribution = normalized_last[gen.name] * effective_weight[gen.name]
            category_scores.setdefault(gen.category, []).append(contribution)

        agreements: list[float] = []
        avg_scores: dict[SignalCategory, float] = {}
        for cat, scores in category_scores.items():
            valid = [s for s in scores if np.isfinite(s)]
            avg = float(np.mean(valid)) if valid else 0.0
            avg_scores[cat] = avg
            if abs(avg) > 0.15:
                agreements.append(float(np.sign(avg)))

        # Sentiment overlay (kept out of category averages to avoid
        # double-counting against dedicated SentimentSignal generators).
        try:
            sent_signal = self.sentiment_engine.generate_signal(symbol)
            if abs(sent_signal["confidence"]) > 0.1:
                agreements.append(float(sent_signal["direction"]))
                avg_scores[SignalCategory.SENTIMENT] = (
                    sent_signal["direction"] * sent_signal["confidence"]
                )
        except Exception as e:
            logger.warning("sentiment_analysis_failed_falling_back", error=str(e))
            sent_signal = {"direction": 0, "confidence": 0.0}

        agree_count = len(agreements)
        if agree_count == 1:
            confidence_multiplier = 0.3
        elif agree_count == 2:
            confidence_multiplier = 0.6
        elif agree_count >= 3:
            confidence_multiplier = 1.0 if len(set(agreements)) == 1 else 0.5
        else:
            confidence_multiplier = 0.2

        # 4. Volatility-adjusted sizing. NAV is set live by the orchestrator.
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        atr = self.risk_sizer.calculate_atr(high, low, close)
        risk_profile = self.risk_sizer.get_position_size(
            current_price=float(close[-1]),
            atr=atr,
            account_balance_usd=float(self._portfolio_nav_usd),
        )
        _ = risk_profile  # surfaced via metadata below when the sizer exposes it

        # 5. Weighted combination (reuses normalized_last / effective_weight).
        total_weight = 0.0
        weighted_sum = 0.0
        contributing: dict[str, float] = {}

        for name, series in signals.items():
            w = effective_weight.get(name, 0.0)
            norm = normalized_last.get(name, 0.0)
            weighted_sum += w * norm
            total_weight += abs(w)

            # Causal validation is reserved for strong signals only and runs
            # against an IN-SAMPLE aligned-return series — no shift(-1) into
            # the live path. If validation fails we null the contribution but
            # still count the weight so direction isn't artificially inflated.
            is_valid = True
            if abs(norm) > 0.4:
                data_causal = data.copy()
                data_causal[name] = series
                # Use an in-sample realized return (already observable at bar t).
                realized = data_causal["close"].pct_change().fillna(0.0)
                data_causal["aligned_returns"] = (realized * np.sign(data_causal[name])).fillna(0.0)
                try:
                    is_valid = self.causal_validator.validate_signal_causality(
                        data_causal, name, target_col="aligned_returns"
                    )
                except Exception as err:
                    logger.debug("causal_validation_errored_treating_as_valid", signal=name, error=str(err))
                    is_valid = True

            contributing[name] = 0.0 if not is_valid else float(norm)
            if not is_valid:
                logger.info("signal_causal_pruned", symbol=symbol, signal=name)

        direction = weighted_sum / total_weight if total_weight > 0 else 0.0
        # Apply the regime multiplier to the confidence (not the direction) so
        # a quiet regime dampens size without flipping sides.
        confidence = min(abs(direction), 1.0) * confidence_multiplier * float(regime_multiplier)

        # ── Guardian AI meta-labeling (soft-veto) ────────────────────────
        feature_vec = [contributing[name] for name in sorted(contributing.keys())]
        safety = self.guardian.predict_safety(feature_vec)
        if not safety["is_safe"]:
            logger.info("guardian_veto_softened", symbol=symbol, probability=safety["probability"])
        confidence *= float(safety["probability"])

        # ── Portfolio heat dampening ─────────────────────────────────────
        current_heat = self.trade_logger.get_portfolio_heat()
        if current_heat > self.max_portfolio_heat:
            logger.info(
                "portfolio_heat_limit_reached",
                heat=current_heat,
                limit=self.max_portfolio_heat,
            )
            confidence *= 0.5

        # ── Structural hysteresis & flip-dampening (single pass) ─────────
        now = datetime.utcnow()
        last_dir = self._last_directions.get(symbol, 0.0)
        last_time = self._last_signal_times.get(symbol, datetime.min)

        is_reversal = (direction > 0 and last_dir < 0) or (direction < 0 and last_dir > 0)
        if is_reversal and (now - last_time) < FLIP_MIN_INTERVAL:
            logger.info("flip_dampening_active", symbol=symbol, cooldown=str(FLIP_MIN_INTERVAL))
            direction = last_dir
            confidence *= 0.5  # hold direction but discount conviction

        if abs(direction - last_dir) < HYSTERESIS_THRESHOLD:
            direction = last_dir

        # Always persist the latest snapshot — the old code only updated when
        # the symbol was already present, so the first decision never armed
        # the hysteresis machinery.
        self._last_directions[symbol] = direction
        self._last_signal_times[symbol] = now

        logger.info(
            "fused_signal_generated",
            symbol=symbol,
            direction=f"{direction:.4f}",
            confidence=f"{confidence:.4f}",
            regime=regime,
            agree_count=agree_count,
        )

        return FusedSignal(
            symbol=symbol,
            direction=float(np.clip(direction, -1, 1)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            contributing_signals=contributing,
            metadata={
                "regime": regime,
                "consensus_multiplier": confidence_multiplier,
                "regime_multiplier": float(regime_multiplier),
                "guardian_probability": safety["probability"],
                "portfolio_heat": current_heat,
                "causal_pruning_active": True,
                "hysteresis_active": True,
                "ic_gate_min_abs": IC_GATE_MIN_ABS,
                "portfolio_nav_usd": float(self._portfolio_nav_usd),
            },
        )

    def _update_regime_ic_weights(self, signals: dict[str, pd.Series], forward_returns: pd.Series, regime: str) -> None:
        """Update signal weights specific to the current market regime based on rolling IC."""
        if regime not in self._regime_ic_history:
            return

        for name, signal_series in signals.items():
            n = min(len(signal_series), len(forward_returns))
            if n < 30:
                continue
            ic, _ = stats.spearmanr(signal_series.values[:n], forward_returns.values[:n])
            if not np.isnan(ic):
                if name not in self._regime_ic_history[regime]:
                    self._regime_ic_history[regime][name] = []
                
                self._regime_ic_history[regime][name].append(ic)
                # Use exponentially-weighted average IC as weight for THIS regime
                recent_ics = self._regime_ic_history[regime][name][-20:]
                weights = np.exp(np.linspace(-1, 0, len(recent_ics)))
                avg_ic = np.average(recent_ics, weights=weights)
                
                # Update the regime-specific weight matrix
                self.regime_weights[regime][name] = max(abs(avg_ic) * 10, 0.1)
