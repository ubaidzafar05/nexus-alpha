"""
Historical backtester — replays OHLCV candles through the full signal pipeline.

Uses the *exact same* logic as the live trading loop:
  - SignalFusionEngine (9 technical generators)
  - ML model predictions (LightweightPredictor)
  - Signal-conflict arbitration
  - Kelly-based position sizing
  - ATR-based SL/TP with min 0.5% floor
  - Trailing stops
  - Max 3 concurrent positions
  - Volume / momentum guards

This gives a realistic estimate of what the bot would have done historically,
including transaction costs and slippage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from nexus_alpha.learning.entry_features import build_augmented_feature_vector
from nexus_alpha.learning.historical_data import build_features, load_ohlcv
from nexus_alpha.learning.offline_trainer import LightweightPredictor
from nexus_alpha.signals.signal_engine import SignalFusionEngine

logger = logging.getLogger(__name__)

# ── Constants (mirror trading_loop.py) ────────────────────────────────────────

MIN_SIGNAL_CONFIDENCE = 0.45
MAX_OPEN_POSITIONS = 3
SL_FLOOR_PCT = 0.025   # 2.5% minimum stop-loss
SL_CAP_PCT = 0.06      # 6% maximum stop-loss
SL_ATR_MULT = 3.0
TP_ATR_MULT = 8.0
TRAILING_TRIGGER_PCT = 0.04  # 4% profit triggers trailing
TRAIL_ATR_MULT = 2.5
FEE_PCT = 0.00075           # Binance discounted/maker-like fee
SLIPPAGE_PCT = 0.0005        # 0.05% slippage per trade
MIN_VOLUME_RATIO = 0.5
MOMENTUM_LOOKBACK = 5
MAX_CLUSTER_POSITIONS = 2

CORRELATION_CLUSTERS = {
    "layer1": {"BTCUSDT", "ETHUSDT", "SOLUSDT"},
    "altcoin": {"BNBUSDT", "ADAUSDT"},
}

REGIME_CONFIDENCE = {
    "strong_trend": 1.2,
    "weak_trend": 1.0,
    "range_bound": 0.7,
    "high_volatility": 0.6,
    "unknown": 0.9,
}


@dataclass
class StrategyParams:
    """Tunable strategy parameters — pass to HistoricalBacktester."""
    min_confidence: float = 0.45
    max_positions: int = 3
    sl_atr_mult: float = 3.0
    sl_floor_pct: float = 0.025
    sl_cap_pct: float = 0.06
    tp_atr_mult: float = 8.0
    min_tp_sl_ratio: float = 2.0      # TP must be ≥ N × SL
    trailing_trigger: float = 0.04
    trail_atr_mult: float = 2.5
    breakeven_trigger: float = 0.02
    time_exit_bars: int = 120
    time_exit_loss_bars: int = 60
    cooldown_bars: int = 48
    require_ml_agreement: bool = True
    use_trend_filter: bool = True
    use_fixed_tp: bool = False
    kelly_fraction: float = 0.5
    regime_dampening: float = 0.7


@dataclass
class BacktestTrade:
    trade_id: int
    symbol: str
    side: Literal["buy", "sell"]
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: float
    take_profit: float | None
    exit_price: float = 0.0
    exit_time: datetime | None = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    peak_favorable: float = 0.0   # max favorable excursion
    peak_adverse: float = 0.0     # max adverse excursion
    holding_bars: int = 0
    confidence: float = 0.0
    ml_agreed: bool = False
    signal_direction: float = 0.0
    feature_vector: list[float] = field(default_factory=list)
    regime: str = "unknown"
    entry_context: dict[str, object] = field(default_factory=dict)
    exit_context: dict[str, object] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Summary statistics from a backtest run."""
    start_date: str = ""
    end_date: str = ""
    symbols: list[str] = field(default_factory=list)
    initial_capital: float = 100_000.0
    final_nav: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_holding_bars: float = 0.0
    total_fees: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)


class HistoricalBacktester:
    """
    Replay historical candles through the full signal + ML pipeline.

    Key design decisions:
    - Walk-forward: models trained on data BEFORE the test period are used.
      We use the already-trained models (trained on full history). For strict
      walk-forward, retrain models at each step — too slow for quick iteration.
    - Signals computed on rolling 200-candle windows (same as live).
    - ATR computed on 14 candles of the replay window.
    - All guards applied: volume, max positions, duplicate, correlation.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        initial_capital: float = 100_000.0,
        fee_pct: float = FEE_PCT,
        slippage_pct: float = SLIPPAGE_PCT,
        params: StrategyParams | None = None,
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.params = params or StrategyParams()

        # Portfolio state
        self.cash = initial_capital
        self.positions: list[BacktestTrade] = []
        self.closed_trades: list[BacktestTrade] = []
        self.equity_curve: list[float] = []
        self.timestamps: list[datetime] = []
        self.trade_counter = 0
        self.total_fees = 0.0

        # Signal engine (same as live)
        self.signal_engine = SignalFusionEngine()
        self.signal_engine.register_defaults()
        self._signal_total_weight = sum(abs(w) for w in self.signal_engine.signal_weights.values()) or 1.0

        # ML models
        self.ml_models: dict[str, LightweightPredictor] = {}
        self._feature_columns: list[str] = []
        self._load_ml_models()

    def _load_ml_models(self) -> None:
        ckpt_dir = Path("data/checkpoints")
        for sym in self.symbols:
            safe = sym.replace("/", "_")
            exchange_sym = sym.replace("/", "")  # BTCUSDT
            for timeframe in ("1h", "4h", "1d"):
                predictor = LightweightPredictor(target_horizon="target_1h")
                path = ckpt_dir / f"lightweight_{safe}_{timeframe}.pkl"
                if predictor.load(path):
                    key = exchange_sym if timeframe == "1h" else f"{exchange_sym}_{timeframe}"
                    self.ml_models[key] = predictor
        logger.info("backtest_ml_models_loaded count=%s", len(self.ml_models))

    def _cluster_for_symbol(self, symbol: str) -> str | None:
        for cluster_name, members in CORRELATION_CLUSTERS.items():
            if symbol in members:
                return cluster_name
        return None

    def _cluster_position_count(self, symbol: str) -> int:
        cluster = self._cluster_for_symbol(symbol)
        if cluster is None:
            return 0
        members = CORRELATION_CLUSTERS[cluster]
        return sum(1 for pos in self.positions if pos.symbol in members)

    def _prepare_symbol_cache(
        self,
        df: pd.DataFrame,
        exchange_sym: str,
        warmup: int,
    ) -> pd.DataFrame:
        cache = df.copy().reset_index(drop=True)
        cache["timestamp"] = pd.to_datetime(cache["timestamp"])
        close = pd.to_numeric(cache["close"], errors="coerce")
        high = pd.to_numeric(cache["high"], errors="coerce")
        low = pd.to_numeric(cache["low"], errors="coerce")
        volume = pd.to_numeric(cache["volume"], errors="coerce")

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        cache["atr"] = tr.rolling(14).mean()
        cache["ema50"] = close.ewm(span=50, adjust=False).mean()
        cache["momentum_return"] = close / close.shift(MOMENTUM_LOOKBACK - 1) - 1

        avg_20 = volume.shift(5).rolling(20).mean()
        recent_5 = volume.rolling(5).mean()
        cache["volume_ok"] = ((recent_5 / avg_20.replace(0, np.nan)) >= MIN_VOLUME_RATIO).fillna(True)

        vol_avg = volume.rolling(20).mean()
        vol_score = (volume / vol_avg.replace(0, np.nan)).clip(upper=2.0).fillna(1.0) / 2.0
        returns = close.pct_change()
        vol_24h = returns.rolling(24).std() * 100
        vol_quality = pd.Series(1.0, index=cache.index, dtype=float)
        vol_quality = vol_quality.mask(vol_24h < 0.3, 0.3)
        vol_quality = vol_quality.mask(vol_24h > 5.0, 0.4)
        mid_mask = (vol_24h >= 0.3) & (vol_24h <= 5.0)
        vol_quality.loc[mid_mask] = 1.0 - (vol_24h.loc[mid_mask] - 1.5).abs() / 5.0
        spread_pct = ((high - low) / close.replace(0, np.nan)).fillna(0.1)
        spread_score = (1.0 - spread_pct * 20).clip(lower=0.0, upper=1.0)
        cache["pair_quality"] = (vol_score * 0.4 + vol_quality * 0.35 + spread_score * 0.25).clip(0.0, 1.0)

        close_24 = close.shift(24)
        cache["trend24"] = np.where(close_24 > 0, close / close_24 - 1, 0.0)
        direction_sign = np.sign(close.pct_change().fillna(0.0))
        cache["directional_persistence_24"] = direction_sign.rolling(24, min_periods=1).mean().clip(-1.0, 1.0)
        vol_short = close.pct_change().rolling(24, min_periods=8).std()
        vol_long = close.pct_change().rolling(168, min_periods=24).std()
        vol_ratio = (vol_short / vol_long.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        cache["volatility_compression"] = (1.0 - vol_ratio).clip(-1.0, 1.0).fillna(0.0)

        all_signals = self.signal_engine.compute_all(cache)
        weighted_sum = pd.Series(0.0, index=cache.index, dtype=float)
        for name, signal_series in all_signals.items():
            series = pd.to_numeric(signal_series, errors="coerce").fillna(0.0)
            rolling_mean = series.rolling(warmup, min_periods=warmup).mean()
            rolling_std = series.rolling(warmup, min_periods=warmup).std(ddof=0)
            normalized = ((series - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-3, 3) / 3
            weighted_sum += self.signal_engine.signal_weights.get(name, 1.0) * normalized.fillna(0.0)
            cache[f"signal_{name}"] = normalized.fillna(0.0)
        cache["fused_direction"] = (weighted_sum / self._signal_total_weight).clip(-1.0, 1.0)
        cache["fused_confidence"] = cache["fused_direction"].abs().clip(0.0, 1.0)

        features = build_features(cache)
        feature_cols = [c for c in features.columns if not c.startswith("target_")]
        if feature_cols and not self._feature_columns:
            self._feature_columns = list(feature_cols)
        if feature_cols:
            for col in feature_cols:
                cache[col] = 0.0
            cache.loc[features.index, feature_cols] = features[feature_cols].astype(float).values
        cache["ml_signal"] = 0.0
        cache["ml_confidence"] = 0.0
        cache["ml_signal_4h"] = 0.0
        cache["ml_signal_1d"] = 0.0
        if len(features) > 0:
            X = features[feature_cols].values.astype(np.float32)
            for key, signal_col, conf_col in (
                (exchange_sym, "ml_signal", "ml_confidence"),
                (f"{exchange_sym}_4h", "ml_signal_4h", None),
                (f"{exchange_sym}_1d", "ml_signal_1d", None),
            ):
                predictor = self.ml_models.get(key)
                if predictor is None:
                    continue
                signals, confidences = predictor.predict_batch(X)
                cache.loc[features.index, signal_col] = signals
                if conf_col is not None:
                    cache.loc[features.index, conf_col] = confidences

        cache["ready"] = False
        cache.loc[warmup - 1 :, "ready"] = True
        return cache.set_index("timestamp", drop=False)

    def _prepare_regime_series(self, ref_cache: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(ref_cache["close"], errors="coerce")
        returns = close.pct_change()
        vol_recent = returns.rolling(24).std()
        vol_long = returns.rolling(168).std()
        vol_ratio = (vol_recent / vol_long.replace(0, np.nan)).fillna(1.0)
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        trend_strength = ((sma20 - sma50).abs() / sma50.replace(0, np.nan)).fillna(0.0)
        regime = pd.Series("unknown", index=ref_cache.index, dtype=object)
        regime.loc[trend_strength > 0.01] = "weak_trend"
        regime.loc[trend_strength > 0.03] = "strong_trend"
        regime.loc[vol_ratio > 1.5] = "high_volatility"
        return regime.map(lambda name: REGIME_CONFIDENCE.get(str(name), 0.9)).fillna(0.9)

    def _regime_name_from_multiplier(self, multiplier: float) -> str:
        for name, value in REGIME_CONFIDENCE.items():
            if abs(value - multiplier) < 1e-9:
                return name
        return "unknown"

    def _row_feature_vector(self, row: pd.Series) -> list[float]:
        if not self._feature_columns:
            return []
        vector: list[float] = []
        for col in self._feature_columns:
            value = row.get(col, 0.0)
            if pd.isna(value):
                value = 0.0
            vector.append(float(value))
        return vector

    def _nav(self, current_prices: dict[str, float]) -> float:
        """Compute net asset value."""
        position_value = 0.0
        for pos in self.positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == "buy":
                position_value += pos.quantity * price
            else:
                position_value -= pos.quantity * price
        return self.cash + position_value

    def _compute_atr(self, candles: pd.DataFrame, period: int = 14) -> float:
        """ATR from OHLCV candles."""
        if len(candles) < period + 1:
            return 0.0
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)
        close = candles["close"].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def _get_sl_tp(
        self, entry_price: float, atr: float, side: str,
    ) -> tuple[float, float | None]:
        """Compute SL/TP with floor and cap from strategy params."""
        p = self.params
        if atr > 0 and entry_price > 0:
            atr_pct = atr / entry_price
            sl_pct = max(p.sl_atr_mult * atr_pct, p.sl_floor_pct)
            sl_pct = min(sl_pct, p.sl_cap_pct)
        else:
            sl_pct = p.sl_floor_pct
        tp_pct = max(p.tp_atr_mult * (atr / entry_price), sl_pct * p.min_tp_sl_ratio) if (
            p.use_fixed_tp and atr > 0 and entry_price > 0
        ) else None

        if side == "buy":
            sl = entry_price * (1 - sl_pct)
            tp = entry_price * (1 + tp_pct) if tp_pct is not None else None
        else:
            sl = entry_price * (1 + sl_pct)
            tp = entry_price * (1 - tp_pct) if tp_pct is not None else None
        return sl, tp

    def _apply_slippage(self, price: float, side: str) -> float:
        """Simulate slippage on entry."""
        if side == "buy":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _kelly_size(self, confidence: float, nav: float) -> float:
        """Simplified Kelly sizing for backtest (no DB dependency)."""
        p = self.params
        win_rate = 0.5 + confidence * 0.15
        avg_wl_ratio = 1.5
        kelly = (win_rate * avg_wl_ratio - (1 - win_rate)) / avg_wl_ratio
        kelly = max(kelly, 0.0)
        fraction = kelly * p.kelly_fraction
        fraction *= p.regime_dampening
        # Clamp: min 2% of NAV, max 15%
        if fraction < 0.02:
            fraction = 0.02
        fraction = min(fraction, 0.15)
        return nav * fraction

    def _get_ml_prediction(
        self, symbol: str, features_df: pd.DataFrame,
    ) -> dict | None:
        """Get ML prediction from pre-trained model."""
        exchange_sym = symbol.replace("/", "")
        predictor = self.ml_models.get(exchange_sym)
        if not predictor:
            return None
        try:
            feature_cols = [c for c in features_df.columns if not c.startswith("target_")]
            if len(features_df) == 0:
                return None
            last_row = features_df[feature_cols].iloc[-1].values.astype(np.float32)
            return predictor.predict(last_row)
        except Exception:
            return None

    def _multi_timeframe_alignment(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        direction_1h: float,
    ) -> float:
        scores = [1.0]
        if len(features_df) == 0:
            return 1.0
        try:
            feature_cols = [c for c in features_df.columns if not c.startswith("target_")]
            last_row = features_df[feature_cols].iloc[-1].values.astype(np.float32)
        except Exception:
            return 1.0

        predictor_4h = self.ml_models.get(f"{symbol}_4h")
        if predictor_4h:
            try:
                pred = predictor_4h.predict(last_row)
                if abs(pred.get("signal", 0.0)) > 0.01:
                    scores.append(1.0 if (pred["signal"] > 0) == (direction_1h > 0) else 0.0)
            except Exception:
                pass

        predictor_1d = self.ml_models.get(f"{symbol}_1d")
        if predictor_1d:
            try:
                pred = predictor_1d.predict(last_row)
                if abs(pred.get("signal", 0.0)) > 0.01:
                    scores.append(1.0 if (pred["signal"] > 0) == (direction_1h > 0) else 0.0)
            except Exception:
                pass

        close = pd.to_numeric(features_df.get("close"), errors="coerce")
        if len(close) > 24 and close.iloc[-24] > 0:
            trend_24h = close.iloc[-1] / close.iloc[-24] - 1
            scores.append(1.0 if (trend_24h > 0) == (direction_1h > 0) else 0.3)

        return sum(scores) / len(scores)

    def _multi_timeframe_alignment_from_row(self, row: pd.Series, direction_1h: float) -> float:
        scores = [1.0]
        ml_4h = float(row.get("ml_signal_4h", 0.0))
        if abs(ml_4h) > 0.01:
            scores.append(1.0 if (ml_4h > 0) == (direction_1h > 0) else 0.0)
        ml_1d = float(row.get("ml_signal_1d", 0.0))
        if abs(ml_1d) > 0.01:
            scores.append(1.0 if (ml_1d > 0) == (direction_1h > 0) else 0.0)
        trend_24h = float(row.get("trend24", 0.0))
        scores.append(1.0 if (trend_24h > 0) == (direction_1h > 0) else 0.3)
        return sum(scores) / len(scores)

    def _volume_confirms(self, window: pd.DataFrame) -> bool:
        if len(window) < 25:
            return True
        try:
            vol = pd.to_numeric(window["volume"], errors="coerce")
            avg_20 = vol.iloc[-25:-5].mean()
            recent_5 = vol.iloc[-5:].mean()
            if avg_20 <= 0:
                return True
            return (recent_5 / avg_20) >= MIN_VOLUME_RATIO
        except Exception:
            return True

    def _momentum_confirms(self, window: pd.DataFrame, direction: float) -> bool:
        if len(window) < MOMENTUM_LOOKBACK + 1:
            return True
        try:
            close = pd.to_numeric(window["close"], errors="coerce")
            recent_return = close.iloc[-1] / close.iloc[-MOMENTUM_LOOKBACK] - 1
            if direction > 0 and recent_return < -0.03:
                return False
            if direction < 0 and recent_return > 0.03:
                return False
            return True
        except Exception:
            return True

    def _pair_quality_score(self, window: pd.DataFrame) -> float:
        if len(window) < 30:
            return 0.5
        close = pd.to_numeric(window["close"], errors="coerce")
        high = pd.to_numeric(window["high"], errors="coerce")
        low = pd.to_numeric(window["low"], errors="coerce")
        volume = pd.to_numeric(window["volume"], errors="coerce")

        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_now = volume.iloc[-1]
        vol_score = min(vol_now / vol_avg, 2.0) / 2.0 if vol_avg > 0 else 0.5

        returns = close.pct_change().dropna()
        vol_24h = returns.tail(24).std() * 100 if len(returns) >= 24 else 1.0
        if vol_24h < 0.3:
            vol_quality = 0.3
        elif vol_24h > 5.0:
            vol_quality = 0.4
        else:
            vol_quality = 1.0 - abs(vol_24h - 1.5) / 5.0

        spread_pct = (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] if close.iloc[-1] > 0 else 0.1
        spread_score = max(0.0, 1.0 - spread_pct * 20)

        quality = vol_score * 0.4 + vol_quality * 0.35 + spread_score * 0.25
        return round(max(0.0, min(quality, 1.0)), 3)

    def _detect_regime(self, ref_window: pd.DataFrame) -> str:
        if len(ref_window) < 50:
            return "unknown"
        close = pd.to_numeric(ref_window["close"], errors="coerce")
        returns = close.pct_change().dropna()
        if len(returns) < 30:
            return "unknown"
        vol_recent = returns.tail(24).std()
        vol_long = returns.tail(168).std() if len(returns) >= 168 else vol_recent
        vol_ratio = vol_recent / vol_long if vol_long > 0 else 1.0
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        trend_strength = abs(sma20 - sma50) / sma50 if sma50 > 0 else 0.0
        if vol_ratio > 1.5:
            return "high_volatility"
        if trend_strength > 0.03:
            return "strong_trend"
        if trend_strength > 0.01:
            return "weak_trend"
        return "range_bound"

    def _regime_confidence_multiplier(self, ref_window: pd.DataFrame) -> float:
        return REGIME_CONFIDENCE.get(self._detect_regime(ref_window), 0.9)

    def _arbitrate_signals(self, signals: list[dict]) -> list[dict]:
        if not signals:
            return []
        scored = []
        for sig in signals:
            composite = abs(sig["direction"]) * sig["confidence"] * sig["pair_quality"]
            scored.append((composite, sig))
        scored.sort(key=lambda item: item[0], reverse=True)

        cluster_direction: dict[str, float] = {}
        accepted: list[dict] = []
        for _, sig in scored:
            cluster = self._cluster_for_symbol(sig["symbol"]) or sig["symbol"]
            if cluster in cluster_direction:
                if (sig["direction"] > 0) != (cluster_direction[cluster] > 0):
                    continue
            else:
                cluster_direction[cluster] = sig["direction"]
            accepted.append(sig)
        return accepted

    def _check_exits(
        self,
        bar: pd.Series,
        current_prices: dict[str, float],
        current_atrs: dict[str, float],
    ) -> None:
        """Check SL/TP/trailing for all open positions."""
        to_close = []
        for pos in self.positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            high = float(bar.get(f"high_{pos.symbol}", price))
            low = float(bar.get(f"low_{pos.symbol}", price))
            atr = float(current_atrs.get(pos.symbol, 0.0))

            pos.holding_bars += 1

            # Track excursions using high/low of the bar
            if pos.side == "buy":
                favorable = (high - pos.entry_price) / pos.entry_price
                adverse = (pos.entry_price - low) / pos.entry_price
                pnl_pct = (price - pos.entry_price) / pos.entry_price
            else:
                favorable = (pos.entry_price - low) / pos.entry_price
                adverse = (high - pos.entry_price) / pos.entry_price
                pnl_pct = (pos.entry_price - price) / pos.entry_price

            pos.peak_favorable = max(pos.peak_favorable, favorable)
            pos.peak_adverse = max(pos.peak_adverse, adverse)

            # Move stop to breakeven after enough profit.
            if pnl_pct >= self.params.breakeven_trigger:
                if pos.side == "buy":
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price * 1.001)
                else:
                    pos.stop_loss = min(pos.stop_loss, pos.entry_price * 0.999)

            # Live parity: ATR-based progressive trailing from peak/trough, not
            # a fixed fraction of peak profit.
            if pnl_pct >= self.params.trailing_trigger and atr > 0:
                trail_dist = self.params.trail_atr_mult * atr
                if pos.side == "buy":
                    peak_price = getattr(pos, "_peak_price", high)
                    pos._peak_price = max(peak_price, high)
                    trail_sl = pos._peak_price - trail_dist
                    if trail_sl > pos.stop_loss:
                        pos.stop_loss = trail_sl
                else:
                    trough_price = getattr(pos, "_trough_price", low)
                    pos._trough_price = min(trough_price, low)
                    trail_sl = pos._trough_price + trail_dist
                    if trail_sl < pos.stop_loss:
                        pos.stop_loss = trail_sl

            # Check SL hit (using intra-bar high/low for realism)
            if pos.side == "buy" and low <= pos.stop_loss:
                exit_price = pos.stop_loss  # assume filled at SL
                to_close.append((pos, exit_price, "stop_loss"))
            elif pos.side == "sell" and high >= pos.stop_loss:
                exit_price = pos.stop_loss
                to_close.append((pos, exit_price, "stop_loss"))
            # Check TP hit
            elif pos.take_profit is not None and pos.side == "buy" and high >= pos.take_profit:
                exit_price = pos.take_profit
                to_close.append((pos, exit_price, "take_profit"))
            elif pos.take_profit is not None and pos.side == "sell" and low <= pos.take_profit:
                exit_price = pos.take_profit
                to_close.append((pos, exit_price, "take_profit"))
            # Time-based exit: close after N bars if neither SL nor TP hit
            elif pos.holding_bars >= (
                self.params.time_exit_loss_bars if pnl_pct < 0 else self.params.time_exit_bars
            ):
                to_close.append((pos, price, "time_exit"))

        for pos, exit_price, reason in to_close:
            self._close_position(pos, exit_price, reason)

    def _close_position(
        self, pos: BacktestTrade, exit_price: float, reason: str,
    ) -> None:
        """Close a position and record the trade."""
        # Apply slippage on exit
        if pos.side == "buy":
            exit_price *= (1 - self.slippage_pct)
        else:
            exit_price *= (1 + self.slippage_pct)

        # Calculate PnL
        if pos.side == "buy":
            raw_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.quantity

        # Deduct exit fee
        exit_fee = abs(pos.quantity * exit_price) * self.fee_pct
        self.total_fees += exit_fee
        net_pnl = raw_pnl - exit_fee

        # Update cash: for longs return sale proceeds, for shorts buy back
        if pos.side == "buy":
            self.cash += pos.quantity * exit_price - exit_fee
        else:
            # Short close: buy back shares (cash already has entry proceeds)
            self.cash -= pos.quantity * exit_price + exit_fee

        pos.exit_price = exit_price
        pos.exit_time = None  # set by caller
        pos.exit_reason = reason
        pos.pnl = net_pnl
        pos.pnl_pct = net_pnl / (pos.entry_price * pos.quantity) if pos.entry_price else 0

        self.positions.remove(pos)
        self.closed_trades.append(pos)

    def run(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2026-04-01",
        timeframe: str = "1h",
        progress_interval: int = 500,
    ) -> BacktestResult:
        """
        Run a backtest over the specified date range.

        Args:
            start_date: Backtest start (need 200 bars before this for warmup).
            end_date: Backtest end.
            timeframe: Candle timeframe (1h recommended).
            progress_interval: Print progress every N bars.
        """
        t0 = time.time()
        print(f"🔄 Loading data for {len(self.symbols)} symbols...")

        # Load all symbol data
        all_data: dict[str, pd.DataFrame] = {}
        all_cache: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            try:
                df = load_ohlcv(sym, timeframe)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                all_data[sym] = df
            except FileNotFoundError:
                print(f"  ⚠️  No data for {sym} — skipping")

        if not all_data:
            raise ValueError("No historical data found for any symbol")

        # Find common date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # Use the first symbol to build the timeline
        ref_sym = list(all_data.keys())[0]
        ref_df = all_data[ref_sym]
        timeline = ref_df[
            (ref_df["timestamp"] >= start_dt) & (ref_df["timestamp"] <= end_dt)
        ]["timestamp"].tolist()

        if not timeline:
            raise ValueError(f"No candles in range {start_date} → {end_date}")

        # Reset state
        self.cash = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.timestamps = []
        self.trade_counter = 0
        self.total_fees = 0.0

        warmup = 300  # bars needed before features are valid (200 indicator lookback + buffer)
        for sym, df in all_data.items():
            exchange_sym = sym.replace("/", "")
            all_cache[sym] = self._prepare_symbol_cache(df, exchange_sym, warmup)

        n_bars = len(timeline)
        print(f"📊 Backtesting {n_bars} bars from {start_date} to {end_date}")
        print(f"   Symbols: {', '.join(all_data.keys())}")
        print(f"   Models: {len(self.ml_models)} loaded")
        print(f"   Params: SL={self.params.sl_atr_mult}×ATR "
              f"[{self.params.sl_floor_pct:.1%}-{self.params.sl_cap_pct:.1%}], "
              f"TP={self.params.tp_atr_mult}×ATR, "
              f"conf≥{self.params.min_confidence}, "
              f"cooldown={self.params.cooldown_bars}h")
        print()

        last_exit_bar: dict[str, int] = {}  # symbol → bar index of last exit
        ref_cache = all_cache.get("BTC/USDT", all_cache[ref_sym])
        regime_mult_series = self._prepare_regime_series(ref_cache)

        for i, ts in enumerate(timeline):
            current_prices: dict[str, float] = {}
            current_atrs: dict[str, float] = {}

            # Build multi-column bar for exit checks
            bar_data = {}

            for sym in all_data:
                cache = all_cache[sym]
                if ts not in cache.index:
                    continue
                row = cache.loc[ts]
                if not bool(row.get("ready", False)):
                    continue

                close_price = float(row["close"])
                exchange_sym = sym.replace("/", "")
                current_prices[exchange_sym] = close_price
                current_atrs[exchange_sym] = float(row.get("atr", 0.0) or 0.0)
                bar_data[f"pair_quality_{exchange_sym}"] = float(row.get("pair_quality", 0.5) or 0.5)

                bar_data[f"high_{exchange_sym}"] = float(row["high"])
                bar_data[f"low_{exchange_sym}"] = float(row["low"])

            regime_mult = float(regime_mult_series.get(ts, 0.9))

            # 1. Check exits on existing positions
            bar_series = pd.Series(bar_data)
            self._check_exits(bar_series, current_prices, current_atrs)
            # Set exit times and record cooldowns
            for t in self.closed_trades:
                if t.exit_time is None:
                    t.exit_time = ts
                    last_exit_bar[t.symbol] = i
                    t.exit_context = {
                        "exit_reason": t.exit_reason,
                        "regime_at_exit": self._regime_name_from_multiplier(regime_mult),
                        "pair_quality_at_exit": float(bar_series.get(f"pair_quality_{t.symbol}", 0.5)),
                        "holding_bars": t.holding_bars,
                    }

            # 2. Generate signals for symbols with enough data
            signals = []
            for sym in all_data:
                cache = all_cache[sym]
                if ts not in cache.index:
                    continue
                row = cache.loc[ts]
                if not bool(row.get("ready", False)):
                    continue

                exchange_sym = sym.replace("/", "")

                # Skip if already have position in this symbol
                if any(p.symbol == exchange_sym for p in self.positions):
                    continue

                # Max positions check
                if len(self.positions) >= self.params.max_positions:
                    break

                # Cooldown: skip if recently exited this symbol
                last_exit = last_exit_bar.get(exchange_sym, -999)
                if (i - last_exit) < self.params.cooldown_bars:
                    continue

                try:
                    direction = float(row.get("fused_direction", 0.0))
                    base_confidence = float(row.get("fused_confidence", 0.0))
                    if np.isnan(direction) or np.isnan(base_confidence):
                        continue

                    # Blend ML prediction
                    ml_agreed = False
                    ml_signal = float(row.get("ml_signal", 0.0))
                    ml_conf = float(row.get("ml_confidence", 0.0))
                    if abs(ml_signal) > 0.01:
                        if (ml_signal > 0 and direction > 0) or (ml_signal < 0 and direction < 0):
                            confidence = min(base_confidence + 0.15 * ml_conf, 1.0)
                            ml_agreed = True
                        else:
                            confidence = max(base_confidence - 0.10 * ml_conf, 0.0)
                    else:
                        confidence = base_confidence

                    # Trend filter: only trade with the 50-EMA.
                    if self.params.use_trend_filter:
                        ema50 = float(row.get("ema50", 0.0))
                        current_close = float(row["close"])
                        if direction > 0 and current_close < ema50:
                            continue
                        if direction < 0 and current_close > ema50:
                            continue

                    # Skip if ML requirement enabled and ML doesn't agree
                    if self.params.require_ml_agreement and not ml_agreed:
                        continue

                    pair_quality = float(row.get("pair_quality", 0.5))
                    mtf_alignment = self._multi_timeframe_alignment_from_row(row, direction)
                    adjusted_confidence = confidence
                    adjusted_confidence *= (0.5 + 0.5 * mtf_alignment)
                    adjusted_confidence *= regime_mult
                    adjusted_confidence *= (0.6 + 0.4 * pair_quality)

                    if not bool(row.get("volume_ok", True)):
                        continue
                    momentum_return = float(row.get("momentum_return", 0.0))
                    if (direction > 0 and momentum_return < -0.03) or (direction < 0 and momentum_return > 0.03):
                        continue

                    final_confidence = min(adjusted_confidence, 1.0)
                    if final_confidence >= self.params.min_confidence:
                        signal_profile = {
                            name: float(row.get(f"signal_{name}", 0.0) or 0.0)
                            for name in self.signal_engine.signal_weights
                        }
                        feature_vector, derived_context = build_augmented_feature_vector(
                            self._row_feature_vector(row),
                            signal_confidence=final_confidence,
                            pair_quality=pair_quality,
                            mtf_alignment=mtf_alignment,
                            regime_multiplier=regime_mult,
                            ml_confidence=ml_conf,
                            ml_signal=ml_signal,
                            contributing_signals=signal_profile,
                            directional_persistence_24=float(row.get("directional_persistence_24", 0.0) or 0.0),
                            volatility_compression=float(row.get("volatility_compression", 0.0) or 0.0),
                            entry_price=float(row["close"]),
                            atr=float(row.get("atr", 0.0) or 0.0),
                            sl_atr_mult=self.params.sl_atr_mult,
                            sl_floor_pct=self.params.sl_floor_pct,
                            sl_cap_pct=self.params.sl_cap_pct,
                            breakeven_trigger_pct=self.params.breakeven_trigger,
                            trailing_trigger_pct=self.params.trailing_trigger,
                            trade_direction=direction,
                        )
                        signals.append({
                            "symbol": exchange_sym,
                            "direction": direction,
                            "confidence": final_confidence,
                            "ml_agreed": ml_agreed,
                            "atr": float(row.get("atr", 0.0) or 0.0),
                            "close": float(row["close"]),
                            "pair_quality": pair_quality,
                            "feature_vector": feature_vector,
                            "regime": self._regime_name_from_multiplier(regime_mult),
                            "entry_context": {
                                "pair_quality": pair_quality,
                                "mtf_alignment": mtf_alignment,
                                "regime_multiplier": regime_mult,
                                "ml_signal": ml_signal,
                                "ml_confidence": ml_conf,
                                "momentum_return": momentum_return,
                                "volume_ok": bool(row.get("volume_ok", True)),
                                "directional_persistence_24": float(row.get("directional_persistence_24", 0.0) or 0.0),
                                "volatility_compression": float(row.get("volatility_compression", 0.0) or 0.0),
                                **derived_context,
                            },
                        })
                except Exception:
                    continue

            # 3. Apply conflict arbitration, then rank by confidence.
            signals = self._arbitrate_signals(signals)
            signals.sort(key=lambda s: s["confidence"], reverse=True)
            remaining_slots = self.params.max_positions - len(self.positions)

            for sig in signals[:remaining_slots]:
                if self._cluster_position_count(sig["symbol"]) >= MAX_CLUSTER_POSITIONS:
                    continue
                nav = self._nav(current_prices)
                size_usd = self._kelly_size(sig["confidence"], nav)

                side = "buy" if sig["direction"] > 0 else "sell"
                entry_price = self._apply_slippage(sig["close"], side)
                quantity = size_usd / entry_price

                # Entry fee
                entry_fee = size_usd * self.fee_pct
                self.total_fees += entry_fee

                # Deduct from cash
                if side == "buy":
                    self.cash -= (quantity * entry_price + entry_fee)
                else:
                    # Short: receive proceeds, will settle later
                    self.cash += (quantity * entry_price - entry_fee)

                sl, tp = self._get_sl_tp(entry_price, sig["atr"], side)

                self.trade_counter += 1
                trade = BacktestTrade(
                    trade_id=self.trade_counter,
                    symbol=sig["symbol"],
                    side=side,
                    entry_price=entry_price,
                    entry_time=ts,
                    quantity=quantity,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=sig["confidence"],
                    ml_agreed=sig["ml_agreed"],
                    signal_direction=sig["direction"],
                    feature_vector=list(sig.get("feature_vector", [])),
                    regime=str(sig.get("regime", "unknown")),
                    entry_context=dict(sig.get("entry_context", {})),
                )
                self.positions.append(trade)

            # 4. Record equity
            nav = self._nav(current_prices)
            self.equity_curve.append(nav)
            self.timestamps.append(ts)

            # Progress
            if i > 0 and i % progress_interval == 0:
                elapsed = time.time() - t0
                pct = i / n_bars * 100
                ret = (nav / self.initial_capital - 1) * 100
                print(
                    f"  [{pct:5.1f}%] Bar {i}/{n_bars}  "
                    f"NAV: ${nav:,.0f} ({ret:+.1f}%)  "
                    f"Trades: {len(self.closed_trades)}  "
                    f"Open: {len(self.positions)}  "
                    f"({elapsed:.0f}s)"
                )

        # Close remaining positions at last price
        for pos in list(self.positions):
            sym_key = None
            for sym in all_data:
                if sym.replace("/", "") == pos.symbol:
                    sym_key = sym
                    break
            if sym_key:
                last_price = float(all_data[sym_key].iloc[-1]["close"])
            else:
                last_price = pos.entry_price
            self._close_position(pos, last_price, "backtest_end")
            self.closed_trades[-1].exit_time = timeline[-1]

        elapsed = time.time() - t0
        result = self._compute_stats(timeline)
        print(f"\n✅ Backtest complete in {elapsed:.1f}s")
        return result

    def export_closed_trades_to_logger(
        self,
        trade_logger,
        run_label: str,
        metadata: dict[str, object] | None = None,
    ) -> int:
        """Export closed replay trades into the persistent learning journal."""
        from nexus_alpha.learning.trade_logger import TradeRecord

        prefix = f"replay:{run_label}:"
        trade_logger.delete_trades_by_prefix(prefix)
        exported = 0
        for trade in self.closed_trades:
            entry_ts = trade.entry_time.isoformat()
            exit_ts = trade.exit_time.isoformat() if trade.exit_time is not None else entry_ts
            trade_id = f"{prefix}{trade.trade_id}"
            entry_ctx = dict(trade.entry_context)
            entry_ctx["source"] = "historical_replay"
            entry_ctx["run_label"] = run_label
            if metadata:
                entry_ctx.update(metadata)
            trade_logger.log_trade_open(
                TradeRecord(
                    trade_id=trade_id,
                    timestamp=entry_ts,
                    symbol=trade.symbol,
                    side=trade.side,
                    entry_price=trade.entry_price,
                    quantity=trade.quantity,
                    notional_usd=trade.entry_price * trade.quantity,
                    signal_direction=trade.signal_direction,
                    signal_confidence=trade.confidence,
                    contributing_signals=json.dumps({
                        "ml_agreed": trade.ml_agreed,
                        "exit_reason": trade.exit_reason,
                    }),
                    sentiment_score=0.0,
                    regime=trade.regime,
                    feature_vector=json.dumps(trade.feature_vector),
                    entry_context=json.dumps(entry_ctx),
                )
            )
            trade_logger.log_trade_close(
                trade_id=trade_id,
                exit_price=trade.exit_price,
                realized_pnl=trade.pnl,
                exit_context=json.dumps(trade.exit_context),
                exit_timestamp=exit_ts,
            )
            exported += 1
        return exported

    def _compute_stats(self, timeline: list) -> BacktestResult:
        """Compute comprehensive backtest statistics."""
        trades = self.closed_trades
        nav = self.equity_curve[-1] if self.equity_curve else self.initial_capital

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        # Equity curve analysis
        eq = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital])
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        # Returns for Sharpe
        if len(eq) > 1:
            returns = np.diff(eq) / eq[:-1]
            sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 24))
        else:
            returns = np.array([0.0])
            sharpe = 0.0

        total_return = (nav / self.initial_capital - 1) * 100
        win_rate = len(winners) / len(trades) if trades else 0.0

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean([t.pnl_pct for t in winners]) * 100 if winners else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losers]) * 100 if losers else 0.0

        calmar = total_return / (max_dd * 100) if max_dd > 0 else 0.0

        result = BacktestResult(
            start_date=str(timeline[0]) if timeline else "",
            end_date=str(timeline[-1]) if timeline else "",
            symbols=[s.replace("/", "") for s in self.symbols],
            initial_capital=self.initial_capital,
            final_nav=nav,
            total_return_pct=total_return,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            avg_win_pct=float(avg_win),
            avg_loss_pct=float(avg_loss),
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd * 100,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            avg_holding_bars=float(np.mean([t.holding_bars for t in trades])) if trades else 0.0,
            total_fees=self.total_fees,
            trades=trades,
            equity_curve=self.equity_curve,
            timestamps=self.timestamps,
        )
        return result


def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report to console."""
    print("\n" + "=" * 70)
    print("  NEXUS-ALPHA BACKTEST REPORT")
    print("=" * 70)
    print(f"  Period:          {result.start_date[:10]} → {result.end_date[:10]}")
    print(f"  Symbols:         {', '.join(result.symbols)}")
    print(f"  Initial Capital: ${result.initial_capital:,.0f}")
    print()
    print(f"  Final NAV:       ${result.final_nav:,.2f}")
    print(f"  Total Return:    {result.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Calmar Ratio:    {result.calmar_ratio:.2f}")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print()
    print(f"  Total Trades:    {result.total_trades}")
    print(f"  Winners:         {result.winning_trades} ({result.win_rate:.1%})")
    print(f"  Losers:          {result.losing_trades}")
    print(f"  Avg Win:         {result.avg_win_pct:+.2f}%")
    print(f"  Avg Loss:        {result.avg_loss_pct:+.2f}%")
    print(f"  Avg Holding:     {result.avg_holding_bars:.1f} bars")
    print(f"  Total Fees:      ${result.total_fees:,.2f}")
    print("=" * 70)

    if result.trades:
        # Top 5 best / worst trades
        sorted_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
        print("\n  TOP 5 WINNERS:")
        for t in sorted_trades[:5]:
            print(
                f"    {t.symbol} {t.side.upper()} "
                f"${t.pnl:+,.2f} ({t.pnl_pct:+.2%}) "
                f"held {t.holding_bars}h  exit={t.exit_reason}"
                f"  ml={'✓' if t.ml_agreed else '✗'}"
            )

        print("\n  TOP 5 LOSERS:")
        for t in sorted_trades[-5:]:
            print(
                f"    {t.symbol} {t.side.upper()} "
                f"${t.pnl:+,.2f} ({t.pnl_pct:+.2%}) "
                f"held {t.holding_bars}h  exit={t.exit_reason}"
                f"  ml={'✓' if t.ml_agreed else '✗'}"
            )

        # ML agreement stats
        ml_agree = [t for t in result.trades if t.ml_agreed]
        ml_disagree = [t for t in result.trades if not t.ml_agreed]
        if ml_agree:
            ml_wr = sum(1 for t in ml_agree if t.pnl > 0) / len(ml_agree)
            print(f"\n  ML Agreement Trades: {len(ml_agree)} (WR: {ml_wr:.1%})")
        if ml_disagree:
            no_ml_wr = sum(1 for t in ml_disagree if t.pnl > 0) / len(ml_disagree)
            print(f"  ML Disagree Trades:  {len(ml_disagree)} (WR: {no_ml_wr:.1%})")

        # Exit reason breakdown
        reasons = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("\n  EXIT REASONS:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pnl = sum(t.pnl for t in result.trades if t.exit_reason == reason)
            print(f"    {reason:20s}: {count:4d} trades  ${pnl:+,.2f}")

    print("=" * 70)
