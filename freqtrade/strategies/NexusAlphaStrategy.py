"""
NexusAlpha Freqtrade Strategy — Free Edition.

Combines:
  - Multi-timeframe trend following (EMA crossover + ADX)
  - Mean reversion (Bollinger Band + RSI)
  - Regime detection (Hurst Exponent + ADX)
  - Order book imbalance gate (Binance REST — free)
  - Kelly criterion position sizing
  - Dynamic ATR trailing stoploss
  - Sentiment gate from NEXUS-ALPHA free pipeline (Redis cache)

FreqAI variant (NexusAlphaMLStrategy) adds:
  - LightGBM directional classifier trained on rolling 90-day window
  - Walk-forward cross-validation via Freqtrade Hyperopt

Usage:
  freqtrade trade --strategy NexusAlphaStrategy --config config/config.json
  freqtrade backtesting --strategy NexusAlphaStrategy --timerange 20230101-20251231
  freqtrade hyperopt --strategy NexusAlphaMLStrategy --hyperopt-loss SharpeHyperOptLoss --epochs 500
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import talib.abstract as ta

from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter


class NexusAlphaStrategy(IStrategy):
    """
    NEXUS-ALPHA Free Edition — production Freqtrade strategy.
    All signal logic is self-contained; no paid data sources required.
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"

    # Multi-timeframe analysis
    informative_timeframes: list[str] = ["4h", "1d"]

    # Risk management
    minimal_roi = {
        "0": 0.08,
        "30": 0.05,
        "60": 0.03,
        "120": 0.01,
    }

    stoploss = -0.05
    use_custom_stoploss = True
    trailing_stop = False

    stake_currency = "USDT"
    stake_amount = "unlimited"
    max_open_trades = 5

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ── Hyperopt search space ─────────────────────────────────────────────────
    ema_fast = IntParameter(8, 30, default=21, space="buy")
    ema_slow = IntParameter(40, 80, default=55, space="buy")
    rsi_oversold = IntParameter(20, 40, default=30, space="buy")
    rsi_overbought = IntParameter(60, 80, default=70, space="sell")
    atr_multiplier = DecimalParameter(1.5, 3.5, default=2.5, space="sell")
    sentiment_threshold = DecimalParameter(0.2, 0.7, default=0.4, space="buy")

    # ── Sentiment cache (written by nexus-alpha intelligence pipeline) ────────
    _REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _sentiment_cache: dict[str, float] = {}

    def _get_sentiment(self, base_asset: str) -> float:
        """Pull pre-computed sentiment from Redis (set by HybridSentimentPipeline)."""
        try:
            import redis

            r = redis.from_url(self._REDIS_URL, decode_responses=True)
            value = r.get(f"sentiment:{base_asset.upper()}")
            return float(value) if value else 0.0
        except Exception:
            return self._sentiment_cache.get(base_asset, 0.0)

    @staticmethod
    def _hurst_exponent(price_series: np.ndarray) -> float:
        if len(price_series) < 20:
            return 0.5
        try:
            lags = range(2, min(20, len(price_series) // 2))
            tau = [
                np.std(np.subtract(price_series[lag:], price_series[:-lag]))
                for lag in lags
            ]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(poly[0])
        except Exception:
            return 0.5

    def _get_ob_imbalance(self, pair: str) -> float:
        """Order book imbalance from Binance REST — free, no API key required."""
        try:
            symbol = pair.replace("/", "")
            resp = requests.get(
                "https://api.binance.com/api/v3/depth",
                params={"symbol": symbol, "limit": 20},
                timeout=2,
            )
            ob = resp.json()
            bid_vol = sum(float(b[1]) for b in ob["bids"][:10])
            ask_vol = sum(float(a[1]) for a in ob["asks"][:10])
            total = bid_vol + ask_vol
            return (bid_vol - ask_vol) / total if total > 0 else 0.0
        except Exception:
            return 0.0

    # ── Indicator computation ─────────────────────────────────────────────────

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        base_asset = pair.split("/")[0]

        # Trend
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # Momentum
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_pct"] = (dataframe["close"] - bb["lowerband"]) / (
            bb["upperband"] - bb["lowerband"] + 1e-8
        )

        # Volume
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["volume_zscore"] = (
            dataframe["volume"] - dataframe["volume"].rolling(20).mean()
        ) / (dataframe["volume"].rolling(20).std() + 1e-8)

        # Regime detection
        dataframe["hurst"] = dataframe["close"].rolling(100).apply(
            self._hurst_exponent, raw=True
        )
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        dataframe["regime"] = "unknown"
        dataframe.loc[
            (dataframe["adx"] > 25) & (dataframe["hurst"] > 0.55), "regime"
        ] = "trending"
        dataframe.loc[
            (dataframe["adx"] < 20) & (dataframe["hurst"] < 0.45), "regime"
        ] = "mean_reverting"
        dataframe.loc[
            dataframe["atr_pct"] > dataframe["atr_pct"].rolling(30).mean() * 1.5,
            "regime",
        ] = "high_vol"

        # External signals (live mode only — skip in backtesting)
        if self.dp.runmode.value not in ("backtest", "hyperopt"):
            dataframe["sentiment"] = self._get_sentiment(base_asset)
            dataframe["ob_imbalance"] = self._get_ob_imbalance(pair)
        else:
            dataframe["sentiment"] = 0.0
            dataframe["ob_imbalance"] = 0.0

        return dataframe

    # ── Entry signal ──────────────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        trend_long = (
            (dataframe["regime"] == "trending")
            & (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1))
            & (dataframe["adx"] > 25)
            & (dataframe["volume_zscore"] > 0.5)
            & (dataframe["close"] > dataframe["ema_200"])
        )

        mean_rev_long = (
            (dataframe["regime"] == "mean_reverting")
            & (dataframe["bb_pct"] < 0.1)
            & (dataframe["rsi"] < self.rsi_oversold.value)
            & (dataframe["macdhist"] > dataframe["macdhist"].shift(1))
        )

        sentiment_ok = dataframe["sentiment"] > -self.sentiment_threshold.value
        ob_ok = dataframe["ob_imbalance"] > -0.3

        dataframe.loc[
            (trend_long | mean_rev_long) & sentiment_ok & ob_ok,
            "enter_long",
        ] = 1

        return dataframe

    # ── Exit signal ───────────────────────────────────────────────────────────

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"])
                | (dataframe["rsi"] > self.rsi_overbought.value)
                | (dataframe["regime"] == "high_vol")
            ),
            "exit_long",
        ] = 1
        return dataframe

    # ── Dynamic ATR stoploss ──────────────────────────────────────────────────

    def custom_stoploss(
        self,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(kwargs["pair"], self.timeframe)
        if dataframe.empty:
            return self.stoploss
        last = dataframe.iloc[-1]
        atr_stop = -(last["atr"] * self.atr_multiplier.value) / current_rate
        return max(atr_stop, self.stoploss)

    # ── Kelly position sizing ─────────────────────────────────────────────────

    def custom_stake_amount(
        self,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        entry_tag: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(kwargs["pair"], self.timeframe)
        if dataframe.empty:
            return proposed_stake * 0.1

        win_rate = 0.55
        win_loss_ratio = 1.5
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        fractional_kelly = kelly * 0.25  # Quarter Kelly — conservative

        regime = dataframe.iloc[-1]["regime"]
        regime_mult = {
            "trending": 1.0,
            "mean_reverting": 0.7,
            "high_vol": 0.3,
            "unknown": 0.5,
        }.get(regime, 0.5)

        atr_pct = dataframe.iloc[-1]["atr_pct"]
        target_vol = 0.02
        vol_scale = min(target_vol / max(atr_pct, 0.001), 1.0)

        nav = self.wallets.get_total_stake_amount()
        final = nav * fractional_kelly * regime_mult * vol_scale

        max_size = nav * 0.20
        min_size = 10.0
        return max(min_size, min(final, max_size))


# ── FreqAI variant ────────────────────────────────────────────────────────────

class NexusAlphaMLStrategy(NexusAlphaStrategy):
    """
    NEXUS-ALPHA ML Edition — adds FreqAI LightGBM classifier.

    FreqAI is built into Freqtrade (no extra cost).
    Trains a directional classifier on rolling 90-day windows with
    walk-forward re-training every 4 hours.
    """

    def feature_engineering_expand_all(
        self,
        dataframe: pd.DataFrame,
        period: int,
        **kwargs,
    ) -> pd.DataFrame:
        dataframe[f"%-hurst-period_{period}"] = dataframe["close"].rolling(100).apply(
            self._hurst_exponent, raw=True
        )
        dataframe[f"%-volume_zscore-period_{period}"] = (
            dataframe["volume"] - dataframe["volume"].rolling(period).mean()
        ) / (dataframe["volume"].rolling(period).std() + 1e-8)
        return dataframe

    def feature_engineering_standard(
        self,
        dataframe: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        pair = kwargs.get("metadata", {}).get("pair", "")
        base = pair.split("/")[0] if pair else ""
        dataframe["%-sentiment"] = self._get_sentiment(base) if base else 0.0
        dataframe["%-ob_imbalance"] = 0.0  # Live-only; filled by populate_indicators
        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dataframe["&-direction"] = np.where(
            dataframe["close"].shift(-24) > dataframe["close"] * 1.02,
            "long",
            np.where(
                dataframe["close"].shift(-24) < dataframe["close"] * 0.98,
                "short",
                "neutral",
            ),
        )
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe["&-direction"] == "long")
            & (dataframe["do_predict"] == 1)
            & (dataframe["sentiment"] > -self.sentiment_threshold.value),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe["&-direction"] == "short")
            | (dataframe["regime"] == "high_vol"),
            "exit_long",
        ] = 1
        return dataframe
