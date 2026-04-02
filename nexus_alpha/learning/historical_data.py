"""
Phase 1: Historical Data Pipeline — download years of OHLCV data for free via CCXT.

Downloads data from Binance public API (no API key needed for OHLCV),
stores as Parquet files for efficient training, and builds feature matrices.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

DATA_DIR = Path("data/ohlcv")
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"]


async def download_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: str = "2022-01-01",
    until: str | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Download historical OHLCV data via CCXT (free, no API key needed).
    Handles pagination automatically and saves to Parquet.
    """
    import ccxt.async_support as ccxt_async

    exchange = ccxt_async.binance({"enableRateLimit": True})
    data_dir.mkdir(parents=True, exist_ok=True)

    since_ts = int(datetime.strptime(since, "%Y-%m-%d").timestamp() * 1000)
    until_ts = int(
        (datetime.strptime(until, "%Y-%m-%d") if until else datetime.utcnow()).timestamp() * 1000
    )

    all_candles: list[list] = []
    current_since = since_ts
    batch_count = 0

    try:
        await exchange.load_markets()
        logger.info("download_started", symbol=symbol, timeframe=timeframe, since=since)

        while current_since < until_ts:
            candles = await exchange.fetch_ohlcv(
                symbol, timeframe, since=current_since, limit=1000
            )
            if not candles:
                break

            all_candles.extend(candles)
            current_since = candles[-1][0] + 1
            batch_count += 1

            if batch_count % 50 == 0:
                logger.info(
                    "download_progress",
                    symbol=symbol,
                    candles=len(all_candles),
                    latest=datetime.fromtimestamp(candles[-1][0] / 1000).isoformat(),
                )

            # Respect rate limits
            await asyncio.sleep(exchange.rateLimit / 1000)

    finally:
        await exchange.close()

    if not all_candles:
        logger.warning("no_data_downloaded", symbol=symbol)
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol

    # Save as Parquet
    safe_symbol = symbol.replace("/", "_")
    output_path = data_dir / f"{safe_symbol}_{timeframe}.parquet"
    df.to_parquet(output_path, index=False)

    logger.info(
        "download_complete",
        symbol=symbol,
        timeframe=timeframe,
        candles=len(df),
        date_range=f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}",
        path=str(output_path),
    )
    return df


def load_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Load previously downloaded OHLCV data from Parquet."""
    safe_symbol = symbol.replace("/", "_")
    path = data_dir / f"{safe_symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} {timeframe}. Run download first.")
    return pd.read_parquet(path)


async def download_all(
    since: str = "2022-01-01",
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Download all symbols and timeframes."""
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    results = {}

    for symbol in symbols:
        for tf in timeframes:
            key = f"{symbol}_{tf}"
            try:
                df = await download_ohlcv(symbol=symbol, timeframe=tf, since=since)
                results[key] = df
            except Exception as err:
                logger.warning("download_failed", symbol=symbol, timeframe=tf, error=repr(err))

    logger.info("download_all_complete", datasets=len(results))
    return results


# ─── Feature Engineering ─────────────────────────────────────────────────────


def build_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Build ML-ready feature matrix from raw OHLCV data.
    All features are normalized and look-back safe (no future leakage).
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ── Returns at multiple horizons ──
    for period in [1, 4, 12, 24, 72]:
        feat[f"return_{period}"] = close.pct_change(period)

    # ── Volatility (rolling std of returns) ──
    ret1 = close.pct_change()
    for window in [12, 24, 72, 168]:
        feat[f"volatility_{window}"] = ret1.rolling(window).std()

    # ── RSI ──
    for period in [7, 14, 28]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        feat[f"rsi_{period}"] = (100 - 100 / (1 + rs)) / 100 - 0.5  # Center at 0

    # ── MACD ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    feat["macd_line"] = macd_line / close
    feat["macd_signal"] = macd_signal / close
    feat["macd_hist"] = macd_hist / close

    # ── Bollinger Band position ──
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["bb_position"] = ((close - bb_mid) / (2 * bb_std.replace(0, np.nan))).clip(-1, 1)

    # ── ATR normalized ──
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat["atr_norm"] = tr.rolling(14).mean() / close

    # ── Volume features ──
    feat["volume_sma_ratio"] = volume / volume.rolling(24).mean().replace(0, np.nan)
    feat["volume_change"] = volume.pct_change(1)

    # ── OBV slope ──
    obv = (np.sign(close.diff()) * volume).cumsum()
    feat["obv_slope"] = obv.diff(12) / (obv.rolling(12).std().replace(0, np.nan))

    # ── Price position in range ──
    for window in [24, 72, 168]:
        roll_high = high.rolling(window).max()
        roll_low = low.rolling(window).min()
        range_size = roll_high - roll_low
        feat[f"price_position_{window}"] = (
            ((close - roll_low) / range_size.replace(0, np.nan)) * 2 - 1
        ).clip(-1, 1)

    # ── Time features (cyclical) ──
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        feat["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        feat["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    # ── Forward returns (TARGET — only used for training labels) ──
    for horizon in [1, 4, 12, 24]:
        feat[f"target_{horizon}h"] = close.pct_change(horizon).shift(-horizon)

    # Drop warmup rows
    feat = feat.iloc[lookback:].copy()
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feat


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "target_1h",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Split features into train/val/test with NO data leakage (chronological split).
    Returns dict with X_train, y_train, X_val, y_val, X_test, y_test.
    """
    features = build_features(df)

    target_cols = [c for c in features.columns if c.startswith("target_")]
    feature_cols = [c for c in features.columns if not c.startswith("target_")]

    X = features[feature_cols].values.astype(np.float32)
    y = features[target_col].values.astype(np.float32)

    # Chronological split (NO shuffle — prevents leakage)
    n = len(X)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
    }
