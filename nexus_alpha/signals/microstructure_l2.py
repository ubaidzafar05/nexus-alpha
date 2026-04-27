"""
L2 Microstructure Intelligence Nodes.
High-fidelity signals based on tick-by-tick order flow and depth.
"""

from __future__ import annotations

import collections
import numpy as np
from typing import Dict, List, Any, Optional

import pandas as pd
from nexus_alpha.log_config import get_logger
from nexus_alpha.signals.base import BaseSignalGenerator, SignalCategory

logger = get_logger(__name__)

class OrderFlowImbalance:
    """
    Measures the net pressure of order book updates.
    Based on Cont & de Larrard (2014) / OFI models.
    """
    
    def __init__(self, window: int = 20):
        self.window = window
        self._prev_bid_price: float | None = None
        self._prev_bid_size: float | None = None
        self._prev_ask_price: float | None = None
        self._prev_ask_size: float | None = None
        self._history = collections.deque(maxlen=window)

    def update(self, bid_price: float, bid_size: float, ask_price: float, ask_size: float) -> float:
        """Update OFI with new top-of-book snapshot."""
        if self._prev_bid_price is None:
            self._prev_bid_price = bid_price
            self._prev_bid_size = bid_size
            self._prev_ask_price = ask_price
            self._prev_ask_size = ask_size
            return 0.0

        # Bid pressure
        if bid_price > self._prev_bid_price:
            bid_delta = bid_size
        elif bid_price == self._prev_bid_price:
            bid_delta = bid_size - self._prev_bid_size
        else:
            bid_delta = -self._prev_bid_size

        # Ask pressure
        if ask_price < self._prev_ask_price:
            ask_delta = ask_size
        elif ask_price == self._prev_ask_price:
            ask_delta = ask_size - self._prev_ask_size
        else:
            ask_delta = -self._prev_ask_size

        ofi = bid_delta - ask_delta
        self._history.append(ofi)
        
        self._prev_bid_price = bid_price
        self._prev_bid_size = bid_size
        self._prev_ask_price = ask_price
        self._prev_ask_size = ask_size
        
        return float(np.mean(self._history)) if self._history else 0.0

class TickVPIN:
    """
    Volume-Synchronized Probability of Informed Trading (Tick-level).
    More precise than OHLCV-VPIN as it uses every trade update.
    """
    
    def __init__(self, bucket_volume: float = 1.0, n_buckets: int = 50):
        self.bucket_volume = bucket_volume
        self.n_buckets = n_buckets
        self._current_buy_vol = 0.0
        self._current_sell_vol = 0.0
        self._current_bucket_vol = 0.0
        self._buckets = collections.deque(maxlen=n_buckets)

    def update(self, price: float, volume: float, prev_price: float | None) -> float:
        """Update VPIN with a new realized trade."""
        if prev_price is None:
            return 0.5
            
        # Tick rule for aggressor identification
        if price > prev_price:
            buy_vol = volume
            sell_vol = 0.0
        elif price < prev_price:
            buy_vol = 0.0
            sell_vol = volume
        else:
            # Zero-tick rule: same as previous
            # For simplicity, split 50/50 on first tie
            buy_vol = volume * 0.5
            sell_vol = volume * 0.5

        self._current_buy_vol += buy_vol
        self._current_sell_vol += sell_vol
        self._current_bucket_vol += volume

        # If bucket is full, finalize it
        if self._current_bucket_vol >= self.bucket_volume:
            imbalance = abs(self._current_buy_vol - self._current_sell_vol)
            self._buckets.append(imbalance)
            
            # Reset for next bucket
            self._current_buy_vol = 0.0
            self._current_sell_vol = 0.0
            self._current_bucket_vol = 0.0

        if not self._buckets:
            return 0.0

        # VPIN = sum(imbalances) / (n_buckets * bucket_volume)
        vpin = sum(self._buckets) / (len(self._buckets) * self.bucket_volume)
        return float(vpin)

# ─── Signal Generator Wrappers ───────────────────────────────────────────────

class OFISignal(BaseSignalGenerator):
    """Signal wrapper for Order Flow Imbalance."""
    def __init__(self, window: int = 20):
        super().__init__("advanced_ofi", SignalCategory.MICROSTRUCTURE)
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if not all(k in data.columns for k in ["bid_price", "bid_depth", "ask_price", "ask_depth"]):
            return pd.Series(0.0, index=data.index)
        
        ofi_calc = OrderFlowImbalance(window=self.window)
        results = []
        for _, row in data.iterrows():
            ofi = ofi_calc.update(
                row["bid_price"], row["bid_depth"],
                row["ask_price"], row["ask_depth"]
            )
            results.append(ofi)
        return pd.Series(results, index=data.index)

class TickVPINSignal(BaseSignalGenerator):
    """Signal wrapper for Tick-level VPIN."""
    def __init__(self, bucket_volume: float = 10.0, n_buckets: int = 50):
        super().__init__("advanced_vpin", SignalCategory.MICROSTRUCTURE)
        self.bucket_volume = bucket_volume
        self.n_buckets = n_buckets

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "close" not in data.columns or "volume" not in data.columns:
            return pd.Series(0.0, index=data.index)
        
        vpin_calc = TickVPIN(bucket_volume=self.bucket_volume, n_buckets=self.n_buckets)
        results = []
        prev_price = None
        for _, row in data.iterrows():
            vpin = vpin_calc.update(row["close"], row["volume"], prev_price)
            results.append(vpin)
            prev_price = row["close"]
        return pd.Series(results, index=data.index)
