"""
NEXUS-ULTRA Microstructure Engine — Tick-Level Signal Generation.

Calculates high-frequency signals:
1. TickVPIN (Volume-Weighted Probability of Informed Trading)
2. Order Flow Imbalance (OFI)
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class MicrostructureSignal:
    symbol: str
    timestamp: str
    vpin: float
    ofi: float
    buy_volume: float
    sell_volume: float
    regime: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


class TickVPINCalculator:
    """
    Volume-Weighted Probability of Informed Trading (VPIN).
    
    VPIN = (1 / n) * sum(|V_buy - V_sell| / V_bucket)
    
    Approximates the presence of informed traders by analyzing the 
    order flow toxicity (imbalance) relative to total volume.
    """

    def __init__(
        self,
        symbol: str,
        bucket_size: float,  # Target volume per bucket
        window_buckets: int = 50,
    ) -> None:
        self.symbol = symbol
        self.bucket_size = bucket_size
        self.window_buckets = window_buckets
        
        self._current_buy_vol = 0.0
        self._current_sell_vol = 0.0
        self._current_total_vol = 0.0
        
        # Stores (abs(buy - sell)) for completed buckets
        self._imbalances: collections.deque[float] = collections.deque(maxlen=window_buckets)
        self._latest_vpin = 0.0

    def update(self, price: float, volume: float, is_buy: bool) -> float:
        """Process a single trade and return the current VPIN."""
        if is_buy:
            self._current_buy_vol += volume
        else:
            self._current_sell_vol += volume
        
        self._current_total_vol += volume
        
        # If bucket is "full", complete it and slide the window
        if self._current_total_vol >= self.bucket_size:
            bucket_imbalance = abs(self._current_buy_vol - self._current_sell_vol)
            self._imbalances.append(bucket_imbalance)
            
            # Reset bucket
            self._current_buy_vol = 0.0
            self._current_sell_vol = 0.0
            self._current_total_vol = 0.0
            
            # Calculate smoothed VPIN
            if len(self._imbalances) > 0:
                self._latest_vpin = sum(self._imbalances) / (len(self._imbalances) * self.bucket_size)
                
        return self._latest_vpin


class OFIEngine:
    """
    Order Flow Imbalance (OFI).
    
    OFI = d(Volume_Bid) - d(Volume_Ask)
    
    Measures the net impact of limit order arrivals, cancellations, 
    and removals at the best bid/ask.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._prev_bid = 0.0
        self._prev_bid_sz = 0.0
        self._prev_ask = 0.0
        self._prev_ask_sz = 0.0
        self._latest_ofi = 0.0

    def update(self, bid: float, bid_sz: float, ask: float, ask_sz: float) -> float:
        """Process an L2 snapshot and return the OFI contribution."""
        if self._prev_bid == 0.0:
            # Initialize state
            self._prev_bid, self._prev_bid_sz = bid, bid_sz
            self._prev_ask, self._prev_ask_sz = ask, ask_sz
            return 0.0

        dbid = 0.0
        if bid > self._prev_bid:
            dbid = bid_sz
        elif bid == self._prev_bid:
            dbid = bid_sz - self._prev_bid_sz
        else:
            dbid = -self._prev_bid_sz

        dask = 0.0
        if ask < self._prev_ask:
            dask = ask_sz
        elif ask == self._prev_ask:
            dask = ask_sz - self._prev_ask_sz
        else:
            dask = -self._prev_ask_sz

        # OFI = dbid - dask
        self._latest_ofi = dbid - dask
        
        # Update state
        self._prev_bid, self._prev_bid_sz = bid, bid_sz
        self._prev_ask, self._prev_ask_sz = ask, ask_sz
        
        return self._latest_ofi


class L2MicrostructureEngine:
    """
    Facade orchestrating VPIN and OFI for a set of symbols.
    """

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols
        # Default bucket size: 50 BTC for VPIN (should be dynamic based on avg daily vol)
        self.vpin_calculators = {s: TickVPINCalculator(s, bucket_size=50.0) for s in symbols}
        self.ofi_engines = {s: OFIEngine(s) for s in symbols}

    def process_trade(self, symbol: str, price: float, volume: float, side: str) -> float:
        calc = self.vpin_calculators.get(symbol)
        if not calc:
            return 0.0
        is_buy = side.lower() in ("buy", "taker_buy")
        return calc.update(price, volume, is_buy)

    def process_orderbook(self, symbol: str, bid: float, bid_sz: float, ask: float, ask_sz: float) -> float:
        eng = self.ofi_engines.get(symbol)
        if not eng:
            return 0.0
        return eng.update(bid, bid_sz, ask, ask_sz)
