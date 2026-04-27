"""
Cross-Asset Microstructure Correlation Engine.
Detects leadership dynamics between different trading symbols.
"""

import collections
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

class LeadLagCorrelationTracker:
    """
    Tracks two symbol microstructure streams and computes rolling correlation
    at various lags to identify Order Flow Leadership.
    """
    
    def __init__(self, leader_symbol: str, follower_symbol: str, window_size: int = 100):
        self.leader = leader_symbol
        self.follower = follower_symbol
        self.window_size = window_size
        
        # Microstructure value buffers
        self._leader_vpin = collections.deque(maxlen=window_size)
        self._follower_vpin = collections.deque(maxlen=window_size)
        
        self._leader_ofi = collections.deque(maxlen=window_size)
        self._follower_ofi = collections.deque(maxlen=window_size)
        
        # Last known values for alignment
        self._last_vpin = {leader_symbol: 0.0, follower_symbol: 0.0}
        self._last_ofi = {leader_symbol: 0.0, follower_symbol: 0.0}

    def update(self, symbol: str, vpin: float, ofi: float) -> None:
        """Update tracker with new data for one of the symbols."""
        self._last_vpin[symbol] = vpin
        self._last_ofi[symbol] = ofi
        
        # Every time a follower updates, we record a snapshot of the pair
        if symbol == self.follower:
            self._leader_vpin.append(self._last_vpin[self.leader])
            self._follower_vpin.append(vpin)
            self._leader_ofi.append(self._last_ofi[self.leader])
            self._follower_ofi.append(ofi)

    def get_correlation_stats(self) -> Dict[str, float]:
        """Compute current lead-lag stats."""
        if len(self._follower_vpin) < 20:
            return {"vpin_corr": 0.0, "ofi_corr": 0.0, "leadership_score": 0.0}
            
        vpin_corr = self._compute_rolling_corr(self._leader_vpin, self._follower_vpin)
        ofi_corr = self._compute_rolling_corr(self._leader_ofi, self._follower_ofi)
        
        # Leadership score: how strongly is the leader moving the follower in unison
        # High positive correlation in OFI suggests direct liquidity propagation
        leadership_score = (vpin_corr + ofi_corr) / 2.0
        
        return {
            "vpin_corr": float(vpin_corr),
            "ofi_corr": float(ofi_corr),
            "leadership_score": float(leadership_score)
        }

    def _compute_rolling_corr(self, leader_buf: collections.deque, follower_buf: collections.deque) -> float:
        l = np.array(leader_buf)
        f = np.array(follower_buf)
        
        if np.std(l) < 1e-10 or np.std(f) < 1e-10:
            return 0.0
            
        corr = np.corrcoef(l, f)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0
