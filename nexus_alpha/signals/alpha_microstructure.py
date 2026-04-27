import numpy as np
from typing import Dict, Any, Optional

class OrderFlowAlpha:
    """Microstructure alpha based on Order Book Imbalance (OBI)."""
    
    def __init__(self, window: int = 10):
        self.window = window
        self.history = []
        
    def calculate_imbalance(self, bids: list, asks: list) -> float:
        """
        Calculate Bid/Ask volume imbalance at top of book.
        bids/asks format: [[price, quantity], ...]
        """
        if not bids or not asks:
            return 0.0
            
        # Consider top 5 levels for a more stable imbalance score
        depth = min(5, len(bids), len(asks))
        bid_vol = sum(float(b[1]) for b in bids[:depth])
        ask_vol = sum(float(a[1]) for a in asks[:depth])
        
        if (bid_vol + ask_vol) == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def generate_signal(self, l2_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a signal segment based on OBI.
        Returns: { 'direction': 1|-1|0, 'confidence': 0.0-1.0, 'metadata': {...} }
        """
        bids = l2_data.get('bids', [])
        asks = l2_data.get('asks', [])
        
        imbalance = self.calculate_imbalance(bids, asks)
        self.history.append(imbalance)
        if len(self.history) > self.window:
            self.history.pop(0)
            
        # Smoothing the imbalance
        smoothed_obi = np.mean(self.history)
        
        # Signal Generation Logic:
        # Strong imbalance (>0.4 or <-0.4) triggers a confidence signal
        direction = 0
        if smoothed_obi > 0.4:
            direction = 1
        elif smoothed_obi < -0.4:
            direction = -1
            
        confidence = min(abs(smoothed_obi), 1.0)
        
        return {
            "direction": direction,
            "confidence": confidence,
            "metadata": {
                "raw_imbalance": float(imbalance),
                "smoothed_obi": float(smoothed_obi),
                "depth_analyzed": min(5, len(bids), len(asks))
            }
        }
