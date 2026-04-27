import numpy as np
from typing import Dict, Any

class ATRPositionSizer:
    """Volatility-adjusted position sizing using ATR."""
    
    def __init__(self, risk_per_trade_usd: float = 100.0, atr_multiplier: float = 2.0):
        self.risk_per_trade_usd = risk_per_trade_usd
        self.atr_multiplier = atr_multiplier
        
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range with numerical safety guards."""
        if len(close) < period + 1:
            return 0.0
            
        try:
            tr = np.maximum(high[1:] - low[1:], 
                            np.maximum(abs(high[1:] - close[:-1]), 
                                       abs(low[1:] - close[:-1])))
            
            # Simple Moving Average of True Range
            atr = np.mean(tr[-period:])
            
            # Numerical Stability Guard
            if not np.isfinite(atr) or atr < 0:
                return 0.0
                
            return float(atr)
        except Exception:
            return 0.0

    def get_position_size(self, current_price: float, atr: float, account_balance_usd: float) -> Dict[str, Any]:
        """
        Calculate units to trade based on ATR.
        Distance to Stop Loss = ATR * multiplier
        Size = Risk_USD / Distance
        """
        # Safety Guard: Fallback if ATR is zero, NaN, or non-finite
        if not np.isfinite(atr) or atr <= 0:
            # Fallback to 1% account risk if ATR is unavailable
            units = (account_balance_usd * 0.01) / current_price
            return {
                "units": float(units) if np.isfinite(units) else 0.0,
                "stop_loss_dist": 0.0,
                "is_fallback": True,
                "reason": "non_finite_atr"
            }
            
        stop_loss_dist = atr * self.atr_multiplier
        
        # Risk / Distance per unit
        # Numerical Guard: Prevent division by zero or extremely small ATR
        if stop_loss_dist < (current_price * 0.0001):
             stop_loss_dist = current_price * 0.01 # Floor at 1% price distance
        
        units = self.risk_per_trade_usd / stop_loss_dist
        
        # Cap at 20% of account balance for safety
        max_notional = account_balance_usd * 0.20
        units = min(units, max_notional / current_price)
        
        # Final Final Safety Guard
        if not np.isfinite(units) or units < 0:
            units = 0.0
            
        return {
            "units": float(units),
            "notional_usd": float(units * current_price),
            "stop_loss_dist": float(stop_loss_dist),
            "stop_loss_price": float(current_price - stop_loss_dist), 
            "is_fallback": False
        }
