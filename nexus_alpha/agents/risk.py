"""
Risk Intelligence Agents — Guardian nodes for tail-risk mitigation.
"""

import collections
import numpy as np
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from nexus_alpha.agents.tournament import BaseAgent
from nexus_alpha.schema_types import Signal
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

class TailHedgeAgent(BaseAgent):
    """
    Guardian Agent that monitors global microstructure stress.
    Generates NEUTRALIZATION signals when VPIN/OFI Z-scores indicate extreme tail risk.
    """
    
    def __init__(self, agent_id: Optional[str] = None, z_threshold: float = 3.5, window: int = 500):
        super().__init__(agent_id=agent_id or "guardian-v7", agent_type="risk-guardian")
        self.z_threshold = z_threshold
        self.window = window
        
        # Microstructure stress trackers
        self._vpin_history = collections.deque(maxlen=window)
        self._ofi_history = collections.deque(maxlen=window)
        
        logger.info("tail_hedge_agent_initialized", agent_id=self.agent_id, threshold=z_threshold)

    def update(self, market_data: dict) -> None:
        """Update stress metrics from incoming ticks."""
        if "tick" in market_data:
            tick = market_data["tick"]
            # We track VPIN/OFI intensity across all symbols to detect global stress
            vpin = tick.get("vpin", 0.5)
            ofi_abs = abs(tick.get("imbalance", 0.0))
            
            self._vpin_history.append(vpin)
            self._ofi_history.append(ofi_abs)

    def generate_signal(self, features: dict[str, np.ndarray]) -> Optional[Signal]:
        """Evaluate tail risk and emit preservation signals."""
        if len(self._vpin_history) < 100:
            return None
            
        vpin_arr = np.array(self._vpin_history)
        vpin_z = (vpin_arr[-1] - np.mean(vpin_arr)) / (np.std(vpin_arr) + 1e-10)
        
        # If stress is extreme, emit a high-confidence neutralization signal
        if vpin_z > self.z_threshold:
            logger.warning("tail_risk_detected", z_score=round(vpin_z, 2), agent_id=self.agent_id)
            return Signal(
                signal_id=uuid.uuid4().hex[:12],
                source=self.agent_id,
                symbol="ALL", # Global signal
                direction=0.0, # Neutralize
                confidence=min(0.5 + (vpin_z - self.z_threshold) * 0.1, 1.0),
                timestamp=datetime.utcnow(),
                timeframe="immediate",
                metadata={"risk_type": "microstructure_heat", "z_score": vpin_z}
            )
            
        return None
