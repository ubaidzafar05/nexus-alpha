"""
Reinforcement Fine-Tuning (RFT) Vault
Local buffering system for ART framework trajectories.

Converts multi-agent debates into standard ChatML format with
thought traces for DeepSeek-R1 GRPO training.
"""

from __future__ import annotations

import json
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

RFT_DIR = Path("data/rft")
TRAJECTORY_FILE = RFT_DIR / "trajectories.jsonl"


class ARTVault:
    """
    Lightweight JSONL logger that acts as local storage for ART trajectories.
    """

    def __init__(self, filepath: Path = TRAJECTORY_FILE):
        self._filepath = filepath
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        # Touch file if it doesn't exist
        if not self._filepath.exists():
            self._filepath.touch()

    def record_debate(
        self,
        symbol: str,
        proposal_text: str,
        challenge_text: str,
        synthesis_text: str,
        system_prompt: str,
        market_context_prompt: str,
    ) -> str:
        """
        Record a complete debate as an ART Trajectory.
        Returns the unique trajectory_id.
        """
        trajectory_id = f"trt_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        # We model the debate as a Multi-turn conversation to teach the model
        # how to synthesize conflicting viewpoints.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"MARKET CONTEXT:\n{market_context_prompt}\n\nPROPOSAL:\n{proposal_text}\n\nCHALLENGE:\n{challenge_text}\n\nSynthesize the debate and output the final proceeding action."},
            {"role": "assistant", "content": synthesis_text}
        ]
        
        record = {
            "id": trajectory_id,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "messages": messages,
            "reward": 0.0,  # Will be updated when trade closes
            "status": "pending_outcome"
        }
        
        self._append_to_jsonl(record)
        logger.info("rft_trajectory_recorded", trajectory_id=trajectory_id, symbol=symbol)
        
        return trajectory_id

    def update_reward(self, trajectory_id: str, reward: float) -> bool:
        """
        Update a pending trajectory with its final PnL reward.
        Since JSONL is append-only for speed, we rewrite the file.
        This is acceptable for the expected volume (< 100 trades/day).
        """
        if not self._filepath.exists():
            return False
            
        updated = False
        lines = []
        try:
            with open(self._filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("id") == trajectory_id:
                            record["reward"] = reward
                            record["status"] = "rewarded"
                            updated = True
                        lines.append(json.dumps(record))
                    except json.JSONDecodeError:
                        lines.append(line.strip()) # keep corrupted lines but dont crash
                        
            if updated:
                # Write back
                with open(self._filepath, 'w', encoding='utf-8') as f:
                    for line in lines:
                        f.write(line + "\n")
                logger.info("rft_reward_updated", trajectory_id=trajectory_id, reward=round(reward,4))
                return True
                
        except Exception as e:
            logger.error("failed_to_update_rft_reward", error=str(e), trajectory_id=trajectory_id)
            
        return False

    def _append_to_jsonl(self, record: dict) -> None:
        try:
            with open(self._filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error("failed_to_write_trajectory", error=str(e))
