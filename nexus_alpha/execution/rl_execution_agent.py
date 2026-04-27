"""
RL Execution Agent — Hybrid Almgren-Chriss + PPO.
Uses Stable Baselines 3 for training and inference.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO  # type: ignore

from nexus_alpha.execution.lob_sim_env import LOBSimEnv
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)


class RLExecutionAgent:
    """
    Reinforcement Learning agent for optimal execution.
    Trained to adjust the Almgren-Chriss baseline quantity based on 
    real-time order book signals.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model: PPO | None = None
        self._env = LOBSimEnv()
        self._pass_through_warned = False
        if model_path and os.path.exists(model_path):
            try:
                self.load(model_path)
            except Exception as err:
                logger.warning("rl_agent_auto_load_failed", path=model_path, error=str(err))
        else:
            logger.warning(
                "rl_execution_agent_untrained",
                model_path=model_path,
                note="No checkpoint loaded; get_action_adjustment will pass-through (multiplier=1.0).",
            )

    def train(self, timesteps: int = 100000):
        """Train the agent in simulation."""
        logger.info("training_rl_execution_agent", timesteps=timesteps)
        self._model = PPO(
            "MlpPolicy",
            self._env,
            verbose=1,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,
        )
        self._model.learn(total_timesteps=timesteps)
        
        if self.model_path:
            self._model.save(self.model_path)
            logger.info("training_complete_saved", path=self.model_path)

    def load(self, model_path: str):
        """Load a pre-trained model."""
        self.model_path = model_path
        if os.path.exists(model_path):
            self._model = PPO.load(model_path)
            logger.info("model_loaded", path=model_path)
        else:
            logger.warning("model_path_not_found", path=model_path)

    def get_action_adjustment(
        self,
        inventory_ratio: float,
        time_ratio: float,
        bid_depth: float,
        ask_depth: float,
        spread_bps: float,
    ) -> float:
        """
        Predict the urgency adjustment based on LOB state.
        Returns a multiplier for the Almgren-Chriss baseline quantity.
        """
        if self._model is None:
            if not self._pass_through_warned:
                logger.info(
                    "rl_execution_pass_through",
                    reason="no_model_loaded",
                    multiplier=1.0,
                )
                self._pass_through_warned = True
            return 1.0  # Pass-through if no model

        obs = np.array([
            inventory_ratio,
            time_ratio,
            bid_depth,
            ask_depth,
            spread_bps,
        ], dtype=np.float32)

        action, _states = self._model.predict(obs, deterministic=True)
        # Action is Discrete(5) -> map 0-4 to multipliers 0.6, 0.8, 1.0, 1.2, 1.4
        multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]
        return multipliers[int(action)]
