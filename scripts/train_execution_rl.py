"""
Training Script for NEXUS-ULTRA Execution RL Agent.
Trains the PPO-based urgency optimizer in the High-Fidelity LOB Simulation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from nexus_alpha.execution.rl_execution_agent import RLExecutionAgent
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Execution RL Agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--save-path", type=str, default="data/checkpoints/rl_execution_v1.zip", help="Path to save the model")
    args = parser.parse_args()

    # Ensure save directory exists
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("starting_execution_rl_training", target_timesteps=args.timesteps, save_path=args.save_path)

    agent = RLExecutionAgent(model_path=args.save_path)
    
    # Run training
    agent.train(timesteps=args.timesteps)

    if os.path.exists(args.save_path):
        logger.info("training_successful", final_path=args.save_path)
    else:
        logger.error("training_failed", path=args.save_path)

if __name__ == "__main__":
    main()
