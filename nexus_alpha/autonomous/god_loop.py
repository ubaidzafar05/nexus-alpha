import time
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from nexus_alpha.learning.trade_logger import TradeLogger
from nexus_alpha.learning.guardian import GuardianAI

logger = logging.getLogger(__name__)

class GodLoop:
    """Master Autonomous Loop: Train -> Eval -> Promote."""
    
    def __init__(self, symbol: str = "BTC/USDT", interval_minutes: int = 60):
        self.symbol = symbol
        self.interval = interval_minutes * 60
        self.base_dir = Path(__file__).parent.parent.parent
        self.registry_path = self.base_dir / "data/tournament/registry.json"
        self.trade_logger = TradeLogger()
        self.guardian = GuardianAI()
        
    def run_cycle(self):
        """Execute one complete improvement cycle."""
        logger.info(f"🌀 STARTING GOD-LOOP CYCLE: {self.symbol}")
        
        # 1. Train Challenger
        # We run a fast Optuna study to find a better configuration
        logger.info("  👉 Phase 1: Training Challenger...")
        subprocess.run([
            sys.executable, "-m", "nexus_alpha.cli", "train-rl", 
            "--symbol", self.symbol, 
            "--episodes", "50" # Faster cycle for autonomous mode
        ])
        
        # 2. Evaluate Candidates
        # This will populate registry.json with a new candidate if found
        logger.info("  👉 Phase 2: Evaluating Candidates...")
        subprocess.run([
            sys.executable, "-m", "nexus_alpha.cli", "walk-forward", 
            "--symbol", self.symbol
        ])
        
        # 3. Decision Logic (Tournament)
        logger.info("  👉 Phase 3: Tournament Arena...")
        self.run_tournament()
        
        # 4. Guardian Retraining (Evolution Phase)
        logger.info("  👉 Phase 4: Guardian AI Retraining...")
        self.train_guardian()
        
    def train_guardian(self):
        """Retrain the Guardian AI if we have enough new data."""
        dataset = self.trade_logger.get_training_data(min_trades=50)
        if dataset:
            results = self.guardian.fit_on_history(dataset)
            if results["status"] == "success":
                logger.info(f"  🛰️  Guardian AI updated. Accuracy: {results['accuracy']:.2f}")
        else:
            logger.info("  🛰️  Guardian AI training skipped (insufficient new trade data).")
        
    def run_tournament(self):
        """
        V6 ULTRA: Competitive Tournament Selection.
        Promotes the best candidate to Champion based on multi-axis T-Score.
        """
        from nexus_alpha.autonomous.tournament import TournamentEngine
        
        logger.info("  ⚖️  Executing Tournament Arena...")
        engine = TournamentEngine(registry_path=str(self.registry_path))
        new_champion = engine.run_tournament()
        
        if new_champion:
            logger.info(f"  🏆 NEW CHAMPION CROWNED: {new_champion['id']} (T-Score: {new_champion.get('t_score', 0):.4f})")
            # We would typically trigger a system reload or hot-swap here if needed
        else:
            logger.info("  ⚖️  Tournament complete: Champion defended status or no decisively better challengers found.")

    def start(self):
        """Infinite autonomous loop."""
        logger.info("🔥 NEXUS GOD-LOOP ACTIVATED")
        while True:
            try:
                self.run_cycle()
                logger.info(f"💤 Cycle complete. Sleeping for {self.interval/60} minutes.")
                time.sleep(self.interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"God-Loop Failure: {e}")
                time.sleep(60) # Cool down before retry

if __name__ == "__main__":
    loop = GodLoop()
    loop.start()
