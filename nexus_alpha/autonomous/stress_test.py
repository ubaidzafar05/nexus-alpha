
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from nexus_alpha.agents.lifecycle import StrategyAgentLifecycle
from nexus_alpha.core.trading_loop import TradingLoopOrchestrator
from nexus_alpha.config import NexusConfig
from nexus_alpha.log_config import get_logger

logger = get_logger("v9_stress_test")

class SwarmStressTester:
    """
    Final v9 Stress Test Orchestrator.
    Simulates 96 hours of market data with synthetic 'Crowding' events
    to verify autonomic Delta-Neutrality.
    """
    
    def __init__(self, hours: int = 96):
        self.hours = hours
        self.lifecycle = StrategyAgentLifecycle()
        self.registry_path = Path("data/tournament/v9_stress_registry.json")
        self.results = []
        
    async def run(self):
        print(f"🔥 STARTING NEXUS-ULTRA v9 96-HOUR STRESS TEST")
        print(f"🧬 Bootstrapping Swarm Intelligence...")
        self.lifecycle.bootstrap()
        
        start_time = datetime.utcnow()
        
        # We simulate cycles (e.g., 5-min intervals)
        total_cycles = (self.hours * 60) // 5 
        
        for cycle in range(total_cycles):
            current_sim_time = start_time + timedelta(minutes=cycle * 5)
            
            # Inject synthetic features
            features = self._generate_synthetic_features(cycle, total_cycles)
            
            print(f"🌀 Cycle {cycle+1}/{total_cycles} | sim_time: {current_sim_time.strftime('%H:%M:%S')}", end="\r")
            
            # Process cycle
            # 1. Update Lifecycle with 'tick'
            self.lifecycle.update({"tick": {"last_price": 50000 + cycle, "timestamp": current_sim_time.isoformat()}})
            
            # 2. Check Crowding & Symmetrize (simulated logic from God-Loop)
            for cluster_id in ["layer1", "alts"]:
                try:
                    cluster_delta = self.lifecycle.tournament.get_cluster_delta(cluster_id, features)
                    if abs(cluster_delta) > 0.75:
                        logger.warning(f"STRESS: Crowding detected in {cluster_id} delta={cluster_delta:.3f}")
                        self.lifecycle.symmetrize_cluster(cluster_id)
                except Exception as e:
                    logger.error(f"STRESS: Cluster check failed: {e}")
            
            # 3. Save state for telemetry
            self.lifecycle.tournament.save_swarm_state(self.registry_path)
            
            # 4. Metric Tracking
            report_snap = self._capture_metrics(cycle, features)
            self.results.append(report_snap)
            
            # Non-blocking yield for async event loop (if needed)
            if cycle % 10 == 0:
                await asyncio.sleep(0)
        
        print("\n\n✅ STRESS TEST COMPLETE.")
        self._generate_final_report()

    def _generate_synthetic_features(self, cycle: int, total: int) -> dict:
        """
        Generate market features.
        Every 48 cycles (4 hours), we inject a 'Flash Crowding' event 
        where all agents see high confidence in one direction.
        """
        is_crowding_event = (cycle % 48 == 0) and cycle > 0
        
        features = {
            "open": np.array([50000.0] * 100),
            "high": np.array([50100.0] * 100),
            "low": np.array([49900.0] * 100),
            "close": np.array([50000.0] * 100),
            "volume": np.array([1000.0] * 100),
            "vpin": np.array([0.8 if is_crowding_event else 0.2] * 100),
            "ofi_l2": np.array([5.0 if is_crowding_event else 1.0] * 100),
            "rsi_7": np.array([80.0 if is_crowding_event else 50.0] * 100)
        }
        return features

    def _capture_metrics(self, cycle: int, features: dict) -> dict:
        snapshot = {
            "cycle": cycle,
            "timestamp": datetime.utcnow().isoformat(),
            "clusters": {}
        }
        for cluster_id in ["layer1", "alts"]:
            delta = self.lifecycle.tournament.get_cluster_delta(cluster_id, features)
            snapshot["clusters"][cluster_id] = {
                "delta": round(delta, 3),
                "num_agents": len([a for a in self.lifecycle.tournament.agents.values() if a.cluster_id == cluster_id]),
                "num_hedges": len([a for a in self.lifecycle.tournament.agents.values() if a.cluster_id == cluster_id and a.metadata.get("hedge")])
            }
        return snapshot

    def _generate_final_report(self):
        report_path = Path("artifacts/v9_stress_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        violation_count = 0
        max_delta = 0.0
        
        for res in self.results:
            for c_id, c_data in res["clusters"].items():
                max_delta = max(max_delta, abs(c_data["delta"]))
                if abs(c_data["delta"]) > 0.85:
                    violation_count += 1
        
        summary = {
            "status": "PASS" if violation_count == 0 else "FAIL",
            "total_cycles": len(self.results),
            "violation_count": violation_count,
            "max_observed_delta": round(max_delta, 3),
            "final_swarm_size": len(self.lifecycle.tournament.agents),
            "report_time": datetime.utcnow().isoformat()
        }
        
        with open(report_path, "w") as f:
            json.dump({
                "summary": summary,
                "history": self.results
            }, f, indent=4)
        
        print(f"📊 STRESS REPORT GENERATED: {report_path}")
        print(f"   Status: {summary['status']}")
        print(f"   Peak Delta: {summary['max_observed_delta']}")

if __name__ == "__main__":
    tester = SwarmStressTester(hours=24) # Start with 24 hours for first verification
    asyncio.run(tester.run())
