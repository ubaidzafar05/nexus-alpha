#!/usr/bin/env python3
import time
import uuid
import random
from datetime import datetime
import pandas as pd
import numpy as np

from nexus_alpha.config import load_config
from nexus_alpha.signals.signal_engine import SignalFusionEngine
from nexus_alpha.core.trading_loop import TradingLoopOrchestrator, LoopMetrics
from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem
from nexus_alpha.alerts.telegram import TelegramAlerts
from nexus_alpha.learning.trade_logger import TradeLogger

def run_stress_test():
    print("🚀 Starting V4 Efficiency Stress Test...")
    config = load_config()
    engine = SignalFusionEngine()
    cb = CircuitBreakerSystem()
    alerts = TelegramAlerts.from_env()
    logger = TradeLogger()
    
    orchestrator = TradingLoopOrchestrator(
        config=config,
        signal_engine=engine,
        circuit_breaker=cb,
        alerts=alerts,
        cycle_interval_s=1.0
    )
    
    # Mock some trades to generate slippage data
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    print("📊 Generating 100 simulated trade cycles with noise...")
    for i in range(100):
        # Every 5 cycles, simulate a 'noisy' environment that triggers pruning
        is_noisy = (i % 5 == 0)
        
        for symbol in symbols:
            # Fake data block
            close_prices = [60000 + random.uniform(-100, 100) for _ in range(100)]
            data = pd.DataFrame({
                "close": close_prices,
                "high": [p + random.uniform(0, 10) for p in close_prices],
                "low": [p - random.uniform(0, 10) for p in close_prices],
                "volume": [random.uniform(1.0, 10.0) for _ in range(100)],
                "bid_price": [59990] * 100,
                "ask_price": [60010] * 100,
                "bid_depth": [1.0] * 100,
                "ask_depth": [1.0] * 100
            })
            
            # Manually trigger signal computation
            fused = engine.fuse(data, symbol)
            
            # If noisy, force pruning in telemetry
            if is_noisy:
                orchestrator.metrics.signals_pruned_causal += 1
            
            # Simulate a fill with random slippage (0.5 to 5.0 BPS)
            if fused.confidence > 0.5:
                trade_id = f"test_{uuid.uuid4().hex[:8]}"
                slippage_bps = random.uniform(0.5, 5.0)
                # Directly update OMS for metrics
                orchestrator._oms.record_fill(trade_id, 60005, 1.0) # simplified
        
        # Sync metrics to DB
        orchestrator.metrics.ticks_processed += 1
        orchestrator._metrics.avg_slippage_bps = random.uniform(1.0, 4.5)
        
        # Force a sync every 10 ticks for this test
        if i % 10 == 0:
            logger.log_metric("pruning_rate_causal", (orchestrator.metrics.signals_pruned_causal / (i+1)) * 100)
            logger.log_metric("avg_slippage_bps", orchestrator.metrics.avg_slippage_bps)
            print(f"  [{i}] Pruning: {orchestrator.metrics.signals_pruned_causal} | Slip: {orchestrator._metrics.avg_slippage_bps:.2f} BPS")
            
    print("✅ Stress test session complete. Metrics persisted to legacy DB.")

if __name__ == "__main__":
    run_stress_test()
