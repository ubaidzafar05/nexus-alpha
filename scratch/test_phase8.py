import asyncio
import unittest
from nexus_alpha.intelligence.microstructure import TickVPINCalculator, OFIEngine

class TestPhase8Microstructure(unittest.TestCase):
    def test_vpin_calculation(self):
        # 50 BTC bucket
        calc = TickVPINCalculator(symbol="BTCUSDT", bucket_size=50.0, window_buckets=10)
        
        # Scenario: Balanced flow (VPIN should be 0)
        # 25 buy @ 10, 25 sell @ 10 = 50 vol. Imbalance = 0.
        vpin = calc.update(price=10.0, volume=25.0, is_buy=True)
        vpin = calc.update(price=10.0, volume=25.0, is_buy=False)
        self.assertEqual(vpin, 0.0)
        
        # Scenario: Toxic flow (VPIN should increase)
        # 50 buy @ 10 = 50 vol. Imbalance = 50.
        # Average imbalance = (0 + 50) / 2 = 25.
        # VPIN = 25 / 50 = 0.5
        vpin = calc.update(price=10.0, volume=50.0, is_buy=True)
        self.assertEqual(vpin, 0.5)

    def test_ofi_calculation(self):
        engine = OFIEngine(symbol="BTCUSDT")
        
        # 1. Price increases, size increases -> positive OFI
        ofi = engine.update(bid=100.0, bid_sz=10.0, ask=101.0, ask_sz=10.0)
        # First update initializes state, OFI is 0
        self.assertEqual(ofi, 0.0)
        
        # 2. Bid increases to 100.1 with 5 size
        ofi = engine.update(bid=100.1, bid_sz=5.0, ask=101.0, ask_sz=10.0)
        # Bid rose -> dbid = new_sz = 5. dask = 0 (no change). OFI = 5.
        self.assertEqual(ofi, 5.0)
        
        # 3. Ask decreases to 100.9 with 8 size
        ofi = engine.update(bid=100.1, bid_sz=5.0, ask=100.9, ask_sz=8.0)
        # Ask dropped -> dask = new_sz = 8. dbid = 0. OFI = 0 - 8 = -8.
        self.assertEqual(ofi, -8.0)

if __name__ == "__main__":
    unittest.main()
