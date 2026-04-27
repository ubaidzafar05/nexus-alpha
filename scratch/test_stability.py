import pandas as pd
import numpy as np
import unittest
from nexus_alpha.learning.causality import CausalSignalValidator

class TestStability(unittest.TestCase):
    def test_singular_matrix_protection(self):
        validator = CausalSignalValidator()
        
        # Create data with zero-variance confounders
        # This used to cause singular matrix errors in statsmodels
        df = pd.DataFrame({
            "signal": [1.0] * 100,
            "future_return": np.random.normal(0, 0.01, 100),
            "volatility": [0.0] * 100,  # STATIC
            "market_regime": [1] * 100  # STATIC
        })
        
        # Should not crash, should handle the static columns gracefully
        # In this specific case, it will actually return True (skipped_static_treatment) 
        # because 'treatment' (abs(signal) > 0.5) will be static [1] * 100.
        result = validator.validate_signal_causality(df, "signal")
        self.assertTrue(result)
        
        # Test with varying treatment but static confounders
        df["signal"] = [1.0, 0.0] * 50
        result = validator.validate_signal_causality(df, "signal")
        # Should now run regression with epsilon-stabilized confounders and not crash
        self.assertIsInstance(result, bool)

if __name__ == "__main__":
    unittest.main()
