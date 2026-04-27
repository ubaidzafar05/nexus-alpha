import pandas as pd
import numpy as np
from nexus_alpha.learning.causality import CausalSignalValidator

df = pd.DataFrame({
    'signal': np.random.normal(0, 1, 100),
    'volatility': np.random.normal(0.5, 0.1, 100),
    'market_regime': np.random.choice([0, 1], 100)
})
df['returns'] = df['signal'] * 0.01 + np.random.normal(0, 0.001, 100)
df['aligned_returns'] = df['returns'] * np.sign(df['signal'])
validator = CausalSignalValidator()
res = validator.validate_signal_causality(df, "signal", target_col="aligned_returns")
print(res)
