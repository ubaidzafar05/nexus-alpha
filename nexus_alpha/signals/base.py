from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
from nexus_alpha.schema_types import Signal

class SignalCategory(str, Enum):
    MICROSTRUCTURE = "microstructure"
    TECHNICAL = "technical"
    STATISTICAL_ARB = "statistical_arb"
    ML_PREDICTION = "ml_prediction"
    OPTIONS_FLOW = "options_flow"
    ON_CHAIN = "on_chain"
    SENTIMENT = "sentiment"
    MACRO = "macro"

class BaseSignalGenerator(ABC):
    """Abstract base for all signal generators."""
    def __init__(self, name: str, category: SignalCategory):
        self.name = name
        self.category = category

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute the signal for the given dataframe (vectorized preferred)."""
        pass
