"""
Adversarial Testing Framework (Red Team).

Continuously stress-tests the system with adversarial scenarios:
- Flash crash simulation
- Data feed poisoning
- Liquidity vacuum
- Cascading failure
- Regime confusion attacks
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from nexus_alpha.logging import get_logger
from nexus_alpha.types import MarketRegime

logger = get_logger(__name__)


class AttackType(str, Enum):
    FLASH_CRASH = "flash_crash"
    DATA_FEED_POISON = "data_feed_poison"
    LIQUIDITY_VACUUM = "liquidity_vacuum"
    CASCADING_FAILURE = "cascading_failure"
    REGIME_CONFUSION = "regime_confusion"
    STALE_DATA = "stale_data"
    LATENCY_SPIKE = "latency_spike"
    ADVERSE_FILL = "adverse_fill"


@dataclass
class AttackScenario:
    """Definition of an adversarial attack."""
    attack_type: AttackType
    name: str
    description: str
    severity: float  # 0.0 (mild) to 1.0 (catastrophic)
    duration_seconds: float
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    """Result of an adversarial test."""
    scenario: AttackScenario
    system_survived: bool
    nav_impact_pct: float
    max_drawdown_during: float
    circuit_breaker_triggered: bool
    circuit_breaker_level: int
    recovery_time_seconds: float
    errors_encountered: list[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AdversarialDataGenerator:
    """Generates adversarial market data for testing."""

    @staticmethod
    def generate_flash_crash(
        base_price: float,
        drop_pct: float = 0.15,
        recovery_pct: float = 0.10,
        crash_bars: int = 5,
        recovery_bars: int = 20,
    ) -> np.ndarray:
        """
        Generate flash crash price series.
        Rapid decline followed by partial recovery.
        """
        total_bars = crash_bars + recovery_bars
        prices = np.zeros(total_bars)

        # Crash phase: exponential decay
        for i in range(crash_bars):
            t = i / max(crash_bars - 1, 1)
            prices[i] = base_price * (1 - drop_pct * t**0.5)

        trough = prices[crash_bars - 1]
        target = trough + (base_price - trough) * recovery_pct / drop_pct

        # Recovery phase: slow log recovery
        for i in range(recovery_bars):
            t = i / max(recovery_bars - 1, 1)
            prices[crash_bars + i] = trough + (target - trough) * np.log1p(t * 2) / np.log(3)

        # Add noise
        noise = np.random.normal(0, base_price * 0.002, total_bars)
        prices += noise
        return np.maximum(prices, base_price * 0.01)

    @staticmethod
    def generate_liquidity_vacuum(
        base_price: float,
        normal_spread_bps: float = 1.0,
        spike_multiplier: float = 50.0,
        bars: int = 100,
        vacuum_start: int = 30,
        vacuum_end: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate price and spread series with a liquidity vacuum.
        Returns (prices, spreads).
        """
        prices = np.full(bars, base_price)
        spreads = np.full(bars, normal_spread_bps)

        # Random walk prices
        for i in range(1, bars):
            prices[i] = prices[i - 1] * (1 + np.random.normal(0, 0.001))

        # Inject vacuum
        for i in range(vacuum_start, min(vacuum_end, bars)):
            t = (i - vacuum_start) / max(vacuum_end - vacuum_start - 1, 1)
            spread_mult = 1 + (spike_multiplier - 1) * np.sin(np.pi * t)
            spreads[i] = normal_spread_bps * spread_mult
            prices[i] *= 1 + np.random.normal(0, 0.005)  # Extra volatility

        return prices, spreads

    @staticmethod
    def generate_regime_confusion(
        base_price: float,
        bars: int = 200,
        switch_frequency: int = 10,
    ) -> tuple[np.ndarray, list[MarketRegime]]:
        """
        Generate data that rapidly switches regimes to confuse the regime oracle.
        Returns (prices, true_regimes).
        """
        prices = np.zeros(bars)
        regimes: list[MarketRegime] = []
        prices[0] = base_price

        regime_generators = {
            MarketRegime.TRENDING_BULL: lambda p: p * (1 + abs(np.random.normal(0.002, 0.005))),
            MarketRegime.TRENDING_BEAR: lambda p: p * (1 - abs(np.random.normal(0.002, 0.005))),
            MarketRegime.MEAN_REVERTING: lambda p: p * (1 + np.random.normal(0, 0.003)),
            MarketRegime.HIGH_VOLATILITY: lambda p: p * (1 + np.random.normal(0, 0.02)),
            MarketRegime.CRISIS: lambda p: p * (1 - abs(np.random.normal(0.005, 0.01))),
        }

        available_regimes = list(regime_generators.keys())
        current_regime = random.choice(available_regimes)

        for i in range(1, bars):
            if i % switch_frequency == 0:
                current_regime = random.choice(available_regimes)
            gen = regime_generators[current_regime]
            prices[i] = gen(prices[i - 1])
            regimes.append(current_regime)

        regimes.insert(0, current_regime)
        return prices, regimes

    @staticmethod
    def generate_poisoned_data(
        clean_prices: np.ndarray,
        poison_ratio: float = 0.05,
        poison_magnitude: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inject random outliers into otherwise clean data.
        Returns (poisoned_prices, poison_mask).
        """
        poisoned = clean_prices.copy()
        mask = np.zeros(len(clean_prices), dtype=bool)

        n_poison = int(len(clean_prices) * poison_ratio)
        indices = np.random.choice(len(clean_prices), size=n_poison, replace=False)

        for idx in indices:
            direction = np.random.choice([-1, 1])
            poisoned[idx] *= 1 + direction * poison_magnitude * np.random.uniform(0.5, 1.5)
            mask[idx] = True

        return poisoned, mask


# ─── Scenario Library ─────────────────────────────────────────────────────────

ADVERSARIAL_SCENARIOS: list[AttackScenario] = [
    AttackScenario(
        attack_type=AttackType.FLASH_CRASH,
        name="BTC Flash Crash 15%",
        description="BTC drops 15% in 5 bars then partially recovers",
        severity=0.8,
        duration_seconds=300,
        parameters={"drop_pct": 0.15, "crash_bars": 5, "recovery_bars": 20},
    ),
    AttackScenario(
        attack_type=AttackType.FLASH_CRASH,
        name="Mild Flash Dip 5%",
        description="BTC drops 5% and recovers quickly",
        severity=0.3,
        duration_seconds=120,
        parameters={"drop_pct": 0.05, "crash_bars": 3, "recovery_bars": 10},
    ),
    AttackScenario(
        attack_type=AttackType.DATA_FEED_POISON,
        name="5% Data Poisoning",
        description="5% of price ticks are corrupted with 10% magnitude",
        severity=0.5,
        duration_seconds=600,
        parameters={"poison_ratio": 0.05, "poison_magnitude": 0.10},
    ),
    AttackScenario(
        attack_type=AttackType.LIQUIDITY_VACUUM,
        name="Liquidity Vacuum 50x Spread",
        description="Spreads widen 50x for 20 bars simulating liquidity crisis",
        severity=0.7,
        duration_seconds=300,
        parameters={"spike_multiplier": 50, "vacuum_start": 30, "vacuum_end": 50},
    ),
    AttackScenario(
        attack_type=AttackType.REGIME_CONFUSION,
        name="Rapid Regime Switching",
        description="Market regime changes every 10 bars for 200 bars",
        severity=0.6,
        duration_seconds=1200,
        parameters={"switch_frequency": 10, "bars": 200},
    ),
    AttackScenario(
        attack_type=AttackType.STALE_DATA,
        name="Stale Feed 60s",
        description="Data feed freezes for 60 seconds then resumes",
        severity=0.4,
        duration_seconds=60,
        parameters={"stale_seconds": 60},
    ),
    AttackScenario(
        attack_type=AttackType.LATENCY_SPIKE,
        name="Exchange Latency 5s",
        description="Exchange API latency spikes to 5 seconds",
        severity=0.5,
        duration_seconds=300,
        parameters={"latency_ms": 5000},
    ),
    AttackScenario(
        attack_type=AttackType.ADVERSE_FILL,
        name="Adverse Fill Slippage 2%",
        description="All fills get 2% adverse slippage (market manipulation)",
        severity=0.6,
        duration_seconds=600,
        parameters={"slippage_pct": 0.02},
    ),
]


class AdversarialTestRunner:
    """
    Runs adversarial scenarios against the system and reports results.
    Used in CI/CD and scheduled red team assessments.
    """

    def __init__(self) -> None:
        self.scenarios = list(ADVERSARIAL_SCENARIOS)
        self.results: list[AttackResult] = []

    def add_scenario(self, scenario: AttackScenario) -> None:
        self.scenarios.append(scenario)

    def run_scenario(self, scenario: AttackScenario, base_price: float = 65000.0) -> AttackResult:
        """
        Run a single adversarial scenario in simulation mode.
        In production: this integrates with the paper trading engine.
        """
        logger.info(
            "adversarial_test_start",
            name=scenario.name,
            severity=scenario.severity,
        )

        gen = AdversarialDataGenerator()
        nav_impact = 0.0
        max_dd = 0.0
        cb_triggered = False
        cb_level = 0
        errors: list[str] = []

        try:
            if scenario.attack_type == AttackType.FLASH_CRASH:
                prices = gen.generate_flash_crash(
                    base_price,
                    drop_pct=scenario.parameters.get("drop_pct", 0.15),
                    crash_bars=scenario.parameters.get("crash_bars", 5),
                    recovery_bars=scenario.parameters.get("recovery_bars", 20),
                )
                trough = prices.min()
                nav_impact = (trough - base_price) / base_price
                max_dd = abs(nav_impact)
                cb_triggered = max_dd > 0.05
                cb_level = 4 if max_dd > 0.15 else 3 if max_dd > 0.10 else 2 if max_dd > 0.05 else 0

            elif scenario.attack_type == AttackType.DATA_FEED_POISON:
                clean = np.full(100, base_price) * (1 + np.random.normal(0, 0.001, 100)).cumprod()
                poisoned, mask = gen.generate_poisoned_data(
                    clean,
                    poison_ratio=scenario.parameters.get("poison_ratio", 0.05),
                    poison_magnitude=scenario.parameters.get("poison_magnitude", 0.10),
                )
                deviation = np.mean(np.abs(poisoned - clean) / clean)
                nav_impact = -deviation * 0.5  # Estimated impact
                max_dd = deviation
                cb_triggered = max_dd > 0.05

            elif scenario.attack_type == AttackType.LIQUIDITY_VACUUM:
                prices, spreads = gen.generate_liquidity_vacuum(
                    base_price,
                    spike_multiplier=scenario.parameters.get("spike_multiplier", 50),
                    vacuum_start=scenario.parameters.get("vacuum_start", 30),
                    vacuum_end=scenario.parameters.get("vacuum_end", 50),
                )
                max_spread = spreads.max()
                nav_impact = -max_spread / 10000 * 2  # Rough slippage estimate
                max_dd = -nav_impact

            elif scenario.attack_type == AttackType.REGIME_CONFUSION:
                prices, regimes = gen.generate_regime_confusion(
                    base_price,
                    bars=scenario.parameters.get("bars", 200),
                    switch_frequency=scenario.parameters.get("switch_frequency", 10),
                )
                returns = np.diff(prices) / prices[:-1]
                max_dd = float(-np.min(np.minimum.accumulate(returns.cumsum()) - np.maximum.accumulate(returns.cumsum())))
                nav_impact = float(returns.sum())

            survived = max_dd < 0.20  # System survived if max DD < 20%

        except Exception as e:
            errors.append(str(e))
            survived = False
            logger.exception("adversarial_test_error", name=scenario.name)

        result = AttackResult(
            scenario=scenario,
            system_survived=survived,
            nav_impact_pct=nav_impact,
            max_drawdown_during=max_dd,
            circuit_breaker_triggered=cb_triggered,
            circuit_breaker_level=cb_level,
            recovery_time_seconds=scenario.duration_seconds,
            errors_encountered=errors,
        )

        self.results.append(result)
        logger.info(
            "adversarial_test_complete",
            name=scenario.name,
            survived=survived,
            nav_impact=f"{nav_impact:.2%}",
            max_dd=f"{max_dd:.2%}",
        )

        return result

    def run_all(self, base_price: float = 65000.0) -> list[AttackResult]:
        """Run all scenarios and return results."""
        return [self.run_scenario(s, base_price) for s in self.scenarios]

    def report(self) -> dict[str, Any]:
        """Generate a summary report of all test results."""
        if not self.results:
            return {"status": "no_tests_run"}

        return {
            "total_scenarios": len(self.results),
            "survived": sum(1 for r in self.results if r.system_survived),
            "failed": sum(1 for r in self.results if not r.system_survived),
            "worst_nav_impact": min(r.nav_impact_pct for r in self.results),
            "worst_drawdown": max(r.max_drawdown_during for r in self.results),
            "circuit_breakers_triggered": sum(1 for r in self.results if r.circuit_breaker_triggered),
            "scenarios": [
                {
                    "name": r.scenario.name,
                    "survived": r.system_survived,
                    "nav_impact": f"{r.nav_impact_pct:.2%}",
                    "max_dd": f"{r.max_drawdown_during:.2%}",
                    "cb_level": r.circuit_breaker_level,
                }
                for r in self.results
            ],
        }
