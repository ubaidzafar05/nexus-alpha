"""Central configuration — all settings injected via environment, never hardcoded."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingMode(str, Enum):
    PAPER = "paper"
    MICRO_LIVE = "micro_live"
    SMALL_LIVE = "small_live"
    PRODUCTION = "production"


class ExchangeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BINANCE_")
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")


class BybitConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BYBIT_")
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")


class KrakenConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KRAKEN_")
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    primary_model: str = "claude-sonnet-4-20250514"
    fallback_model: str = "gpt-4o"


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")
    timescaledb_url: SecretStr = SecretStr(
        "postgresql+asyncpg://nexus:changeme@localhost:5432/nexus_alpha"
    )
    redis_url: str = "redis://localhost:6379/0"


class KafkaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_")
    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "nexus-alpha"
    tick_topic: str = "market.ticks"
    signal_topic: str = "signals.raw"
    order_topic: str = "orders.submitted"


class WorldModelConfig(BaseSettings):
    """Temporal Fusion Transformer configuration."""

    model_config = SettingsConfigDict(env_prefix="WORLD_MODEL_")
    input_size: int = 128
    hidden_size: int = 256
    attention_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    output_quantiles: list[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    mc_dropout_samples: int = 50
    ewc_lambda: float = 0.4
    episodic_memory_per_regime: int = 5000
    online_update_interval_hours: int = 4


class RLAgentConfig(BaseSettings):
    """Reinforcement learning agent configuration."""

    model_config = SettingsConfigDict(env_prefix="RL_")
    state_dim: int = 85
    action_dim: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    noise_std: float = 0.2
    batch_size: int = 256
    replay_buffer_size: int = 1_000_000


class RiskConfig(BaseSettings):
    """Risk management configuration."""

    model_config = SettingsConfigDict(env_prefix="RISK_")
    max_portfolio_drawdown_pct: float = 0.15
    max_single_position_pct: float = 0.20
    max_correlated_exposure_pct: float = 0.40
    cvar_confidence: float = 0.99
    max_daily_loss_pct: float = 0.05
    max_market_participation_rate: float = 0.05
    max_order_book_impact: float = 0.001
    circuit_breaker_enabled: bool = True


class TournamentConfig(BaseSettings):
    """Agent tournament configuration."""

    model_config = SettingsConfigDict(env_prefix="TOURNAMENT_")
    rolling_window_days: int = 60
    cull_bottom_pct: float = 0.20
    cull_frequency_days: int = 30
    min_agents: int = 5
    graduation_stages: list[str] = ["paper", "micro", "small", "production"]
    paper_to_micro_days: int = 60
    micro_to_small_days: int = 30
    small_to_prod_days: int = 60


class NexusConfig(BaseSettings):
    """Root configuration — single source of truth for all settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.DEVELOPMENT
    trading_mode: TradingMode = TradingMode.PAPER
    log_level: str = "INFO"
    project_root: Path = Path(__file__).parent.parent

    # Sub-configs
    binance: ExchangeConfig = Field(default_factory=ExchangeConfig)
    bybit: BybitConfig = Field(default_factory=BybitConfig)
    kraken: KrakenConfig = Field(default_factory=KrakenConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    rl_agent: RLAgentConfig = Field(default_factory=RLAgentConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    tournament: TournamentConfig = Field(default_factory=TournamentConfig)

    @property
    def is_live(self) -> bool:
        return self.trading_mode != TradingMode.PAPER

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


def load_config() -> NexusConfig:
    """Load configuration from environment. Call once at startup."""
    return NexusConfig()
