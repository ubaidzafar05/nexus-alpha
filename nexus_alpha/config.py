"""Central configuration — all settings injected via environment, never hardcoded."""

from __future__ import annotations

import os
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from pydantic import Field, SecretStr, field_validator
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
    testnet: bool = False


class BybitConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BYBIT_")
    enabled: bool = False
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")


class KrakenConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KRAKEN_")
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")


class LLMConfig(BaseSettings):
    """Free LLM stack: Ollama local-first, with optional Groq fallback."""

    model_config = SettingsConfigDict(env_prefix="")

    # Ollama — local LLM server (runs on Oracle Cloud free VM or localhost)
    ollama_base_url: str = "http://localhost:11434"
    ollama_primary_model: str = "qwen3:8b"          # Best reasoning in 8B class
    ollama_fast_model: str = "mistral:7b"            # Fast structured output
    ollama_reasoning_model: str = "deepseek-r1:8b"  # Debate / cross-validation
    ollama_embed_model: str = "nomic-embed-text"     # Free local embeddings

    # Groq — optional cloud fallback. Disabled by default for pure Ollama mode.
    use_groq_fallback: bool = False
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.3-70b-versatile"     # Free 70B model via Groq

    # FinBERT — specialized finance sentiment (110M params, CPU-fast, Apache 2.0)
    finbert_enabled: bool = True
    finbert_model_name: str = "ProsusAI/finbert"
    
    # HuggingFace Token — for unauthenticated model access (prevents 429s)
    hf_token: SecretStr = SecretStr("")

    # Legacy field aliases — kept for backwards compat with any code that reads them
    primary_model: str = "qwen3:8b"
    fallback_model: str = "llama-3.3-70b-versatile"

    # These are intentionally empty — no paid keys required
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")

    @property
    def model_name(self) -> str:
        """Canonical primary model accessor used across subsystems.

        Returns the explicitly set primary_model if overridden, otherwise the
        Ollama primary model (default free-stack behaviour).
        """
        # primary_model default matches ollama_primary_model; if they differ
        # it means the caller explicitly overrode primary_model — respect that.
        if self.primary_model != self.ollama_primary_model:
            return self.primary_model
        return self.ollama_primary_model

    @property
    def has_groq(self) -> bool:
        return self.use_groq_fallback and bool(self.groq_api_key.get_secret_value())


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")
    timescaledb_url: SecretStr = SecretStr(
        "postgresql+asyncpg://nexus:changeme@localhost:5432/nexus_alpha"
    )
    # Format: redis://default:PASSWORD@host:port/db
    redis_url: str = "redis://default:nexus_dev@localhost:6379/0"

    @field_validator("redis_url", mode="before")
    @classmethod
    def _normalize_redis_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "redis" or parsed.password is None or parsed.username:
            return value

        netloc = parsed.hostname or "localhost"
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        netloc = f"default:{parsed.password}@{netloc}"
        return urlunparse(parsed._replace(netloc=netloc))


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_")
    url: str = "http://localhost:6333"
    collection_name: str = "nexus_memories"
    vector_size: int = 768  # Matches nomic-embed-text
    distance: str = "Cosine"


class KafkaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_")
    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "nexus-alpha"
    tick_topic: str = "market.ticks"
    signal_topic: str = "signals.raw"
    order_topic: str = "orders.submitted"


class DataQualityConfig(BaseSettings):
    """Phase 0 quality + observability thresholds from roadmap targets."""

    model_config = SettingsConfigDict(env_prefix="DATA_")
    max_future_skew_seconds: float = 1.0
    max_quality_failures_per_hour: int = 10
    max_tick_latency_p99_ms: float = 10.0
    max_feature_staleness_seconds: float = 300.0
    max_quality_check_p99_ms: float = 2.0


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
    max_order_notional_usd: float = 100_000.0  # Fat-finger limit
    max_asset_concentration_pct: float = 0.25   # Max exposure to 1 asset
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
    paper_min_signal_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    paper_max_position_age_hours: float | None = Field(default=None, gt=0.0)

    # Sub-configs
    binance: ExchangeConfig = Field(default_factory=ExchangeConfig)
    bybit: BybitConfig = Field(default_factory=BybitConfig)
    kraken: KrakenConfig = Field(default_factory=KrakenConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    rl_agent: RLAgentConfig = Field(default_factory=RLAgentConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    tournament: TournamentConfig = Field(default_factory=TournamentConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)

    @property
    def is_live(self) -> bool:
        return self.trading_mode != TradingMode.PAPER

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


def load_config(env_file: str = ".env") -> NexusConfig:
    """Load configuration from environment. Call once at startup."""
    env_path = Path(env_file)
    if not env_path.exists():
        return NexusConfig(_env_file=None)

    with _temporary_env_overlay(_read_env_file(env_path)):
        return NexusConfig(_env_file=None)


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


@contextmanager
def _temporary_env_overlay(values: dict[str, str]) -> None:
    original: dict[str, str | None] = {}
    inserted: set[str] = set()
    try:
        for key, value in values.items():
            previous = os.environ.get(key)
            original[key] = previous
            if previous is None:
                os.environ[key] = value
                inserted.add(key)
        yield
    finally:
        for key, previous in original.items():
            if key in inserted:
                os.environ.pop(key, None)
            elif previous is not None:
                os.environ[key] = previous
