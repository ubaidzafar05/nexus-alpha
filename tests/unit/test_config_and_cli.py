from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from nexus_alpha.cli import _collect_health_status, _exchange_credentials, cli
from nexus_alpha.config import LLMConfig, load_config
from nexus_alpha.intelligence.openclaw_agents import IntelligenceCategory, IntelligenceReport, Urgency


def test_load_config_respects_custom_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("LOG_LEVEL=DEBUG\nTRADING_MODE=paper\n", encoding="utf-8")
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("TRADING_MODE", raising=False)
    cfg = load_config(env_file=str(env_file))
    assert cfg.log_level == "DEBUG"


def test_load_config_preserves_exported_env_over_env_file(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("TRADING_MODE=paper\n", encoding="utf-8")
    monkeypatch.setenv("TRADING_MODE", "micro_live")
    cfg = load_config(env_file=str(env_file))
    assert cfg.trading_mode.value == "micro_live"


def test_cli_accepts_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("LOG_LEVEL=WARNING\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--env-file", str(env_file), "health"])
    assert result.exit_code == 0
    assert "System Health" in result.output


def test_crawl_intel_command_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    report = IntelligenceReport(
        report_id="abc123",
        category=IntelligenceCategory.BREAKING_NEWS,
        urgency=Urgency.MEDIUM,
        headline="Sample headline",
        summary="Sample summary",
        sentiment_score=0.1,
        confidence=0.8,
        affected_symbols=["BTC"],
        source_urls=["https://example.test/article"],
        raw_data={},
    )
    monkeypatch.setattr(
        "nexus_alpha.intelligence.crawl4ai_agents.Crawl4AINewsAgent.fetch",
        AsyncMock(return_value=[report]),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["crawl-intel", "--url", "https://example.test"])

    assert result.exit_code == 0
    assert "Crawl complete: 1 reports" in result.output
    assert "Sample headline" in result.output


def test_sentiment_once_command_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyRunner:
        def __init__(self, config):
            self.config = config

        def _init_redis(self) -> bool:
            return True

        def _init_kafka(self) -> None:
            return None

        async def _run_once(self, max_articles: int | None = None, deep_analysis_enabled: bool = True):
            from nexus_alpha.data.sentiment_pipeline import SentimentScore
            from datetime import datetime

            return {
                "BTC": SentimentScore(
                    asset="BTC",
                    score=0.42,
                    confidence=0.8,
                    source_count=3,
                    method="hybrid_finbert_qwen3_macro",
                    timestamp=datetime.utcnow(),
                )
            }

        def _write_to_redis(self, scores) -> None:
            return None

        def _publish_to_kafka(self, scores) -> None:
            return None

    monkeypatch.setattr("nexus_alpha.data.sentiment_pipeline.SentimentPipelineRunner", DummyRunner)
    runner = CliRunner()
    result = runner.invoke(cli, ["sentiment-once", "--max-articles", "5"])

    assert result.exit_code == 0
    assert "Sentiment cycle complete" in result.output


def test_llm_config_exposes_canonical_model_property() -> None:
    cfg = LLMConfig(primary_model="test-primary-model")
    assert cfg.model_name == "test-primary-model"


def test_exchange_credentials_returns_empty_for_unknown() -> None:
    cfg = load_config()
    api_key, api_secret = _exchange_credentials(cfg, "unknown")
    assert api_key == ""
    assert api_secret == ""


@pytest.mark.asyncio
async def test_collect_health_status_aggregates_probes(monkeypatch) -> None:
    cfg = load_config()

    monkeypatch.setattr(
        "nexus_alpha.cli._probe_tcp_endpoint",
        AsyncMock(side_effect=["ok", "ok", "down"]),
    )

    class DummyLLM:
        async def health_check(self) -> dict[str, str]:
            return {"status": "degraded"}

    class DummyTelegram:
        is_configured = True

    monkeypatch.setattr(
        "nexus_alpha.intelligence.free_llm.FreeLLMClient.from_config",
        lambda cfg: DummyLLM(),
    )
    monkeypatch.setattr(
        "nexus_alpha.alerts.telegram.TelegramAlerts.from_env",
        classmethod(lambda cls: DummyTelegram()),
    )

    status = await _collect_health_status(cfg)

    assert status["timescaledb"] == "ok"
    assert status["redis"] == "ok"
    assert status["kafka"] == "down"
    assert status["ollama"] == "degraded"
    assert status["telegram"] == "configured"
