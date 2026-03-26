from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from nexus_alpha.cli import _collect_health_status, _exchange_credentials, cli
from nexus_alpha.config import LLMConfig, load_config


def test_load_config_respects_custom_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("LOG_LEVEL=DEBUG\nTRADING_MODE=paper\n", encoding="utf-8")
    cfg = load_config(env_file=str(env_file))
    assert cfg.log_level == "DEBUG"


def test_cli_accepts_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("LOG_LEVEL=WARNING\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--env-file", str(env_file), "health"])
    assert result.exit_code == 0
    assert "System Health" in result.output


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
