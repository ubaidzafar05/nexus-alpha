from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from nexus_alpha.cli import cli
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

