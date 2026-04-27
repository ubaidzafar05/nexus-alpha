from __future__ import annotations

import asyncio
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


def test_paper_eval_command_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AsyncMock(return_value=None)
    monkeypatch.setattr("nexus_alpha.cli._run_system", runtime)

    class DummyTradeLogger:
        def __init__(self):
            self._count = 0

        def get_performance_summary(self):
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl_usd": 0.0,
                "avg_pnl_pct": 0.0,
            }

        def count_closed_trades(self):
            return self._count

        def get_open_trades(self):
            return []

        def get_symbol_performance(self):
            return {"BTCUSDT": {"total_trades": 6}}

        def get_symbol_learning_scores(self, min_trades: int = 5):
            return {"BTCUSDT": 1.05}

    monkeypatch.setattr("nexus_alpha.learning.trade_logger.TradeLogger", DummyTradeLogger)

    runner = CliRunner()
    result = runner.invoke(cli, ["paper-eval", "--seconds", "1"])

    assert result.exit_code == 0
    assert runtime.await_args.args[0].paper_min_signal_confidence == 0.35
    assert runtime.await_args.args[0].paper_max_position_age_hours == 0.25
    assert "Paper Evaluation Summary" in result.output
    assert "Symbol learning scores" in result.output
    assert "min signal confidence 0.35" in result.output
    assert "max age 15.0m" in result.output


def test_paper_command_accepts_min_signal_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AsyncMock(return_value=None)
    monkeypatch.setattr("nexus_alpha.cli._run_system", runtime)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["paper", "--min-signal-confidence", "0.33", "--max-position-age-minutes", "12"],
    )

    assert result.exit_code == 0
    assert runtime.await_args.args[0].paper_min_signal_confidence == 0.33
    assert runtime.await_args.args[0].paper_max_position_age_hours == 0.2


def test_paper_eval_waits_for_cancelled_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"cancelled": False}

    async def fake_run_system(_config) -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def fake_wait_for(task, timeout: float):
        await asyncio.sleep(0)
        raise asyncio.TimeoutError

    class DummyTradeLogger:
        def get_performance_summary(self):
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl_usd": 0.0,
                "avg_pnl_pct": 0.0,
            }

        def count_closed_trades(self):
            return 0

        def get_open_trades(self):
            return []

        def get_symbol_performance(self):
            return {}

        def get_symbol_learning_scores(self, min_trades: int = 5):
            return {}

    monkeypatch.setattr("nexus_alpha.cli._run_system", fake_run_system)
    monkeypatch.setattr("nexus_alpha.cli.asyncio.wait_for", fake_wait_for)
    monkeypatch.setattr("nexus_alpha.learning.trade_logger.TradeLogger", DummyTradeLogger)

    runner = CliRunner()
    result = runner.invoke(cli, ["paper-eval", "--seconds", "1"])

    assert result.exit_code == 0
    assert state["cancelled"] is True


def test_replay_train_command_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyBacktester:
        def __init__(self, symbols, initial_capital, params=None):
            self.symbols = symbols
            self.initial_capital = initial_capital
            self.params = params

        def run(self, start_date: str, end_date: str, timeframe: str, progress_interval: int):
            class Result:
                total_trades = 12

            return Result()

        def export_closed_trades_to_logger(self, trade_logger, run_label: str, metadata=None) -> int:
            return 12

    class DummyTradeLogger:
        def __init__(self, db_path=None):
            self.db_path = db_path

        def count_closed_trades(self) -> int:
            return 72

    class DummyLearner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def retrain_from_journal(self, trade_logger):
            return {
                "updated": True,
                "n_trades": 72,
                "new_trades": 12,
                "val_direction_accuracy": 0.58,
                "val_balanced_accuracy": 0.56,
                "val_mae": 0.11,
            }

    monkeypatch.setattr("nexus_alpha.backtesting.engine.HistoricalBacktester", DummyBacktester)
    monkeypatch.setattr("nexus_alpha.backtesting.engine.print_report", lambda result: None)
    monkeypatch.setattr("nexus_alpha.learning.trade_logger.TradeLogger", DummyTradeLogger)
    monkeypatch.setattr("nexus_alpha.learning.offline_trainer.OnlineLearner", DummyLearner)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "replay-train",
            "--start-date", "2025-01-01",
            "--end-date", "2025-02-01",
            "--db-path", str(tmp_path / "trades.db"),
        ],
    )

    assert result.exit_code == 0
    assert "Replay trades exported: 12" in result.output
    assert "Journal closed trades: 72" in result.output
    assert "Online retrain complete" in result.output


def test_benchmark_replay_models_command_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyTradeLogger:
        def __init__(self, db_path=None):
            self.db_path = db_path

    monkeypatch.setattr("nexus_alpha.learning.trade_logger.TradeLogger", DummyTradeLogger)
    monkeypatch.setattr(
        "nexus_alpha.learning.offline_trainer.benchmark_trade_outcome_models",
        lambda trade_logger, min_trades=30, min_quality_score=0.0, balanced=False, target_metric="pnl_pct", target_threshold=None, regime_slice=None: {
            "n_trades": 38,
            "baseline_accuracy": 0.75,
            "baseline_balanced_accuracy": 0.5,
            "quality_mean": 0.61,
            "balanced_dataset": balanced,
            "target_metric": target_metric,
            "regime_slice": regime_slice or "all",
            "models": {
                "logistic_regression": {"accuracy": 0.625, "balanced_accuracy": 0.55, "macro_f1": 0.5},
                "random_forest": {"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.44},
                "gradient_boosting": {"accuracy": 0.25, "balanced_accuracy": 0.3, "macro_f1": 0.2},
            },
            "best_model": "logistic_regression",
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["benchmark-replay-models", "--db-path", str(tmp_path / "trades.db"), "--min-trades", "30"],
    )

    assert result.exit_code == 0
    assert "Replay outcome model benchmark" in result.output
    assert "best model: logistic_regression" in result.output


def test_benchmark_bucket_models_command_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyTradeLogger:
        def __init__(self, db_path=None):
            self.db_path = db_path

    monkeypatch.setattr("nexus_alpha.learning.trade_logger.TradeLogger", DummyTradeLogger)
    monkeypatch.setattr(
        "nexus_alpha.learning.offline_trainer.benchmark_trade_bucket_models",
        lambda trade_logger, min_trades=30, strong_move_pct=0.02, min_quality_score=0.0, balanced=False, target_metric="pnl_pct", target_threshold=None, regime_slice=None: {
            "n_trades": 38,
            "strong_move_pct": strong_move_pct,
            "majority_class_accuracy": 0.625,
            "majority_class_balanced_accuracy": 0.25,
            "quality_mean": 0.58,
            "balanced_dataset": balanced,
            "target_metric": target_metric,
            "regime_slice": regime_slice or "all",
            "train_class_counts": [13, 4, 2, 11],
            "val_class_counts": [2, 0, 1, 5],
            "models": {
                "logistic_regression": {"balanced_accuracy": 0.33, "macro_f1": 0.13},
                "random_forest": {"balanced_accuracy": 0.33, "macro_f1": 0.13},
            },
            "best_model": "logistic_regression",
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["benchmark-bucket-models", "--db-path", str(tmp_path / "trades.db"), "--min-trades", "30"],
    )

    assert result.exit_code == 0
    assert "Replay bucketed-outcome benchmark" in result.output
    assert "best model: logistic_regression" in result.output


def test_benchmark_learning_targets_command_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyTradeLogger:
        def __init__(self, db_path=None):
            self.db_path = db_path

    monkeypatch.setattr("nexus_alpha.learning.trade_logger.TradeLogger", DummyTradeLogger)
    monkeypatch.setattr(
        "nexus_alpha.learning.offline_trainer.benchmark_learning_targets",
        lambda trade_logger, min_trades=30, strong_move_pct=0.02, min_quality_score=0.0, target_metric="pnl_pct", target_threshold=None: {
            "variants": {
                "binary_chronological": {
                    "n_trades": 40,
                    "best_model": "logistic_regression",
                    "baseline_balanced_accuracy": 0.5,
                },
                "quaternary_balanced": None,
            }
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["benchmark-learning-targets", "--db-path", str(tmp_path / "trades.db"), "--min-trades", "30"],
    )

    assert result.exit_code == 0
    assert "Learning target comparison" in result.output
    assert "binary_chronological" in result.output


def test_diagnose_learning_features_command_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyTradeLogger:
        def __init__(self, db_path=None):
            self.db_path = db_path

    monkeypatch.setattr("nexus_alpha.learning.trade_logger.TradeLogger", DummyTradeLogger)
    monkeypatch.setattr(
        "nexus_alpha.learning.offline_trainer.diagnose_learning_features",
        lambda trade_logger, min_trades=30, target_mode="quaternary", strong_move_pct=0.02, min_quality_score=0.0, top_n=8, target_metric="pnl_pct", target_threshold=None, regime_slice=None: {
            "target_mode": target_mode,
            "target_metric": target_metric,
            "regime_slice": regime_slice or "all",
            "n_trades": 44,
            "class_counts": {0: 10, 1: 12, 2: 11, 3: 11},
            "slice_counts": {"bear": 11, "bull": 11, "mixed": 11, "recovery": 11},
            "top_importance_features": [
                {"feature": "f0", "importance": 0.21, "separation": 0.8},
            ],
            "top_separation_features": [
                {"feature": "f1", "importance": 0.11, "separation": 0.9},
            ],
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["diagnose-learning-features", "--db-path", str(tmp_path / "trades.db"), "--min-trades", "30"],
    )

    assert result.exit_code == 0
    assert "Learning feature diagnostics" in result.output
    assert "top importance" in result.output


def test_benchmark_regime_slices_command_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyTradeLogger:
        def __init__(self, db_path=None):
            self.db_path = db_path

    monkeypatch.setattr("nexus_alpha.learning.trade_logger.TradeLogger", DummyTradeLogger)
    monkeypatch.setattr(
        "nexus_alpha.learning.offline_trainer.benchmark_regime_slices",
        lambda trade_logger, min_trades=10, target_mode="binary", strong_move_pct=0.02, min_quality_score=0.0, target_metric="pnl_pct", target_threshold=None: {
            "target_mode": target_mode,
            "target_metric": target_metric,
            "slices": {
                "bear": {"n_trades": 12, "best_model": "logistic_regression", "baseline_balanced_accuracy": 0.5},
                "bull": {"n_trades": 11, "best_model": "random_forest", "baseline_balanced_accuracy": 0.45},
            },
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["benchmark-regime-slices", "--db-path", str(tmp_path / "trades.db"), "--min-trades", "10"],
    )

    assert result.exit_code == 0
    assert "Regime-slice benchmark" in result.output
    assert "bear" in result.output


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
