"""
NEXUS-ALPHA CLI — Main Entry Point.

Usage:
    nexus run           Start the full trading system
    nexus paper         Start in paper trading mode
    nexus backtest      Run backtesting engine
    nexus crawl-intel   Run Crawl4AI on curated intelligence targets
    nexus agents        Manage agent tournament
    nexus health        Show system health
    nexus adversarial   Run adversarial test suite
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from typing import Any

# Set before any HuggingFace/transformers imports to prevent tokenizer subprocesses
# crashing when stdin is closed in daemon/nohup mode (macOS bad-fd issue).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from urllib.parse import urlparse

import click

from nexus_alpha.config import NexusConfig, TradingMode, load_config
from nexus_alpha.log_config import get_logger, setup_logging

logger = get_logger(__name__)


async def _probe_tcp_endpoint(host: str, port: int, timeout: float = 2.0) -> str:
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        await writer.wait_closed()
        return "ok"
    except Exception:
        return "down"


def _parse_host_port_from_url(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    return host, port


def _parse_kafka_bootstrap(bootstrap_servers: str) -> tuple[str, int]:
    first = bootstrap_servers.split(",")[0].strip()
    if ":" in first:
        host, port = first.rsplit(":", 1)
        return host, int(port)
    return first, 9092


def _exchange_credentials(config: NexusConfig, exchange: str) -> tuple[str, str]:
    exchange_key = exchange.lower()
    if exchange_key == "binance":
        return (
            config.binance.api_key.get_secret_value(),
            config.binance.api_secret.get_secret_value(),
        )
    if exchange_key == "bybit":
        return (
            config.bybit.api_key.get_secret_value(),
            config.bybit.api_secret.get_secret_value(),
        )
    if exchange_key == "kraken":
        return (
            config.kraken.api_key.get_secret_value(),
            config.kraken.api_secret.get_secret_value(),
        )
    return "", ""


async def _collect_health_status(config: NexusConfig) -> dict[str, str]:
    from nexus_alpha.alerts.telegram import TelegramAlerts
    from nexus_alpha.intelligence.free_llm import FreeLLMClient

    db_host, db_port = _parse_host_port_from_url(
        config.database.timescaledb_url.get_secret_value(),
        5432,
    )
    redis_host, redis_port = _parse_host_port_from_url(config.database.redis_url, 6379)
    kafka_host, kafka_port = _parse_kafka_bootstrap(config.kafka.bootstrap_servers)

    llm_client = FreeLLMClient.from_config(config.llm)
    llm_status = await llm_client.health_check()
    telegram = TelegramAlerts.from_env()

    timescaledb_status, redis_status, kafka_status = await asyncio.gather(
        _probe_tcp_endpoint(db_host, db_port),
        _probe_tcp_endpoint(redis_host, redis_port),
        _probe_tcp_endpoint(kafka_host, kafka_port),
    )

    return {
        "timescaledb": timescaledb_status,
        "redis": redis_status,
        "kafka": kafka_status,
        "ollama": llm_status.get("status", "down"),
        "telegram": "configured" if telegram.is_configured else "not_configured",
    }


@click.group()
@click.option("--env-file", default=".env", help="Path to environment file")
@click.option("--log-level", default=None, help="Override log level")
@click.pass_context
def cli(ctx: click.Context, env_file: str, log_level: str | None) -> None:
    """NEXUS-ALPHA v3.0 — Autonomous Crypto Trading System."""
    ctx.ensure_object(dict)
    config = load_config(env_file=env_file)
    if log_level:
        config.log_level = log_level
    setup_logging(config.log_level)
    ctx.obj["config"] = config
    ctx.obj["env_file"] = env_file


@cli.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Start the full trading system."""
    config: NexusConfig = ctx.obj["config"]
    logger.info(
        "nexus_starting",
        environment=config.environment.value,
        trading_mode=config.trading_mode.value,
    )

    asyncio.run(_run_system(config))


@cli.command()
@click.option(
    "--min-signal-confidence",
    default=None,
    type=click.FloatRange(0.0, 1.0),
    help="Optional paper-only entry threshold override.",
)
@click.option(
    "--max-position-age-minutes",
    default=None,
    type=click.FloatRange(min=0.1),
    help="Optional paper-only max hold time before time-exit.",
)
@click.pass_context
def paper(
    ctx: click.Context,
    min_signal_confidence: float | None,
    max_position_age_minutes: float | None,
) -> None:
    """Start in paper trading mode (no real money)."""
    config: NexusConfig = ctx.obj["config"]
    config.trading_mode = TradingMode.PAPER
    config.paper_min_signal_confidence = min_signal_confidence
    config.paper_max_position_age_hours = (
        max_position_age_minutes / 60.0 if max_position_age_minutes is not None else None
    )
    logger.info("nexus_paper_mode")
    asyncio.run(_run_system(config))


@cli.command("paper-eval")
@click.option("--seconds", default=180, type=int, help="How long to run the paper session")
@click.option(
    "--min-signal-confidence",
    default=0.35,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Paper-only entry threshold override for bounded evaluation.",
)
@click.option(
    "--max-position-age-minutes",
    default=15.0,
    show_default=True,
    type=click.FloatRange(min=0.1),
    help="Paper-only max hold time before forcing a time-exit in bounded evaluation.",
)
@click.pass_context
def paper_eval(
    ctx: click.Context,
    seconds: int,
    min_signal_confidence: float,
    max_position_age_minutes: float,
) -> None:
    """Run a bounded paper-trading session and summarize the result."""
    from nexus_alpha.learning.trade_logger import TradeLogger

    config: NexusConfig = ctx.obj["config"]
    config.trading_mode = TradingMode.PAPER
    config.paper_min_signal_confidence = min_signal_confidence
    config.paper_max_position_age_hours = max_position_age_minutes / 60.0
    tl = TradeLogger()
    before = tl.get_performance_summary()
    before_closed = tl.count_closed_trades()

    async def _run_for_duration() -> None:
        task = asyncio.create_task(_run_system(config))
        try:
            await asyncio.wait_for(task, timeout=float(seconds))
        except asyncio.TimeoutError:
            logger.info("paper_eval_window_complete", seconds=seconds)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    click.echo(
        f"🧪 Running paper evaluation for {seconds}s "
        f"(min signal confidence {min_signal_confidence:.2f}, "
        f"max age {max_position_age_minutes:.1f}m)..."
    )
    asyncio.run(_run_for_duration())

    after = tl.get_performance_summary()
    after_closed = tl.count_closed_trades()
    delta_closed = after_closed - before_closed
    open_trades = tl.get_open_trades()
    symbol_perf = tl.get_symbol_performance()

    click.echo("\n📊 Paper Evaluation Summary")
    click.echo(f"  new closed trades: {delta_closed}")
    click.echo(f"  open positions: {len(open_trades)}")
    click.echo(f"  total closed trades: {after.get('total_trades', 0)}")
    click.echo(f"  win rate: {after.get('win_rate', 0):.1%}" if isinstance(after.get("win_rate"), float) else f"  win rate: {after.get('win_rate')}")
    click.echo(f"  total pnl usd: {after.get('total_pnl_usd', 0)}")
    click.echo(f"  avg pnl pct: {after.get('avg_pnl_pct', 0)}")

    if symbol_perf:
        click.echo("\n🧠 Symbol learning scores:")
        for symbol, score in sorted(tl.get_symbol_learning_scores().items()):
            click.echo(f"  {symbol}: {score:.3f}")


@cli.command()
@click.option("--start-date", required=True, help="Backtest start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="Backtest end date (YYYY-MM-DD)")
@click.option("--initial-capital", default=100000.0, help="Starting capital")
@click.option("--symbols", default="BTCUSDT,ETHUSDT", help="Comma-separated symbols")
@click.option("--strategy", default="NexusAlphaStrategy", help="Freqtrade strategy class")
@click.option("--timeframe", default="1h", help="Candle timeframe (e.g. 1h, 4h, 1d)")
@click.option("--download-data", is_flag=True, default=False, help="Download OHLCV data first")
@click.pass_context
def backtest(
    ctx: click.Context,
    start_date: str,
    end_date: str,
    initial_capital: float,
    symbols: str,
    strategy: str,
    timeframe: str,
    download_data: bool,
) -> None:
    """Run backtesting via Freqtrade + NexusAlphaStrategy."""
    import shutil
    import subprocess

    ft = shutil.which("freqtrade")
    if ft is None:
        click.echo("❌  Freqtrade not found. Install it or use the Docker service.")
        click.echo("    docker-compose run --rm freqtrade backtesting ...")
        raise SystemExit(1)

    config_path = "freqtrade/config/config.json"

    if download_data:
        click.echo(f"📥  Downloading {timeframe} OHLCV data ({start_date} → {end_date})...")
        dl_cmd = [
            ft, "download-data",
            "--config", config_path,
            "--timeframe", timeframe,
            "--timerange", f"{start_date.replace('-', '')}-{end_date.replace('-', '')}",
        ]
        subprocess.run(dl_cmd, check=True)  # noqa: S603

    click.echo(f"🔬  Backtesting {strategy}: {start_date} → {end_date}")
    bt_cmd = [
        ft, "backtesting",
        "--config", config_path,
        "--strategy", strategy,
        "--timerange", f"{start_date.replace('-', '')}-{end_date.replace('-', '')}",
        "--starting-balance", str(int(initial_capital)),
        "--timeframe", timeframe,
    ]
    subprocess.run(bt_cmd, check=True)  # noqa: S603


@cli.command()
@click.option("--list", "list_agents", is_flag=True, help="List all tournament agents")
@click.option("--leaderboard", is_flag=True, help="Show tournament leaderboard")
@click.pass_context
def agents(ctx: click.Context, list_agents: bool, leaderboard: bool) -> None:
    """Manage agent tournament."""
    if list_agents or leaderboard:
        click.echo("Agent Tournament Status")
        click.echo("=" * 40)
        click.echo("Tournament not yet running — start with `nexus run` first.")
    else:
        click.echo("Use --list or --leaderboard. Run `nexus agents --help` for options.")


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Show system health status."""
    config: NexusConfig = ctx.obj["config"]
    status = asyncio.run(_collect_health_status(config))

    click.echo("NEXUS-ALPHA System Health")


@cli.command("telegram-test")
@click.option("--message", default="Test message from NEXUS-ALPHA", help="Custom test message")
@click.pass_context
def telegram_test(ctx: click.Context, message: str) -> None:
    """Send a test Telegram message to verify alerts are configured."""
    from nexus_alpha.alerts.telegram import TelegramAlerts

    env_file = ctx.obj.get("env_file", ".env")
    alerts = TelegramAlerts.from_env(env_file=env_file)
    if not alerts.is_configured:
        click.echo("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env or environment.")
        return
    click.echo("Sending test message to Telegram...")
    asyncio.run(alerts.send(message))
    click.echo("Done.")


@cli.command("run-retrain-watcher")
@click.option("--interval", default=3600, type=int, help="Seconds between checks")
@click.pass_context
def run_retrain_watcher(ctx: click.Context, interval: int) -> None:
    """Start the background retrain watcher (blocks)."""
    click.echo(f"Starting retrain watcher (interval={interval}s). Ctrl-C to exit")
    # Import locally to avoid heavy imports on CLI startup
    from infra.self_healing.retrain_watcher import retrain_watcher_main

    retrain_watcher_main(interval_s=interval)


@cli.command("maintenance-timescale")
@click.option("--retention-days", default=90, type=int, help="Retention window in days")
@click.option("--execute", is_flag=True, help="Execute maintenance (default is dry-run)")
@click.pass_context
def maintenance_timescale(ctx: click.Context, retention_days: int, execute: bool) -> None:
    """Run TimescaleDB maintenance (drop_chunks + vacuum) for hypertables."""
    env_file = ctx.obj.get("env_file", ".env")
    script = "infra/maintenance/timescale_maintenance.py"
    cmd = ["python", script, "--retention-days", str(retention_days)]
    if not execute:
        cmd.append("--dry-run")
    click.echo(f"Running maintenance script: {' '.join(cmd)}")
    subprocess.run(cmd)
    click.echo("=" * 40)
    for component, state in status.items():
        icon = "✅" if state in {"ok", "configured"} else ("⚠️" if state == "degraded" else "❌")
        click.echo(f"  {icon} {component}: {state}")

    healthy = sum(1 for state in status.values() if state in {"ok", "configured"})
    click.echo(f"\nHealthy checks: {healthy}/{len(status)}")


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host interface")
@click.option("--port", default=8080, type=int, help="Port")
def api(host: str, port: int) -> None:
    """Run health/readiness API surface for infrastructure checks."""
    import uvicorn

    uvicorn.run("nexus_alpha.api.health:app", host=host, port=port, log_level="info")


@cli.command("materialize-features")
@click.option("--once", is_flag=True, help="Run one worker cycle")
@click.option("--cycles", default=60, type=int, help="Max cycles in loop mode")
@click.option("--interval-seconds", default=1.0, type=float, help="Delay between cycles")
@click.option("--prefer-kafka/--no-prefer-kafka", default=True, help="Prefer Kafka transport")
@click.option("--seed-demo-ticks", default=0, type=int, help="Publish demo ticks before processing")
@click.pass_context
def materialize_features(
    ctx: click.Context,
    once: bool,
    cycles: int,
    interval_seconds: float,
    prefer_kafka: bool,
    seed_demo_ticks: int,
) -> None:
    """Consume tick events and emit feature snapshots."""
    from nexus_alpha.data.streaming import FeatureStreamingLoop

    config: NexusConfig = ctx.obj["config"]
    if prefer_kafka and seed_demo_ticks > 0:
        suffix = uuid.uuid4().hex[:8]
        config.kafka.tick_topic = f"{config.kafka.tick_topic}.demo.{suffix}"
        config.kafka.signal_topic = f"{config.kafka.signal_topic}.demo.{suffix}"
        config.kafka.consumer_group = f"{config.kafka.consumer_group}-demo-{suffix}"
    loop = FeatureStreamingLoop.from_config(config, prefer_kafka=prefer_kafka)
    try:
        if seed_demo_ticks > 0:
            loop.seed_demo_ticks(n=seed_demo_ticks)

        if once:
            stats = loop.run_cycle()
            if loop.mode == "kafka" and seed_demo_ticks > 0 and stats.emitted_snapshots == 0:
                for _ in range(10):
                    time.sleep(0.2)
                    stats = loop.run_cycle()
                    if stats.emitted_snapshots >= seed_demo_ticks:
                        break
            slo_ok = loop.metrics()["slo"]["ok"]
            click.echo(
                "Feature worker run complete: "
                f"{stats.emitted_snapshots} snapshots emitted ({loop.mode}) "
                f"[slo_ok={slo_ok}]"
            )
            return

        metrics = loop.run_for(cycles=cycles, interval_seconds=interval_seconds)
        worker = metrics["worker"]
        slo_ok = metrics["slo"]["ok"]
        click.echo(
            "Feature worker loop complete: "
            f"{worker['emitted_snapshots']} snapshots from {worker['processed_ticks']} ticks"
            f" ({loop.mode}) [slo_ok={slo_ok}]"
        )
    finally:
        loop.close()


@cli.command()
@click.option("--base-price", default=65000.0, help="Base price for simulations")
@click.pass_context
def adversarial(ctx: click.Context, base_price: float) -> None:
    """Run the adversarial (red team) test suite."""
    from nexus_alpha.infrastructure.adversarial import AdversarialTestRunner

    runner = AdversarialTestRunner()
    click.echo("Running adversarial test suite...")
    click.echo("=" * 50)

    runner.run_all(base_price=base_price)
    report = runner.report()

    click.echo(f"\nTotal scenarios:  {report['total_scenarios']}")
    click.echo(f"Survived:         {report['survived']}")
    click.echo(f"Failed:           {report['failed']}")
    click.echo(f"Worst NAV impact: {report['worst_nav_impact']}")
    click.echo(f"Worst drawdown:   {report['worst_drawdown']}")
    click.echo(f"CB triggered:     {report['circuit_breakers_triggered']}x")

    click.echo("\nScenario Results:")
    for s in report["scenarios"]:
        status = "✓" if s["survived"] else "✗"
        click.echo(
            f"  {status} {s['name']}: NAV {s['nav_impact']}, "
            f"DD {s['max_dd']}, CB L{s['cb_level']}"
        )


# ─── Live Ingest Command ─────────────────────────────────────────────────────

@cli.command("live-ingest")
@click.option("--exchange", default="binance", help="Exchange ID (binance, bybit, kraken...)")
@click.option("--symbols", default="BTC/USDT,ETH/USDT", help="Comma-separated symbols")
@click.option("--multi/--single", default=True, help="Multi-exchange (Binance+Bybit) or single")
@click.pass_context
def live_ingest(ctx: click.Context, exchange: str, symbols: str, multi: bool) -> None:
    """Start the live WebSocket market data ingestor (ccxt.pro → Kafka)."""
    from nexus_alpha.data.live_ingestor import LiveMarketIngestor, MultiExchangeIngestor

    config: NexusConfig = ctx.obj["config"]
    symbol_list = [s.strip() for s in symbols.split(",")]

    async def _run() -> None:
        if multi:
            ingestor = MultiExchangeIngestor(config)
        else:
            api_key, api_secret = _exchange_credentials(config, exchange)
            ingestor = LiveMarketIngestor(
                exchange_id=exchange,
                symbols=symbol_list,
                kafka_bootstrap=config.kafka.bootstrap_servers,
                kafka_tick_topic=config.kafka.tick_topic,
                redis_url=config.database.redis_url,
                exchange_api_key=api_key,
                exchange_api_secret=api_secret,
                use_testnet=config.binance.testnet,
            )
        click.echo(f"📡  Live ingestor starting — {'multi-exchange' if multi else exchange}")
        click.echo(f"    Symbols: {symbol_list}")
        click.echo(f"    Kafka:   {config.kafka.bootstrap_servers}")
        if config.binance.testnet:
            click.echo("    Sandbox: Binance testnet enabled")
        await ingestor.run()

    asyncio.run(_run())


@cli.command("crawl-intel")
@click.option("--url", "urls", multiple=True, help="Target URL to crawl (repeatable)")
@click.option("--max-items", default=5, type=int, help="Max extracted items per target")
@click.option("--publish/--no-publish", default=False, help="Publish reports to Kafka")
@click.option("--json-output/--no-json-output", default=False, help="Print full JSON payloads")
@click.pass_context
def crawl_intel(
    ctx: click.Context,
    urls: tuple[str, ...],
    max_items: int,
    publish: bool,
    json_output: bool,
) -> None:
    """Run curated Crawl4AI extraction against production news targets."""
    import json

    from nexus_alpha.intelligence.crawl4ai_agents import (
        DEFAULT_CRAWL_TARGETS,
        Crawl4AINewsAgent,
        intelligence_report_payload,
        publish_reports_to_kafka,
    )

    config: NexusConfig = ctx.obj["config"]
    target_urls = list(urls) or list(DEFAULT_CRAWL_TARGETS)

    async def _run() -> list[Any]:
        agent = Crawl4AINewsAgent(
            target_urls=target_urls,
            ollama_base_url=config.llm.ollama_base_url,
            model=config.llm.ollama_primary_model,
            max_items_per_target=max_items,
        )
        return await agent.fetch()

    reports = asyncio.run(_run())
    click.echo(f"🕸️  Crawl complete: {len(reports)} reports from {len(target_urls)} target(s)")

    if json_output:
        click.echo(json.dumps([intelligence_report_payload(report) for report in reports], indent=2))
    else:
        for report in reports[:20]:
            url = report.source_urls[0] if report.source_urls else ""
            click.echo(f"- [{report.category.value}/{report.urgency.value}] {report.headline} :: {url}")

    if publish:
        published = publish_reports_to_kafka(
            reports,
            config.kafka.bootstrap_servers,
            topic="nexus.intelligence",
        )
        click.echo(f"📨  Published {published} report(s) to nexus.intelligence")


@cli.command("sentiment-once")
@click.option("--max-articles", default=25, type=int, help="Cap articles processed in this one-off run")
@click.option("--deep-analysis/--no-deep-analysis", default=False, help="Enable Ollama deep analysis during one-off validation")
@click.pass_context
def sentiment_once(ctx: click.Context, max_articles: int, deep_analysis: bool) -> None:
    """Run one sentiment cycle and write results to Redis/Kafka."""
    from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner

    config: NexusConfig = ctx.obj["config"]

    async def _run() -> dict[str, Any]:
        runner = SentimentPipelineRunner(config)
        if not runner._init_redis():
            raise RuntimeError("Redis unavailable for sentiment-once")
        runner._init_kafka()
        scores = await runner._run_once(
            max_articles=max_articles,
            deep_analysis_enabled=deep_analysis,
        )
        runner._write_to_redis(scores)
        runner._publish_to_kafka(scores)
        return {
            "asset_count": len(scores),
            "sample": {asset: score.score for asset, score in list(scores.items())[:5]},
        }

    result = asyncio.run(_run())
    click.echo(
        "Sentiment cycle complete: "
        f"{result['asset_count']} assets scored "
        f"sample={result['sample']}"
    )


# ─── Heartbeat — Periodic Health & Metrics ────────────────────────────────────


async def _heartbeat(
    alerts: Any,
    trading_loop: Any,
    circuit_breaker: Any,
    config: NexusConfig,
    interval_s: float = 900.0,
) -> None:
    """Send periodic health/metrics to Telegram every 15 minutes."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            m = trading_loop.metrics
            cb = circuit_breaker.state
            lines = [
                "📊 *NEXUS-ALPHA Heartbeat*",
                f"• Cycles: `{m.ticks_processed}`",
                "• Status: " + ("⚠️ *BLIND-HALT (INFRA)*" if m.is_blind_halt else "✅ *ACTIVE*"),
                f"• Signals: `{m.signals_generated}`",
                f"• Debates: `{m.debates_triggered}`",
                f"• Orders: `{m.orders_submitted}` (rejected: `{m.orders_rejected}`)",
                f"• Errors: `{m.errors}`",
                f"• Circuit Breaker: `{cb.level.name}`",
            ]
            await alerts.send("\n".join(lines))

            health = await _collect_health_status(config)
            degraded = [k for k, v in health.items() if v not in {"ok", "configured"}]
            if degraded:
                await alerts.send(
                    f"⚠️ Degraded services: {', '.join(degraded)}"
                )
        except Exception:
            logger.debug("heartbeat_error")  # heartbeat must never crash


# ─── System Runner ────────────────────────────────────────────────────────────

async def _run_system(config: NexusConfig) -> None:
    """
    Initialize and run all NEXUS-ALPHA system components.

    Startup order (dependency-safe):
    1. Infrastructure health check
    2. Regime oracle + World model
    3. Signal engine
    4. Circuit breaker
    5. Live data ingestor (ccxt WebSocket → Kafka)
    6. Sentiment pipeline (RSS → FinBERT → Redis)
    7. Intelligence agents (Crawl4AI, RSS, DeFiLlama, SEC)
    8. Portfolio optimizer
    9. Agent tournament
    10. System watchdog
    11. Telegram health notification
    """
    import uvicorn
    api_port = int(os.getenv("NEXUS_API_PORT", "8000"))
    print(f"DEBUG: Preparing API Server on port {api_port}...")
    
    api_config = uvicorn.Config(
        "dashboard.backend.main:app",
        host="0.0.0.0",
        port=api_port,
        log_level="warning",
        loop="none",           # Don't install a new event loop — reuse the running one
    )
    api_server = uvicorn.Server(api_config)
    api_server.install_signal_handlers = lambda: None  # Prevent signal handler conflict
    
    async def _safe_api_start():
        try:
            print(f"DEBUG: Task api_server starting on port {api_port}...")
            logger.info("api_server_starting", port=api_port)
            await api_server.serve()
        except Exception as e:
            print(f"DEBUG: API Server FAILED: {str(e)}")
            logger.error("api_server_failed", error=str(e))

    logger.info("initializing_components")

    # ── Core intelligence ─────────────────────────────────────────────────────
    from nexus_alpha.agents.tournament import TournamentOrchestrator
    from nexus_alpha.alerts.telegram import TelegramAlerts
    from nexus_alpha.core.regime_oracle import RegimeOracle
    from nexus_alpha.core.trading_loop import TradingLoopOrchestrator
    from nexus_alpha.data.live_ingestor import MultiExchangeIngestor
    from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner
    from nexus_alpha.infrastructure.self_healing import SystemWatchdog
    from nexus_alpha.intelligence.crawl4ai_agents import FreeIntelligenceOrchestrator
    from nexus_alpha.intelligence.openclaw_agents import OpenClawNetwork
    from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem
    from nexus_alpha.signals.signal_engine import SignalFusionEngine

    # ── Alerts first — so we can notify on startup/failure ───────────────────
    alerts = TelegramAlerts.from_env()

    # REASONING: Concurrent health checks to prevent boot deadlock.
    async def _boot_health():
        status = await _collect_health_status(config)
        if status.get("kafka") != "ok" and config.kafka.bootstrap_servers:
            logger.warning("kafka_runtime_disabled", bootstrap=config.kafka.bootstrap_servers)
            config.kafka.bootstrap_servers = ""
        await alerts.system_health(status)
        return status

    health_task = asyncio.create_task(_boot_health())

    # ── Core modules ──────────────────────────────────────────────────────────
    circuit_breaker = CircuitBreakerSystem(risk_config=config.risk)
    _regime_oracle = RegimeOracle(n_regimes=5, lookback_window=200)  # noqa: F841
    _world_model = None
    try:
        from nexus_alpha.core.world_model import WorldModel

        _world_model = WorldModel(config.world_model)  # noqa: F841
    except ModuleNotFoundError as err:
        if err.name not in {"torch", "torch.nn", "torch.optim", "torch.utils.data"}:
            raise
        logger.warning("world_model_disabled_missing_dependency", dependency=err.name)
    _tournament = TournamentOrchestrator(config.tournament)  # noqa: F841

    signal_engine = SignalFusionEngine()
    signal_engine.register_defaults()

    # ── Trading Loop — Signal → Debate → Portfolio → Execution ───────────────
    trading_loop = TradingLoopOrchestrator(
        config=config,
        signal_engine=signal_engine,
        circuit_breaker=circuit_breaker,
        alerts=alerts,
        cycle_interval_s=60.0,
        regime_oracle=_regime_oracle,
    )

    # ── Intelligence network ──────────────────────────────────────────────────
    # Primary: free agents (RSS, CryptoPanic, DeFiLlama, SEC EDGAR)
    free_intel = FreeIntelligenceOrchestrator(
        cryptopanic_token=os.getenv("CRYPTOPANIC_API_TOKEN", ""),
        ollama_base_url=config.llm.ollama_base_url,
    )
    # Secondary: OpenClaw network (uses free LLM backend)
    openclaw = OpenClawNetwork(config=config)

    # ── Data pipelines ────────────────────────────────────────────────────────
    ingestor = MultiExchangeIngestor(config)
    sentiment_runner = SentimentPipelineRunner(config)

    # ── System watchdog ───────────────────────────────────────────────────────
    watchdog = SystemWatchdog(check_interval_seconds=30.0)

    logger.info(
        "system_initialized",
        trading_mode=config.trading_mode.value,
        circuit_breaker="enabled" if config.risk.circuit_breaker_enabled else "disabled",
        llm_backend=config.llm.ollama_base_url,
    )

    # Initial reports in background too
    async def _bootstrap_intel():
        try:
            initial_reports = await free_intel.run_all()
            # We'll just use the config value directly, the loop handles offline kafka
            free_intel.publish_reports(initial_reports, bootstrap_servers=config.kafka.bootstrap_servers)
            logger.info("free_intelligence_bootstrap_complete", reports=len(initial_reports))
        except Exception as err:
            logger.warning("free_intelligence_bootstrap_failed", error=repr(err))
        free_intel.start_scheduler(bootstrap_servers=config.kafka.bootstrap_servers)

    intel_task = asyncio.create_task(_bootstrap_intel())

    # ── Start all subsystems ──────────────────────────────────────────────────
    tasks = [
        asyncio.create_task(ingestor.run(), name="live_ingestor"),
        asyncio.create_task(sentiment_runner.run(), name="sentiment_pipeline"),
        asyncio.create_task(openclaw.start_all(), name="openclaw"),
        asyncio.create_task(watchdog.run(), name="watchdog"),
        asyncio.create_task(trading_loop.run(), name="trading_loop"),
        asyncio.create_task(
            _heartbeat(alerts, trading_loop, circuit_breaker, config),
            name="heartbeat",
        ),
        asyncio.create_task(_safe_api_start(), name="api_server")
    ]

    # uvicorn handling moved to top for boot-priority

    click.echo("╔══════════════════════════════════════════╗")
    click.echo("║   NEXUS-ALPHA v3.0 — Free Edition        ║")
    click.echo("╚══════════════════════════════════════════╝")
    click.echo(f"  Mode:     {config.trading_mode.value}")
    click.echo(f"  LLM:      Ollama ({config.llm.ollama_primary_model})")
    click.echo(f"  Kafka:    {config.kafka.bootstrap_servers or 'disabled'}")
    click.echo("  Monthly cost: $0.00")
    click.echo("  Press Ctrl+C to stop.\n")

    # Notify Telegram on startup in background
    async def _notify_startup():
        msg = (
            "✅ NEXUS-ALPHA started\n"
            f"Mode: `{config.trading_mode.value}`\n"
            f"LLM: `{config.llm.ollama_primary_model}`"
        )
        await alerts.send(msg)

    asyncio.create_task(_notify_startup())

    # ── Task Supervision Loop ────────────────────────────────────────────────
    logger.info("system_supervision_active")
    try:
        while True:
            # Audit task health every 30 seconds
            await asyncio.sleep(30)
            
            for t in tasks:
                if t.done():
                    try:
                        # This will re-raise the exception if the task failed
                        t.result()
                        logger.warning("task_finished_unexpectedly", task_name=t.get_name())
                    except asyncio.CancelledError:
                        logger.info("task_cancelled", task_name=t.get_name())
                    except Exception as err:
                        logger.critical(
                            "critical_task_failed",
                            task_name=t.get_name(),
                            error=str(err),
                            exc_info=True
                        )
                        # Optionally: trigger a full system restart by raising
                        # or try to restart the specific task.
                        # For now, we've hardened the ingestor internally.
            
            # If all core tasks are dead, trigger a terminal exit so Docker restarts us
            active_core_tasks = [
                t for t in tasks 
                if not t.done() and t.get_name() in {"live_ingestor", "trading_loop"}
            ]
            if not active_core_tasks:
                logger.critical("all_core_tasks_dead_triggering_reboot")
                break

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("shutdown_requested")
    finally:
        trading_loop.stop()
        sentiment_runner.stop()
        ingestor.stop()
        free_intel.stop_scheduler()
        await openclaw.stop_all()
        await watchdog.stop()
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await alerts.send("🛑 NEXUS-ALPHA stopped")
        logger.info("nexus_stopped")


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


# ── Learning Pipeline CLI Commands ────────────────────────────────────────


@cli.command()
@click.option("--since", default="2022-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--symbols", default=None, help="Comma-separated symbols (e.g. BTC/USDT,ETH/USDT)")
@click.pass_context
def download_data(ctx: click.Context, since: str, symbols: str | None) -> None:
    """Download historical OHLCV data for ML training (free, no API key needed)."""
    from nexus_alpha.learning.historical_data import download_all

    symbol_list = symbols.split(",") if symbols else None
    click.echo(f"📥 Downloading historical data since {since}...")
    results = asyncio.run(download_all(since=since, symbols=symbol_list))
    for key, df in results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            click.echo(f"  ✅ {key}: {len(df)} candles")
        else:
            click.echo(f"  ⚠️  {key}: no data")
    click.echo(f"\n✅ Downloaded {len(results)} datasets to data/ohlcv/")


@cli.command()
@click.option("--symbols", default=None, help="Comma-separated symbols")
@click.option("--timeframe", default="1h", help="Timeframe to train on")
@click.pass_context
def train(ctx: click.Context, symbols: str | None, timeframe: str) -> None:
    """Train ML models on downloaded historical data."""
    from nexus_alpha.learning.offline_trainer import train_all_symbols

    symbol_list = symbols.split(",") if symbols else None
    click.echo(f"🧠 Training ML models on {timeframe} data...")
    results = train_all_symbols(symbols=symbol_list, timeframe=timeframe)
    for sym, stats in results.items():
        if "error" in stats:
            click.echo(f"  ⚠️  {sym}: {stats['error']}")
        else:
            dir_acc = stats.get("test_direction_accuracy", 0)
            click.echo(f"  ✅ {sym}: direction accuracy={dir_acc:.1%}, R²={stats.get('test_r2', 0):.4f}")
    click.echo("\n✅ Models saved to data/checkpoints/")


@cli.command("walk-forward")
@click.option("--symbol", default="BTC/USDT", help="Symbol to evaluate")
@click.option("--timeframe", default="1h", help="Timeframe to evaluate")
@click.option("--target-col", default="target_1h", help="Target return column")
@click.option("--train-bars", default=1500, type=int, help="Rolling training window size")
@click.option("--test-bars", default=250, type=int, help="Out-of-sample test window size")
@click.option("--step-bars", default=250, type=int, help="Window step size")
@click.option("--min-confidence", default=0.15, type=float, help="Only score trades above this confidence")
@click.option("--fee-pct", default=0.00075, type=float, help="Per-side fee assumption")
@click.option("--save-json", default=None, help="Optional path to save JSON results")
@click.pass_context
def walk_forward(
    ctx: click.Context,
    symbol: str,
    timeframe: str,
    target_col: str,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    min_confidence: float,
    fee_pct: float,
    save_json: str | None,
) -> None:
    """Run rolling walk-forward evaluation on historical data."""
    from pathlib import Path

    from nexus_alpha.learning.walk_forward import recommend_learning_policy, run_walk_forward

    click.echo(f"🔁 Walk-forward evaluation: {symbol} {timeframe}")
    result = run_walk_forward(
        symbol=symbol,
        timeframe=timeframe,
        target_col=target_col,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        min_confidence=min_confidence,
        fee_pct=fee_pct,
    )

    click.echo(f"  windows: {result.windows}")
    click.echo(f"  traded samples: {result.traded_samples} ({result.traded_coverage:.1%} coverage)")
    click.echo(f"  avg accuracy: {result.avg_direction_accuracy:.1%}")
    click.echo(f"  traded accuracy: {result.avg_traded_direction_accuracy:.1%}")
    click.echo(f"  total net return: {result.total_net_return_pct:.2f}%")
    click.echo(f"  avg window net: {result.avg_window_net_return_pct:.2f}%")
    click.echo(f"  avg confidence: {result.avg_confidence:.3f}")
    click.echo(f"  avg MAE: {result.avg_mae:.6f}")

    if result.window_results:
        best = max(result.window_results, key=lambda window: window.net_return_pct)
        worst = min(result.window_results, key=lambda window: window.net_return_pct)
        click.echo(
            f"  best window: #{best.window_index} {best.test_start[:10]}→{best.test_end[:10]} "
            f"net={best.net_return_pct:.2f}%"
        )
        click.echo(
            f"  worst window: #{worst.window_index} {worst.test_start[:10]}→{worst.test_end[:10]} "
            f"net={worst.net_return_pct:.2f}%"
        )

    policy = recommend_learning_policy(result)
    click.echo("\n🧭 Suggested learning policy:")
    click.echo(f"  min confidence: {policy.min_confidence:.2f}")
    click.echo(f"  min new trades: {policy.min_new_trades}")
    click.echo(f"  retrain interval: {policy.retrain_interval_hours}h")

    if save_json:
        output_path = result.save_json(Path(save_json))
        click.echo(f"\n💾 Saved JSON results to {output_path}")


@cli.command("train-all")
@click.option("--symbols", default=None, help="Comma-separated symbols")
@click.pass_context
def train_all(ctx: click.Context, symbols: str | None) -> None:
    """Train ML models on all timeframes (1h, 4h, 1d)."""
    from nexus_alpha.learning.offline_trainer import train_all_symbols

    symbol_list = symbols.split(",") if symbols else None
    for tf in ["1h", "4h", "1d"]:
        click.echo(f"\n🧠 Training {tf} models...")
        results = train_all_symbols(symbols=symbol_list, timeframe=tf)
        for sym, stats in results.items():
            if "error" in stats:
                click.echo(f"  ⚠️  {sym}: {stats['error']}")
            else:
                dir_acc = stats.get("test_direction_accuracy", 0)
                click.echo(f"  ✅ {sym}: direction accuracy={dir_acc:.1%}")
    click.echo("\n✅ All models saved to data/checkpoints/")


@cli.command("replay-train")
@click.option("--start-date", default="2022-01-01", help="Replay start date (YYYY-MM-DD)")
@click.option("--end-date", default="2026-04-01", help="Replay end date (YYYY-MM-DD)")
@click.option("--timeframe", default="1h", help="Replay timeframe")
@click.option(
    "--symbols",
    default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT",
    help="Comma-separated symbols",
)
@click.option("--initial-capital", default=100000.0, type=float, help="Initial capital for replay")
@click.option("--min-confidence", default=0.45, type=float, help="Backtest entry confidence threshold")
@click.option("--progress-interval", default=500, type=int, help="Backtest progress interval")
@click.option("--run-label", default=None, help="Optional replay label for journal entries")
@click.option("--regime-slice", default=None, help="Optional regime slice label (bear/recovery/bull/mixed)")
@click.option("--db-path", default=None, help="Optional trade journal SQLite path")
@click.option("--min-total-trades", default=50, type=int, help="Minimum closed trades required to retrain")
@click.option("--min-new-trades", default=10, type=int, help="Minimum new trades expected between retrains")
@click.option("--retrain/--no-retrain", default=True, help="Retrain reward model after exporting replay trades")
@click.option("--target-mode", default="binary", type=click.Choice(["binary", "ternary", "quaternary"]), help="Target mode for replay retraining")
@click.option("--target-metric", default="pnl_pct", type=click.Choice(["pnl_pct", "risk_multiple"]), help="Target metric for replay retraining")
@click.option("--target-threshold", default=None, type=float, help="Optional threshold for target buckets")
@click.option("--strong-move-pct", default=0.02, type=float, help="Absolute pnl pct for strong win/loss targets")
@click.option("--min-quality-score", default=0.0, type=float, help="Minimum sample-quality score for retraining")
@click.option("--balanced-replay/--no-balanced-replay", default=False, help="Balance replay retraining across slice/target groups")
@click.pass_context
def replay_train(
    ctx: click.Context,
    start_date: str,
    end_date: str,
    timeframe: str,
    symbols: str,
    initial_capital: float,
    min_confidence: float,
    progress_interval: int,
    run_label: str | None,
    regime_slice: str | None,
    db_path: str | None,
    min_total_trades: int,
    min_new_trades: int,
    retrain: bool,
    target_mode: str,
    target_metric: str,
    target_threshold: float | None,
    strong_move_pct: float,
    min_quality_score: float,
    balanced_replay: bool,
) -> None:
    """Replay historical trades into the learning journal, then retrain from them."""
    from pathlib import Path

    from nexus_alpha.backtesting.engine import HistoricalBacktester, StrategyParams, print_report
    from nexus_alpha.learning.offline_trainer import OnlineLearner
    from nexus_alpha.learning.trade_logger import TradeLogger

    symbol_list = [symbol.strip() for symbol in symbols.split(",") if symbol.strip()]
    label = run_label or (
        f"{timeframe}_{start_date}_{end_date}_conf{min_confidence:.2f}_"
        f"{'_'.join(symbol.replace('/', '') for symbol in symbol_list)}"
    )

    click.echo(f"🔁 Running historical replay accumulation: {label}")
    params = StrategyParams(min_confidence=min_confidence)
    backtester = HistoricalBacktester(
        symbols=symbol_list,
        initial_capital=initial_capital,
        params=params,
    )
    result = backtester.run(
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        progress_interval=progress_interval,
    )
    print_report(result)

    trade_logger = TradeLogger(Path(db_path) if db_path else Path("data/trade_logs/trades.db"))
    exported = backtester.export_closed_trades_to_logger(
        trade_logger,
        run_label=label,
        metadata={
            "regime_slice": regime_slice or "unspecified",
            "confidence_policy": round(min_confidence, 4),
            "replay_timeframe": timeframe,
            "replay_start": start_date,
            "replay_end": end_date,
        },
    )
    total_closed = trade_logger.count_closed_trades()
    click.echo(f"\n🧠 Replay trades exported: {exported}")
    click.echo(f"📚 Journal closed trades: {total_closed}")

    if not retrain:
        return

    learner = OnlineLearner(
        retrain_interval_hours=0,
        min_new_trades=min_new_trades,
        min_total_trades=min_total_trades,
        target_mode=target_mode,
        target_metric=target_metric,
        target_threshold=target_threshold,
        strong_move_pct=strong_move_pct,
        min_quality_score=min_quality_score,
        balanced_replay=balanced_replay,
        regime_slice=regime_slice,
    )
    stats = learner.retrain_from_journal(trade_logger)
    if stats is None:
        click.echo(
            "⚠️  Not enough closed-trade evidence for retraining yet "
            f"(need {min_total_trades}+)."
        )
        return

    status = "updated" if stats.get("updated") else "rejected"
    click.echo(
        "✅ Online retrain complete"
        if stats.get("updated")
        else "⚠️  Online retrain finished but model update was rejected"
    )
    click.echo(
        f"  status={status} trades={stats['n_trades']} "
        f"new={stats['new_trades']} dir_acc={stats['val_direction_accuracy']:.1%} "
        f"bal_acc={stats.get('val_balanced_accuracy', 0.0):.1%} "
        f"mae={stats['val_mae']:.4f}"
    )


@cli.command("benchmark-replay-models")
@click.option("--db-path", default=None, help="Optional trade journal SQLite path")
@click.option("--min-trades", default=30, type=int, help="Minimum qualified trades required")
@click.option("--min-quality-score", default=0.0, type=float, help="Minimum sample-quality score")
@click.option("--balanced/--no-balanced", default=False, help="Balance evaluation across target/slice groups")
@click.option("--target-metric", default="pnl_pct", type=click.Choice(["pnl_pct", "risk_multiple"]), help="Target metric for labels")
@click.option("--target-threshold", default=None, type=float, help="Optional threshold for target buckets")
@click.option("--regime-slice", default=None, help="Optional regime slice filter")
@click.pass_context
def benchmark_replay_models(
    ctx: click.Context,
    db_path: str | None,
    min_trades: int,
    min_quality_score: float,
    balanced: bool,
    target_metric: str,
    target_threshold: float | None,
    regime_slice: str | None,
) -> None:
    """Benchmark simple classifiers on the filtered replay/live journal dataset."""
    from pathlib import Path

    from nexus_alpha.learning.offline_trainer import benchmark_trade_outcome_models
    from nexus_alpha.learning.trade_logger import TradeLogger

    trade_logger = TradeLogger(Path(db_path) if db_path else Path("data/trade_logs/trades.db"))
    results = benchmark_trade_outcome_models(
        trade_logger,
        min_trades=min_trades,
        min_quality_score=min_quality_score,
        balanced=balanced,
        target_metric=target_metric,
        target_threshold=target_threshold,
        regime_slice=regime_slice,
    )
    if results is None:
        click.echo(f"⚠️  Not enough qualified trades to benchmark ({min_trades}+ required).")
        return

    click.echo("🧪 Replay outcome model benchmark")
    click.echo(f"  qualified trades: {results['n_trades']}")
    click.echo(f"  naive baseline accuracy: {results['baseline_accuracy']:.1%}")
    click.echo(f"  naive balanced accuracy: {results['baseline_balanced_accuracy']:.1%}")
    click.echo(f"  quality mean: {results['quality_mean']:.3f}")
    click.echo(f"  balanced dataset: {results['balanced_dataset']}")
    click.echo(f"  target metric: {results['target_metric']}")
    click.echo(f"  regime slice: {results['regime_slice']}")
    for model_name, stats in results["models"].items():
        click.echo(
            f"  {model_name}: accuracy={stats['accuracy']:.1%} "
            f"bal_acc={stats['balanced_accuracy']:.1%} "
            f"macro_f1={stats['macro_f1']:.4f}"
        )
    click.echo(f"  best model: {results['best_model']}")


@cli.command("benchmark-bucket-models")
@click.option("--db-path", default=None, help="Optional trade journal SQLite path")
@click.option("--min-trades", default=30, type=int, help="Minimum qualified trades required")
@click.option("--strong-move-pct", default=0.02, type=float, help="Absolute pnl pct for strong win/loss buckets")
@click.option("--min-quality-score", default=0.0, type=float, help="Minimum sample-quality score")
@click.option("--balanced/--no-balanced", default=False, help="Balance evaluation across target/slice groups")
@click.option("--target-metric", default="pnl_pct", type=click.Choice(["pnl_pct", "risk_multiple"]), help="Target metric for labels")
@click.option("--target-threshold", default=None, type=float, help="Optional threshold for target buckets")
@click.option("--regime-slice", default=None, help="Optional regime slice filter")
@click.pass_context
def benchmark_bucket_models(
    ctx: click.Context,
    db_path: str | None,
    min_trades: int,
    strong_move_pct: float,
    min_quality_score: float,
    balanced: bool,
    target_metric: str,
    target_threshold: float | None,
    regime_slice: str | None,
) -> None:
    """Benchmark bucketed outcome models on the filtered replay/live journal dataset."""
    from pathlib import Path

    from nexus_alpha.learning.offline_trainer import benchmark_trade_bucket_models
    from nexus_alpha.learning.trade_logger import TradeLogger

    trade_logger = TradeLogger(Path(db_path) if db_path else Path("data/trade_logs/trades.db"))
    results = benchmark_trade_bucket_models(
        trade_logger,
        min_trades=min_trades,
        strong_move_pct=strong_move_pct,
        min_quality_score=min_quality_score,
        balanced=balanced,
        target_metric=target_metric,
        target_threshold=target_threshold,
        regime_slice=regime_slice,
    )
    if results is None:
        click.echo(f"⚠️  Not enough qualified trades to benchmark buckets ({min_trades}+ required).")
        return

    click.echo("🧪 Replay bucketed-outcome benchmark")
    click.echo(f"  qualified trades: {results['n_trades']}")
    click.echo(f"  strong move pct: {results['strong_move_pct']:.2%}")
    click.echo(f"  majority-class accuracy: {results['majority_class_accuracy']:.1%}")
    click.echo(f"  majority-class balanced accuracy: {results['majority_class_balanced_accuracy']:.1%}")
    click.echo(f"  quality mean: {results['quality_mean']:.3f}")
    click.echo(f"  balanced dataset: {results['balanced_dataset']}")
    click.echo(f"  target metric: {results['target_metric']}")
    click.echo(f"  regime slice: {results['regime_slice']}")
    click.echo(f"  train class counts: {results['train_class_counts']}")
    click.echo(f"  val class counts: {results['val_class_counts']}")
    for model_name, stats in results["models"].items():
        click.echo(
            f"  {model_name}: bal_acc={stats['balanced_accuracy']:.1%} "
            f"macro_f1={stats['macro_f1']:.4f}"
        )
    click.echo(f"  best model: {results['best_model']}")


@cli.command("benchmark-learning-targets")
@click.option("--db-path", default=None, help="Optional trade journal SQLite path")
@click.option("--min-trades", default=30, type=int, help="Minimum qualified trades required")
@click.option("--strong-move-pct", default=0.02, type=float, help="Absolute pnl pct for strong win/loss buckets")
@click.option("--min-quality-score", default=0.0, type=float, help="Minimum sample-quality score")
@click.option("--target-metric", default="pnl_pct", type=click.Choice(["pnl_pct", "risk_multiple"]), help="Target metric for labels")
@click.option("--target-threshold", default=None, type=float, help="Optional threshold for target buckets")
@click.pass_context
def benchmark_learning_targets_cmd(
    ctx: click.Context,
    db_path: str | None,
    min_trades: int,
    strong_move_pct: float,
    min_quality_score: float,
    target_metric: str,
    target_threshold: float | None,
) -> None:
    """Compare target schemes on chronological and balanced learning datasets."""
    from pathlib import Path

    from nexus_alpha.learning.offline_trainer import benchmark_learning_targets
    from nexus_alpha.learning.trade_logger import TradeLogger

    trade_logger = TradeLogger(Path(db_path) if db_path else Path("data/trade_logs/trades.db"))
    results = benchmark_learning_targets(
        trade_logger,
        min_trades=min_trades,
        strong_move_pct=strong_move_pct,
        min_quality_score=min_quality_score,
        target_metric=target_metric,
        target_threshold=target_threshold,
    )
    if results is None:
        click.echo(f"⚠️  Not enough qualified trades to compare targets ({min_trades}+ required).")
        return

    click.echo("🧪 Learning target comparison")
    for variant_name, variant_result in results["variants"].items():
        if variant_result is None:
            click.echo(f"  {variant_name}: unavailable")
            continue
        click.echo(
            f"  {variant_name}: trades={variant_result['n_trades']} "
            f"best={variant_result['best_model']} "
            f"baseline_bal={variant_result.get('baseline_balanced_accuracy', variant_result.get('majority_class_balanced_accuracy', 0.0)):.1%}"
        )


@cli.command("diagnose-learning-features")
@click.option("--db-path", default=None, help="Optional trade journal SQLite path")
@click.option("--min-trades", default=30, type=int, help="Minimum qualified trades required")
@click.option("--target-mode", default="quaternary", type=click.Choice(["binary", "ternary", "quaternary"]))
@click.option("--strong-move-pct", default=0.02, type=float, help="Absolute pnl pct for strong win/loss buckets")
@click.option("--min-quality-score", default=0.0, type=float, help="Minimum sample-quality score")
@click.option("--top-n", default=8, type=int, help="Number of top features to display")
@click.option("--target-metric", default="pnl_pct", type=click.Choice(["pnl_pct", "risk_multiple"]), help="Target metric for labels")
@click.option("--target-threshold", default=None, type=float, help="Optional threshold for target buckets")
@click.option("--regime-slice", default=None, help="Optional regime slice filter")
@click.pass_context
def diagnose_learning_features_cmd(
    ctx: click.Context,
    db_path: str | None,
    min_trades: int,
    target_mode: str,
    strong_move_pct: float,
    min_quality_score: float,
    top_n: int,
    target_metric: str,
    target_threshold: float | None,
    regime_slice: str | None,
) -> None:
    """Inspect which journal features separate the chosen learning target."""
    from pathlib import Path

    from nexus_alpha.learning.offline_trainer import diagnose_learning_features
    from nexus_alpha.learning.trade_logger import TradeLogger

    trade_logger = TradeLogger(Path(db_path) if db_path else Path("data/trade_logs/trades.db"))
    results = diagnose_learning_features(
        trade_logger,
        min_trades=min_trades,
        target_mode=target_mode,
        strong_move_pct=strong_move_pct,
        min_quality_score=min_quality_score,
        top_n=top_n,
        target_metric=target_metric,
        target_threshold=target_threshold,
        regime_slice=regime_slice,
    )
    if results is None:
        click.echo(f"⚠️  Not enough qualified trades to diagnose features ({min_trades}+ required).")
        return

    click.echo("🧪 Learning feature diagnostics")
    click.echo(f"  target mode: {results['target_mode']}")
    click.echo(f"  target metric: {results['target_metric']}")
    click.echo(f"  regime slice: {results['regime_slice']}")
    click.echo(f"  qualified trades: {results['n_trades']}")
    click.echo(f"  class counts: {results['class_counts']}")
    click.echo(f"  slice counts: {results['slice_counts']}")
    click.echo("  top importance:")
    for item in results["top_importance_features"]:
        click.echo(f"    {item['feature']}: importance={item['importance']:.4f} separation={item['separation']:.4f}")
    click.echo("  top separation:")
    for item in results["top_separation_features"]:
        click.echo(f"    {item['feature']}: separation={item['separation']:.4f} importance={item['importance']:.4f}")


@cli.command("benchmark-regime-slices")
@click.option("--db-path", default=None, help="Optional trade journal SQLite path")
@click.option("--min-trades", default=10, type=int, help="Minimum qualified trades required per slice")
@click.option("--target-mode", default="binary", type=click.Choice(["binary", "quaternary"]), help="Target mode to benchmark")
@click.option("--strong-move-pct", default=0.02, type=float, help="Absolute pnl pct for strong win/loss buckets")
@click.option("--min-quality-score", default=0.0, type=float, help="Minimum sample-quality score")
@click.option("--target-metric", default="pnl_pct", type=click.Choice(["pnl_pct", "risk_multiple"]), help="Target metric for labels")
@click.option("--target-threshold", default=None, type=float, help="Optional threshold for target buckets")
@click.pass_context
def benchmark_regime_slices_cmd(
    ctx: click.Context,
    db_path: str | None,
    min_trades: int,
    target_mode: str,
    strong_move_pct: float,
    min_quality_score: float,
    target_metric: str,
    target_threshold: float | None,
) -> None:
    """Benchmark learnability separately for each replay regime slice."""
    from pathlib import Path

    from nexus_alpha.learning.offline_trainer import benchmark_regime_slices
    from nexus_alpha.learning.trade_logger import TradeLogger

    trade_logger = TradeLogger(Path(db_path) if db_path else Path("data/trade_logs/trades.db"))
    results = benchmark_regime_slices(
        trade_logger,
        min_trades=min_trades,
        target_mode=target_mode,
        strong_move_pct=strong_move_pct,
        min_quality_score=min_quality_score,
        target_metric=target_metric,
        target_threshold=target_threshold,
    )
    if results is None:
        click.echo(f"⚠️  Not enough qualified trades to benchmark slices ({min_trades}+ required per slice).")
        return

    click.echo("🧪 Regime-slice benchmark")
    click.echo(f"  target mode: {results['target_mode']}")
    click.echo(f"  target metric: {results['target_metric']}")
    for slice_name, slice_result in results["slices"].items():
        baseline = slice_result.get("baseline_balanced_accuracy", slice_result.get("majority_class_balanced_accuracy", 0.0))
        click.echo(
            f"  {slice_name}: trades={slice_result['n_trades']} "
            f"best={slice_result['best_model']} baseline_bal={baseline:.1%}"
        )


@cli.command()
@click.option("--symbol", default="BTC/USDT", help="Symbol to train on")
@click.option("--episodes", default=100, help="Number of training episodes")
@click.pass_context
def train_rl(ctx: click.Context, symbol: str, episodes: int) -> None:
    """Train RL agent on historical data (requires torch)."""
    from nexus_alpha.learning.rl_environment import train_rl_agent

    click.echo(f"🤖 Training RL agent on {symbol} for {episodes} episodes...")
    result = train_rl_agent(symbol=symbol, n_episodes=episodes)
    if "error" in result:
        click.echo(f"  ❌ {result['error']}")
    else:
        click.echo(f"  ✅ Best reward: {result['best_reward']}")
        click.echo(f"  ✅ Checkpoint: {result['checkpoint']}")


@cli.command()
@click.pass_context
def trade_stats(ctx: click.Context) -> None:
    """Show trading performance and learning stats."""
    from nexus_alpha.learning.trade_logger import TradeLogger

    tl = TradeLogger()
    stats = tl.get_performance_summary()
    click.echo("\n📊 Trading Performance:")
    for key, val in stats.items():
        click.echo(f"  {key}: {val}")

    open_trades = tl.get_open_trades()
    click.echo(f"\n📈 Open positions: {len(open_trades)}")
    for t in open_trades[:5]:
        click.echo(f"  {t['symbol']} {t['side']} @ {t['entry_price']:.2f}")


@cli.command()
@click.option("--symbol", default="BTC/USDT", help="Symbol to optimize")
@click.option("--timeframe", default="1h", help="Timeframe to optimize")
@click.option("--trials", default=20, type=int, help="Number of Optuna trials")
@click.pass_context
def optimize(ctx: click.Context, symbol: str, timeframe: str, trials: int) -> None:
    """Run Optuna hyperparameter optimization for signal weights."""
    from nexus_alpha.learning.signal_optimizer import SignalOptimizer
    
    click.echo(f"🎯 Starting optimization for {symbol} ({timeframe})...")
    optimizer = SignalOptimizer(symbol=symbol, timeframe=timeframe, n_trials=trials)
    best_params = optimizer.run()
    
    click.echo("\n✅ Optimization Complete!")
    click.echo(f"  Best Parameters: {best_params}")
    click.echo(f"  Results saved to data/optimization/best_params_{symbol.replace('/', '_')}.json")


@cli.command("tournament-run")
@click.option("--symbol", default="BTC/USDT", help="Symbol for tournament")
@click.option("--timeframe", default="1h", help="Timeframe for tournament")
@click.pass_context
def tournament_run(ctx: click.Context, symbol: str, timeframe: str) -> None:
    """Run the model tournament and promote champions."""
    from nexus_alpha.learning.tournament_engine import TournamentEngine
    
    engine = TournamentEngine(symbol=symbol, timeframe=timeframe)
    click.echo(f"🏟️  Running Tournament for {symbol}...")
    
    results = engine.evaluate_candidates()
    click.echo(f"  Evaluated {len(results)} candidates.")
    
    new_champ = engine.promote_new_champion()
    if new_champ:
        click.echo(f"  🏆 NEW CHAMPION PROMOTED: {new_champ}")
    else:
        click.echo("  ⚖️  No promotion candidate found (Champion retained or no candidates).")


@cli.command()
@click.option("--port", default=8501, help="Backend port")
def dashboard(port: int):
    """Launch the NEXUS-ALPHA Dashboard (UI + Backend)."""
    import subprocess
    import sys
    import os
    
    logger.info("launching_dashboard", port=port)
    
    # 1. Start Backend in background
    backend_cmd = [sys.executable, "-m", "uvicorn", "dashboard.backend.main:app", "--host", "0.0.0.0", "--port", str(port), "--reload"]
    subprocess.Popen(backend_cmd)
    
    # 2. Start Frontend server in background
    frontend_port = port + 1
    frontend_dir = os.path.join(os.getcwd(), "dashboard")
    frontend_cmd = [sys.executable, "-m", "http.server", str(frontend_port), "--directory", frontend_dir]
    subprocess.Popen(frontend_cmd)
    
    click.echo(f"🚀 NEXUS Dashboard Launched at http://localhost:{frontend_port}")
    click.echo(f"📡 API Backend running at http://localhost:{port}")

@cli.command()
@click.option("--symbol", default="BTC/USDT", help="Primary symbol")
def control(symbol: str):
    """MASTER COMMAND: Launch the full autonomous trade ecosystem."""
    import subprocess
    import sys
    
    click.echo(f"🔥 INITIALIZING NEXUS CONTROL: {symbol}")
    
    # 1. Dashboard
    subprocess.Popen([sys.executable, "-m", "nexus_alpha.cli", "dashboard"])
    
    # 2. RL Trainer (episodes=100 for safety in master mode)
    subprocess.Popen([sys.executable, "-m", "nexus_alpha.cli", "train-rl", "--symbol", symbol, "--episodes", "100"])
    
    # 3. Walk-forward paper eval
    subprocess.Popen([sys.executable, "-m", "nexus_alpha.cli", "walk-forward", "--symbol", symbol])
    
    click.echo("✅ All systems initialized. Monitor progress at http://localhost:8501")


@cli.command()
@click.option("--symbol", default="BTC/USDT", help="Symbol to optimize")
@click.option("--interval", default=60, help="Interval in minutes between cycles")
def god_loop(symbol: str, interval: int):
    """ACTIVATE GOD-MODE: Start the master autonomous self-improvement loop."""
    from nexus_alpha.autonomous.god_loop import GodLoop
    click.echo(f"🔥 ACTIVATING NEXUS GOD-LOOP FOR {symbol}")
    loop = GodLoop(symbol=symbol, interval_minutes=interval)
    loop.start()

if __name__ == "__main__":
    cli()
