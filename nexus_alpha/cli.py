"""
NEXUS-ALPHA CLI — Main Entry Point.

Usage:
    nexus run           Start the full trading system
    nexus paper         Start in paper trading mode
    nexus backtest      Run backtesting engine
    nexus agents        Manage agent tournament
    nexus health        Show system health
    nexus adversarial   Run adversarial test suite
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlparse

import click

from nexus_alpha.config import NexusConfig, TradingMode, load_config
from nexus_alpha.logging import get_logger, setup_logging

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
@click.pass_context
def paper(ctx: click.Context) -> None:
    """Start in paper trading mode (no real money)."""
    config: NexusConfig = ctx.obj["config"]
    config.trading_mode = TradingMode.PAPER
    logger.info("nexus_paper_mode")
    asyncio.run(_run_system(config))


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
    loop = FeatureStreamingLoop.from_config(config, prefer_kafka=prefer_kafka)

    if seed_demo_ticks > 0:
        loop.seed_demo_ticks(n=seed_demo_ticks)

    if once:
        stats = loop.run_cycle()
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
            )
        click.echo(f"📡  Live ingestor starting — {'multi-exchange' if multi else exchange}")
        click.echo(f"    Symbols: {symbol_list}")
        click.echo(f"    Kafka:   {config.kafka.bootstrap_servers}")
        await ingestor.run()

    asyncio.run(_run())


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
    logger.info("initializing_components")

    # ── Core intelligence ─────────────────────────────────────────────────────
    from nexus_alpha.agents.tournament import TournamentOrchestrator
    from nexus_alpha.alerts.telegram import TelegramAlerts
    from nexus_alpha.core.regime_oracle import RegimeOracle
    from nexus_alpha.core.world_model import WorldModel
    from nexus_alpha.data.live_ingestor import MultiExchangeIngestor
    from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner
    from nexus_alpha.infrastructure.self_healing import SystemWatchdog
    from nexus_alpha.intelligence.crawl4ai_agents import FreeIntelligenceOrchestrator
    from nexus_alpha.intelligence.openclaw_agents import OpenClawNetwork
    from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem
    from nexus_alpha.signals.signal_engine import SignalFusionEngine

    # ── Alerts first — so we can notify on startup/failure ───────────────────
    alerts = TelegramAlerts.from_env()

    # ── Core modules ──────────────────────────────────────────────────────────
    core_components = (
        CircuitBreakerSystem(risk_config=config.risk),
        RegimeOracle(n_regimes=5, lookback_window=200),
        WorldModel(config.world_model),
        TournamentOrchestrator(config.tournament),
    )
    signal_engine = SignalFusionEngine()
    signal_engine.register_defaults()

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
        core_components=len(core_components),
    )

    try:
        initial_reports = await free_intel.run_all()
        logger.info("free_intelligence_bootstrap_complete", reports=len(initial_reports))
    except Exception as err:
        logger.warning("free_intelligence_bootstrap_failed", error=str(err))
    free_intel.start_scheduler()

    health_status = await _collect_health_status(config)

    # ── Start all subsystems ──────────────────────────────────────────────────
    tasks = [
        asyncio.create_task(ingestor.run(), name="live_ingestor"),
        asyncio.create_task(sentiment_runner.run(), name="sentiment_pipeline"),
        asyncio.create_task(openclaw.start_all(), name="openclaw"),
        asyncio.create_task(watchdog.run(), name="watchdog"),
    ]

    click.echo("╔══════════════════════════════════════════╗")
    click.echo("║   NEXUS-ALPHA v3.0 — Free Edition        ║")
    click.echo("╚══════════════════════════════════════════╝")
    click.echo(f"  Mode:     {config.trading_mode.value}")
    click.echo(f"  LLM:      Ollama ({config.llm.ollama_primary_model})")
    click.echo(f"  Kafka:    {config.kafka.bootstrap_servers}")
    click.echo("  Monthly cost: $0.00")
    click.echo("  Press Ctrl+C to stop.\n")

    # Notify Telegram on startup
    startup_msg = (
        "✅ NEXUS-ALPHA started\n"
        f"Mode: `{config.trading_mode.value}`\n"
        f"LLM: `{config.llm.ollama_primary_model}`"
    )
    await alerts.send(startup_msg)
    await alerts.system_health(health_status)

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("shutdown_requested")
        sentiment_runner.stop()
        ingestor.stop()
        free_intel.stop_scheduler()
        await openclaw.stop_all()
        await watchdog.stop()
        for t in tasks:
            t.cancel()
        await alerts.send("🛑 NEXUS-ALPHA stopped (manual shutdown)")
        logger.info("nexus_stopped")


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
