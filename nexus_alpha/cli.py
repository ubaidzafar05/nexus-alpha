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

import click

from nexus_alpha.config import NexusConfig, TradingMode, load_config
from nexus_alpha.logging import get_logger, setup_logging

logger = get_logger(__name__)


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
@click.pass_context
def backtest(
    ctx: click.Context,
    start_date: str,
    end_date: str,
    initial_capital: float,
    symbols: str,
) -> None:
    """Run backtesting engine."""
    config: NexusConfig = ctx.obj["config"]
    symbol_list = [s.strip() for s in symbols.split(",")]
    logger.info(
        "backtest_starting",
        start_date=start_date,
        end_date=end_date,
        capital=initial_capital,
        symbols=symbol_list,
    )
    click.echo(f"Backtest: {start_date} → {end_date}")
    click.echo(f"Capital: ${initial_capital:,.0f}")
    click.echo(f"Symbols: {', '.join(symbol_list)}")
    click.echo("Backtest engine not yet fully wired — all modules are ready.")


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
    click.echo("NEXUS-ALPHA System Health")
    click.echo("=" * 40)

    components = [
        "TimescaleDB", "Redis", "Kafka", "World Model",
        "Regime Oracle", "Signal Engine", "Execution Engine",
        "Circuit Breaker", "OpenClaw Network",
    ]
    for comp in components:
        click.echo(f"  {comp}: not connected (system not running)")

    click.echo("\nStart with `nexus run` to see live health.")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host interface")
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

    results = runner.run_all(base_price=base_price)
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
        click.echo(f"  {status} {s['name']}: NAV {s['nav_impact']}, DD {s['max_dd']}, CB L{s['cb_level']}")


# ─── System Runner ────────────────────────────────────────────────────────────

async def _run_system(config: NexusConfig) -> None:
    """Initialize and run all system components."""
    logger.info("initializing_components")

    # Import components
    from nexus_alpha.agents.tournament import TournamentOrchestrator
    from nexus_alpha.core.regime_oracle import RegimeOracle
    from nexus_alpha.core.world_model import WorldModel
    from nexus_alpha.infrastructure.self_healing import SystemWatchdog
    from nexus_alpha.intelligence.openclaw_agents import OpenClawNetwork
    from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem
    from nexus_alpha.signals.signal_engine import SignalFusionEngine

    # Initialize
    circuit_breaker = CircuitBreakerSystem(risk_config=config.risk)
    regime_oracle = RegimeOracle(n_regimes=5, lookback_window=200)
    world_model = WorldModel(config.world_model)
    signal_engine = SignalFusionEngine()
    signal_engine.register_defaults()
    openclaw = OpenClawNetwork(config=config)
    watchdog = SystemWatchdog(check_interval_seconds=30.0)

    logger.info(
        "system_initialized",
        trading_mode=config.trading_mode.value,
        circuit_breaker="enabled" if config.risk.circuit_breaker_enabled else "disabled",
    )

    # Start subsystems
    tasks = [
        asyncio.create_task(openclaw.start_all()),
        asyncio.create_task(watchdog.run()),
    ]

    click.echo("NEXUS-ALPHA v3.0 is running.")
    click.echo(f"Mode: {config.trading_mode.value}")
    click.echo("Press Ctrl+C to stop.")

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("shutdown_requested")
        await openclaw.stop_all()
        await watchdog.stop()
        for t in tasks:
            t.cancel()
        logger.info("nexus_stopped")


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
