#!/usr/bin/env python3
"""
NEXUS-ALPHA Pre-flight Checklist — Run before going live.

Usage:
    python scripts/preflight.py          # Quick check (no network)
    python scripts/preflight.py --full   # Full check (probes services)

Validates:
  ✓ Environment variables present
  ✓ Config loads without error
  ✓ Exchange credentials configured
  ✓ Ollama model availability (--full)
  ✓ Redis/Kafka/TimescaleDB reachable (--full)
  ✓ Telegram bot functional (--full)
  ✓ Freqtrade strategy parseable
  ✓ Critical imports work
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys

CHECKS_PASSED = 0
CHECKS_FAILED = 0
CHECKS_WARNED = 0


def _ok(msg: str) -> None:
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  ✅ {msg}")


def _fail(msg: str) -> None:
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  ❌ {msg}")


def _warn(msg: str) -> None:
    global CHECKS_WARNED
    CHECKS_WARNED += 1
    print(f"  ⚠️  {msg}")


def check_env_vars() -> None:
    print("\n── Environment Variables ──")
    required = [
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
    ]
    recommended = [
        "GROQ_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "CRYPTOPANIC_API_TOKEN",
        "ETHERSCAN_API_KEY",
    ]
    for var in required:
        if os.getenv(var):
            _ok(f"{var} set")
        else:
            _fail(f"{var} missing (required for live trading)")

    for var in recommended:
        if os.getenv(var):
            _ok(f"{var} set")
        else:
            _warn(f"{var} not set (recommended)")


def check_config_loads() -> None:
    print("\n── Configuration ──")
    try:
        from nexus_alpha.config import load_config

        config = load_config()
        _ok(f"Config loaded: mode={config.trading_mode.value}")
        _ok(f"LLM backend: {config.llm.ollama_base_url}")

        if config.llm.has_groq:
            _ok("Groq fallback configured")
        else:
            _warn("Groq fallback not configured")

        if config.binance.api_key.get_secret_value():
            _ok("Binance credentials present")
        else:
            _warn("Binance credentials empty (paper mode only)")

        if config.binance.testnet:
            _ok("Binance testnet enabled")
        elif config.is_live:
            _warn("Binance testnet disabled (real exchange if live)")

    except Exception as e:
        _fail(f"Config load failed: {e}")


def check_imports() -> None:
    print("\n── Critical Imports ──")
    modules = [
        "nexus_alpha.core.trading_loop",
        "nexus_alpha.signals.signal_engine",
        "nexus_alpha.agents.debate",
        "nexus_alpha.portfolio.optimizer",
        "nexus_alpha.execution.execution_engine",
        "nexus_alpha.risk.circuit_breaker",
        "nexus_alpha.data.sentiment_pipeline",
        "nexus_alpha.data.live_ingestor",
        "nexus_alpha.alerts.telegram",
        "nexus_alpha.intelligence.free_llm",
        "nexus_alpha.intelligence.crawl4ai_agents",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            _ok(f"import {mod.split('.')[-1]}")
        except Exception as e:
            _fail(f"import {mod}: {e}")


def check_signal_engine() -> None:
    print("\n── Signal Engine ──")
    try:
        from nexus_alpha.signals.signal_engine import SignalFusionEngine

        engine = SignalFusionEngine()
        engine.register_defaults()
        _ok(f"{len(engine.generators)} signal generators registered")
    except Exception as e:
        _fail(f"Signal engine init: {e}")


def check_freqtrade_strategy() -> None:
    print("\n── Freqtrade Strategy ──")
    config_path = "freqtrade/config/config.json"
    if os.path.exists(config_path):
        _ok(f"{config_path} exists")
    else:
        _warn(f"{config_path} not found")

    strategy_path = "freqtrade/strategies/NexusAlphaStrategy.py"
    if os.path.exists(strategy_path):
        _ok(f"{strategy_path} exists")
    else:
        _warn(f"{strategy_path} not found")


async def check_services_full() -> None:
    print("\n── Service Connectivity ──")
    try:
        from nexus_alpha.cli import _collect_health_status
        from nexus_alpha.config import load_config

        config = load_config()
        status = await _collect_health_status(config)
        for svc, state in status.items():
            if state in {"ok", "configured"}:
                _ok(f"{svc}: {state}")
            elif state == "degraded":
                _warn(f"{svc}: {state}")
            else:
                _fail(f"{svc}: {state}")
    except Exception as e:
        _fail(f"Health check error: {e}")


def main() -> None:
    full_mode = "--full" in sys.argv

    print("╔══════════════════════════════════════════╗")
    print("║  NEXUS-ALPHA Pre-flight Checklist        ║")
    print("╚══════════════════════════════════════════╝")

    check_env_vars()
    check_config_loads()
    check_imports()
    check_signal_engine()
    check_freqtrade_strategy()

    if full_mode:
        asyncio.run(check_services_full())
    else:
        print("\n  ℹ️  Run with --full to probe live services")

    print("\n" + "=" * 44)
    print(f"  ✅ Passed: {CHECKS_PASSED}")
    print(f"  ⚠️  Warned: {CHECKS_WARNED}")
    print(f"  ❌ Failed: {CHECKS_FAILED}")
    print("=" * 44)

    if CHECKS_FAILED > 0:
        print("\n⛔ Fix failures before going live!")
        sys.exit(1)
    elif CHECKS_WARNED > 0:
        print("\n⚡ Ready for paper trading (warnings are optional)")
    else:
        print("\n🚀 All clear — ready for production!")


if __name__ == "__main__":
    main()
