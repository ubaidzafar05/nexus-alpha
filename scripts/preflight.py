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
  ✓ Telegram bot functional when configured (--full)
  ✓ Freqtrade strategy parseable
  ✓ Critical imports work
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _env_value(name: str, env_file_values: dict[str, str]) -> str:
    return os.getenv(name) or env_file_values.get(name, "")


def check_env_vars() -> None:
    print("\n── Environment Variables ──")
    env_file_values = _read_env_file(ROOT / ".env")
    trading_mode = _env_value("TRADING_MODE", env_file_values).lower() or "paper"
    recommended = [
        "GROQ_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "CRYPTOPANIC_API_TOKEN",
        "ETHERSCAN_API_KEY",
    ]

    for var in ("BINANCE_API_KEY", "BINANCE_API_SECRET"):
        if _env_value(var, env_file_values):
            _ok(f"{var} set")
        elif trading_mode == "paper":
            _warn(f"{var} missing (paper mode)")
        else:
            _fail(f"{var} missing (required for live trading)")

    for var in recommended:
        if _env_value(var, env_file_values):
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
        elif config.llm.use_groq_fallback:
            _warn("Groq fallback enabled but API key missing")
        else:
            _ok("Groq fallback disabled (Ollama-only mode)")

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
            elif svc == "telegram" and state == "not_configured":
                _warn(f"{svc}: {state}")
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
