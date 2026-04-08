#!/usr/bin/env python3
"""Simple Ollama monitor: polls /api/tags and alerts on failure.

Run as: python infra/self_healing/ollama_monitor.py --interval 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from nexus_alpha.alerts.telegram import TelegramAlerts
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


async def check_once(base_url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base_url.rstrip('/')}/api/tags")
            r.raise_for_status()
            models = [m.get('name') for m in r.json().get('models', [])]
            return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


async def monitor_loop(base_url: str, interval: int, alerts: TelegramAlerts):
    fail_count = 0
    while True:
        res = await check_once(base_url)
        if res.get("ok"):
            if fail_count >= 1:
                logger.info("ollama_recovered", models=res.get("models"))
                await alerts.ollama_status(True, res.get("models"))
            fail_count = 0
        else:
            fail_count += 1
            logger.warning("ollama_check_failed", error=res.get("error"), fail_count=fail_count)
            # Alert on first failure and every 3 consecutive fails
            if fail_count == 1 or fail_count % 3 == 0:
                await alerts.ollama_status(False, None)
        await asyncio.sleep(interval)


def main(interval: int = 30):
    cfg_path = Path('.env')
    from nexus_alpha.config import load_config

    cfg = load_config(str(cfg_path))
    alerts = TelegramAlerts.from_env()
    base = cfg.llm.ollama_base_url
    try:
        asyncio.run(monitor_loop(base, interval, alerts))
    except KeyboardInterrupt:
        logger.info("ollama_monitor_stopped")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=30)
    args = p.parse_args()
    main(args.interval)
