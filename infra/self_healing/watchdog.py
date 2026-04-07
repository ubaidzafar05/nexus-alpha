#!/usr/bin/env python3
"""Lightweight Docker Compose watchdog for NEXUS-ALPHA.

Runs on the host and periodically checks container health for core services
(timescaledb, kafka, ollama, redis, nexus). If a service is unhealthy, the
watchdog will restart it via "docker-compose restart <service>" and send a
Telegram alert using nexus_alpha.alerts.telegram.TelegramAlerts (if configured).

Usage:
  PYTHONPATH=. python infra/self_healing/watchdog.py

Environment variables:
  WATCH_SERVICES    comma-separated service names to monitor (default: timescaledb,kafka,ollama,redis,nexus)
  MONITOR_INTERVAL  seconds between checks (default: 60)
  MIN_RESTART_INTERVAL  minimum seconds between restarts for the same service (default: 300)
  TELEGRAM env vars are read by TelegramAlerts.from_env (.env is also read)

This script is intentionally simple so it can run as a cron job, systemd service
or inside a small container. It avoids extra Python dependencies and shells
out to docker/docker-compose.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List

from nexus_alpha.logging import get_logger
from nexus_alpha.alerts.telegram import TelegramAlerts

logger = get_logger(__name__)

WATCH_SERVICES = os.getenv("WATCH_SERVICES", "timescaledb,kafka,ollama,redis,nexus").split(",")
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "60"))
MIN_RESTART_INTERVAL = int(os.getenv("MIN_RESTART_INTERVAL", "300"))
ENV_FILE = os.getenv("ENV_FILE", ".env")

_shutdown = False


def _run(cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=False, stdout=subprocess.PIPE if capture_output else None, stderr=subprocess.PIPE if capture_output else None, text=True)
    except FileNotFoundError:
        logger.error("command_not_found", cmd=cmd[0])
        raise


async def _send_alert(client: TelegramAlerts, message: str) -> None:
    try:
        if client.is_configured:
            await client.system_health({"watchdog": message})
        else:
            logger.info("watchdog_alert", msg=message)
    except Exception as err:
        logger.warning("alert_send_failed", error=repr(err))


def _container_id_for_service(service: str) -> str | None:
    # Use docker-compose to find the container id for the service
    proc = _run(["docker-compose", "ps", "-q", service])
    if proc.returncode != 0:
        logger.debug("docker_compose_ps_failed", service=service, stderr=proc.stderr.strip())
        return None
    cid = proc.stdout.strip()
    return cid or None


def _health_status_for_container(cid: str) -> str | None:
    # Prefer detailed Health.Status if present
    proc = _run(["docker", "inspect", "--format", "{{json .State.Health.Status}}", cid])
    if proc.returncode == 0 and proc.stdout:
        out = proc.stdout.strip()
        try:
            return json.loads(out)
        except Exception:
            return out.strip('"')
    # Fallback to general state (running/exited)
    proc2 = _run(["docker", "inspect", "--format", "{{.State.Status}}", cid])
    if proc2.returncode == 0 and proc2.stdout:
        return proc2.stdout.strip()
    return None


def _restart_service(service: str) -> bool:
    logger.info("restarting_service", service=service)
    proc = _run(["docker-compose", "restart", service], capture_output=True)
    if proc.returncode == 0:
        logger.info("service_restarted", service=service)
        return True
    logger.warning("service_restart_failed", service=service, stderr=proc.stderr.strip())
    return False


async def monitor_loop() -> None:
    last_restart: Dict[str, float] = defaultdict(lambda: 0.0)
    alerts = TelegramAlerts.from_env(env_file=ENV_FILE)

    while not _shutdown:
        for svc in WATCH_SERVICES:
            svc = svc.strip()
            if not svc:
                continue
            cid = _container_id_for_service(svc)
            if not cid:
                logger.warning("service_not_found", service=svc)
                continue
            status = _health_status_for_container(cid)
            logger.debug("service_health", service=svc, container=cid, status=status)
            if status not in ("healthy", "running"):
                now = time.time()
                if now - last_restart[svc] < MIN_RESTART_INTERVAL:
                    logger.info("skip_restart_rate_limit", service=svc, since_last=int(now - last_restart[svc]))
                    continue
                ok = _restart_service(svc)
                last_restart[svc] = now if ok else last_restart[svc]
                txt = f"Watchdog restarted service {svc} (health={status})" if ok else f"Watchdog failed to restart {svc} (health={status})"
                await _send_alert(alerts, txt)
        await asyncio.sleep(MONITOR_INTERVAL)


def _on_signal(signum, frame):
    global _shutdown
    logger.info("watchdog_shutdown_signal", signal=signum)
    _shutdown = True


def main() -> int:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    logger.info("watchdog_starting", services=WATCH_SERVICES, interval=MONITOR_INTERVAL)
    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        logger.info("watchdog_terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
