#!/usr/bin/env python3
"""Validate required production inputs and optional network reachability."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_ENV_KEYS = [
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "BYBIT_API_KEY",
    "BYBIT_API_SECRET",
    "KRAKEN_API_KEY",
    "KRAKEN_API_SECRET",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "TIMESCALEDB_URL",
    "REDIS_URL",
    "KAFKA_BOOTSTRAP_SERVERS",
    "VAULT_ADDR",
    "VAULT_TOKEN",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env", help="Path to env file")
    parser.add_argument(
        "--inputs-file",
        default="config/production_inputs.yaml",
        help="Path to production inputs YAML",
    )
    parser.add_argument(
        "--check-network",
        action="store_true",
        help="Check TCP reachability for configured services",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if env_path.exists():
        load_env_file(env_path)

    failures: list[str] = []
    warnings: list[str] = []

    for key in REQUIRED_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if not value:
            failures.append(f"missing_env:{key}")

    inputs_path = Path(args.inputs_file)
    if not inputs_path.exists():
        warnings.append(f"missing_inputs_file:{inputs_path}")
    else:
        failures.extend(validate_inputs_yaml(inputs_path))

    if args.check_network:
        failures.extend(run_network_checks())

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"- {warning}")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("OK: production readiness preflight passed")
    return 0


def load_env_file(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def validate_inputs_yaml(path: Path) -> list[str]:
    failures: list[str] = []
    try:
        import yaml
    except ImportError:
        return ["missing_dependency:pyyaml"]

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("project", {}).get("environment") != "production":
        failures.append("inputs_environment_not_production")

    approved_by = payload.get("policy", {}).get("approved_by", "")
    if not str(approved_by).strip():
        failures.append("policy_approval_missing")

    stage_owners = payload.get("rollout", {}).get("stage_owners", {})
    for stage in ("paper", "micro_live", "small_live", "production"):
        if not str(stage_owners.get(stage, "")).strip():
            failures.append(f"missing_stage_owner:{stage}")

    return failures


def run_network_checks() -> list[str]:
    failures: list[str] = []
    targets = []
    targets.extend(parse_bootstrap_servers(os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")))
    targets.append(parse_url_host_port(os.getenv("REDIS_URL", ""), 6379))
    targets.append(parse_url_host_port(os.getenv("TIMESCALEDB_URL", ""), 5432))
    targets.append(parse_url_host_port(os.getenv("VAULT_ADDR", ""), 8200))

    for host, port, label in targets:
        if not host:
            failures.append(f"invalid_target:{label}")
            continue
        if not tcp_check(host, port, timeout=1.0):
            failures.append(f"unreachable:{label}:{host}:{port}")
    return failures


def parse_bootstrap_servers(value: str) -> list[tuple[str, int, str]]:
    items: list[tuple[str, int, str]] = []
    for idx, server in enumerate([s.strip() for s in value.split(",") if s.strip()]):
        if ":" in server:
            host, port_text = server.rsplit(":", 1)
            try:
                port = int(port_text)
            except ValueError:
                items.append(("", 0, f"kafka_{idx}"))
                continue
        else:
            host, port = server, 9092
        items.append((host, port, f"kafka_{idx}"))
    return items


def parse_url_host_port(value: str, default_port: int) -> tuple[str, int, str]:
    if not value:
        return "", 0, "url"
    normalized = value.replace("postgresql+asyncpg://", "postgresql://", 1)
    parsed = urlparse(normalized)
    return parsed.hostname or "", parsed.port or default_port, parsed.scheme or "url"


def tcp_check(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())
