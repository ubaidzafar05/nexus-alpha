#!/usr/bin/env python3
"""Lightweight Prometheus exporter that exposes simple Ollama metrics by tailing the
ollama_monitor log file. This is a scaffold for metrics; install prometheus_client
(if not present) via pip install prometheus_client.

Run: python infra/monitoring/ollama_prometheus_exporter.py --port 8001 --log /tmp/ollama_monitor.log
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from prometheus_client import start_http_server, Gauge
except Exception:
    raise SystemExit("prometheus_client is required: pip install prometheus_client")


def parse_counts(log_path: Path) -> dict:
    text = log_path.read_text() if log_path.exists() else ""
    return {
        "fail_count": text.count("ollama_check_failed"),
        "recover_count": text.count("ollama_recovered"),
        "retries_exhausted": text.count("ollama_retries_exhausted"),
    }


def main(port: int = 8001, log_path: str = "/tmp/ollama_monitor.log", interval: int = 10):
    lp = Path(log_path)
    g_fail = Gauge("nexus_ollama_fail_count", "Number of ollama check failures seen in log")
    g_recover = Gauge("nexus_ollama_recover_count", "Number of ollama recoveries seen in log")
    g_retries = Gauge("nexus_ollama_retries_exhausted", "Number of ollama retries exhausted entries")

    start_http_server(port)
    print(f"Prometheus exporter listening on :{port}, tailing {lp}")

    last_counts = {"fail_count": 0, "recover_count": 0, "retries_exhausted": 0}
    while True:
        counts = parse_counts(lp)
        # set gauges
        g_fail.set(counts["fail_count"])
        g_recover.set(counts["recover_count"])
        g_retries.set(counts["retries_exhausted"])
        last_counts = counts
        time.sleep(interval)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--log", type=str, default="/tmp/ollama_monitor.log")
    p.add_argument("--interval", type=int, default=10)
    args = p.parse_args()
    main(port=args.port, log_path=args.log, interval=args.interval)
