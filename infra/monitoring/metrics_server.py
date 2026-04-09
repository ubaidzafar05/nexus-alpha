#!/usr/bin/env python3
"""Start an HTTP metrics endpoint for Prometheus.

This script imports the FreeLLMClient module to ensure metrics are registered
and starts prometheus_client's HTTP server. Run as a detached background process.
"""
from __future__ import annotations

import argparse
import time

try:
    from prometheus_client import start_http_server
except Exception:
    raise SystemExit("prometheus_client is required: pip install prometheus_client")

# Import the module so that metrics defined there are registered
import nexus_alpha.intelligence.free_llm as _free


def main(port: int = 8000, interval: int = 5):
    print(f"Starting metrics server on :{port}")
    start_http_server(port)
    try:
        while True:
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Metrics server stopped")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    main(port=args.port)
