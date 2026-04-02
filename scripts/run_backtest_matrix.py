#!/usr/bin/env python3
"""Run reproducible Freqtrade backtest/ablation matrix for NexusAlphaStrategy."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOTAL_ROW_RE = re.compile(
    r"│\s*TOTAL\s*│\s*(?P<trades>\d+)\s*│\s*(?P<avg_profit>[-\d.]+)\s*│\s*(?P<profit_usdt>[-\d.]+)\s*│\s*(?P<profit_pct>[-\d.]+)"
)

DEFAULT_VARIANTS: dict[str, dict[str, str]] = {
    "base": {},
    "no_regime_filter": {"NEXUS_ENABLE_REGIME_FILTER": "false"},
    "trend_only": {"NEXUS_ENABLE_MEAN_REVERSION": "false"},
    "mean_reversion_only": {"NEXUS_ENABLE_TREND": "false"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--pairs", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--strategy", default="NexusAlphaStrategy")
    parser.add_argument("--config", default="/freqtrade/user_data/config/config.json")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--run-hyperopt", action="store_true")
    parser.add_argument("--hyperopt-epochs", default=10, type=int)
    parser.add_argument("--output-json", default="backtest_matrix_results.json")
    return parser.parse_args()


def _timerange(start_date: str, end_date: str) -> str:
    return f"{start_date.replace('-', '')}-{end_date.replace('-', '')}"


def _run_command(args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(  # noqa: S603
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _extract_total_metrics(output: str) -> dict[str, float | int] | None:
    in_report = False
    for line in output.splitlines():
        if "BACKTESTING REPORT" in line:
            in_report = True
            continue
        if not in_report:
            continue
        match = TOTAL_ROW_RE.search(line)
        if match:
            return {
                "trades": int(match.group("trades")),
                "avg_profit_pct": float(match.group("avg_profit")),
                "total_profit_usdt": float(match.group("profit_usdt")),
                "total_profit_pct": float(match.group("profit_pct")),
            }
    return None


def run_download_data(config_path: str, pairs: list[str], timeframe: str, timerange: str) -> None:
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "freqtrade",
        "download-data",
        "--config",
        config_path,
        "--pairs",
        *pairs,
        "--timeframes",
        timeframe,
        "--timerange",
        timerange,
    ]
    result = _run_command(cmd)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    print("download_data: ok")


def run_backtest_variant(
    *,
    config_path: str,
    strategy: str,
    pairs: list[str],
    timeframe: str,
    timerange: str,
    variant_name: str,
    variant_env: dict[str, str],
) -> dict[str, object]:
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        *sum((["-e", f"{key}={value}"] for key, value in variant_env.items()), []),
        "freqtrade",
        "backtesting",
        "--config",
        config_path,
        "--strategy",
        strategy,
        "--pairs",
        *pairs,
        "--timeframe",
        timeframe,
        "--timerange",
        timerange,
    ]
    result = _run_command(cmd)
    metrics = _extract_total_metrics(result.stdout)
    return {
        "variant": variant_name,
        "env": variant_env,
        "returncode": result.returncode,
        "metrics": metrics,
        "stdout_tail": result.stdout[-6000:],
        "stderr_tail": result.stderr[-2000:],
    }


def run_hyperopt(
    *,
    config_path: str,
    strategy: str,
    pairs: list[str],
    timeframe: str,
    timerange: str,
    epochs: int,
) -> dict[str, object]:
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "freqtrade",
        "hyperopt",
        "--config",
        config_path,
        "--strategy",
        strategy,
        "--pairs",
        *pairs,
        "--timeframe",
        timeframe,
        "--timerange",
        timerange,
        "--spaces",
        "buy",
        "sell",
        "--epochs",
        str(epochs),
        "--hyperopt-loss",
        "SharpeHyperOptLoss",
    ]
    result = _run_command(cmd)
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-6000:],
        "stderr_tail": result.stderr[-2000:],
    }


def main() -> int:
    args = parse_args()
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    timerange = _timerange(args.start_date, args.end_date)

    if args.download_data:
        run_download_data(args.config, pairs, args.timeframe, timerange)

    results = {
        "timerange": timerange,
        "pairs": pairs,
        "timeframe": args.timeframe,
        "strategy": args.strategy,
        "variants": [],
    }

    for name, env_vars in DEFAULT_VARIANTS.items():
        outcome = run_backtest_variant(
            config_path=args.config,
            strategy=args.strategy,
            pairs=pairs,
            timeframe=args.timeframe,
            timerange=timerange,
            variant_name=name,
            variant_env=env_vars,
        )
        results["variants"].append(outcome)
        metrics = outcome["metrics"]
        if metrics is None:
            print(f"{name}: failed (see output)")
        else:
            print(
                f"{name}: trades={metrics['trades']} "
                f"profit_pct={metrics['total_profit_pct']:.2f} "
                f"profit_usdt={metrics['total_profit_usdt']:.3f}"
            )

    if args.run_hyperopt:
        results["hyperopt"] = run_hyperopt(
            config_path=args.config,
            strategy=args.strategy,
            pairs=pairs,
            timeframe=args.timeframe,
            timerange=timerange,
            epochs=args.hyperopt_epochs,
        )
        print(f"hyperopt: returncode={results['hyperopt']['returncode']}")

    output_path = ROOT / args.output_json
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved_results={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
