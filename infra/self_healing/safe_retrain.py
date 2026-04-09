#!/usr/bin/env python3
"""Safe retrain pipeline:
- Run OnlineLearner.retrain_from_journal saving candidate to temp path
- Run benchmark_learning_targets to sanity-check learnability
- Promote candidate to production path only if both retrain accepted and benchmark indicates models learnable
- Send Telegram alerts with summary
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from nexus_alpha.learning.trade_logger import TradeLogger
from nexus_alpha.learning.offline_trainer import OnlineLearner, benchmark_learning_targets
from nexus_alpha.alerts.telegram import TelegramAlerts
from nexus_alpha.logging import get_logger

logger = get_logger("safe_retrain")


def main(
    candidate_path: str = "/tmp/lightweight_candidate.pkl",
    promote_path: str = "data/checkpoints/lightweight_online_reward.pkl",
    min_trades: int = 30,
):
    candidate = Path(candidate_path)
    promote = Path(promote_path)
    alerts = TelegramAlerts.from_env()
    tl = TradeLogger()

    ol = OnlineLearner(model_path=candidate)
    logger.info("starting_candidate_retrain", candidate=str(candidate))
    stats = ol.retrain_from_journal(tl)
    if stats is None:
        msg = "No retrain dataset available; aborting safe_retrain"
        logger.info(msg)
        if alerts.is_configured:
            try:
                # TelegramAlerts.send is async; provide sync wrapper
                __import__('asyncio').run(alerts.send(msg))
            except Exception:
                logger.exception('telegram_send_failed')
        return 1

    # Summarize
    updated = bool(stats.get("updated", False))
    n_trades = int(stats.get("n_trades", 0))
    val_bal = float(stats.get("val_balanced_accuracy", 0.0))
    val_acc = float(stats.get("val_direction_accuracy", 0.0))

    if not updated:
        msg = f"Retrain rejected by validation: updated={updated} n_trades={n_trades} val_acc={val_acc} val_bal={val_bal}"
        logger.info(msg)
        if alerts.is_configured:
            try:
                __import__('asyncio').run(alerts.send(msg))
            except Exception:
                logger.exception('telegram_send_failed')
        # cleanup candidate if exists
        try:
            if candidate.exists():
                candidate.unlink()
        except Exception:
            logger.exception("failed_cleanup_candidate")
        return 2

    # If updated, run benchmark_learning_targets
    logger.info("retrain_candidate_saved", path=str(candidate))
    bench = benchmark_learning_targets(tl, min_trades=min_trades)
    if bench is None:
        msg = "Benchmark unavailable or insufficient trades; rejecting promotion"
        logger.info(msg)
        if alerts.is_configured:
            try:
                __import__('asyncio').run(alerts.send(msg))
            except Exception:
                logger.exception('telegram_send_failed')
        try:
            if candidate.exists():
                candidate.unlink()
        except Exception:
            logger.exception("failed_cleanup_candidate")
        return 3

    # Accept if any variant has a best_model and baseline improved
    variants = bench.get("variants", {})
    any_good = False
    for vname, vres in variants.items():
        if not vres:
            continue
        baseline_bal = float(vres.get("baseline_balanced_accuracy", 0.0))
        # pick model metric: check best_model and its balanced accuracy
        best = vres.get("best_model")
        if not best:
            continue
        models = vres.get("models", {})
        mstats = models.get(best, {})
        model_bal = float(mstats.get("balanced_accuracy", 0.0))
        if model_bal > baseline_bal + 0.02:
            any_good = True
            break

    if not any_good:
        msg = f"Benchmark did not show consistent improvement; rejecting promotion (bench={bench})"
        logger.info(msg)
        if alerts.is_configured:
            try:
                __import__('asyncio').run(alerts.send(msg))
            except Exception:
                logger.exception('telegram_send_failed')
        try:
            if candidate.exists():
                candidate.unlink()
        except Exception:
            logger.exception("failed_cleanup_candidate")
        return 4

    # Promote: atomically move candidate -> promote path
    promote.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(candidate), str(promote))
        msg = f"✅ Retrain promoted to {promote} (n_trades={n_trades} val_acc={val_acc} val_bal={val_bal})"
        logger.info(msg)
        if alerts.is_configured:
            try:
                __import__('asyncio').run(alerts.send(msg))
            except Exception:
                logger.exception('telegram_send_failed')
        return 0
    except Exception as e:
        logger.exception("promotion_failed")
        if alerts.is_configured:
            try:
                __import__('asyncio').run(alerts.send(f"Promotion failed: {e}"))
            except Exception:
                logger.exception('telegram_send_failed')
        return 5


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="/tmp/lightweight_candidate.pkl")
    p.add_argument("--promote", default="data/checkpoints/lightweight_online_reward.pkl")
    p.add_argument("--min-trades", type=int, default=30)
    args = p.parse_args()
    sys.exit(main(candidate_path=args.candidate, promote_path=args.promote, min_trades=args.min_trades))
