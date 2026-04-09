#!/usr/bin/env python3
"""Retrain watcher: periodically checks trade journal and triggers OnlineLearner retrains.

Run with: python infra/self_healing/retrain_watcher.py --interval 3600
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any

from nexus_alpha.learning.trade_logger import TradeLogger
from nexus_alpha.learning.offline_trainer import OnlineLearner
from nexus_alpha.alerts.telegram import TelegramAlerts


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrain_watcher")


def _send_telegram_alert(alerts: TelegramAlerts, message: str) -> None:
    try:
        if alerts and alerts.is_configured:
            asyncio.run(alerts.send(message))
    except Exception:
        logger.exception("telegram_alert_failed")


def retrain_watcher_main(interval_s: int = 3600) -> None:
    """Main loop: check for retrain condition and execute retrain when needed."""
    tl = TradeLogger()
    ol = OnlineLearner()
    alerts = TelegramAlerts.from_env()

    logger.info("retrain_watcher_started", interval_s=interval_s)
    try:
        while True:
            try:
                if ol.should_retrain(tl):
                    logger.info("retrain_condition_met")
                    # Record attempt metric
                    try:
                        from nexus_alpha.monitoring.metrics import RETRAIN_ATTEMPTS
                        RETRAIN_ATTEMPTS.inc()
                    except Exception:
                        pass
                    stats = ol.retrain_from_journal(tl)
                    if stats is None:
                        logger.info("retrain_no_dataset")
                    else:
                        updated = stats.get("updated", False)
                        n_trades = stats.get("n_trades", 0)
                        val_acc = stats.get("val_direction_accuracy")
                        val_bal = stats.get("val_balanced_accuracy")
                        msg = (
                            f"🔁 Online retrain completed — updated={updated} "
                            f"n_trades={n_trades} val_acc={val_acc} val_bal={val_bal}"
                        )
                        logger.info(msg)
                        tl.log_metric("last_retrain_summary", float(n_trades), details=json.dumps(stats))
                        _send_telegram_alert(alerts, msg)
                        try:
                            from nexus_alpha.monitoring.metrics import RETRAIN_ACCEPTED, RETRAIN_REJECTED
                            if updated:
                                RETRAIN_ACCEPTED.inc()
                            else:
                                RETRAIN_REJECTED.inc()
                        except Exception:
                            pass
                time.sleep(interval_s)
            except Exception:
                logger.exception("retrain_iteration_error")
                time.sleep(60)
    except KeyboardInterrupt:
        logger.info("retrain_watcher_stopped")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=3600, help="Seconds between checks")
    args = p.parse_args()
    retrain_watcher_main(interval_s=args.interval)
