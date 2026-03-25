"""
Telegram alerting system — replaces PagerDuty at zero cost.

Setup:
  1. Message @BotFather on Telegram → /newbot → get TELEGRAM_BOT_TOKEN
  2. Message @userinfobot to find your TELEGRAM_CHAT_ID
  3. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Supports:
  - Trade notifications (entry, exit, P&L)
  - Risk alerts (drawdown, circuit breaker)
  - Daily summary reports
  - System health alerts
  - Rate limiting to avoid Telegram API spam
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    TRADE = "trade"
    SYSTEM = "system"

    @property
    def icon(self) -> str:
        return {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
            "trade": "💹",
            "system": "🤖",
        }[self.value]


@dataclass
class Alert:
    message: str
    level: AlertLevel = AlertLevel.INFO
    timestamp: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class TelegramAlerts:
    """
    Async Telegram alert client.

    Includes:
    - Per-message rate limiting (Telegram allows ~30 msg/sec, we cap at 1/sec)
    - Message deduplication (won't re-send identical message within 60s)
    - Graceful degradation (logs instead of raising on failure)
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        min_interval_s: float = 1.0,
        dedup_window_s: float = 60.0,
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._min_interval = min_interval_s
        self._dedup_window = dedup_window_s
        self._last_sent: float = 0.0
        self._recent: deque[tuple[str, float]] = deque(maxlen=50)
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "TelegramAlerts":
        """Load credentials from environment variables."""
        import os
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            logger.warning("telegram_not_configured", hint="Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return cls(bot_token=token, chat_id=chat_id)

    @property
    def is_configured(self) -> bool:
        return bool(self._token and self._chat_id)

    # ── Core send ────────────────────────────────────────────────────────────

    async def send(self, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """Send a message. Returns True on success, False on failure."""
        if not self.is_configured:
            logger.debug("telegram_not_configured_skipping_alert")
            return False

        formatted = f"{level.icon} *NEXUS-ALPHA*\n\n{message}"

        async with self._lock:
            import time
            now = time.monotonic()

            # Deduplication check
            for prev_msg, prev_time in self._recent:
                if prev_msg == formatted and now - prev_time < self._dedup_window:
                    logger.debug("telegram_dedup_skip")
                    return True

            # Rate limiting
            wait = self._min_interval - (now - self._last_sent)
            if wait > 0:
                await asyncio.sleep(wait)

            success = await self._do_send(formatted)
            if success:
                self._last_sent = time.monotonic()
                self._recent.append((formatted, time.monotonic()))
            return success

    async def _do_send(self, text: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{TELEGRAM_API}/bot{self._token}/sendMessage",
                    json={
                        "chat_id": self._chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                )
                resp.raise_for_status()
                return True
        except Exception as err:
            logger.warning("telegram_send_failed", error=str(err))
            return False

    # ── High-level alert methods ──────────────────────────────────────────────

    async def trade_opened(self, trade: dict[str, Any]) -> None:
        msg = (
            f"*Trade Opened* 📈\n\n"
            f"Pair: `{trade.get('pair', 'N/A')}`\n"
            f"Direction: {trade.get('direction', 'long').upper()}\n"
            f"Entry: `${trade.get('entry_price', 0):,.4f}`\n"
            f"Size: `${trade.get('size_usd', 0):,.0f}` "
            f"({trade.get('size_pct_nav', 0):.1f}% NAV)\n"
            f"Strategy: `{trade.get('strategy', 'N/A')}`\n"
            f"Regime: `{trade.get('regime', 'N/A')}`\n"
            f"Confidence: `{trade.get('confidence', 0):.0%}`"
        )
        await self.send(msg, AlertLevel.TRADE)

    async def trade_closed(self, trade: dict[str, Any]) -> None:
        pnl = trade.get("pnl_pct", 0)
        icon = "✅" if pnl >= 0 else "❌"
        msg = (
            f"*Trade Closed* {icon}\n\n"
            f"Pair: `{trade.get('pair', 'N/A')}`\n"
            f"P&L: `{pnl:+.2f}%` (`${trade.get('pnl_usd', 0):+,.2f}`)\n"
            f"Duration: {trade.get('duration_hours', 0):.1f}h\n"
            f"Exit Reason: `{trade.get('exit_reason', 'N/A')}`"
        )
        await self.send(msg, AlertLevel.TRADE)

    async def risk_alert(self, alert_type: str, details: dict[str, Any]) -> None:
        msg = (
            f"*Risk Alert* — `{alert_type}`\n\n"
            + "\n".join(f"`{k}`: {v}" for k, v in details.items())
        )
        await self.send(msg, AlertLevel.CRITICAL)

    async def circuit_breaker_triggered(self, reason: str, drawdown_pct: float) -> None:
        msg = (
            f"🛑 *Circuit Breaker TRIGGERED*\n\n"
            f"Reason: `{reason}`\n"
            f"Current Drawdown: `{drawdown_pct:.2f}%`\n"
            f"All trading halted until manual review."
        )
        await self.send(msg, AlertLevel.CRITICAL)

    async def daily_report(self, stats: dict[str, Any]) -> None:
        pnl_pct = stats.get("daily_pnl_pct", 0)
        pnl_icon = "📈" if pnl_pct >= 0 else "📉"
        msg = (
            f"*Daily P&L Report* {pnl_icon}\n\n"
            f"💰 NAV: `${stats.get('nav', 0):,.2f}`\n"
            f"Daily PnL: `{pnl_pct:+.2f}%` (`${stats.get('daily_pnl_usd', 0):+,.2f}`)\n"
            f"Drawdown: `{stats.get('drawdown', 0):.2f}%`\n"
            f"Sharpe (30d): `{stats.get('sharpe_30d', 0):.2f}`\n"
            f"Win Rate (30d): `{stats.get('win_rate_30d', 0):.0%}`\n\n"
            f"Active Strategies: {stats.get('active_strategies', 0)}\n"
            f"News Processed: {stats.get('news_processed', 0)}\n"
            f"Sentiment: `{stats.get('current_sentiment', 0):+.2f}`\n"
            f"Regime: `{stats.get('current_regime', 'unknown')}`"
        )
        await self.send(msg, AlertLevel.INFO)

    async def system_health(self, status: dict[str, Any]) -> None:
        msg = (
            f"*System Health Check*\n\n"
            + "\n".join(
                f"{'✅' if v == 'ok' else '❌'} {k}: `{v}`"
                for k, v in status.items()
            )
        )
        await self.send(msg, AlertLevel.SYSTEM)

    async def ollama_status(self, healthy: bool, models: list[str] | None = None) -> None:
        if healthy:
            msg = f"✅ Ollama LLM server online\nModels: {', '.join(models or [])}"
        else:
            msg = "⚠️ Ollama LLM server unreachable — falling back to Groq free tier"
        await self.send(msg, AlertLevel.SYSTEM)
