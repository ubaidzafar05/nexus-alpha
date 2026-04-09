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
import contextlib
import os
import random
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
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_env(cls, env_file: str = ".env") -> TelegramAlerts:
        """Load credentials from environment variables or a .env file.

        Will attempt to extract a numeric chat id if the provided value contains extra
        content (e.g. pasted JSON or bot responses). This prevents common misconfiguration
        where users paste API responses into TELEGRAM_CHAT_ID.
        """
        import os
        import re
        from pathlib import Path

        from dotenv import dotenv_values

        file_values = dotenv_values(Path(env_file)) if Path(env_file).exists() else {}
        token = os.getenv("TELEGRAM_BOT_TOKEN", "") or str(file_values.get("TELEGRAM_BOT_TOKEN", ""))
        chat_id_raw = os.getenv("TELEGRAM_CHAT_ID", "") or str(file_values.get("TELEGRAM_CHAT_ID", ""))

        # Normalize chat_id: prefer an integer string if one exists inside the provided value
        chat_id = ""
        if chat_id_raw:
            m = re.search(r"(-?\d+)", chat_id_raw)
            if m:
                chat_id = m.group(1)
            else:
                chat_id = chat_id_raw

        if not token or not chat_id:
            logger.warning(
                "telegram_not_configured",
                hint="Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (numeric chat id).",
            )
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

        # Use plain-text messages (no Markdown) to avoid parse errors from unescaped content.
        formatted = f"{level.icon} NEXUS-ALPHA\n\n{message}"
        # Truncate to Telegram limits (~4096 chars).
        if len(formatted) > 3900:
            formatted = formatted[:3897] + "..."

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
        # Exponential backoff with jitter for transport-level failures
        max_attempts = 6
        base_backoff = 1.0
        for attempt in range(max_attempts):
            try:
                client = self._get_client()
                # Per-request timeout slightly larger than client default to allow retries
                resp = await client.post(
                    f"{TELEGRAM_API}/bot{self._token}/sendMessage",
                    json={
                        "chat_id": self._chat_id,
                        "text": text,
                    },
                    timeout=15.0,
                )
                resp.raise_for_status()
                return True
            except httpx.HTTPStatusError as err:
                # Permanent failure (4xx) — don't retry
                body = err.response.text[:300] if err.response is not None else ""
                logger.warning(
                    "telegram_send_failed",
                    error=repr(err),
                    status_code=err.response.status_code if err.response else None,
                    response_body=body,
                )
                return False
            except (httpx.RequestError, httpx.TransportError) as err:
                # Transient network error — retry with backoff
                await self._reset_client()
                if attempt < max_attempts - 1:
                    jitter = random.uniform(0, base_backoff)
                    sleep_for = base_backoff * (2 ** attempt) + jitter
                    logger.info("telegram_send_retry", attempt=attempt + 1, sleep_for=f"{sleep_for:.2f}s", error=repr(err))
                    await asyncio.sleep(sleep_for)
                    continue
                logger.warning("telegram_send_failed", error=repr(err))
                return False
            except Exception as err:
                logger.warning("telegram_send_failed", error=repr(err))
                return False
        return False

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Respect environment proxy settings (HTTP_PROXY / HTTPS_PROXY)
            proxies = None
            https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
            if https_proxy or http_proxy:
                proxies = {}
                if https_proxy:
                    proxies["https"] = https_proxy
                if http_proxy:
                    proxies["http"] = http_proxy

            # Use a slightly higher timeout and ensure we close idle connections more aggressively
            try:
                self._client = httpx.AsyncClient(
                    timeout=15.0,
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=2),
                    proxies=proxies,
                    trust_env=True,
                )
            except TypeError:
                # Older httpx versions do not accept 'proxies' in the constructor.
                # Fall back to creating the client without proxies; httpx will pick
                # up environment proxy variables automatically if present.
                self._client = httpx.AsyncClient(
                    timeout=15.0,
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=2),
                    trust_env=True,
                )
        return self._client

    async def _reset_client(self) -> None:
        if self._client is None:
            return
        client = self._client
        self._client = None
        with contextlib.suppress(Exception):
            await client.aclose()

    async def aclose(self) -> None:
        await self._reset_client()

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
            "*System Health Check*\n\n"
            + "\n".join(
                (
                    f"{'✅' if v in {'ok', 'configured'} else ('⚠️' if v == 'degraded' else '❌')} "
                    f"{k}: `{v}`"
                )
                for k, v in status.items()
            )
        )
        await self.send(msg, AlertLevel.SYSTEM)

    async def ollama_status(self, healthy: bool, models: list[str] | None = None) -> None:
        if healthy:
            msg = f"✅ Ollama LLM server online\nModels: {', '.join(models or [])}"
        else:
            msg = "⚠️ Ollama LLM server unreachable"
        await self.send(msg, AlertLevel.SYSTEM)
