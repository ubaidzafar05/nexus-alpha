"""
OpenClaw Intelligence Network — 10 Specialized Web Agents.

Each agent has a defined domain, data sources, and output schema.
Agents run autonomously on configurable schedules and publish results
to the intelligence bus (Kafka topic: nexus.intelligence).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from nexus_alpha.config import LLMConfig, NexusConfig
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


# ─── Intelligence Types ──────────────────────────────────────────────────────

class IntelligenceCategory(str, Enum):
    BREAKING_NEWS = "breaking_news"
    SOCIAL_PULSE = "social_pulse"
    CRYPTO_MEDIA = "crypto_media"
    OPTIONS_FLOW = "options_flow"
    WHALE_ALERT = "whale_alert"
    SEC_FILING = "sec_filing"
    ON_CHAIN = "on_chain"
    MACRO_CALENDAR = "macro_calendar"
    EXCHANGE_STATUS = "exchange_status"
    COMPETITOR_INTEL = "competitor_intel"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntelligenceReport:
    """Structured output from an OpenClaw agent."""
    report_id: str
    category: IntelligenceCategory
    urgency: Urgency
    headline: str
    summary: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float       # 0.0 to 1.0
    affected_symbols: list[str]
    source_urls: list[str]
    raw_data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 300  # How long this intel is actionable

    @property
    def is_expired(self) -> bool:
        elapsed = (datetime.utcnow() - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds


# ─── Base Agent ───────────────────────────────────────────────────────────────

class BaseIntelligenceAgent(ABC):
    """
    Abstract base for all OpenClaw intelligence agents.

    Each agent:
    1. Polls its data sources on a schedule
    2. Extracts + normalizes relevant signals
    3. Uses LLM for analysis/summarization when needed
    4. Publishes IntelligenceReport to the intelligence bus
    """

    def __init__(
        self,
        category: IntelligenceCategory,
        poll_interval_seconds: int = 60,
        llm_config: LLMConfig | None = None,
    ):
        self.category = category
        self.poll_interval = poll_interval_seconds
        self.llm_config = llm_config or LLMConfig()
        self._recent_reports: deque[IntelligenceReport] = deque(maxlen=500)
        self._seen_hashes: set[str] = set()
        self._running = False
        self._http_client: httpx.AsyncClient | None = None

    @abstractmethod
    async def gather(self) -> list[IntelligenceReport]:
        """Gather intelligence from data sources. Must be implemented by subclasses."""
        ...

    async def start(self) -> None:
        """Start the agent's polling loop."""
        self._running = True
        self._http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("openclaw_agent_started", category=self.category.value)

        while self._running:
            try:
                reports = await self.gather()
                for report in reports:
                    content_hash = hashlib.sha256(
                        f"{report.headline}{report.category}".encode()
                    ).hexdigest()[:16]
                    if content_hash not in self._seen_hashes:
                        self._seen_hashes.add(content_hash)
                        self._recent_reports.append(report)
                        await self._publish(report)
            except Exception:
                logger.exception("openclaw_agent_error", category=self.category.value)
            await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
        logger.info("openclaw_agent_stopped", category=self.category.value)

    async def _publish(self, report: IntelligenceReport) -> None:
        """Publish to intelligence bus (Kafka in production)."""
        logger.info(
            "intelligence_report",
            category=report.category.value,
            urgency=report.urgency.value,
            headline=report.headline[:80],
            symbols=report.affected_symbols,
        )

    def get_recent(self, limit: int = 10) -> list[IntelligenceReport]:
        return list(self._recent_reports)[-limit:]

    async def _llm_analyze(self, system_prompt: str, user_content: str) -> str:
        """Call LLM for analysis/summarization."""
        if not self._http_client or not self.llm_config.anthropic_api_key:
            return ""
        try:
            response = await self._http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.llm_config.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.llm_config.model_name,
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_content}],
                },
            )
            response.raise_for_status()
            data = response.json()
            content_blocks = data.get("content", [])
            return content_blocks[0].get("text", "") if content_blocks else ""
        except Exception:
            logger.exception("llm_analysis_failed")
            return ""


# ─── Agent Implementations ───────────────────────────────────────────────────

class BreakingNewsAgent(BaseIntelligenceAgent):
    """Agent 1: Breaking news monitoring via RSS/API aggregation."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.BREAKING_NEWS, poll_interval_seconds=30, llm_config=llm_config)
        self.feeds: list[str] = []  # RSS/API endpoints injected at runtime

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: fetch RSS feeds, crypto news APIs
        # Analyze with LLM for sentiment + urgency
        return reports


class SocialPulseAgent(BaseIntelligenceAgent):
    """Agent 2: Social media sentiment (Twitter/X, Reddit, Telegram)."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.SOCIAL_PULSE, poll_interval_seconds=60, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: aggregate trending topics, sentiment shifts, KOL tracking
        return reports


class CryptoMediaAgent(BaseIntelligenceAgent):
    """Agent 3: Crypto media analysis (CoinDesk, The Block, DL News)."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.CRYPTO_MEDIA, poll_interval_seconds=120, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        return reports


class OptionsFlowAgent(BaseIntelligenceAgent):
    """Agent 4: Options flow monitoring (Deribit, large block trades)."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.OPTIONS_FLOW, poll_interval_seconds=30, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: monitor Deribit API for large options trades,
        # detect unusual put/call ratios, large notional block trades
        return reports


class WhaleAlertAgent(BaseIntelligenceAgent):
    """Agent 5: On-chain whale movement tracking."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.WHALE_ALERT, poll_interval_seconds=30, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: Whale Alert API, Etherscan, large transfers
        return reports


class SECFilingAgent(BaseIntelligenceAgent):
    """Agent 6: SEC EDGAR & regulatory filing monitoring."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.SEC_FILING, poll_interval_seconds=300, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: EDGAR RSS, parse 8-K, 10-K, S-1 filings for crypto exposure
        return reports


class OnChainAgent(BaseIntelligenceAgent):
    """Agent 7: On-chain analytics (exchange flows, DeFi TVL, staking)."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.ON_CHAIN, poll_interval_seconds=60, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: Glassnode, DefiLlama, Dune Analytics APIs
        return reports


class MacroCalendarAgent(BaseIntelligenceAgent):
    """Agent 8: Macroeconomic calendar events (FOMC, CPI, NFP)."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.MACRO_CALENDAR, poll_interval_seconds=600, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: economic calendar APIs, Fed speech tracking
        return reports


class ExchangeStatusAgent(BaseIntelligenceAgent):
    """Agent 9: Exchange health & status monitoring."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.EXCHANGE_STATUS, poll_interval_seconds=30, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: exchange status pages, API latency monitoring,
        # withdrawal/deposit status, maintenance schedules
        return reports


class CompetitorIntelAgent(BaseIntelligenceAgent):
    """Agent 10: Competitor strategy detection (MEV, copytrading patterns)."""

    def __init__(self, llm_config: LLMConfig | None = None):
        super().__init__(IntelligenceCategory.COMPETITOR_INTEL, poll_interval_seconds=120, llm_config=llm_config)

    async def gather(self) -> list[IntelligenceReport]:
        reports: list[IntelligenceReport] = []
        # In production: on-chain bot detection, order pattern analysis
        return reports


# ─── Intelligence Network Orchestrator ────────────────────────────────────────

class OpenClawNetwork:
    """
    Orchestrates all 10 intelligence agents.
    Manages lifecycle, aggregates reports, and provides fused intelligence.
    """

    AGENT_CLASSES = [
        BreakingNewsAgent,
        SocialPulseAgent,
        CryptoMediaAgent,
        OptionsFlowAgent,
        WhaleAlertAgent,
        SECFilingAgent,
        OnChainAgent,
        MacroCalendarAgent,
        ExchangeStatusAgent,
        CompetitorIntelAgent,
    ]

    def __init__(self, config: NexusConfig | None = None):
        llm_config = config.llm if config else None
        self.agents: list[BaseIntelligenceAgent] = [cls(llm_config=llm_config) for cls in self.AGENT_CLASSES]
        self._tasks: list[asyncio.Task[None]] = []
        logger.info("openclaw_network_initialized", agent_count=len(self.agents))

    async def start_all(self) -> None:
        """Start all intelligence agents concurrently."""
        for agent in self.agents:
            task = asyncio.create_task(agent.start())
            self._tasks.append(task)
        logger.info("openclaw_network_started", agents=len(self._tasks))

    async def stop_all(self) -> None:
        """Gracefully stop all agents."""
        for agent in self.agents:
            await agent.stop()
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        logger.info("openclaw_network_stopped")

    def get_all_recent(self, limit: int = 50) -> list[IntelligenceReport]:
        """Get recent reports from all agents, sorted by timestamp descending."""
        all_reports: list[IntelligenceReport] = []
        for agent in self.agents:
            all_reports.extend(agent.get_recent(limit=limit))
        all_reports.sort(key=lambda r: r.timestamp, reverse=True)
        return all_reports[:limit]

    def get_critical_alerts(self) -> list[IntelligenceReport]:
        """Get all non-expired CRITICAL urgency reports."""
        return [
            r for r in self.get_all_recent(limit=200)
            if r.urgency == Urgency.CRITICAL and not r.is_expired
        ]

    def get_by_category(self, category: IntelligenceCategory, limit: int = 10) -> list[IntelligenceReport]:
        for agent in self.agents:
            if agent.category == category:
                return agent.get_recent(limit=limit)
        return []

    def compute_aggregate_sentiment(self, symbol: str) -> float:
        """Compute weighted sentiment across all agents for a given symbol."""
        reports = [
            r for r in self.get_all_recent(limit=200)
            if symbol in r.affected_symbols and not r.is_expired
        ]
        if not reports:
            return 0.0

        urgency_weights = {
            Urgency.LOW: 0.25,
            Urgency.MEDIUM: 0.5,
            Urgency.HIGH: 1.0,
            Urgency.CRITICAL: 2.0,
        }

        weighted_sum = sum(
            r.sentiment_score * r.confidence * urgency_weights.get(r.urgency, 0.5)
            for r in reports
        )
        weight_total = sum(
            r.confidence * urgency_weights.get(r.urgency, 0.5) for r in reports
        )

        return weighted_sum / weight_total if weight_total > 0 else 0.0
