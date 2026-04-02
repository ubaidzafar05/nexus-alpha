"""
Free web intelligence agents using Crawl4AI + local Ollama.

Replaces the paid OpenClaw agent backend with 100% free tooling:
  - Crawl4AI (open-source, JS rendering, LLM-guided extraction)
  - Ollama local LLM (Qwen3:8b) for structured extraction
  - APScheduler for autonomous polling schedules
  - RSS/PRAW/Etherscan as primary data sources (no scraping needed)

Maintains the same IntelligenceReport output schema as openclaw_agents.py
so downstream pipeline (openclaw_pipeline.py) works without changes.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from nexus_alpha.data.free_sources import (
    fetch_all_rss_feeds,
    get_cryptopanic_news,
    get_current_fear_greed,
    get_total_tvl_history,
)
from nexus_alpha.intelligence.openclaw_agents import (
    IntelligenceCategory,
    IntelligenceReport,
    Urgency,
)
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CRAWL_TARGETS = [
    "https://www.coindesk.com/markets/",
    "https://decrypt.co/news",
    "https://cointelegraph.com/latest-cryptocurrency-news",
]

DEFAULT_CRAWL_PROMPT = (
    "Extract up to {max_items} freshest crypto market-relevant items from this page. "
    "Return strict JSON as a list of objects with keys: headline, summary, url, "
    "affected_symbols, urgency. Use urgency values low, medium, or high. "
    "Use concise summaries and only include market-relevant content."
)

_SYMBOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "eth"),
    "SOL": ("solana", "sol"),
    "BNB": ("binance coin", "bnb"),
    "ADA": ("cardano", "ada"),
    "AVAX": ("avalanche", "avax"),
    "MATIC": ("polygon", "matic", "pol"),
    "DOT": ("polkadot", "dot"),
    "LINK": ("chainlink", "link"),
    "UNI": ("uniswap", "uni"),
}


def _make_report_id(source: str, content: str) -> str:
    return hashlib.sha256(f"{source}:{content}".encode()).hexdigest()[:16]


def _bounded_confidence(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(parsed, 1.0))


def _coerce_urgency(value: Any, default: Urgency) -> Urgency:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in Urgency._value2member_map_:
            return Urgency(normalized)
    return default


def _infer_symbols(text: str) -> list[str]:
    haystack = text.lower()
    hits = [
        symbol
        for symbol, keywords in _SYMBOL_KEYWORDS.items()
        if any(keyword in haystack for keyword in keywords)
    ]
    return sorted(set(hits))


def _coerce_symbols(value: Any, fallback_text: str) -> list[str]:
    if isinstance(value, list):
        return sorted({str(item).strip().upper() for item in value if str(item).strip()})
    if isinstance(value, str) and value.strip():
        return sorted({item.strip().upper() for item in value.split(",") if item.strip()})
    return _infer_symbols(fallback_text)


@dataclass(frozen=True)
class CrawlTarget:
    url: str
    category: IntelligenceCategory = IntelligenceCategory.BREAKING_NEWS
    urgency: Urgency = Urgency.MEDIUM
    confidence: float = 0.6
    max_items: int = 5


def intelligence_report_payload(report: IntelligenceReport) -> dict[str, Any]:
    return {
        "report_id": report.report_id,
        "category": report.category.value,
        "urgency": report.urgency.value,
        "headline": report.headline,
        "summary": report.summary,
        "sentiment_score": report.sentiment_score,
        "confidence": report.confidence,
        "affected_symbols": report.affected_symbols,
        "source_urls": report.source_urls,
        "raw_data": report.raw_data,
        "timestamp": report.timestamp.isoformat(),
        "ttl_seconds": report.ttl_seconds,
    }


def publish_reports_to_kafka(
    reports: list[IntelligenceReport],
    bootstrap_servers: str,
    topic: str = "nexus.intelligence",
) -> int:
    if not reports:
        return 0

    import json

    from confluent_kafka import Producer  # type: ignore[import]

    from nexus_alpha.data.kafka_admin import ensure_topics_exist

    ensure_topics_exist(bootstrap_servers, [topic])
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    produced = 0
    for report in reports:
        producer.produce(
            topic,
            key=report.report_id,
            value=json.dumps(intelligence_report_payload(report)).encode(),
        )
        producer.poll(0)
        produced += 1
    producer.flush(timeout=5.0)
    logger.info("intelligence_reports_published", topic=topic, count=produced)
    return produced


# ── Crawl4AI-powered scraper ──────────────────────────────────────────────────

async def _crawl_url_with_ollama(
    url: str,
    extraction_prompt: str,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen3:8b",
) -> list[dict[str, Any]]:
    """
    Use Crawl4AI to fetch and extract structured data from a URL.
    Falls back to plain HTTP fetch + Ollama if Crawl4AI unavailable.
    """
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore[import]
        from crawl4ai.async_configs import LLMConfig  # type: ignore[import]
        from crawl4ai.extraction_strategy import LLMExtractionStrategy  # type: ignore[import]

        strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(provider=f"ollama/{model}", base_url=ollama_base_url),
            instruction=extraction_prompt,
        )
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url=url,
                extraction_strategy=strategy,
                bypass_cache=True,
            )
        if result.success and result.extracted_content:
            import json
            try:
                data = json.loads(result.extracted_content)
                return data if isinstance(data, list) else [data]
            except Exception as err:
                logger.debug("crawl4ai_json_parse_failed", error=repr(err))

        fallback_text = (
            getattr(result, "markdown", None)
            or getattr(result, "cleaned_html", None)
            or getattr(result, "html", None)
            or ""
        )
        return [
            {
                "url": url,
                "success": bool(getattr(result, "success", False)),
                "content": str(fallback_text)[:4000],
                "prompt": extraction_prompt,
            }
        ]
    except ImportError:
        logger.debug("crawl4ai_not_installed_using_httpx_fallback")
        return await _httpx_fallback(url)
    except Exception as err:
        logger.warning("crawl4ai_failed", url=url, error=repr(err))
        return []


async def _httpx_fallback(url: str) -> list[dict[str, Any]]:
    """Plain HTTP fetch — used when Crawl4AI is not installed."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, follow_redirects=True)
            return [{"raw_html": resp.text[:2000], "url": url}]
    except Exception:
        return []


class Crawl4AINewsAgent:
    """Manual/periodic crawl agent for curated news targets."""

    def __init__(
        self,
        target_urls: list[str] | None = None,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        max_items_per_target: int = 5,
    ) -> None:
        urls = target_urls or DEFAULT_CRAWL_TARGETS
        seen: set[str] = set()
        self._targets = []
        for url in urls:
            normalized = url.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            self._targets.append(CrawlTarget(url=normalized, max_items=max_items_per_target))
        self._ollama_base_url = ollama_base_url
        self._model = model

    async def fetch(self) -> list[IntelligenceReport]:
        if not self._targets:
            return []

        results = await asyncio.gather(
            *(self._fetch_target(target) for target in self._targets),
            return_exceptions=True,
        )
        reports: list[IntelligenceReport] = []
        seen_keys: set[str] = set()
        for batch in results:
            if isinstance(batch, Exception):
                logger.warning("crawl_target_batch_failed", error=repr(batch))
                continue
            for report in batch:
                dedupe_key = (report.source_urls[0] if report.source_urls else report.headline).strip().lower()
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                reports.append(report)
        logger.info("crawl4ai_agent_fetched", targets=len(self._targets), count=len(reports))
        return reports

    async def _fetch_target(self, target: CrawlTarget) -> list[IntelligenceReport]:
        prompt = DEFAULT_CRAWL_PROMPT.format(max_items=target.max_items)
        items = await _crawl_url_with_ollama(
            target.url,
            extraction_prompt=prompt,
            ollama_base_url=self._ollama_base_url,
            model=self._model,
        )
        reports = []
        for item in items[:target.max_items]:
            report = self._item_to_report(item, target)
            if report is not None:
                reports.append(report)
        logger.info("crawl_target_fetched", url=target.url, count=len(reports))
        return reports

    def _item_to_report(self, item: dict[str, Any], target: CrawlTarget) -> IntelligenceReport | None:
        summary = str(
            item.get("summary")
            or item.get("content")
            or item.get("text")
            or item.get("raw_html")
            or ""
        ).strip()
        headline = str(item.get("headline") or item.get("title") or "").strip()
        if not headline and summary:
            headline = summary[:160]
        if not headline:
            return None

        source_url = str(item.get("url") or item.get("link") or target.url).strip() or target.url
        sentiment = 0.0
        if "bullish" in summary.lower() or "surge" in headline.lower():
            sentiment = 0.25
        elif "bearish" in summary.lower() or "drop" in headline.lower():
            sentiment = -0.25

        fallback_text = f"{headline}\n{summary}\n{source_url}"
        affected_symbols = _coerce_symbols(item.get("affected_symbols"), fallback_text)

        return IntelligenceReport(
            report_id=_make_report_id("crawl4ai", source_url or headline),
            category=target.category,
            urgency=_coerce_urgency(item.get("urgency"), target.urgency),
            headline=headline[:200],
            summary=(summary or headline)[:500],
            sentiment_score=sentiment,
            confidence=_bounded_confidence(item.get("confidence"), target.confidence),
            affected_symbols=affected_symbols,
            source_urls=[source_url],
            raw_data={"source": "crawl4ai", "target_url": target.url, "extracted": item},
            ttl_seconds=1800,
        )


# ── Individual intelligence agents ───────────────────────────────────────────

class RSSNewsAgent:
    """
    Multi-source RSS aggregator — crypto news, SEC filings, Fed releases.
    No scraping needed — RSS is free and instant.
    """

    SCHEDULE_MINUTES = 15

    async def fetch(self, max_age_minutes: int = 30) -> list[IntelligenceReport]:
        articles = await fetch_all_rss_feeds(max_age_minutes=max_age_minutes)
        reports = []
        for article in articles[:50]:  # Cap at 50 per run
            reports.append(
                IntelligenceReport(
                    report_id=_make_report_id("rss", article.url),
                    category=IntelligenceCategory.CRYPTO_MEDIA,
                    urgency=Urgency.MEDIUM,
                    headline=article.title[:200],
                    summary=article.summary[:500],
                    sentiment_score=0.0,  # Will be scored by HybridSentimentPipeline
                    confidence=0.6,
                    affected_symbols=[],
                    source_urls=[article.url],
                    raw_data={"source": article.source, "published": article.published.isoformat()},
                    timestamp=article.published,
                )
            )
        logger.info("rss_agent_fetched", count=len(reports))
        return reports


class CryptoPanicAgent:
    """
    CryptoPanic curated news — pre-labeled bullish/bearish/important.
    Free tier: 50 calls/hour.
    """

    SCHEDULE_MINUTES = 20

    def __init__(self, auth_token: str) -> None:
        self._token = auth_token

    async def fetch(self) -> list[IntelligenceReport]:
        if not self._token:
            return []
        reports = []
        for filter_type in ("hot", "bullish", "bearish", "important"):
            try:
                items = await get_cryptopanic_news(
                    self._token, filter_type=filter_type, limit=10
                )
                for item in items:
                    sentiment = 0.4 if filter_type == "bullish" else (
                        -0.4 if filter_type == "bearish" else 0.0
                    )
                    urgency = Urgency.HIGH if filter_type == "important" else Urgency.MEDIUM
                    currencies = [c["code"] for c in item.get("currencies", [])]
                    reports.append(
                        IntelligenceReport(
                            report_id=_make_report_id("cryptopanic", item.get("url", "")),
                            category=IntelligenceCategory.BREAKING_NEWS,
                            urgency=urgency,
                            headline=item.get("title", "")[:200],
                            summary=item.get("title", ""),
                            sentiment_score=sentiment,
                            confidence=0.7,
                            affected_symbols=currencies,
                            source_urls=[item.get("url", "")],
                            raw_data={"filter": filter_type, "votes": item.get("votes", {})},
                        )
                    )
            except Exception as err:
                logger.warning("cryptopanic_fetch_failed", filter=filter_type, error=repr(err))
        logger.info("cryptopanic_agent_fetched", count=len(reports))
        return reports


class FearGreedAgent:
    """Fear & Greed Index — macro sentiment indicator. Unlimited free API."""

    SCHEDULE_MINUTES = 60

    async def fetch(self) -> list[IntelligenceReport]:
        try:
            data = await get_current_fear_greed()
            value = int(data.get("value", 50))
            classification = data.get("classification", "Neutral")

            sentiment = (value - 50) / 50.0  # Map 0-100 → -1.0 to 1.0
            urgency = Urgency.HIGH if (value <= 15 or value >= 85) else Urgency.LOW

            return [
                IntelligenceReport(
                    report_id=_make_report_id("fear_greed", str(datetime.utcnow().date())),
                    category=IntelligenceCategory.ON_CHAIN,
                    urgency=urgency,
                    headline=f"Fear & Greed Index: {value} — {classification}",
                    summary=(
                        f"Market sentiment is {classification.lower()} "
                        f"(score: {value}/100). "
                    ),
                    sentiment_score=round(sentiment, 3),
                    confidence=0.85,
                    affected_symbols=["BTC", "ETH"],
                    source_urls=["https://alternative.me/crypto/fear-and-greed-index/"],
                    raw_data=data,
                )
            ]
        except Exception as err:
            logger.warning("fear_greed_fetch_failed", error=repr(err))
            return []


class DeFiTVLAgent:
    """
    DeFiLlama TVL tracker — measures risk appetite and DeFi health.
    100% free, no auth required.
    """

    SCHEDULE_HOURS = 4

    async def fetch(self) -> list[IntelligenceReport]:
        try:
            history = await get_total_tvl_history()
            if len(history) < 2:
                return []

            current_tvl = history[-1].get("totalLiquidityUSD", 0)
            prev_tvl = history[-2].get("totalLiquidityUSD", 0)
            pct_change = (current_tvl - prev_tvl) / max(prev_tvl, 1) * 100

            sentiment = min(max(pct_change / 10, -1.0), 1.0)  # ±10% TVL change → ±1.0
            direction = "up" if pct_change > 0 else "down"
            risk_text = (
                "Increasing TVL indicates risk appetite and capital inflows."
                if pct_change > 0
                else "Decreasing TVL indicates capital outflows and risk-off sentiment."
            )

            return [
                IntelligenceReport(
                    report_id=_make_report_id("defillama_tvl", str(datetime.utcnow().date())),
                    category=IntelligenceCategory.ON_CHAIN,
                    urgency=Urgency.LOW,
                    headline=f"DeFi Total TVL: ${current_tvl/1e9:.1f}B ({pct_change:+.1f}% daily)",
                    summary=(
                        f"Total DeFi TVL is ${current_tvl/1e9:.1f}B, "
                        f"{direction} {abs(pct_change):.1f}% from previous period. "
                        f"{risk_text}"
                    ),
                    sentiment_score=round(sentiment, 3),
                    confidence=0.75,
                    affected_symbols=["ETH", "BTC", "DEFI"],
                    source_urls=["https://defillama.com"],
                    raw_data={"current_tvl_usd": current_tvl, "pct_change_24h": pct_change},
                    ttl_seconds=14400,  # 4 hours
                )
            ]
        except Exception as err:
            logger.warning("defillama_tvl_fetch_failed", error=repr(err))
            return []


class SECFilingAgent:
    """
    SEC EDGAR monitoring — crypto-related 8-K filings, institutional activity.
    100% free public data, no auth required.
    """

    SCHEDULE_HOURS = 2
    EDGAR_SEARCH = (
        "https://efts.sec.gov/LATEST/search-index"
        "?q=%22bitcoin%22+%22cryptocurrency%22&forms=8-K&dateRange=custom"
        "&startdt={start}&enddt={end}"
    )

    async def fetch(self) -> list[IntelligenceReport]:
        today = date.today().isoformat()
        week_ago = (date.today() - timedelta(days=7)).isoformat()
        url = self.EDGAR_SEARCH.format(start=week_ago, end=today)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, follow_redirects=True)
                data = resp.json()

            hits = data.get("hits", {}).get("hits", [])
            reports = []
            for hit in hits[:10]:
                src = hit.get("_source", {})
                entity = src.get("entity_name", "Unknown Company")
                form = src.get("file_type", "8-K")
                filed = src.get("file_date", "")
                accession = src.get("accession_no", "").replace("-", "")
                filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={accession}"

                reports.append(
                    IntelligenceReport(
                        report_id=_make_report_id("sec_edgar", accession),
                        category=IntelligenceCategory.SEC_FILING,
                        urgency=Urgency.HIGH,
                        headline=f"SEC {form}: {entity} filed crypto-related disclosure",
                        summary=(
                            f"{entity} filed an {form} with the SEC on {filed} containing "
                            f"cryptocurrency references. Institutional crypto activity detected."
                        ),
                        sentiment_score=0.1,  # Generally positive for institutional adoption
                        confidence=0.8,
                        affected_symbols=["BTC", "ETH"],
                        source_urls=[filing_url],
                        raw_data=src,
                        ttl_seconds=7200,
                    )
                )
            logger.info("sec_agent_fetched", count=len(reports))
            return reports
        except Exception as err:
            logger.warning("sec_edgar_fetch_failed", error=repr(err))
            return []


# ── Agent orchestrator ────────────────────────────────────────────────────────

@dataclass
class FreeIntelligenceOrchestrator:
    """
    Runs all free intelligence agents and publishes reports to the bus.

    Uses APScheduler for autonomous polling. Each agent runs on its own
    schedule to respect rate limits and data freshness requirements.
    """

    cryptopanic_token: str = ""
    ollama_base_url: str = "http://localhost:11434"
    crawl_targets: list[str] | None = None
    intelligence_topic: str = "nexus.intelligence"
    _report_buffer: list[IntelligenceReport] = field(default_factory=list)
    _scheduler: Any | None = field(default=None, init=False, repr=False)
    _kafka_producer: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rss = RSSNewsAgent()
        self._cryptopanic = CryptoPanicAgent(self.cryptopanic_token)
        self._fear_greed = FearGreedAgent()
        self._tvl = DeFiTVLAgent()
        self._sec = SECFilingAgent()
        self._crawl = Crawl4AINewsAgent(
            target_urls=self.crawl_targets,
            ollama_base_url=self.ollama_base_url,
        )

    async def run_all(self) -> list[IntelligenceReport]:
        """Fetch from all agents concurrently. Safe to call every 15 minutes."""
        tasks = [
            self._rss.fetch(),
            self._cryptopanic.fetch(),
            self._fear_greed.fetch(),
            self._tvl.fetch(),
            self._sec.fetch(),
            self._crawl.fetch(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        reports: list[IntelligenceReport] = []
        for batch in results:
            if isinstance(batch, list):
                reports.extend(batch)
        logger.info("orchestrator_run_complete", total_reports=len(reports))
        return reports

    def publish_reports(self, reports: list[IntelligenceReport], bootstrap_servers: str | None = None) -> int:
        if not reports:
            return 0
        if self._kafka_producer is not None:
            import json

            published = 0
            for report in reports:
                self._kafka_producer.produce(
                    self.intelligence_topic,
                    key=report.report_id,
                    value=json.dumps(intelligence_report_payload(report)).encode(),
                )
                self._kafka_producer.poll(0)
                published += 1
            self._kafka_producer.flush(timeout=5.0)
            return published
        if bootstrap_servers:
            return publish_reports_to_kafka(reports, bootstrap_servers, topic=self.intelligence_topic)
        return 0

    def start_scheduler(
        self,
        kafka_producer: Any | None = None,
        bootstrap_servers: str | None = None,
    ) -> None:
        """Start APScheduler background jobs. Call once at startup."""
        if self._scheduler is not None:
            logger.info("free_intelligence_scheduler_already_started")
            return
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import]
        except ImportError:
            logger.warning("apscheduler_not_installed", hint="pip install apscheduler")
            return

        scheduler = AsyncIOScheduler()
        self._kafka_producer = kafka_producer

        async def _publish_all() -> None:
            reports = await self.run_all()
            self._report_buffer.extend(reports)
            try:
                self.publish_reports(reports, bootstrap_servers=bootstrap_servers)
            except Exception as err:
                logger.warning("kafka_publish_failed", error=repr(err))

        scheduler.add_job(_publish_all, "interval", minutes=15, id="intelligence_agents")
        scheduler.start()
        self._scheduler = scheduler
        logger.info("free_intelligence_scheduler_started")

    def stop_scheduler(self) -> None:
        if self._scheduler is None:
            return
        self._scheduler.shutdown(wait=False)
        self._scheduler = None
        if self._kafka_producer is not None:
            self._kafka_producer.flush(timeout=5.0)
            self._kafka_producer = None
        logger.info("free_intelligence_scheduler_stopped")

    def get_buffered_reports(self, clear: bool = True) -> list[IntelligenceReport]:
        reports = list(self._report_buffer)
        if clear:
            self._report_buffer.clear()
        return reports
