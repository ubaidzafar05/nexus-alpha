"""
Sentiment Pipeline Runner — writes live sentiment scores to Redis.

This is the critical glue between:
  HybridSentimentPipeline (FinBERT + Qwen3) + FreeIntelligenceOrchestrator
  → Redis (`sentiment:{ASSET}` keys)
  → FreqtradeStrategy._get_sentiment() reads from Redis
  → NexusAlpha SignalEngine reads from Redis

Without this, sentiment is always 0.0 in Freqtrade. This service closes the loop.

Architecture:
  [RSS/CryptoPanic/Reddit] → [FinBERT fast-path] → [Redis]
                          ↘ [Qwen3 deep (high-stakes)] ↗

Runs on a 15-minute cycle (RSS refresh interval).
Kafka publishes processed articles to alt-data.sentiment topic.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nexus_alpha.config import NexusConfig
from nexus_alpha.data.free_sources import (
    CRYPTO_SUBREDDITS,
    KNOWN_EXCHANGE_WALLETS,
    fetch_all_rss_feeds,
    get_cryptopanic_news,
    get_current_fear_greed,
    get_exchange_flows,
    get_gas_price,
    get_total_tvl_history,
)
from nexus_alpha.data.reddit_client import fetch_new_posts
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

# Assets we track sentiment for
TRACKED_ASSETS = ["BTC", "ETH", "SOL", "BNB", "ADA", "AVAX", "MATIC", "DOT", "LINK", "UNI"]

# Keyword → asset mapping for RSS article routing
ASSET_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "satoshi", "crypto"],
    "ETH": ["ethereum", "eth", "vitalik", "solidity", "erc-20", "defi"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance", "bnb"],
    "ADA": ["cardano", "ada"],
    "AVAX": ["avalanche", "avax"],
}

# Redis TTL for sentiment scores (20 minutes — RSS refresh + processing time)
SENTIMENT_TTL_SECONDS = 1200


def _extract_assets(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for asset, keywords in ASSET_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(asset)
    return found or ["BTC"]  # Default to BTC if no asset found


@dataclass
class SentimentScore:
    asset: str
    score: float  # -1.0 to 1.0
    confidence: float
    source_count: int
    method: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MacroFactors:
    global_score: float
    confidence: float
    source_count: int
    details: dict[str, float | int | str] = field(default_factory=dict)


class SentimentPipelineRunner:
    """
    Continuously fetches news, scores sentiment, and writes to Redis.

    Score aggregation: exponentially weighted moving average across sources.
    Recent high-confidence scores weighted 3x vs routine FinBERT scores.
    """

    def __init__(
        self,
        config: NexusConfig,
        run_interval_minutes: int = 15,
    ) -> None:
        self._config = config
        self._interval = run_interval_minutes * 60
        self._redis: Any = None
        self._kafka_producer: Any = None
        self._running = False

        # Lazy-imported to allow startup without GPU
        self._sentiment_pipeline: Any = None

    async def _collect_rss_articles(self) -> list[dict[str, Any]]:
        articles: list[dict[str, Any]] = []
        try:
            rss_articles = await fetch_all_rss_feeds(max_age_minutes=30)
            for article in rss_articles:
                articles.append(
                    {
                        "title": article.title,
                        "text": article.summary,
                        "source": article.source,
                        "url": article.url,
                    }
                )
        except Exception as err:
            logger.warning("rss_fetch_error", error=str(err))
        return articles

    async def _collect_cryptopanic_articles(self) -> list[dict[str, Any]]:
        cryptopanic_token = os.getenv("CRYPTOPANIC_API_TOKEN", "")
        if not cryptopanic_token:
            return []

        try:
            cp_items = await get_cryptopanic_news(cryptopanic_token)
            return [
                {
                    "title": item.get("title", ""),
                    "text": item.get("title", ""),
                    "source": "cryptopanic",
                    "url": item.get("url", ""),
                    "votes": item.get("votes", {}),
                }
                for item in cp_items
            ]
        except Exception as err:
            logger.warning("cryptopanic_fetch_error", error=str(err))
            return []

    async def _collect_reddit_articles(self) -> list[dict[str, Any]]:
        subreddits_raw = os.getenv("REDDIT_SUBREDDITS", "")
        subreddits = [
            item.strip()
            for item in subreddits_raw.split(",")
            if item.strip()
        ] or CRYPTO_SUBREDDITS
        post_limit = int(os.getenv("REDDIT_POST_LIMIT", "15"))

        tasks = [fetch_new_posts(subreddit, limit=post_limit) for subreddit in subreddits]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[dict[str, Any]] = []
        for subreddit, batch in zip(subreddits, results, strict=False):
            if isinstance(batch, Exception):
                logger.warning("reddit_fetch_error", subreddit=subreddit, error=str(batch))
                continue

            for post in batch:
                title = str(post.get("title", "")).strip()
                body = str(post.get("selftext", "")).strip()
                if not title:
                    continue
                articles.append(
                    {
                        "title": title,
                        "text": body[:1000],
                        "source": f"reddit:{subreddit}",
                        "url": post.get("url", ""),
                        "score": int(post.get("score", 0) or 0),
                        "num_comments": int(post.get("num_comments", 0) or 0),
                        "author": post.get("author"),
                    }
                )

        return articles

    async def _collect_raw_articles(self) -> list[dict[str, Any]]:
        rss_articles, cryptopanic_articles, reddit_articles = await asyncio.gather(
            self._collect_rss_articles(),
            self._collect_cryptopanic_articles(),
            self._collect_reddit_articles(),
        )
        raw_articles = rss_articles + cryptopanic_articles + reddit_articles
        logger.info(
            "sentiment_sources_collected",
            rss_articles=len(rss_articles),
            cryptopanic_articles=len(cryptopanic_articles),
            reddit_articles=len(reddit_articles),
            total_articles=len(raw_articles),
        )
        return raw_articles

    def _article_weight(self, article: dict[str, Any], confidence: float) -> float:
        weight = max(confidence, 0.1)
        source = str(article.get("source", ""))
        if source.startswith("reddit:"):
            score = max(int(article.get("score", 0) or 0), 0)
            num_comments = max(int(article.get("num_comments", 0) or 0), 0)
            engagement_boost = min(2.0, 1.0 + (score / 500.0) + (num_comments / 200.0))
            return weight * engagement_boost
        return weight

    def _score_fear_greed(self, payload: dict[str, Any]) -> float:
        value = int(payload.get("value", 50))
        return max(-1.0, min(1.0, (value - 50) / 50.0))

    def _score_gas_conditions(self, payload: dict[str, Any]) -> float:
        propose = float(payload.get("propose_gwei") or 0.0)
        if propose <= 0:
            return 0.0
        if propose < 20:
            return 0.15
        if propose < 40:
            return 0.08
        if propose < 80:
            return -0.05
        return -0.18

    def _score_tvl_history(self, history: list[dict[str, Any]]) -> float:
        if len(history) < 2:
            return 0.0

        def _extract_tvl(entry: dict[str, Any]) -> float:
            for key in ("totalLiquidityUSD", "tvl", "liquidity"):
                if key in entry:
                    return float(entry[key] or 0.0)
            return 0.0

        latest = _extract_tvl(history[-1])
        previous = _extract_tvl(history[-2])
        if previous <= 0:
            return 0.0
        pct_change = (latest - previous) / previous
        return max(-0.25, min(0.25, pct_change * 5.0))

    def _score_exchange_flow_pressure(
        self,
        wallet_flows: dict[str, list[dict[str, Any]]],
    ) -> float:
        total_signed_eth = 0.0
        total_abs_eth = 0.0

        for wallet_address, flows in wallet_flows.items():
            wallet_lower = wallet_address.lower()
            for tx in flows:
                value_eth = float(tx.get("value_eth", 0.0) or 0.0)
                if value_eth <= 0:
                    continue
                inbound = str(tx.get("to", "")).lower() == wallet_lower
                outbound = str(tx.get("from", "")).lower() == wallet_lower
                if inbound:
                    total_signed_eth -= value_eth
                elif outbound:
                    total_signed_eth += value_eth
                total_abs_eth += value_eth

        if total_abs_eth <= 0:
            return 0.0
        normalized = total_signed_eth / total_abs_eth
        return max(-0.2, min(0.2, normalized * 0.2))

    async def _collect_macro_factors(self) -> MacroFactors:
        etherscan_key = os.getenv("ETHERSCAN_API_KEY", "")
        results: dict[str, Any] = {}

        tasks: dict[str, Any] = {
            "fear_greed": get_current_fear_greed(),
            "tvl_history": get_total_tvl_history(),
        }
        if etherscan_key:
            tasks["gas"] = get_gas_price(etherscan_key)
            for wallet_name, wallet_address in KNOWN_EXCHANGE_WALLETS.items():
                tasks[f"wallet:{wallet_name}"] = get_exchange_flows(
                    wallet_address=wallet_address,
                    api_key=etherscan_key,
                    min_eth=100.0,
                )

        raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for key, value in zip(tasks.keys(), raw_results, strict=False):
            if isinstance(value, Exception):
                logger.warning("macro_source_failed", source=key, error=str(value))
                continue
            results[key] = value

        component_scores: list[tuple[float, float]] = []
        details: dict[str, float | int | str] = {}

        fear_greed = results.get("fear_greed")
        if isinstance(fear_greed, dict):
            score = self._score_fear_greed(fear_greed)
            component_scores.append((score, 0.35))
            details["fear_greed"] = round(score, 4)

        tvl_history = results.get("tvl_history")
        if isinstance(tvl_history, list):
            score = self._score_tvl_history(tvl_history)
            component_scores.append((score, 0.30))
            details["tvl_trend"] = round(score, 4)

        gas = results.get("gas")
        if isinstance(gas, dict):
            score = self._score_gas_conditions(gas)
            component_scores.append((score, 0.15))
            details["gas_pressure"] = round(score, 4)

        wallet_flows = {
            KNOWN_EXCHANGE_WALLETS[key.split(":", 1)[1]]: value
            for key, value in results.items()
            if key.startswith("wallet:") and isinstance(value, list)
        }
        if wallet_flows:
            score = self._score_exchange_flow_pressure(wallet_flows)
            component_scores.append((score, 0.20))
            details["exchange_flow_pressure"] = round(score, 4)

        if not component_scores:
            return MacroFactors(global_score=0.0, confidence=0.0, source_count=0)

        total_weight = sum(weight for _, weight in component_scores)
        blended = sum(score * weight for score, weight in component_scores) / total_weight
        confidence = min(0.9, 0.35 + (0.1 * len(component_scores)))
        return MacroFactors(
            global_score=round(blended, 4),
            confidence=round(confidence, 4),
            source_count=len(component_scores),
            details=details,
        )

    def _ensure_sentiment_pipeline(self) -> Any:
        if self._sentiment_pipeline is None:
            from nexus_alpha.intelligence.free_llm import FreeLLMClient
            from nexus_alpha.intelligence.sentiment import HybridSentimentPipeline

            llm = FreeLLMClient.from_config(self._config.llm)
            self._sentiment_pipeline = HybridSentimentPipeline(
                llm_client=llm,
                finbert_model=self._config.llm.finbert_model_name,
            )
        return self._sentiment_pipeline

    def _init_redis(self) -> bool:
        try:
            import redis  # type: ignore[import]
            self._redis = redis.from_url(
                self._config.database.redis_url, decode_responses=True
            )
            self._redis.ping()
            logger.info("sentiment_redis_connected")
            return True
        except Exception as err:
            logger.warning("sentiment_redis_unavailable", error=str(err))
            return False

    def _init_kafka(self) -> None:
        try:
            from confluent_kafka import Producer  # type: ignore[import]
            self._kafka_producer = Producer({
                "bootstrap.servers": self._config.kafka.bootstrap_servers,
                "acks": "1",
            })
            logger.info("sentiment_kafka_producer_ready")
        except Exception as err:
            logger.warning("sentiment_kafka_unavailable", error=str(err))

    def _write_to_redis(self, scores: dict[str, SentimentScore]) -> None:
        if not self._redis:
            return
        pipe = self._redis.pipeline()
        for asset, score in scores.items():
            key = f"sentiment:{asset}"
            pipe.setex(key, SENTIMENT_TTL_SECONDS, str(round(score.score, 4)))
            # Also write full score object for dashboards
            pipe.setex(
                f"sentiment_full:{asset}",
                SENTIMENT_TTL_SECONDS,
                json.dumps({
                    "score": score.score,
                    "confidence": score.confidence,
                    "source_count": score.source_count,
                    "method": score.method,
                    "timestamp": score.timestamp.isoformat(),
                }),
            )
        pipe.execute()
        logger.info("sentiment_written_to_redis", assets=list(scores.keys()))

    def _publish_to_kafka(self, scores: dict[str, SentimentScore]) -> None:
        if not self._kafka_producer:
            return
        for asset, score in scores.items():
            payload = json.dumps({
                "asset": asset,
                "score": score.score,
                "confidence": score.confidence,
                "timestamp": score.timestamp.isoformat(),
            }).encode()
            self._kafka_producer.produce(
                "alt-data.sentiment",
                key=asset.encode(),
                value=payload,
            )
        self._kafka_producer.poll(0)

    async def _run_once(self) -> dict[str, SentimentScore]:
        """Single pipeline run: fetch → score → aggregate → write."""
        pipeline = self._ensure_sentiment_pipeline()

        raw_articles, macro_factors = await asyncio.gather(
            self._collect_raw_articles(),
            self._collect_macro_factors(),
        )

        if not raw_articles and macro_factors.source_count == 0:
            logger.info("no_articles_fetched_this_cycle")
            return {}

        enriched = await pipeline.process_articles(raw_articles) if raw_articles else []

        # Aggregate scores per asset
        asset_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for article in enriched:
            sentiment = article.get("sentiment", {})
            score = float(sentiment.get("score", 0.0))
            confidence = float(sentiment.get("confidence", 0.5))
            article_weight = self._article_weight(article, confidence)
            assets = _extract_assets(article.get("title", "") + " " + article.get("text", ""))
            for asset in assets:
                if asset in TRACKED_ASSETS:
                    asset_scores[asset].append((score, article_weight))

        # Weighted average (confidence-weighted)
        result: dict[str, SentimentScore] = {}
        for asset, score_pairs in asset_scores.items():
            if not score_pairs:
                continue
            total_weight = sum(conf for _, conf in score_pairs)
            if total_weight == 0:
                continue
            weighted_score = sum(s * c for s, c in score_pairs) / total_weight
            avg_confidence = total_weight / len(score_pairs)

            blended = weighted_score
            if macro_factors.source_count > 0:
                blended = weighted_score * 0.7 + macro_factors.global_score * 0.3

            result[asset] = SentimentScore(
                asset=asset,
                score=round(blended, 4),
                confidence=round(avg_confidence, 4),
                source_count=len(score_pairs),
                method=(
                    "hybrid_finbert_qwen3_macro"
                    if macro_factors.source_count > 0
                    else "hybrid_finbert_qwen3"
                ),
            )

        # For assets with no articles, use macro factors as fallback
        for asset in TRACKED_ASSETS:
            if asset not in result:
                result[asset] = SentimentScore(
                    asset=asset,
                    score=round(macro_factors.global_score * 0.5, 4),
                    confidence=max(0.4, macro_factors.confidence * 0.7),
                    source_count=macro_factors.source_count,
                    method=(
                        "macro_factors_only"
                        if macro_factors.source_count > 0
                        else "neutral_fallback"
                    ),
                )

        return result

    async def run(self) -> None:
        """Run the sentiment pipeline on a recurring schedule."""
        self._init_redis()
        self._init_kafka()
        self._running = True

        logger.info(
            "sentiment_pipeline_starting",
            interval_minutes=self._interval // 60,
            tracked_assets=TRACKED_ASSETS,
        )

        while self._running:
            try:
                scores = await self._run_once()
                if scores:
                    self._write_to_redis(scores)
                    self._publish_to_kafka(scores)
                    logger.info(
                        "sentiment_cycle_complete",
                        assets_scored=len(scores),
                        sample={k: v.score for k, v in list(scores.items())[:3]},
                    )
            except Exception as err:
                logger.exception("sentiment_pipeline_error", error=str(err))

            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from nexus_alpha.config import load_config
    from nexus_alpha.logging import setup_logging

    cfg = load_config()
    setup_logging(cfg.log_level)
    runner = SentimentPipelineRunner(cfg)
    asyncio.run(runner.run())
