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
    fetch_all_rss_feeds,
    get_cryptopanic_news,
    get_current_fear_greed,
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
        for subreddit, batch in zip(subreddits, results):
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

        raw_articles = await self._collect_raw_articles()

        # 3. Fear & Greed (macro sentiment — unlimited free)
        fear_greed_score = 0.0
        try:
            fg = await get_current_fear_greed()
            fear_greed_score = (int(fg.get("value", 50)) - 50) / 50.0
        except Exception:
            pass

        if not raw_articles:
            logger.info("no_articles_fetched_this_cycle")
            return {}

        # Score all articles
        enriched = await pipeline.process_articles(raw_articles)

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

            # Blend with Fear & Greed (30% weight for macro context)
            blended = weighted_score * 0.7 + fear_greed_score * 0.3

            result[asset] = SentimentScore(
                asset=asset,
                score=round(blended, 4),
                confidence=round(avg_confidence, 4),
                source_count=len(score_pairs),
                method="hybrid_finbert_qwen3",
            )

        # For assets with no articles, use Fear & Greed as fallback
        for asset in TRACKED_ASSETS:
            if asset not in result:
                result[asset] = SentimentScore(
                    asset=asset,
                    score=round(fear_greed_score * 0.5, 4),  # Diluted — less conviction
                    confidence=0.4,
                    source_count=0,
                    method="fear_greed_only",
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
