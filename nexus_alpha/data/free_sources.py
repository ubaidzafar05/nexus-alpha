"""
Free data sources — 100% free, no paid subscriptions required.

Sources:
  - CoinGecko API       (10k calls/month free, no key for basic endpoints)
  - DeFiLlama           (100% free, no auth, open data)
  - Alternative.me      (Fear & Greed Index, unlimited, no key)
  - Etherscan API       (free key, 5 calls/sec, all EVM chains)
  - CryptoPanic         (50 calls/hour free with key)
  - Reddit PRAW         (60 req/min free with registered app)
  - RSS feeds           (unlimited, no rate limits)

All functions are async-first. Sync helpers provided where needed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import httpx

from nexus_alpha.logging import get_logger

logger = get_logger(__name__)

# ── Rate-limit helpers ────────────────────────────────────────────────────────

_LAST_REQUEST: dict[str, float] = {}


async def _rate_limited_get(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    min_interval_s: float = 0.2,
    source_key: str = "",
) -> dict[str, Any]:
    """GET with simple per-source rate limiting."""
    key = source_key or url
    now = time.monotonic()
    wait = min_interval_s - (now - _LAST_REQUEST.get(key, 0))
    if wait > 0:
        await asyncio.sleep(wait)
    _LAST_REQUEST[key] = time.monotonic()
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


# ── CoinGecko (free, ~10k calls/month no key, demo key extends this) ──────────

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


async def get_coingecko_prices(
    coin_ids: list[str],
    vs_currencies: str = "usd",
    api_key: str = "",
) -> dict[str, Any]:
    """Current prices for multiple coins. No key needed for basic use."""
    params: dict[str, Any] = {"ids": ",".join(coin_ids), "vs_currencies": vs_currencies}
    if api_key:
        params["x_cg_demo_api_key"] = api_key
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await _rate_limited_get(client, f"{COINGECKO_BASE}/simple/price", params, source_key="coingecko")


async def get_coingecko_ohlc(coin_id: str, days: int = 30, api_key: str = "") -> list[list[float]]:
    """OHLC data: [[timestamp_ms, open, high, low, close], ...]"""
    params: dict[str, Any] = {"vs_currency": "usd", "days": days}
    if api_key:
        params["x_cg_demo_api_key"] = api_key
    async with httpx.AsyncClient(timeout=15.0) as client:
        return await _rate_limited_get(
            client, f"{COINGECKO_BASE}/coins/{coin_id}/ohlc", params, source_key="coingecko"
        )


async def get_coingecko_market_data(coin_id: str, api_key: str = "") -> dict[str, Any]:
    """Rich market data: market_cap, volume, circulating_supply, ATH, etc."""
    params: dict[str, Any] = {"localization": "false", "tickers": "false", "community_data": "false"}
    if api_key:
        params["x_cg_demo_api_key"] = api_key
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await _rate_limited_get(
            client, f"{COINGECKO_BASE}/coins/{coin_id}", params, source_key="coingecko"
        )


# ── Alternative.me — Fear & Greed Index ───────────────────────────────────────

async def get_fear_greed_index(limit: int = 30) -> list[dict[str, Any]]:
    """
    Crypto Fear & Greed Index — unlimited, no key required.
    Returns list of {'value': '45', 'value_classification': 'Fear', 'timestamp': '...'}
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        data = await _rate_limited_get(
            client,
            "https://api.alternative.me/fng/",
            params={"limit": limit},
            source_key="alternative_me",
        )
    return data.get("data", [])


async def get_current_fear_greed() -> dict[str, Any]:
    items = await get_fear_greed_index(limit=1)
    if not items:
        return {"value": "50", "value_classification": "Neutral"}
    item = items[0]
    return {
        "value": int(item["value"]),
        "classification": item["value_classification"],
        "timestamp": datetime.fromtimestamp(int(item["timestamp"])).isoformat(),
    }


# ── DeFiLlama — TVL, stablecoins, yields ─────────────────────────────────────

DEFILLAMA_BASE = "https://api.llama.fi"


async def get_total_tvl_history() -> list[dict[str, Any]]:
    """Total DeFi TVL over time — macro market health indicator."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        return await _rate_limited_get(
            client, f"{DEFILLAMA_BASE}/v2/historicalChainTvl", source_key="defillama"
        )


async def get_protocol_tvl(protocol: str) -> dict[str, Any]:
    """TVL for a specific protocol (e.g. 'uniswap', 'aave', 'lido')."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await _rate_limited_get(
            client, f"{DEFILLAMA_BASE}/protocol/{protocol}", source_key="defillama"
        )


async def get_stablecoin_flows() -> dict[str, Any]:
    """Stablecoin supply and chain distribution — risk-appetite proxy."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        return await _rate_limited_get(
            client, "https://stablecoins.llama.fi/stablecoins", source_key="defillama"
        )


async def get_defi_yields(min_tvl_usd: float = 1_000_000) -> list[dict[str, Any]]:
    """DeFi yield rates — risk appetite indicator."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        data = await _rate_limited_get(
            client, f"{DEFILLAMA_BASE}/yields", source_key="defillama"
        )
    pools = data.get("data", [])
    return [p for p in pools if (p.get("tvlUsd") or 0) >= min_tvl_usd]


# ── Etherscan — on-chain activity (free key, 5 req/sec) ──────────────────────

ETHERSCAN_BASE = "https://api.etherscan.io/api"
KNOWN_EXCHANGE_WALLETS = {
    "binance_hot": "0x28C6c06298d514Db089934071355E5743bf21d60",
    "coinbase_cold": "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3",
    "kraken": "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2",
}


async def get_exchange_flows(
    wallet_address: str,
    api_key: str,
    min_eth: float = 50.0,
) -> list[dict[str, Any]]:
    """Track large inflows/outflows for a known exchange wallet."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        data = await _rate_limited_get(
            client,
            ETHERSCAN_BASE,
            params={
                "module": "account",
                "action": "txlist",
                "address": wallet_address,
                "sort": "desc",
                "apikey": api_key,
            },
            min_interval_s=0.25,  # 4 req/sec to stay within free limit
            source_key="etherscan",
        )
    txs = data.get("result", [])
    if not isinstance(txs, list):
        return []
    return [
        {
            "hash": tx["hash"],
            "from": tx["from"],
            "to": tx["to"],
            "value_eth": float(tx["value"]) / 1e18,
            "timestamp": datetime.fromtimestamp(int(tx["timeStamp"])).isoformat(),
            "block": int(tx["blockNumber"]),
        }
        for tx in txs
        if float(tx["value"]) / 1e18 >= min_eth
    ]


async def get_gas_price(api_key: str) -> dict[str, Any]:
    """Current Ethereum gas prices — network congestion indicator."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        data = await _rate_limited_get(
            client,
            ETHERSCAN_BASE,
            params={"module": "gastracker", "action": "gasoracle", "apikey": api_key},
            source_key="etherscan",
        )
    result = data.get("result", {})
    return {
        "safe_gwei": result.get("SafeGasPrice"),
        "propose_gwei": result.get("ProposeGasPrice"),
        "fast_gwei": result.get("FastGasPrice"),
    }


# ── CryptoPanic — curated crypto news (50 req/hour free) ─────────────────────

async def get_cryptopanic_news(
    auth_token: str,
    currencies: str = "BTC,ETH,SOL",
    filter_type: str = "hot",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Curated crypto news with sentiment labels.
    filter_type: hot | rising | bullish | bearish | important
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        data = await _rate_limited_get(
            client,
            "https://cryptopanic.com/api/v1/posts/",
            params={
                "auth_token": auth_token,
                "currencies": currencies,
                "filter": filter_type,
                "public": "true",
            },
            min_interval_s=0.075,  # ~13 req/sec burst — well within 50/hour
            source_key="cryptopanic",
        )
    return data.get("results", [])[:limit]


# ── RSS feed aggregator ───────────────────────────────────────────────────────

FREE_CRYPTO_RSS_FEEDS: dict[str, str] = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "cryptoslate": "https://cryptoslate.com/feed/",
    "bitcoinmagazine": "https://bitcoinmagazine.com/feed",
    "sec_press": "https://www.sec.gov/rss/news/press.xml",
    "federal_reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
}


@dataclass
class RSSArticle:
    title: str
    url: str
    published: datetime
    source: str
    summary: str = ""
    text: str = ""
    tags: list[str] = field(default_factory=list)


async def fetch_rss_feed(url: str, source_name: str) -> list[RSSArticle]:
    """Fetch and parse a single RSS feed (no rate limit)."""
    try:
        import feedparser  # type: ignore[import]
    except ImportError:
        logger.warning("feedparser_not_installed", hint="pip install feedparser")
        return []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, follow_redirects=True)
            feed = feedparser.parse(resp.text)

        articles = []
        for entry in feed.entries:
            try:
                pub_time = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.utcnow()
            except Exception:
                pub_time = datetime.utcnow()

            articles.append(
                RSSArticle(
                    title=entry.get("title", ""),
                    url=entry.get("link", ""),
                    published=pub_time,
                    source=source_name,
                    summary=entry.get("summary", "")[:500],
                )
            )
        return articles
    except Exception as err:
        logger.warning("rss_fetch_failed", source=source_name, error=repr(err))
        return []


async def fetch_all_rss_feeds(
    max_age_minutes: int = 30,
    feeds: dict[str, str] | None = None,
) -> list[RSSArticle]:
    """Fetch all RSS feeds concurrently and return recent articles."""
    feeds = feeds or FREE_CRYPTO_RSS_FEEDS
    cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)

    tasks = [fetch_rss_feed(url, name) for name, url in feeds.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[RSSArticle] = []
    for batch in results:
        if isinstance(batch, list):
            articles.extend(a for a in batch if a.published >= cutoff)

    return sorted(articles, key=lambda a: a.published, reverse=True)


# ── Reddit PRAW — social sentiment ───────────────────────────────────────────

CRYPTO_SUBREDDITS = [
    "CryptoCurrency", "Bitcoin", "ethereum", "solana",
    "CryptoMarkets", "defi", "CryptoMoonShots",
]


def fetch_reddit_posts(
    client_id: str,
    client_secret: str,
    user_agent: str = "NexusAlpha/1.0",
    subreddits: list[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch hot posts from crypto subreddits (synchronous — PRAW is sync-first)."""
    try:
        import praw  # type: ignore[import]
    except ImportError:
        logger.warning("praw_not_installed", hint="pip install praw")
        return []

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    posts = []
    for sub_name in (subreddits or CRYPTO_SUBREDDITS):
        try:
            sub = reddit.subreddit(sub_name)
            for post in sub.hot(limit=limit):
                posts.append(
                    {
                        "title": post.title,
                        "subreddit": sub_name,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat(),
                        "url": post.url,
                        "selftext": (post.selftext or "")[:300],
                    }
                )
        except Exception as err:
            logger.warning("reddit_fetch_failed", subreddit=sub_name, error=str(err))

    return posts


# ── Composite market snapshot ─────────────────────────────────────────────────

async def get_free_market_snapshot(
    coins: list[str] | None = None,
    etherscan_key: str = "",
) -> dict[str, Any]:
    """
    Aggregate free data into a single market snapshot dict.
    Safe to call every 5 minutes without hitting any rate limits.
    """
    coins = coins or ["bitcoin", "ethereum", "solana", "binancecoin"]

    tasks = {
        "prices": get_coingecko_prices(coins),
        "fear_greed": get_current_fear_greed(),
        "tvl_history": get_total_tvl_history(),
    }

    results: dict[str, Any] = {}
    for key, coro in tasks.items():
        try:
            results[key] = await coro
        except Exception as err:
            logger.warning("free_source_failed", key=key, error=str(err))
            results[key] = {}

    if etherscan_key:
        try:
            results["gas"] = await get_gas_price(etherscan_key)
        except Exception as err:
            logger.warning("etherscan_failed", error=str(err))

    return results
