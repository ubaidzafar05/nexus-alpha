"""
Reddit client wrapper for NEXUS-ALPHA.

Features:
- Uses PRAW (synchronous) when credentials are provided — executed in a thread to
  avoid blocking the asyncio event loop.
- Falls back to unauthenticated public JSON endpoints (/r/<sub>/new.json) when
  credentials are not configured.
- RSS fallback via /r/<sub>/new/.rss if JSON fails.

Placement: nexus_alpha/data/reddit_client.py

Usage:
    from nexus_alpha.data.reddit_client import RedditClient
    client = RedditClient()
    posts = await client.fetch_new("cryptocurrency", limit=50)

This module intentionally avoids adding asyncpraw to dependencies; it uses praw
(which is already in pyproject) and runs blocking calls in a thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "nexus-alpha/0.1 by anonymous")


class RedditClient:
    def __init__(self) -> None:
        self._client: Optional[praw.Reddit] = None
        self._client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self._client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self._username = os.getenv("REDDIT_USERNAME", "")
        self._password = os.getenv("REDDIT_PASSWORD", "")
        self._user_agent = os.getenv("REDDIT_USER_AGENT", DEFAULT_USER_AGENT)

    @property
    def has_credentials(self) -> bool:
        return bool(self._client_id and self._client_secret and self._username and self._password)

    def _ensure_praw(self):
        try:
            import praw  # type: ignore
        except Exception as e:
            raise RuntimeError("praw library is required for authenticated Reddit access") from e

        if self._client is None:
            logger.info("initializing_praw_client")
            self._client = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                username=self._username,
                password=self._password,
                user_agent=self._user_agent,
            )
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), retry=retry_if_exception_type(Exception))
    async def fetch_new(self, subreddit: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch newest posts for a subreddit.

        Tries in order:
        1. Authenticated PRAW (threaded)
        2. Public JSON endpoint (async httpx)
        3. RSS feed parsing (threaded feedparser)
        """
        if self.has_credentials:
            try:
                return await asyncio.to_thread(self._fetch_new_praw, subreddit, limit)
            except Exception as e:
                logger.warning("praw_fetch_failed: %s", str(e))

        # Fallback: public JSON endpoint
        try:
            return await self._fetch_new_public_json(subreddit, limit)
        except Exception as e:
            logger.warning("public_json_failed: %s", str(e))

        # Final fallback: RSS
        try:
            return await asyncio.to_thread(self._fetch_new_rss, subreddit, limit)
        except Exception as e:
            logger.exception("rss_fallback_failed: %s", str(e))
            raise

    def _fetch_new_praw(self, subreddit: str, limit: int = 50) -> List[Dict[str, Any]]:
        reddit = self._ensure_praw()
        posts = []
        for p in reddit.subreddit(subreddit).new(limit=limit):
            posts.append(self._praw_post_to_dict(p))
        return posts

    def _praw_post_to_dict(self, post: Any) -> Dict[str, Any]:
        return {
            "id": getattr(post, "id", ""),
            "title": getattr(post, "title", ""),
            "selftext": getattr(post, "selftext", ""),
            "created_utc": getattr(post, "created_utc", 0),
            "score": getattr(post, "score", 0),
            "num_comments": getattr(post, "num_comments", 0),
            "url": getattr(post, "url", ""),
            "author": getattr(getattr(post, "author", None), "name", None),
        }

    async def _fetch_new_public_json(self, subreddit: str, limit: int = 50) -> List[Dict[str, Any]]:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
        headers = {"User-Agent": self._user_agent}
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            body = resp.json()
            items = body.get("data", {}).get("children", [])
            posts = [self._json_child_to_post(c.get("data", {})) for c in items]
            return posts

    def _json_child_to_post(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "selftext": data.get("selftext", ""),
            "created_utc": data.get("created_utc", 0),
            "score": data.get("score", 0),
            "num_comments": data.get("num_comments", 0),
            "url": data.get("url", ""),
            "author": data.get("author"),
        }

    def _fetch_new_rss(self, subreddit: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Parse RSS using feedparser if available, otherwise fallback to basic XML parsing."""
        url = f"https://www.reddit.com/r/{subreddit}/new/.rss"
        try:
            import feedparser  # type: ignore

            parsed = feedparser.parse(url)
            entries = parsed.get("entries", [])[:limit]
            posts = []
            for e in entries:
                posts.append({
                    "id": e.get("id", ""),
                    "title": e.get("title", ""),
                    "selftext": e.get("summary", ""),
                    "created_utc": 0,
                    "score": 0,
                    "num_comments": 0,
                    "url": e.get("link", ""),
                    "author": e.get("author", None),
                })
            return posts
        except Exception:
            # Minimal fallback: simple XML parse to extract title/link
            try:
                import xml.etree.ElementTree as ET
                import httpx

                r = httpx.get(url, timeout=10.0)
                r.raise_for_status()
                root = ET.fromstring(r.content)
                items = root.findall(".//item")[:limit]
                posts = []
                for it in items:
                    title = it.findtext("title") or ""
                    link = it.findtext("link") or ""
                    posts.append({
                        "id": link,
                        "title": title,
                        "selftext": "",
                        "created_utc": 0,
                        "score": 0,
                        "num_comments": 0,
                        "url": link,
                        "author": None,
                    })
                return posts
            except Exception as e:
                logger.exception("rss_parse_error: %s", str(e))
                return []


# Module-level convenience
_default_client: Optional[RedditClient] = None


def get_reddit_client() -> RedditClient:
    global _default_client
    if _default_client is None:
        _default_client = RedditClient()
    return _default_client


async def fetch_new_posts(subreddit: str, limit: int = 50) -> List[Dict[str, Any]]:
    client = get_reddit_client()
    return await client.fetch_new(subreddit, limit)
