import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from nexus_alpha.data.reddit_client import RedditClient, fetch_new_posts, get_reddit_client


@pytest.mark.asyncio
async def test_fetch_new_uses_praw_when_credentials(monkeypatch):
    client = RedditClient()
    client._client_id = "id"
    client._client_secret = "secret"
    client._username = "user"
    client._password = "pass"

    class DummyPost:
        def __init__(self):
            self.id = "abc"
            self.title = "hello"
            self.selftext = "body"
            self.created_utc = 123
            self.score = 10
            self.num_comments = 1
            self.url = "https://reddit.com/r/test/abc"
            self.author = MagicMock()
            self.author.name = "u/testuser"

    dummy_sub = MagicMock()
    dummy_sub.new.return_value = [DummyPost()]
    dummy_reddit = MagicMock()
    dummy_reddit.subreddit.return_value = dummy_sub

    monkeypatch.setattr(client, "_ensure_praw", lambda: dummy_reddit)

    posts = await client.fetch_new("test", limit=1)
    assert isinstance(posts, list)
    assert posts[0]["id"] == "abc"


@pytest.mark.asyncio
async def test_fetch_new_public_json(monkeypatch):
    client = RedditClient()
    client._client_id = ""
    # Mock httpx.AsyncClient.get
    sample = {"data": {"children": [{"data": {"id": "x1", "title": "t", "selftext": "s"}}]}}

    class DummyResponse:
        def __init__(self, json_data):
            self._json = json_data

        def raise_for_status(self):
            return None

        def json(self):
            return self._json

    async def dummy_get(url):
        return DummyResponse(sample)

    class DummyClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return DummyResponse(sample)

    monkeypatch.setattr("nexus_alpha.data.reddit_client.httpx.AsyncClient", DummyClient)

    posts = await client.fetch_new("test", limit=1)
    assert posts[0]["id"] == "x1"


@pytest.mark.asyncio
async def test_fetch_new_rss_fallback(monkeypatch):
    client = RedditClient()
    client._client_id = ""

    sample_feed = {"entries": [{"id": "rss1", "title": "rtitle", "summary": "s", "link": "u"}]}
    # Ensure feedparser.parse is available when reddit_client imports it lazily
    import types, sys

    fake = types.SimpleNamespace(parse=lambda url: sample_feed)
    monkeypatch.setitem(sys.modules, "feedparser", fake)

    async def _fail(sub, lim):
        raise Exception("force_public_fail")

    monkeypatch.setattr(client, "_fetch_new_public_json", _fail)

    posts = await client.fetch_new("test", limit=1)
    assert posts[0]["id"] == "rss1"
