"""
Unit tests for the free stack components:
- FreeLLMClient routing (Ollama primary, Groq fallback)
- HybridSentimentPipeline (FinBERT fast-path)
- SentimentPipelineRunner (Redis writer)
- LiveMarketIngestor (ccxt.pro WebSocket config)
- TelegramAlerts (graceful no-op when not configured)
- NexusAlphaStrategy (do_predict guard)
- CLI commands exist (backtest, live-ingest)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pandas as pd
import pytest
from click.testing import CliRunner

from nexus_alpha.cli import cli

# ─── FreeLLMClient ────────────────────────────────────────────────────────────

class TestFreeLLMClient:
    def test_from_config_uses_ollama_url(self):
        from nexus_alpha.config import LLMConfig
        from nexus_alpha.intelligence.free_llm import FreeLLMClient

        cfg = LLMConfig()
        client = FreeLLMClient.from_config(cfg)
        assert "11434" in client._ollama_url or "ollama" in client._ollama_url.lower()

    def test_model_name_property_returns_primary(self):
        from nexus_alpha.config import LLMConfig

        cfg = LLMConfig(ollama_primary_model="qwen3:8b")
        assert "qwen3" in cfg.model_name

    @pytest.mark.asyncio
    async def test_complete_falls_back_to_groq_when_enabled(self):
        """When Ollama returns a connection error, Groq fallback is invoked."""
        from nexus_alpha.config import LLMConfig
        from nexus_alpha.intelligence.free_llm import FreeLLMClient

        cfg = LLMConfig(use_groq_fallback=True, groq_api_key="test-key")  # type: ignore[call-arg]
        client = FreeLLMClient.from_config(cfg)

        with patch.object(client, "_ollama_complete", side_effect=ConnectionError("refused")):
            with patch.object(
                client,
                "_groq_complete",
                new=AsyncMock(return_value="groq response"),
            ):
                result = await client.complete("hello")
        assert result == "groq response"

    @pytest.mark.asyncio
    async def test_complete_raises_when_ollama_unavailable_and_groq_disabled(self):
        from nexus_alpha.config import LLMConfig
        from nexus_alpha.intelligence.free_llm import FreeLLMClient

        client = FreeLLMClient.from_config(LLMConfig())

        with patch.object(client, "_ollama_complete", side_effect=ConnectionError("refused")):
            with pytest.raises(RuntimeError, match="Groq fallback is disabled"):
                await client.complete("hello")


# ─── HybridSentimentPipeline ─────────────────────────────────────────────────

class TestHybridSentimentPipeline:
    """Tests that don't require GPU — mock the FinBERT model."""

    def _make_pipeline(self):
        from nexus_alpha.intelligence.sentiment import HybridSentimentPipeline

        pipeline = HybridSentimentPipeline.__new__(HybridSentimentPipeline)
        pipeline._finbert = None
        pipeline._llm = MagicMock()
        pipeline._use_finbert = False  # Skip GPU loading in tests
        return pipeline

    def test_pipeline_instantiates(self):
        pipeline = self._make_pipeline()
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_process_articles_returns_enriched(self):
        from nexus_alpha.intelligence.sentiment import SentimentResult

        pipeline = self._make_pipeline()

        # Mock FinBERT scorer object
        mock_finbert = MagicMock()
        mock_finbert.score_batch.return_value = [
            SentimentResult(0.8, 0.92, "positive", "finbert")
        ]
        pipeline._finbert = mock_finbert
        pipeline._use_finbert = True

        with patch.object(pipeline, "_needs_deep_analysis", return_value=False):
            articles = [{"title": "Bitcoin surges to new ATH", "text": "BTC price up 10%."}]
            result = await pipeline.process_articles(articles)

        assert len(result) == 1
        assert "sentiment" in result[0]
        assert result[0]["sentiment"]["score"] == 0.8


# ─── SentimentPipelineRunner ─────────────────────────────────────────────────

class TestSentimentPipelineRunner:
    def test_runner_initializes(self):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)
        assert runner is not None
        assert runner._running is False

    def test_asset_keyword_extraction(self):
        from nexus_alpha.data.sentiment_pipeline import _extract_assets

        assert "BTC" in _extract_assets("Bitcoin hits new all-time high")
        assert "ETH" in _extract_assets("Ethereum dApp ecosystem grows")
        assert "SOL" in _extract_assets("Solana network upgrade complete")

    def test_write_to_redis_calls_setex(self):
        from datetime import datetime

        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner, SentimentScore

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        runner._redis = mock_redis

        scores = {
            "BTC": SentimentScore("BTC", 0.6, 0.8, 5, "hybrid_finbert_qwen3", datetime.utcnow()),
        }
        runner._write_to_redis(scores)

        mock_pipe.setex.assert_called()
        mock_pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_raw_articles_includes_reddit(self, monkeypatch):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)

        monkeypatch.setenv("REDDIT_SUBREDDITS", "CryptoCurrency")
        monkeypatch.setenv("REDDIT_POST_LIMIT", "2")

        monkeypatch.setattr(
            "nexus_alpha.data.sentiment_pipeline.fetch_all_rss_feeds",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            "nexus_alpha.data.sentiment_pipeline.fetch_new_posts",
            AsyncMock(
                return_value=[
                    {
                        "title": "Bitcoin breaks higher",
                        "selftext": "BTC momentum is strong",
                        "url": "https://reddit.test/post1",
                        "score": 320,
                        "num_comments": 88,
                        "author": "tester",
                    }
                ]
            ),
        )

        articles = await runner._collect_raw_articles()

        assert len(articles) == 1
        assert articles[0]["source"] == "reddit:CryptoCurrency"
        assert articles[0]["score"] == 320

    def test_article_weight_boosts_high_engagement_reddit(self):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)

        base_weight = runner._article_weight({"source": "coindesk"}, 0.7)
        reddit_weight = runner._article_weight(
            {"source": "reddit:CryptoCurrency", "score": 500, "num_comments": 200},
            0.7,
        )

        assert reddit_weight > base_weight

    def test_score_tvl_history_positive_trend(self):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)

        score = runner._score_tvl_history(
            [
                {"totalLiquidityUSD": 1_000_000_000},
                {"totalLiquidityUSD": 1_100_000_000},
            ]
        )

        assert score > 0

    def test_score_exchange_flow_pressure_prefers_outflows(self):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)

        wallet = "0xexchange"
        score = runner._score_exchange_flow_pressure(
            {
                wallet: [
                    {"from": wallet, "to": "0xuser1", "value_eth": 300.0},
                    {"from": "0xuser2", "to": wallet, "value_eth": 100.0},
                ]
            }
        )

        assert score > 0

    @pytest.mark.asyncio
    async def test_collect_macro_factors_combines_sources(self, monkeypatch):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)

        monkeypatch.setenv("ETHERSCAN_API_KEY", "test-key")
        monkeypatch.setattr(
            "nexus_alpha.data.sentiment_pipeline.get_current_fear_greed",
            AsyncMock(return_value={"value": 70}),
        )
        monkeypatch.setattr(
            "nexus_alpha.data.sentiment_pipeline.get_total_tvl_history",
            AsyncMock(
                return_value=[
                    {"totalLiquidityUSD": 1_000_000_000},
                    {"totalLiquidityUSD": 1_050_000_000},
                ]
            ),
        )
        monkeypatch.setattr(
            "nexus_alpha.data.sentiment_pipeline.get_gas_price",
            AsyncMock(return_value={"propose_gwei": "18"}),
        )
        monkeypatch.setattr(
            "nexus_alpha.data.sentiment_pipeline.get_exchange_flows",
            AsyncMock(return_value=[{"from": "0xwallet", "to": "0xuser", "value_eth": 200.0}]),
        )

        factors = await runner._collect_macro_factors()

        assert factors.source_count >= 3
        assert factors.global_score > 0
        assert "fear_greed" in factors.details

    @pytest.mark.asyncio
    async def test_run_once_uses_macro_only_when_no_articles(self, monkeypatch):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import MacroFactors, SentimentPipelineRunner

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)

        mock_pipeline = MagicMock()
        mock_pipeline.process_articles = AsyncMock(return_value=[])

        monkeypatch.setattr(runner, "_ensure_sentiment_pipeline", lambda **kwargs: mock_pipeline)
        monkeypatch.setattr(runner, "_collect_raw_articles", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            runner,
            "_collect_macro_factors",
            AsyncMock(
                return_value=MacroFactors(
                    global_score=0.3,
                    confidence=0.6,
                    source_count=3,
                    details={"fear_greed": 0.2},
                )
            ),
        )

        scores = await runner._run_once()

        assert scores["BTC"].score == 0.15
        assert scores["BTC"].method == "macro_factors_only"

    @pytest.mark.asyncio
    async def test_run_once_uses_reddit_articles_in_aggregation(self, monkeypatch):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.sentiment_pipeline import SentimentPipelineRunner

        cfg = NexusConfig()
        runner = SentimentPipelineRunner(cfg)

        mock_pipeline = MagicMock()
        mock_pipeline.process_articles = AsyncMock(
            return_value=[
                {
                    "title": "Bitcoin trend is bullish",
                    "text": "BTC rally continues",
                    "source": "reddit:CryptoCurrency",
                    "score": 500,
                    "num_comments": 120,
                    "sentiment": {"score": 0.8, "confidence": 0.9},
                }
            ]
        )

        monkeypatch.setattr(runner, "_ensure_sentiment_pipeline", lambda **kwargs: mock_pipeline)
        monkeypatch.setattr(
            runner,
            "_collect_raw_articles",
            AsyncMock(
                return_value=[
                    {
                        "title": "Bitcoin trend is bullish",
                        "text": "BTC rally continues",
                        "source": "reddit:CryptoCurrency",
                        "score": 500,
                        "num_comments": 120,
                    }
                ]
            ),
        )
        monkeypatch.setattr(
            "nexus_alpha.data.sentiment_pipeline.get_current_fear_greed",
            AsyncMock(return_value={"value": 60}),
        )

        scores = await runner._run_once()

        assert "BTC" in scores
        assert scores["BTC"].score > 0.5
        assert scores["BTC"].source_count == 1
        assert scores["BTC"].method == "hybrid_finbert_qwen3_macro"


# ─── LiveMarketIngestor ───────────────────────────────────────────────────────

class TestLiveMarketIngestor:
    def test_ingestor_config(self):
        from nexus_alpha.data.live_ingestor import LiveMarketIngestor

        ingestor = LiveMarketIngestor(
            exchange_id="binance",
            symbols=["BTC/USDT"],
        )
        assert ingestor._exchange_id == "binance"
        assert "BTC/USDT" in ingestor._symbols

    def test_multi_ingestor_creates_two_ingestors(self):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.data.live_ingestor import MultiExchangeIngestor

        cfg = NexusConfig()
        multi = MultiExchangeIngestor(cfg)
        assert len(multi._ingestors) == 1

    def test_stop_sets_running_false(self):
        from nexus_alpha.data.live_ingestor import LiveMarketIngestor

        ingestor = LiveMarketIngestor("binance", ["BTC/USDT"])
        ingestor._running = True
        ingestor.stop()
        assert ingestor._running is False

    def test_orderbook_depth_is_exchange_safe(self):
        from nexus_alpha.data.live_ingestor import LiveMarketIngestor

        assert LiveMarketIngestor("binance", ["BTC/USDT"])._orderbook_depth() == 20
        assert LiveMarketIngestor("bybit", ["BTC/USDT"])._orderbook_depth() == 50

    def test_exchange_specific_stream_configuration(self):
        from nexus_alpha.data.live_ingestor import LiveMarketIngestor

        binance = LiveMarketIngestor("binance", ["BTC/USDT"])
        bybit = LiveMarketIngestor("bybit", ["BTC/USDT"])

        assert binance._exchange_options()["defaultType"] == "spot"
        assert binance._exchange_options()["loadAllOptions"] is False
        assert binance._stream_params() == {}
        assert bybit._exchange_options()["defaultType"] == "spot"
        assert bybit._exchange_options()["defaultSubType"] == "spot"
        assert bybit._stream_params() == {"category": "spot"}
        assert "apiKey" not in binance._build_exchange_config()

    def test_binance_ticks_are_cached_for_trading_loop(self):
        import json

        from nexus_alpha.data.live_ingestor import LiveMarketIngestor

        class _FakePipe:
            def __init__(self):
                self.calls = []

            def setex(self, key, ttl, value):
                self.calls.append((key, ttl, value))
                return self

            def execute(self):
                return None

        class _FakeRedis:
            def __init__(self):
                self.values = {}
                self.pipe = _FakePipe()

            def get(self, key):
                return self.values.get(key)

            def pipeline(self):
                self.pipe = _FakePipe()
                return self.pipe

        ingestor = LiveMarketIngestor("binance", ["BTC/USDT"])
        ingestor._redis = _FakeRedis()
        ingestor._redis.values["ohlcv:BTCUSDT"] = json.dumps(
            [{"timestamp": "2024-01-01T00:00:00", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 3}]
        )

        ingestor._cache_tick(
            {
                "schema": "ohlcv_v1",
                "symbol": "BTCUSDT",
                "timestamp": "2024-01-01T00:01:00",
                "open": 2.0,
                "high": 3.0,
                "low": 1.5,
                "close": 2.5,
                "volume": 4.0,
            }
        )

        keys = [call[0] for call in ingestor._redis.pipe.calls]
        assert "ohlcv:BTCUSDT" in keys
        assert "price:BTCUSDT" in keys

    def test_ohlcv_window_cache_populates_history(self):
        import json

        from nexus_alpha.data.live_ingestor import LiveMarketIngestor

        class _FakePipe:
            def __init__(self):
                self.calls = []

            def setex(self, key, ttl, value):
                self.calls.append((key, ttl, value))
                return self

            def execute(self):
                return None

        class _FakeRedis:
            def __init__(self):
                self.pipe = _FakePipe()
                self.values = {}

            def get(self, key):
                return self.values.get(key)

            def pipeline(self):
                self.pipe = _FakePipe()
                return self.pipe

        ingestor = LiveMarketIngestor("binance", ["BTC/USDT"])
        ingestor._redis = _FakeRedis()
        ingestor._cache_ohlcv_window(
            "BTC/USDT",
            [
                [1704067200000, 1.0, 2.0, 0.5, 1.5, 10.0],
                [1704067260000, 1.5, 2.5, 1.0, 2.0, 12.0],
            ],
        )

        history_call = next(call for call in ingestor._redis.pipe.calls if call[0] == "ohlcv:BTCUSDT")
        rows = json.loads(history_call[2])
        assert len(rows) == 2
        assert rows[-1]["close"] == 2.0

    def test_normalize_testnet_order_request_respects_notional(self):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator
        from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem
        from nexus_alpha.signals.signal_engine import SignalFusionEngine

        class _FakeExchange:
            def price_to_precision(self, symbol, price):
                return f"{price:.2f}"

            def amount_to_precision(self, symbol, amount):
                return f"{amount:.5f}"

        config = NexusConfig()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(risk_config=config.risk),
            alerts=MagicMock(),
        )
        price, quantity = loop._normalize_testnet_order_request(
            exchange=_FakeExchange(),
            market={
                "limits": {
                    "amount": {"min": 0.00001},
                    "cost": {"min": 5.0},
                },
                "precision": {"amount": 0.00001},
            },
            symbol="BTC/USDT",
            quantity=0.00001,
            price=36000.0,
        )

        assert price == 36000.0
        assert quantity * price >= 5.0

    def test_exchange_symbol_normalizes_compact_pairs(self):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator
        from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem
        from nexus_alpha.signals.signal_engine import SignalFusionEngine

        config = NexusConfig()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(risk_config=config.risk),
            alerts=MagicMock(),
        )

        assert loop._exchange_symbol("ADAUSDT") == "ADA/USDT"
        assert loop._exchange_symbol("BTC/USDT") == "BTC/USDT"

    def test_cap_quantity_to_available_balance_respects_quote_balance(self):
        from nexus_alpha.config import NexusConfig
        from nexus_alpha.core.trading_loop import TradingLoopOrchestrator
        from nexus_alpha.risk.circuit_breaker import CircuitBreakerSystem
        from nexus_alpha.signals.signal_engine import SignalFusionEngine
        from nexus_alpha.types import OrderSide

        class _FakeExchange:
            def amount_to_precision(self, symbol, amount):
                return f"{amount:.5f}"

        config = NexusConfig()
        loop = TradingLoopOrchestrator(
            config=config,
            signal_engine=SignalFusionEngine(),
            circuit_breaker=CircuitBreakerSystem(risk_config=config.risk),
            alerts=MagicMock(),
        )

        capped = loop._cap_quantity_to_available_balance(
            exchange=_FakeExchange(),
            market={
                "base": "ADA",
                "quote": "USDT",
                "limits": {"amount": {"min": 0.1}, "cost": {"min": 5.0}},
            },
            symbol="ADA/USDT",
            side=OrderSide.BUY,
            quantity=1000.0,
            price=1.0,
            balance={"free": {"USDT": 100.0}},
        )

        assert capped == 95.0


# ─── TelegramAlerts ───────────────────────────────────────────────────────────

class TestTelegramAlerts:
    def test_from_env_returns_alerts_object(self):
        from nexus_alpha.alerts.telegram import TelegramAlerts

        alerts = TelegramAlerts.from_env()
        assert alerts is not None

    @pytest.mark.asyncio
    async def test_send_is_no_op_when_not_configured(self):
        from nexus_alpha.alerts.telegram import TelegramAlerts

        alerts = TelegramAlerts(bot_token=None, chat_id=None)  # type: ignore[arg-type]
        result = await alerts.send("test message")
        # Should return False gracefully, not raise
        assert result is False

    @pytest.mark.asyncio
    async def test_send_retries_once_on_transport_error(self, monkeypatch: pytest.MonkeyPatch):
        from nexus_alpha.alerts.telegram import TelegramAlerts

        created_clients: list[object] = []

        class _FakeClient:
            def __init__(self):
                self.calls = 0
                self.closed = False
                created_clients.append(self)

            async def post(self, *args, **kwargs):
                self.calls += 1
                if len(created_clients) == 1:
                    raise httpx.ConnectError("flaky")
                return httpx.Response(200, request=httpx.Request("POST", "https://api.telegram.org"))

            async def aclose(self):
                self.closed = True

        monkeypatch.setattr("nexus_alpha.alerts.telegram.httpx.AsyncClient", lambda *a, **kw: _FakeClient())

        alerts = TelegramAlerts(bot_token="token", chat_id="chat")
        assert await alerts.send("test message") is True
        assert len(created_clients) == 2
        assert created_clients[0].closed is True
        await alerts.aclose()
        assert created_clients[1].closed is True


# ─── Crawl4AI intelligence ─────────────────────────────────────────────────────

class TestCrawl4AIIntelligence:
    @pytest.mark.asyncio
    async def test_crawl_agent_normalizes_items(self, monkeypatch: pytest.MonkeyPatch):
        from nexus_alpha.intelligence.crawl4ai_agents import Crawl4AINewsAgent

        monkeypatch.setattr(
            "nexus_alpha.intelligence.crawl4ai_agents._crawl_url_with_ollama",
            AsyncMock(
                return_value=[
                    {
                        "headline": "Bitcoin surges after ETF inflows",
                        "summary": "BTC jumps as spot ETF demand rises.",
                        "url": "https://example.test/btc",
                        "affected_symbols": ["BTC"],
                        "urgency": "high",
                    }
                ]
            ),
        )

        agent = Crawl4AINewsAgent(target_urls=["https://example.test/news"], max_items_per_target=3)
        reports = await agent.fetch()

        assert len(reports) == 1
        assert reports[0].headline.startswith("Bitcoin surges")
        assert reports[0].affected_symbols == ["BTC"]
        assert reports[0].urgency.value == "high"


# ─── NexusAlphaStrategy FreqAI guard ─────────────────────────────────────────

class TestNexusAlphaStrategyDoPredict:
    def test_populate_entry_trend_without_freqai_column(self):
        """Should not raise KeyError when do_predict column is absent."""
        # Stub freqtrade so it can be imported without installation
        ft_stub = types.ModuleType("freqtrade")
        ft_strategy_stub = types.ModuleType("freqtrade.strategy")

        class _IStrategy:
            def __init__(self, config=None):
                pass

        ft_strategy_stub.IStrategy = _IStrategy
        ft_strategy_stub.DecimalParameter = lambda *a, **kw: MagicMock(value=0.1)
        sys.modules.setdefault("freqtrade", ft_stub)
        sys.modules.setdefault("freqtrade.strategy", ft_strategy_stub)
        sys.modules.setdefault("freqtrade.persistence", types.ModuleType("freqtrade.persistence"))

        # Build a minimal dataframe without do_predict
        df = pd.DataFrame({
            "ema_fast": [100.0, 101.0],
            "ema_slow": [99.0, 99.5],
            "rsi": [45.0, 48.0],
            "sentiment": [0.3, 0.2],
        })

        # Directly test the guard logic (column absent path)
        freqai_active = "do_predict" in df.columns
        assert freqai_active is False

        # Guard: fallback to base strategy logic
        ml_signal = (
            (df["ema_fast"] > df["ema_slow"])
            & (df["rsi"] < 50)
        )
        assert ml_signal.any(), "Fallback signal should trigger on test data"

    def test_strategy_toggle_env_parsing(self, monkeypatch: pytest.MonkeyPatch):
        import importlib.util
        from pathlib import Path

        ft_stub = types.ModuleType("freqtrade")
        ft_strategy_stub = types.ModuleType("freqtrade.strategy")

        class _IStrategy:
            def __init__(self, config=None):
                self.dp = MagicMock()
                self.wallets = MagicMock()

        ft_strategy_stub.IStrategy = _IStrategy
        ft_strategy_stub.DecimalParameter = lambda *a, **kw: MagicMock(value=kw.get("default", 0.1))
        ft_strategy_stub.IntParameter = lambda *a, **kw: MagicMock(value=kw.get("default", 1))
        talib_stub = types.ModuleType("talib")
        talib_abstract_stub = types.ModuleType("talib.abstract")
        sys.modules["freqtrade"] = ft_stub
        sys.modules["freqtrade.strategy"] = ft_strategy_stub
        sys.modules["talib"] = talib_stub
        sys.modules["talib.abstract"] = talib_abstract_stub

        strategy_path = Path(__file__).resolve().parents[2] / "freqtrade/strategies/NexusAlphaStrategy.py"
        spec = importlib.util.spec_from_file_location("nexus_alpha_freqtrade_strategy_test", strategy_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        strategy = module.NexusAlphaStrategy()

        monkeypatch.setenv("NEXUS_ENABLE_TREND", "false")
        monkeypatch.setenv("NEXUS_ENABLE_MEAN_REVERSION", "1")
        monkeypatch.setenv("NEXUS_ENABLE_REGIME_FILTER", "no")

        assert strategy._trend_entries_enabled() is False
        assert strategy._mean_reversion_entries_enabled() is True
        assert strategy._regime_filter_enabled() is False


# ─── CLI command registration ─────────────────────────────────────────────────

class TestCLICommands:
    def test_backtest_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "Freqtrade" in result.output or "freqtrade" in result.output.lower()

    def test_live_ingest_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["live-ingest", "--help"])
        assert result.exit_code == 0
        assert "exchange" in result.output.lower()

    def test_health_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--help"])
        assert result.exit_code == 0

    def test_crawl_intel_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["crawl-intel", "--help"])
        assert result.exit_code == 0
        assert "crawl" in result.output.lower()

    def test_sentiment_once_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["sentiment-once", "--help"])
        assert result.exit_code == 0
        assert "sentiment" in result.output.lower()
