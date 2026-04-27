"""
Live Market Data Ingestor — ccxt.pro WebSocket → Kafka.

This is the real-time data heartbeat of the entire system.
Connects to exchange WebSockets (Binance, Bybit, Kraken) via ccxt.pro,
normalizes ticks, and publishes to Kafka for all downstream consumers.

ccxt.pro WebSocket is 100% free — it's the same data the exchange
provides to all users at no cost.

Usage (from CLI or Docker):
    python -m nexus_alpha.data.live_ingestor

Or from code:
    ingestor = LiveMarketIngestor.from_config(config)
    await ingestor.run()
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nexus_alpha.config import NexusConfig
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class IngestorStats:
    ticks_published: int = 0
    ticks_dropped: int = 0
    reconnects: int = 0
    last_tick_at: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ticks_per_second(self) -> float:
        elapsed = time.monotonic() - self._start
        return self.ticks_published / max(elapsed, 1.0)

    def __post_init__(self) -> None:
        self._start = time.monotonic()


class LiveMarketIngestor:
    """
    Streams live OHLCV + order book + trade data from exchanges via ccxt.pro
    WebSocket and publishes normalized ticks to Kafka.

    Handles:
    - Automatic reconnection with exponential backoff
    - Multi-exchange, multi-symbol fan-out
    - Normalization to NEXUS-ALPHA tick schema
    - Kafka publishing with delivery confirmation
    - Sentiment score injection from Redis (set by HybridSentimentPipeline)
    """

    DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
    RECONNECT_BASE_DELAY = 1.0
    RECONNECT_MAX_DELAY = 60.0
    PRICE_TTL_SECONDS = 60 * 60
    OHLCV_TTL_SECONDS = 6 * 60 * 60
    OHLCV_MAX_ROWS = 500

    def __init__(
        self,
        exchange_id: str = "binance",
        symbols: list[str] | None = None,
        kafka_bootstrap: str = "localhost:9092",
        kafka_tick_topic: str = "market.ticks",
        redis_url: str = "redis://localhost:6379/0",
        exchange_api_key: str = "",
        exchange_api_secret: str = "",
        use_testnet: bool = False,
    ) -> None:
        self._exchange_id = exchange_id
        self._symbols = symbols or self.DEFAULT_SYMBOLS
        self._kafka_bootstrap = kafka_bootstrap
        self._kafka_topic = kafka_tick_topic
        self._redis_url = redis_url
        self._api_key = exchange_api_key
        self._api_secret = exchange_api_secret
        self._use_testnet = use_testnet
        self._stats = IngestorStats()
        self._running = False
        self._producer: Any = None
        self._redis: Any = None

    @classmethod
    def from_config(cls, config: NexusConfig) -> LiveMarketIngestor:
        return cls(
            exchange_id="binance",
            symbols=cls.DEFAULT_SYMBOLS,
            kafka_bootstrap=config.kafka.bootstrap_servers,
            kafka_tick_topic=config.kafka.tick_topic,
            redis_url=config.database.redis_url,
            exchange_api_key=config.binance.api_key.get_secret_value(),
            exchange_api_secret=config.binance.api_secret.get_secret_value(),
            use_testnet=config.binance.testnet,
        )

    # ── Kafka producer ────────────────────────────────────────────────────────

    def _init_kafka(self) -> None:
        if not self._kafka_bootstrap:
            logger.info("kafka_producer_disabled", exchange=self._exchange_id)
            return
        try:
            from confluent_kafka import Producer  # type: ignore[import]
            self._producer = Producer({
                "bootstrap.servers": self._kafka_bootstrap,
                "linger.ms": 5,           # Batch for 5ms before sending
                "compression.type": "none",
                "acks": "1",              # Leader ack — fast but reliable enough for ticks
            })
            logger.info("kafka_producer_ready", bootstrap=self._kafka_bootstrap)
        except ImportError:
            logger.warning("confluent_kafka_not_installed_using_in_memory_fallback")

    def _publish_tick(self, tick: dict[str, Any]) -> None:
        payload = json.dumps(tick, default=str).encode()
        if self._producer:
            self._producer.produce(
                self._kafka_topic,
                key=tick.get("symbol", "").encode(),
                value=payload,
            )
            # Poll to trigger delivery callbacks (non-blocking)
            self._producer.poll(0)
        self._cache_tick(tick)
        self._stats.ticks_published += 1
        self._stats.last_tick_at = time.monotonic()

    # ── Redis sentiment reader ────────────────────────────────────────────────

    def _init_redis(self) -> None:
        try:
            import redis  # type: ignore[import]
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("redis_connected_for_sentiment")
        except Exception as err:
            logger.warning("redis_unavailable_sentiment_disabled", error=str(err))

    def _get_sentiment(self, base_asset: str) -> float:
        if not self._redis:
            return 0.0
        try:
            value = self._redis.get(f"sentiment:{base_asset.upper()}")
            return float(value) if value else 0.0
        except Exception:
            return 0.0

    def _should_cache_tick(self, tick: dict[str, Any]) -> bool:
        # The trading loop reads symbol-only Redis keys (no exchange suffix), so
        # keep the execution venue canonical to avoid mixing Binance and Bybit rows.
        return self._exchange_id == "binance" and bool(self._redis)

    def _cache_tick(self, tick: dict[str, Any]) -> None:
        if not self._should_cache_tick(tick):
            return

        symbol = tick.get("symbol")
        if not symbol:
            return

        try:
            pipe = self._redis.pipeline()

            if tick.get("schema") == "ohlcv_v1":
                cached_rows = self._redis.get(f"ohlcv:{symbol}")
                rows = json.loads(cached_rows) if cached_rows else []
                candle = {
                    "timestamp": tick["timestamp"],
                    "open": tick["open"],
                    "high": tick["high"],
                    "low": tick["low"],
                    "close": tick["close"],
                    "volume": tick["volume"],
                }
                if rows and rows[-1].get("timestamp") == candle["timestamp"]:
                    rows[-1] = candle
                else:
                    rows.append(candle)
                rows = rows[-self.OHLCV_MAX_ROWS :]
                pipe.setex(
                    f"ohlcv:{symbol}",
                    self.OHLCV_TTL_SECONDS,
                    json.dumps(rows),
                )
                pipe.setex(
                    f"price:{symbol}",
                    self.PRICE_TTL_SECONDS,
                    str(tick["close"]),
                )
            elif tick.get("schema") == "orderbook_v1":
                bid_top = float(tick.get("bid_top", 0.0))
                ask_top = float(tick.get("ask_top", 0.0))
                if bid_top > 0 and ask_top > 0:
                    mid_price = (bid_top + ask_top) / 2.0
                    pipe.setex(
                        f"price:{symbol}",
                        self.PRICE_TTL_SECONDS,
                        str(round(mid_price, 8)),
                    )

            pipe.execute()
        except Exception as err:
            logger.warning("redis_tick_cache_failed", symbol=symbol, error=str(err))

    def _cache_ohlcv_window(self, symbol: str, ohlcv_rows: list[list[Any]]) -> None:
        if not self._should_cache_tick({"symbol": symbol}):
            return

        try:
            symbol_key = symbol.replace("/", "")
            incoming_rows = [
                {
                    "timestamp": datetime.utcfromtimestamp(row[0] / 1000).isoformat(),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
                for row in ohlcv_rows[-self.OHLCV_MAX_ROWS :]
            ]
            if not incoming_rows:
                return

            cached_rows = self._redis.get(f"ohlcv:{symbol_key}")
            existing_rows = json.loads(cached_rows) if cached_rows else []
            merged_by_ts = {row["timestamp"]: row for row in existing_rows}
            for row in incoming_rows:
                merged_by_ts[row["timestamp"]] = row
            rows = [merged_by_ts[key] for key in sorted(merged_by_ts.keys())][-self.OHLCV_MAX_ROWS :]

            latest_close = incoming_rows[-1]["close"]
            pipe = self._redis.pipeline()
            pipe.setex(
                f"ohlcv:{symbol_key}",
                self.OHLCV_TTL_SECONDS,
                json.dumps(rows),
            )
            pipe.setex(
                f"price:{symbol_key}",
                self.PRICE_TTL_SECONDS,
                str(latest_close),
            )
            pipe.execute()
        except Exception as err:
            logger.warning("redis_ohlcv_window_cache_failed", symbol=symbol, error=str(err))

    # ── Tick normalization ────────────────────────────────────────────────────

    def _normalize_ohlcv(self, symbol: str, ohlcv: list[Any]) -> dict[str, Any]:
        """
        Normalize ccxt OHLCV to NEXUS-ALPHA tick schema.
        ohlcv = [timestamp_ms, open, high, low, close, volume]
        """
        base = symbol.split("/")[0]
        return {
            "schema": "ohlcv_v1",
            "symbol": symbol.replace("/", ""),
            "exchange": self._exchange_id,
            "timestamp": datetime.utcfromtimestamp(ohlcv[0] / 1000).isoformat(),
            "open": float(ohlcv[1]),
            "high": float(ohlcv[2]),
            "low": float(ohlcv[3]),
            "close": float(ohlcv[4]),
            "volume": float(ohlcv[5]),
            "sentiment": self._get_sentiment(base),
            "ingested_at": datetime.utcnow().isoformat(),
        }

    def _normalize_trade(self, symbol: str, trade: dict[str, Any]) -> dict[str, Any]:
        """Normalize raw trade event for VPIN/OFI consumption."""
        return {
            "schema": "trade_v1",
            "symbol": symbol.replace("/", ""),
            "exchange": self._exchange_id,
            "timestamp": datetime.utcfromtimestamp(trade["timestamp"] / 1000).isoformat(),
            "price": float(trade["price"]),
            "amount": float(trade["amount"]),
            "side": trade["side"],  # 'buy' or 'sell'
            "trade_id": str(trade["id"]),
            "ingested_at": datetime.utcnow().isoformat(),
        }

    def _normalize_orderbook(self, symbol: str, ob: dict[str, Any]) -> dict[str, Any]:
        """Normalize order book snapshot — compute imbalance signal."""
        bids = ob.get("bids", [])[:10]
        asks = ob.get("asks", [])[:10]
        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

        return {
            "schema": "orderbook_v1",
            "symbol": symbol.replace("/", ""),
            "exchange": self._exchange_id,
            "timestamp": datetime.utcnow().isoformat(),
            "bid_top": float(bids[0][0]) if bids else 0.0,
            "ask_top": float(asks[0][0]) if asks else 0.0,
            "bid_volume_10": bid_vol,
            "ask_volume_10": ask_vol,
            "imbalance": round(imbalance, 4),
            "spread_bps": round(
                (float(asks[0][0]) - float(bids[0][0])) / float(bids[0][0]) * 10000, 2
            ) if bids and asks else 0.0,
        }

    def _orderbook_depth(self) -> int:
        if self._exchange_id == "bybit":
            return 50
        return 20

    def _exchange_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "defaultType": "spot",
        }
        if self._exchange_id == "binance":
            # Avoid probing unrelated derivatives endpoints during spot-only ingest.
            options["loadAllOptions"] = False
        if self._exchange_id == "bybit":
            options["defaultSubType"] = "spot"
        return options

    def _stream_params(self) -> dict[str, Any]:
        if self._exchange_id == "bybit":
            # Bybit's websocket APIs need the category pinned for spot symbols.
            return {"category": "spot"}
        return {}

    def _build_exchange_config(self) -> dict[str, Any]:
        return {
            "enableRateLimit": True,
            "options": self._exchange_options(),
        }

    async def _ensure_markets_loaded(self, exchange: Any) -> None:
        """Load exchange markets with persistent retries."""
        backoff = 1.0
        while self._running:
            try:
                await exchange.load_markets()
                logger.info("exchange_markets_loaded", exchange=self._exchange_id)
                return
            except Exception as err:
                logger.warning(
                    "exchange_markets_load_failed_retrying",
                    exchange=self._exchange_id,
                    error=str(err),
                    next_retry_s=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    # ── WebSocket streaming ───────────────────────────────────────────────────

    async def _stream_ohlcv(self, exchange: Any, symbol: str) -> None:
        """Stream 1-minute OHLCV for a symbol with auto-reconnect."""
        backoff = self.RECONNECT_BASE_DELAY
        params = self._stream_params()
        history_seeded = False
        while self._running:
            try:
                if not history_seeded:
                    history = await exchange.fetch_ohlcv(symbol, "1m", limit=120)
                    if history:
                        self._cache_ohlcv_window(symbol, history)
                    history_seeded = True
                ohlcv_list = await exchange.watch_ohlcv(symbol, "1m", None, None, params)
                if ohlcv_list:
                    self._cache_ohlcv_window(symbol, ohlcv_list)
                    latest = ohlcv_list[-1]
                    tick = self._normalize_ohlcv(symbol, latest)
                    self._publish_tick(tick)
                backoff = self.RECONNECT_BASE_DELAY  # Reset on success
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.warning("ohlcv_stream_error", symbol=symbol, error=str(err))
                self._stats.reconnects += 1
                self._stats.errors.append(f"{symbol}: {err}")
                await asyncio.sleep(min(backoff, self.RECONNECT_MAX_DELAY))
                backoff *= 2

    async def _stream_orderbook(self, exchange: Any, symbol: str) -> None:
        """Stream order book updates with auto-reconnect."""
        backoff = self.RECONNECT_BASE_DELAY
        params = self._stream_params()
        while self._running:
            try:
                ob = await exchange.watch_order_book(symbol, self._orderbook_depth(), params)
                tick = self._normalize_orderbook(symbol, ob)
                self._publish_tick(tick)
                backoff = self.RECONNECT_BASE_DELAY
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.warning("orderbook_stream_error", symbol=symbol, error=str(err))
                self._stats.reconnects += 1
                await asyncio.sleep(min(backoff, self.RECONNECT_MAX_DELAY))
                backoff *= 2

    async def _stream_trades(self, exchange: Any, symbol: str) -> None:
        """Stream real-time trades for VPIN calculation."""
        backoff = self.RECONNECT_BASE_DELAY
        params = self._stream_params()
        while self._running:
            try:
                trades = await exchange.watch_trades(symbol, None, None, params)
                for t in trades:
                    tick = self._normalize_trade(symbol, t)
                    self._publish_tick(tick)
                backoff = self.RECONNECT_BASE_DELAY
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.warning("trade_stream_error", symbol=symbol, error=str(err))
                self._stats.reconnects += 1
                await asyncio.sleep(min(backoff, self.RECONNECT_MAX_DELAY))
                backoff *= 2

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Start streaming all symbols concurrently with external supervisor logic.
        """
        self._running = True
        supervisor_backoff = 2.0
        while self._running:
            try:
                await self._run_internal()
                # If _run_internal returns without error but we are still 'running',
                # it means the internal tasks finished (e.g. connectivity lost).
                # We should wait and retry instead of breaking.
                if self._running:
                    logger.warning("ingestor_internal_loop_exited_restarting", exchange=self._exchange_id)
                    await asyncio.sleep(supervisor_backoff)
                    supervisor_backoff = min(supervisor_backoff * 2, 60.0)
            except Exception as err:
                logger.error(
                    "ingestor_task_crashed_restarting",
                    exchange=self._exchange_id,
                    error=str(err),
                    next_restart_s=supervisor_backoff,
                )
                await asyncio.sleep(supervisor_backoff)
                supervisor_backoff = min(supervisor_backoff * 2, 60.0)

    async def _run_internal(self) -> None:
        """Implementation of the streaming loop."""
        try:
            import ccxt.pro as ccxtpro  # type: ignore[import]
        except ImportError:
            logger.error(
                "ccxt_pro_not_installed",
                hint="pip install 'ccxt[async]'",
            )
            return

        self._init_kafka()
        self._init_redis()
        tasks: list[asyncio.Task[Any]] = []

        exchange = getattr(ccxtpro, self._exchange_id)(self._build_exchange_config())
        try:
            if self._use_testnet:
                logger.info(
                    "exchange_testnet_execution_enabled",
                    exchange=self._exchange_id,
                )

            logger.info(
                "live_ingestor_starting",
                exchange=self._exchange_id,
                symbols=self._symbols,
            )
            await self._ensure_markets_loaded(exchange)

            # Create concurrent streams for all symbols
            for symbol in self._symbols:
                tasks.append(asyncio.create_task(self._stream_ohlcv(exchange, symbol)))
                tasks.append(asyncio.create_task(self._stream_orderbook(exchange, symbol)))
                tasks.append(asyncio.create_task(self._stream_trades(exchange, symbol)))

            tasks.append(asyncio.create_task(self._stats_reporter()))

            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await exchange.close()
            if self._producer:
                self._producer.flush(timeout=5)

    async def _stats_reporter(self) -> None:
        """Log ingestion stats every 60 seconds."""
        while self._running:
            await asyncio.sleep(60)
            logger.info(
                "ingestor_stats",
                ticks_published=self._stats.ticks_published,
                ticks_per_second=round(self._stats.ticks_per_second, 2),
                reconnects=self._stats.reconnects,
            )

    def stop(self) -> None:
        self._running = False


# ── Multi-exchange fan-out ────────────────────────────────────────────────────

class MultiExchangeIngestor:
    """
    Runs LiveMarketIngestor for multiple exchanges concurrently.
    Binance (spot) + Bybit (futures proxy) = better price discovery.
    """

    DEFAULT_EXCHANGE_CONFIGS = [
        {
            "exchange_id": "binance",
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"],
        },
        {"exchange_id": "bybit", "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]},
    ]

    def __init__(self, config: NexusConfig) -> None:
        self._config = config
        exchange_configs = [self.DEFAULT_EXCHANGE_CONFIGS[0]]
        if config.bybit.enabled:
            exchange_configs.append(self.DEFAULT_EXCHANGE_CONFIGS[1])
        self._ingestors = [
            LiveMarketIngestor(
                exchange_id=ec["exchange_id"],
                symbols=ec["symbols"],
                kafka_bootstrap=config.kafka.bootstrap_servers,
                kafka_tick_topic=config.kafka.tick_topic,
                redis_url=config.database.redis_url,
                use_testnet=config.binance.testnet and ec["exchange_id"] == "binance",
            )
            for ec in exchange_configs
        ]

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(i.run(), name=f"{i._exchange_id}_ingestor")
            for i in self._ingestors
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def stop(self) -> None:
        for i in self._ingestors:
            i.stop()


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from nexus_alpha.config import load_config
    from nexus_alpha.log_config import setup_logging

    cfg = load_config()
    setup_logging(cfg.log_level)
    ingestor = LiveMarketIngestor.from_config(cfg)
    asyncio.run(ingestor.run())
