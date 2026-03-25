from __future__ import annotations

from datetime import datetime

from nexus_alpha.execution.execution_engine import ExchangeLiquidity, IntelligentExchangeRouter
from nexus_alpha.types import ExchangeName, Order, OrderSide, OrderType


def _build_order(side: OrderSide) -> Order:
    return Order(
        order_id="o-1",
        symbol="BTCUSDT",
        exchange=ExchangeName.BINANCE,
        side=side,
        order_type=OrderType.MARKET,
        quantity=10.0,
        created_at=datetime.utcnow(),
    )


def test_buy_prefers_ask_depth() -> None:
    router = IntelligentExchangeRouter()
    router.update_liquidity(
        ExchangeName.BINANCE,
        ExchangeLiquidity(
            exchange=ExchangeName.BINANCE,
            bid_depth=1000,
            ask_depth=1,
            spread_bps=1.0,
            maker_fee_bps=1.0,
            taker_fee_bps=1.0,
            latency_ms=20,
            recent_fill_quality=0.7,
        ),
    )
    router.update_liquidity(
        ExchangeName.BYBIT,
        ExchangeLiquidity(
            exchange=ExchangeName.BYBIT,
            bid_depth=1,
            ask_depth=1000,
            spread_bps=1.0,
            maker_fee_bps=1.0,
            taker_fee_bps=1.0,
            latency_ms=20,
            recent_fill_quality=0.7,
        ),
    )
    decision = router.route_order(_build_order(OrderSide.BUY))
    assert decision.primary_exchange == ExchangeName.BYBIT


def test_sell_prefers_bid_depth() -> None:
    router = IntelligentExchangeRouter()
    router.update_liquidity(
        ExchangeName.BINANCE,
        ExchangeLiquidity(
            exchange=ExchangeName.BINANCE,
            bid_depth=1000,
            ask_depth=1,
            spread_bps=1.0,
            maker_fee_bps=1.0,
            taker_fee_bps=1.0,
            latency_ms=20,
            recent_fill_quality=0.7,
        ),
    )
    router.update_liquidity(
        ExchangeName.BYBIT,
        ExchangeLiquidity(
            exchange=ExchangeName.BYBIT,
            bid_depth=1,
            ask_depth=1000,
            spread_bps=1.0,
            maker_fee_bps=1.0,
            taker_fee_bps=1.0,
            latency_ms=20,
            recent_fill_quality=0.7,
        ),
    )
    decision = router.route_order(_build_order(OrderSide.SELL))
    assert decision.primary_exchange == ExchangeName.BINANCE

