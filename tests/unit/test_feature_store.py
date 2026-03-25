from __future__ import annotations

from datetime import datetime, timedelta

from nexus_alpha.data.contracts import FeatureSnapshotPayload
from nexus_alpha.data.feature_store import InMemoryFeatureStore
from nexus_alpha.types import ExchangeName


def test_point_in_time_returns_latest_snapshot_without_lookahead() -> None:
    store = InMemoryFeatureStore()
    base = datetime.utcnow()
    store.upsert_snapshot(
        FeatureSnapshotPayload(
            symbol="BTCUSDT",
            exchange=ExchangeName.BINANCE,
            timestamp=base,
            features={"mid_price": 100.0},
        )
    )
    store.upsert_snapshot(
        FeatureSnapshotPayload(
            symbol="BTCUSDT",
            exchange=ExchangeName.BINANCE,
            timestamp=base + timedelta(seconds=10),
            features={"mid_price": 110.0},
        )
    )
    result = store.get_point_in_time(
        symbol="BTCUSDT",
        exchange=ExchangeName.BINANCE,
        as_of=base + timedelta(seconds=5),
    )
    assert result is not None
    assert result.features["mid_price"] == 100.0


def test_point_in_time_returns_none_if_only_future_data() -> None:
    store = InMemoryFeatureStore()
    base = datetime.utcnow()
    store.upsert_snapshot(
        FeatureSnapshotPayload(
            symbol="BTCUSDT",
            exchange=ExchangeName.BINANCE,
            timestamp=base + timedelta(seconds=5),
            features={"mid_price": 120.0},
        )
    )
    result = store.get_point_in_time(
        symbol="BTCUSDT",
        exchange=ExchangeName.BINANCE,
        as_of=base,
    )
    assert result is None
