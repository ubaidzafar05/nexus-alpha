"""Feature store primitives with point-in-time retrieval guarantees."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from nexus_alpha.data.contracts import FeatureSnapshotPayload
from nexus_alpha.schema_types import ExchangeName


@dataclass(frozen=True)
class FeatureStoreStats:
    snapshots_written: int = 0
    point_in_time_queries: int = 0
    symbols_tracked: int = 0


class FeatureStore(Protocol):
    def upsert_snapshot(self, snapshot: FeatureSnapshotPayload) -> None: ...
    def get_point_in_time(
        self,
        symbol: str,
        exchange: ExchangeName,
        as_of: datetime,
    ) -> FeatureSnapshotPayload | None: ...
    def metrics(self) -> dict[str, int]: ...


class InMemoryFeatureStore:
    """In-memory feature store with no-lookahead point-in-time retrieval."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, ExchangeName], list[FeatureSnapshotPayload]] = defaultdict(list)
        self._stats = FeatureStoreStats()

    def upsert_snapshot(self, snapshot: FeatureSnapshotPayload) -> None:
        key = (snapshot.symbol, snapshot.exchange)
        rows = self._rows[key]
        insert_idx = bisect_right([item.timestamp for item in rows], snapshot.timestamp)
        rows.insert(insert_idx, snapshot)
        self._stats = FeatureStoreStats(
            snapshots_written=self._stats.snapshots_written + 1,
            point_in_time_queries=self._stats.point_in_time_queries,
            symbols_tracked=len(self._rows),
        )

    def get_point_in_time(
        self,
        symbol: str,
        exchange: ExchangeName,
        as_of: datetime,
    ) -> FeatureSnapshotPayload | None:
        key = (symbol, exchange)
        rows = self._rows.get(key, [])
        timestamps = [item.timestamp for item in rows]
        idx = bisect_right(timestamps, as_of) - 1
        self._stats = FeatureStoreStats(
            snapshots_written=self._stats.snapshots_written,
            point_in_time_queries=self._stats.point_in_time_queries + 1,
            symbols_tracked=len(self._rows),
        )
        if idx < 0:
            return None
        return rows[idx]

    def metrics(self) -> dict[str, int]:
        return {
            "snapshots_written": self._stats.snapshots_written,
            "point_in_time_queries": self._stats.point_in_time_queries,
            "symbols_tracked": self._stats.symbols_tracked,
        }
