"""
Phase 2: Trade Logger — records every decision and outcome for learning.

Every paper/live trade is logged with:
- The full feature state at decision time
- The signal that triggered it
- The actual P&L outcome (filled in when position closes)
- Reward computed for RL training

This is the memory that lets the bot learn from its own wins and losses.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import urllib.request

from nexus_alpha.learning.entry_features import augmented_feature_names
from nexus_alpha.learning.historical_data import get_feature_column_names
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

DB_PATH = Path("data/trade_logs/trades.db")


@dataclass
class TradeRecord:
    """Complete record of a single trading decision + outcome."""

    trade_id: str
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    notional_usd: float

    # Decision context
    signal_direction: float
    signal_confidence: float
    contributing_signals: str  # JSON
    sentiment_score: float
    regime: str

    # Feature state at decision time (serialized)
    feature_vector: str  # JSON array

    # Outcome (filled in when position closes)
    exit_price: float = 0.0
    exit_timestamp: str = ""
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    hold_duration_seconds: float = 0.0

    # RL reward (computed from outcome)
    reward: float = 0.0

    # Status
    status: str = "open"  # open, closed, stopped_out

    # G3: Rich telemetry (JSON-encoded)
    entry_context: str = ""  # JSON: mtf_alignment, pair_quality, regime, guards passed
    exit_context: str = ""   # JSON: exit_reason, regime_at_exit, mtf_at_exit


class TradeLogger:
    """
    SQLite-backed trade logger for persistent learning memory.
    Thread-safe and survives restarts.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    notional_usd REAL NOT NULL,
                    signal_direction REAL,
                    signal_confidence REAL,
                    contributing_signals TEXT,
                    sentiment_score REAL DEFAULT 0,
                    regime TEXT DEFAULT 'unknown',
                    feature_vector TEXT,
                    exit_price REAL DEFAULT 0,
                    exit_timestamp TEXT DEFAULT '',
                    realized_pnl REAL DEFAULT 0,
                    realized_pnl_pct REAL DEFAULT 0,
                    hold_duration_seconds REAL DEFAULT 0,
                    reward REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    entry_context TEXT DEFAULT '',
                    exit_context TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)
            """)
            # Migrate: add telemetry columns to existing tables
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN entry_context TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN exit_context TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    details TEXT
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _notify_ui(self, record: dict) -> None:
        """Ping the dashboard internal API to trigger real-time UI updates."""
        try:
            url = "http://localhost:8500/api/internal/notify_trade"
            # Standardize for the UI model
            payload = {
                "trade_id": record.get("trade_id"),
                "timestamp": record.get("timestamp"),
                "symbol": record.get("symbol"),
                "side": record.get("side"),
                "entry_price": float(record.get("entry_price", 0)),
                "exit_price": float(record.get("exit_price", 0)),
                "realized_pnl_pct": float(record.get("realized_pnl_pct", 0)),
                "status": record.get("status", "open")
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=1) as response:
                pass
        except Exception:
            # Silent fail — internal UI notification should NEVER crash the trading bot
            pass

    def log_trade_open(self, record: TradeRecord) -> None:
        """Log a new trade entry with full telemetry context."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades
                (trade_id, timestamp, symbol, side, entry_price, quantity, notional_usd,
                 signal_direction, signal_confidence, contributing_signals,
                 sentiment_score, regime, feature_vector, entry_context, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """, (
                record.trade_id, record.timestamp, record.symbol, record.side,
                record.entry_price, record.quantity, record.notional_usd,
                record.signal_direction, record.signal_confidence,
                record.contributing_signals, record.sentiment_score,
                record.regime, record.feature_vector,
                record.entry_context,
            ))
        logger.info("trade_logged", trade_id=record.trade_id, symbol=record.symbol)
        self._notify_ui(asdict(record))

    def log_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        realized_pnl: float,
        reward: float | None = None,
        exit_context: str = "",
        exit_timestamp: str | None = None,
    ) -> None:
        """Update a trade with exit info, compute reward, and store exit context."""
        close_ts = exit_timestamp or datetime.utcnow().isoformat()

        with self._conn() as conn:
            row = conn.execute(
                "SELECT entry_price, quantity, side, timestamp, entry_context FROM trades WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()

            if not row:
                logger.warning("trade_not_found_for_close", trade_id=trade_id)
                return

            entry_price, quantity, side, open_ts, entry_context_str = row
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
            if side == "sell":
                pnl_pct = -pnl_pct

            try:
                open_dt = datetime.fromisoformat(open_ts)
                close_dt = datetime.fromisoformat(close_ts)
                hold_seconds = max(0.0, (close_dt - open_dt).total_seconds())
            except (ValueError, TypeError):
                hold_seconds = 0.0

            if reward is None:
                reward = self._compute_reward(pnl_pct, hold_seconds)

            conn.execute("""
                UPDATE trades SET
                    exit_price = ?,
                    exit_timestamp = ?,
                    realized_pnl = ?,
                    realized_pnl_pct = ?,
                    hold_duration_seconds = ?,
                    reward = ?,
                    status = 'closed',
                    exit_context = ?
                WHERE trade_id = ?
            """, (exit_price, close_ts, realized_pnl, pnl_pct, hold_seconds, reward, exit_context, trade_id))

        # --- RFT: Update ARTVault Reward ---
        try:
            if entry_context_str:
                entry_ctx = json.loads(entry_context_str)
                traj_id = entry_ctx.get("trajectory_id")
                if traj_id:
                    from nexus_alpha.learning.rft_vault import ARTVault
                    ARTVault().update_reward(traj_id, reward)
        except Exception as e:
            logger.error("failed_to_update_trajectory_reward", error=str(e), trade_id=trade_id)
        # -----------------------------------

        logger.info(
            "trade_closed",
            trade_id=trade_id,
            pnl_pct=f"{pnl_pct:.4f}",
            reward=f"{reward:.4f}",
        )
        
        # Notify UI of the closure
        self._notify_ui({
            "trade_id": trade_id,
            "exit_price": exit_price,
            "realized_pnl_pct": pnl_pct,
            "status": "closed"
        })

    def delete_trades_by_prefix(self, prefix: str) -> int:
        """Delete trades whose IDs start with a given prefix."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM trades WHERE trade_id LIKE ?",
                (f"{prefix}%",),
            )
        return int(cursor.rowcount or 0)

    def _compute_reward(self, pnl_pct: float, hold_seconds: float) -> float:
        """
        Risk-adjusted reward for RL training.
        Rewards profitable trades, penalizes losses harder, and discourages
        holding losing positions too long.
        """
        # Base: profit = positive reward, loss = amplified negative
        if pnl_pct >= 0:
            base_reward = pnl_pct * 10  # Scale up small returns
        else:
            base_reward = pnl_pct * 15  # Losses hurt 1.5x more (asymmetric)

        # Time penalty: penalize holding losers, but not winners
        hold_hours = hold_seconds / 3600
        if pnl_pct < 0 and hold_hours > 24:
            base_reward -= 0.1 * (hold_hours / 24)  # -0.1 per day of holding a loser

        return float(np.clip(base_reward, -5.0, 5.0))

    # ── Query methods for training ────────────────────────────────────────
    
    def get_trade_record(self, trade_id: str) -> dict[str, Any] | None:
        """Fetch a single trade record by ID."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_closed_trades(self, limit: int = 10000) -> list[dict[str, Any]]:
        """Get closed trades for offline training."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_closed_trades(self) -> int:
        """Return the number of closed trades available for learning."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status = 'closed'"
            ).fetchone()
        return int(row[0]) if row else 0

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Get currently open positions."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_portfolio_heat(self) -> float:
        """Calculate the total notional exposure (USD) of all open trades."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT SUM(notional_usd) FROM trades WHERE status = 'open'"
            ).fetchone()
        return float(row[0] or 0.0)

    def get_symbol_performance(self) -> dict[str, dict[str, Any]]:
        """Aggregate closed-trade performance by symbol."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    symbol,
                    COUNT(*) AS total_trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    AVG(realized_pnl_pct) AS avg_pnl_pct,
                    AVG(reward) AS avg_reward,
                    SUM(realized_pnl) AS total_pnl
                FROM trades
                WHERE status = 'closed'
                GROUP BY symbol
                ORDER BY total_pnl DESC
                """
            ).fetchall()
        results: dict[str, dict[str, Any]] = {}
        for row in rows:
            total_trades = int(row["total_trades"] or 0)
            wins = int(row["wins"] or 0)
            results[str(row["symbol"])] = {
                "total_trades": total_trades,
                "wins": wins,
                "win_rate": wins / total_trades if total_trades > 0 else 0.0,
                "avg_pnl_pct": float(row["avg_pnl_pct"] or 0.0),
                "avg_reward": float(row["avg_reward"] or 0.0),
                "total_pnl": float(row["total_pnl"] or 0.0),
            }
        return results

    def get_symbol_learning_scores(self, min_trades: int = 5) -> dict[str, float]:
        """
        Convert symbol-level trade outcomes into conservative confidence multipliers.

        Returns values centered near 1.0, capped to avoid overreacting to small
        samples. Intended for paper/live confidence scaling, not hard filtering.
        """
        performance = self.get_symbol_performance()
        scores: dict[str, float] = {}
        for symbol, stats in performance.items():
            trades = int(stats["total_trades"])
            if trades < min_trades:
                continue
            shrinkage = min(1.0, trades / 20.0)
            win_rate_edge = (float(stats["win_rate"]) - 0.5) * 0.6
            reward_edge = float(stats["avg_reward"]) * 0.25
            pnl_edge = float(stats["avg_pnl_pct"]) * 4.0
            raw_edge = win_rate_edge + reward_edge + pnl_edge
            multiplier = 1.0 + shrinkage * raw_edge
            scores[symbol] = round(float(np.clip(multiplier, 0.80, 1.15)), 4)
        return scores

    def get_training_data(
        self,
        min_trades: int = 50,
        min_abs_pnl_pct: float = 0.003,
        live_min_abs_pnl_pct: float = 0.0005,
        excluded_exit_reasons: set[str] | None = None,
        replay_min_mtf_alignment: float | None = 1.0,
    ) -> dict[str, np.ndarray] | None:
        """
        Extract feature vectors and rewards from closed trades for RL training.
        Returns None if not enough data yet.
        """
        dataset = self.build_learning_dataset(
            min_trades=min_trades,
            target_mode="binary",
            min_abs_pnl_pct=min_abs_pnl_pct,
            live_min_abs_pnl_pct=live_min_abs_pnl_pct,
            excluded_exit_reasons=excluded_exit_reasons,
            replay_min_mtf_alignment=replay_min_mtf_alignment,
        )
        if dataset is None:
            return None
        return {
            "features": dataset["features"],
            "rewards": dataset["rewards"],
            "directions": dataset["directions"],
            "pnl_pcts": dataset["pnl_pcts"],
            "n_trades": dataset["n_trades"],
        }

    def build_learning_dataset(
        self,
        min_trades: int = 50,
        target_mode: str = "binary",
        strong_move_pct: float = 0.02,
        target_metric: str = "pnl_pct",
        target_threshold: float | None = None,
        min_abs_pnl_pct: float = 0.003,
        live_min_abs_pnl_pct: float = 0.0005,
        excluded_exit_reasons: set[str] | None = None,
        replay_min_mtf_alignment: float | None = 1.0,
        min_quality_score: float = 0.0,
        top_fraction: float | None = None,
        balanced: bool = False,
        regime_slice: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a filtered learning dataset with targets, metadata, and sample scores."""
        excluded_exit_reasons = excluded_exit_reasons or {
            "orphaned_on_restart",
            "take_profit_partial",
        }
        effective_threshold = target_threshold
        if effective_threshold is None:
            effective_threshold = 1.0 if target_metric == "risk_multiple" else strong_move_pct
        trades = self.get_closed_trades()
        if len(trades) < min_trades:
            logger.info("insufficient_trades_for_training", count=len(trades), required=min_trades)
            return None

        features = []
        rewards = []
        directions = []
        pnl_pcts = []
        timestamps = []
        quality_scores = []
        regime_slices = []
        sources = []
        targets = []
        target_values = []
        feature_lengths: Counter[int] = Counter()
        for trade in trades:
            try:
                fv = json.loads(trade["feature_vector"]) if trade["feature_vector"] else None
                if fv is None:
                    continue
                if not isinstance(fv, list) or len(fv) == 0:
                    continue
                if not any(abs(float(value)) > 1e-9 for value in fv):
                    continue
                entry_ctx = json.loads(trade["entry_context"]) if trade["entry_context"] else {}
                exit_ctx = json.loads(trade["exit_context"]) if trade["exit_context"] else {}
                exit_reason = str(exit_ctx.get("exit_reason", ""))
                if exit_reason in excluded_exit_reasons:
                    continue
                realized_pnl_pct = float(trade.get("realized_pnl_pct", 0.0) or 0.0)
                source = str(entry_ctx.get("source", "live"))
                source_min_abs_pnl_pct = min_abs_pnl_pct if source == "historical_replay" else live_min_abs_pnl_pct
                if abs(realized_pnl_pct) < source_min_abs_pnl_pct:
                    continue
                if source == "historical_replay" and replay_min_mtf_alignment is not None:
                    mtf_alignment = float(entry_ctx.get("mtf_alignment", 0.0) or 0.0)
                    if mtf_alignment < replay_min_mtf_alignment:
                        continue
                inferred_slice = self._infer_regime_slice(entry_ctx, trade)
                if regime_slice and inferred_slice != regime_slice:
                    continue
                quality_score = self._sample_quality_score(
                    trade=trade,
                    entry_ctx=entry_ctx,
                    exit_ctx=exit_ctx,
                    realized_pnl_pct=realized_pnl_pct,
                )
                if quality_score < min_quality_score:
                    continue
                target_value = self._build_target_value(
                    realized_pnl_pct=realized_pnl_pct,
                    entry_ctx=entry_ctx,
                    target_metric=target_metric,
                )
                if target_value is None:
                    continue
                features.append(fv)
                rewards.append(trade["reward"])
                directions.append(float(np.sign(trade.get("signal_direction", 0.0)) or 0.0))
                pnl_pcts.append(realized_pnl_pct)
                timestamps.append(str(trade.get("timestamp", "")))
                quality_scores.append(quality_score)
                regime_slices.append(inferred_slice)
                sources.append(source)
                targets.append(self._build_target(target_value, target_mode, effective_threshold))
                target_values.append(target_value)
                feature_lengths[len(fv)] += 1
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        if not feature_lengths:
            logger.info(
                "no_qualified_trades_for_training",
                min_abs_pnl_pct=min_abs_pnl_pct,
                live_min_abs_pnl_pct=live_min_abs_pnl_pct,
                excluded_exit_reasons=sorted(excluded_exit_reasons),
                replay_min_mtf_alignment=replay_min_mtf_alignment,
                target_mode=target_mode,
            )
            return None

        eligible_feature_lengths = [
            (feature_len, count)
            for feature_len, count in feature_lengths.items()
            if count >= min_trades
        ]
        if eligible_feature_lengths:
            target_len, _ = max(
                eligible_feature_lengths,
                key=lambda item: (item[0], item[1]),
            )
        else:
            target_len, _ = feature_lengths.most_common(1)[0]
        filtered = [
            (ts, fv, reward, direction, pnl_pct, quality_score, regime_slice_name, source, target, target_value)
            for ts, fv, reward, direction, pnl_pct, quality_score, regime_slice_name, source, target, target_value in zip(
                timestamps,
                features,
                rewards,
                directions,
                pnl_pcts,
                quality_scores,
                regime_slices,
                sources,
                targets,
                target_values,
            )
            if len(fv) == target_len
        ]
        filtered.sort(key=lambda item: item[0])

        if top_fraction is not None:
            keep_n = max(min_trades, int(len(filtered) * top_fraction))
            filtered = sorted(filtered, key=lambda item: item[5], reverse=True)[:keep_n]
            filtered.sort(key=lambda item: item[0])

        if balanced:
            grouped: dict[tuple[int, str], list[tuple[Any, ...]]] = {}
            for item in filtered:
                group_key = (int(item[8]), str(item[6]))
                grouped.setdefault(group_key, []).append(item)
            if grouped:
                min_group = min(len(group_items) for group_items in grouped.values())
                if min_group > 0:
                    balanced_rows = []
                    for group_items in grouped.values():
                        balanced_rows.extend(group_items[:min_group])
                    filtered = sorted(balanced_rows, key=lambda item: item[0])

        if len(filtered) < min_trades:
            logger.info(
                "insufficient_consistent_feature_trades",
                count=len(filtered),
                required=min_trades,
                feature_length=target_len,
                target_mode=target_mode,
            )
            return None

        timestamps = [ts for ts, _, _, _, _, _, _, _, _, _ in filtered]
        features = [fv for _, fv, _, _, _, _, _, _, _, _ in filtered]
        rewards = [reward for _, _, reward, _, _, _, _, _, _, _ in filtered]
        directions = [direction for _, _, _, direction, _, _, _, _, _, _ in filtered]
        pnl_pcts = [pnl_pct for _, _, _, _, pnl_pct, _, _, _, _, _ in filtered]
        quality_scores = [score for _, _, _, _, _, score, _, _, _, _ in filtered]
        regime_slices = [slice_name for _, _, _, _, _, _, slice_name, _, _, _ in filtered]
        sources = [source_name for _, _, _, _, _, _, _, source_name, _, _ in filtered]
        targets = [target for _, _, _, _, _, _, _, _, target, _ in filtered]
        target_values = [target_value for _, _, _, _, _, _, _, _, _, target_value in filtered]
        class_counts = Counter(targets)
        slice_counts = Counter(regime_slices)
        base_feature_names = get_feature_column_names()
        feature_names: list[str]
        if target_len == len(base_feature_names):
            feature_names = base_feature_names
        elif target_len == len(augmented_feature_names(base_feature_names)):
            feature_names = augmented_feature_names(base_feature_names)
        else:
            feature_names = [f"f{i}" for i in range(target_len)]

        logger.info(
            "training_data_prepared",
            raw_trades=len(trades),
            qualified_trades=len(filtered),
            feature_length=target_len,
            min_abs_pnl_pct=min_abs_pnl_pct,
            live_min_abs_pnl_pct=live_min_abs_pnl_pct,
            replay_min_mtf_alignment=replay_min_mtf_alignment,
            min_quality_score=min_quality_score,
            target_mode=target_mode,
            balanced=balanced,
        )
        return {
            "features": np.array(features, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "directions": np.array(directions, dtype=np.float32),
            "pnl_pcts": np.array(pnl_pcts, dtype=np.float32),
            "quality_scores": np.array(quality_scores, dtype=np.float32),
            "targets": np.array(targets, dtype=np.int32),
            "timestamps": timestamps,
            "regime_slices": regime_slices,
            "sources": sources,
            "class_counts": dict(class_counts),
            "slice_counts": dict(slice_counts),
            "feature_names": feature_names,
            "target_mode": target_mode,
            "target_metric": target_metric,
            "target_threshold": effective_threshold,
            "strong_move_pct": strong_move_pct,
            "target_values": np.array(target_values, dtype=np.float32),
            "n_trades": len(features),
        }

    def _build_target_value(
        self,
        realized_pnl_pct: float,
        entry_ctx: dict[str, Any],
        target_metric: str,
    ) -> float | None:
        if target_metric == "pnl_pct":
            return realized_pnl_pct
        if target_metric == "risk_multiple":
            stop_distance_pct = float(entry_ctx.get("ctx_stop_distance_pct", 0.0) or 0.0)
            if stop_distance_pct <= 0:
                return None
            return realized_pnl_pct / stop_distance_pct
        raise ValueError(f"Unsupported target metric: {target_metric}")

    def _build_target(self, target_value: float, target_mode: str, threshold: float) -> int:
        mode = target_mode.lower()
        if mode == "binary":
            return int(target_value > 0)
        if mode == "ternary":
            if target_value <= -threshold:
                return 0
            if target_value >= threshold:
                return 2
            return 1
        if mode == "quaternary":
            if target_value <= -threshold:
                return 0
            if target_value < 0:
                return 1
            if target_value < threshold:
                return 2
            return 3
        raise ValueError(f"Unsupported target mode: {target_mode}")

    def _infer_regime_slice(self, entry_ctx: dict[str, Any], trade: dict[str, Any]) -> str:
        explicit = str(entry_ctx.get("regime_slice", "")).strip()
        if explicit:
            return explicit
        run_label = str(entry_ctx.get("run_label", "")).lower()
        if "bear" in run_label:
            return "bear"
        if "recovery" in run_label:
            return "recovery"
        if "bull" in run_label:
            return "bull"
        if "mixed" in run_label:
            return "mixed"
        source = str(entry_ctx.get("source", "live"))
        return source

    def _sample_quality_score(
        self,
        trade: dict[str, Any],
        entry_ctx: dict[str, Any],
        exit_ctx: dict[str, Any],
        realized_pnl_pct: float,
    ) -> float:
        signal_conf = float(trade.get("signal_confidence", 0.0) or 0.0)
        pair_quality = float(entry_ctx.get("pair_quality", 0.5) or 0.5)
        mtf_alignment = float(entry_ctx.get("mtf_alignment", 0.5) or 0.5)
        regime_mult = float(entry_ctx.get("regime_multiplier", 0.9) or 0.9)
        regime_score = float(np.clip((regime_mult - 0.6) / 0.6, 0.0, 1.0))
        move_score = float(np.clip(abs(realized_pnl_pct) / 0.05, 0.0, 1.0))
        hold_hours = float(trade.get("hold_duration_seconds", 0.0) or 0.0) / 3600.0
        if hold_hours <= 0:
            duration_score = 0.0
        elif hold_hours <= 96:
            duration_score = 1.0
        elif hold_hours <= 168:
            duration_score = 0.7
        else:
            duration_score = 0.4
        exit_reason = str(exit_ctx.get("exit_reason", ""))
        exit_score = 1.0
        if exit_reason in {"backtest_end", "unknown"}:
            exit_score = 0.4
        elif exit_reason == "time_exit":
            exit_score = 0.75
        quality = (
            0.20 * signal_conf
            + 0.20 * pair_quality
            + 0.20 * mtf_alignment
            + 0.15 * regime_score
            + 0.15 * move_score
            + 0.05 * duration_score
            + 0.05 * exit_score
        )
        return round(float(np.clip(quality, 0.0, 1.0)), 4)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall trading performance stats."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN realized_pnl = 0 THEN 1 ELSE 0 END) as breakeven,
                    AVG(realized_pnl_pct) as avg_pnl_pct,
                    SUM(realized_pnl) as total_pnl,
                    AVG(reward) as avg_reward,
                    AVG(hold_duration_seconds) / 3600 as avg_hold_hours,
                    MAX(realized_pnl_pct) as best_trade_pct,
                    MIN(realized_pnl_pct) as worst_trade_pct
                FROM trades WHERE status = 'closed'
            """).fetchone()

        if not row or row[0] == 0:
            return {"total_trades": 0, "message": "No closed trades yet"}

        total, wins, losses, be, avg_pnl, total_pnl, avg_reward, avg_hold, best, worst = row
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "avg_pnl_pct": round(avg_pnl * 100, 4),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_reward": round(avg_reward, 4),
            "avg_hold_hours": round(avg_hold, 1),
            "best_trade_pct": round(best * 100, 4),
            "worst_trade_pct": round(worst * 100, 4),
        }

    def audit_performance(self) -> None:
        """Autonomous Performance Audit: Summarizes recent activity for the Evaluation Heartbeat."""
        summary = self.get_performance_summary()
        if summary.get("total_trades", 0) < 5:
            logger.info("audit_skipped_insufficient_data", count=summary.get("total_trades", 0))
            return

        # Enhanced metrics for the heartbeat
        win_rate = summary["win_rate"]
        avg_pnl = summary["avg_pnl_pct"]
        total = summary["total_trades"]

        logger.info("📊 NEXUS-ULTRA EVALUATION HEARTBEAT",
                    total_trades=total,
                    win_rate=f"{win_rate*100:.1f}%",
                    avg_pnl=f"{avg_pnl:.4f}%",
                    total_pnl=f"${summary['total_pnl_usd']:.2f}")

        # G4: Risk-Control — issue warning if performance degrades
        if win_rate < 0.35 and total >= 20:
             logger.warning("📉 CRITICAL_PERFORMANCE_DEGRADATION", 
                            reason="Win rate below 35% baseline. Market regime shift likely.")

    def log_metric(self, name: str, value: float, details: str = "") -> None:
        """Log a learning/performance metric."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO learning_metrics (metric_name, metric_value, details) VALUES (?, ?, ?)",
                (name, value, details),
            )

    def get_latest_metric(self, name: str) -> dict[str, Any] | None:
        """Return the latest logged metric row for a metric name."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT timestamp, metric_name, metric_value, details
                FROM learning_metrics
                WHERE metric_name = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
        return dict(row) if row else None

    # ── G4: Feature importance tracking ───────────────────────────────────

    def log_feature_importances(
        self, symbol: str, timeframe: str, importances: list[tuple[str, float]]
    ) -> None:
        """Log feature importances from a training run for trend analysis."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_importances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fi_symbol
                ON feature_importances(symbol, timeframe)
            """)
            for feat_name, imp_val in importances:
                conn.execute(
                    "INSERT INTO feature_importances (symbol, timeframe, feature_name, importance) "
                    "VALUES (?, ?, ?, ?)",
                    (symbol, timeframe, feat_name, imp_val),
                )
        logger.info(
            "feature_importances_logged",
            symbol=symbol,
            timeframe=timeframe,
            top_3=[f"{n}={v:.4f}" for n, v in importances[:3]],
        )

    def get_feature_importance_trends(self, n_latest: int = 3) -> dict[str, float]:
        """
        Aggregate feature importances across latest N training runs.
        Returns {feature_name: avg_importance} sorted desc.
        Useful for pruning low-value features.
        """
        with self._conn() as conn:
            # Check if table exists
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_importances'"
            ).fetchone()
            if not table_check:
                return {}

            # Get the latest N distinct (symbol, timeframe, timestamp) combos
            runs = conn.execute("""
                SELECT DISTINCT symbol, timeframe, timestamp
                FROM feature_importances
                ORDER BY timestamp DESC
                LIMIT ?
            """, (n_latest * 5,)).fetchall()  # 5 symbols × n_latest

            if not runs:
                return {}

            # Get all importances from these runs
            timestamps = [r[2] for r in runs]
            placeholders = ",".join(["?"] * len(timestamps))
            rows = conn.execute(
                f"SELECT feature_name, importance FROM feature_importances "
                f"WHERE timestamp IN ({placeholders})",
                timestamps,
            ).fetchall()

        # Aggregate
        from collections import defaultdict
        sums: dict[str, list[float]] = defaultdict(list)
        for name, imp in rows:
            sums[name].append(imp)

        averages = {name: sum(vals) / len(vals) for name, vals in sums.items()}
        return dict(sorted(averages.items(), key=lambda x: x[1], reverse=True))

    def get_low_value_features(self, threshold: float = 0.005) -> list[str]:
        """Return features with average importance below threshold (pruning candidates)."""
        trends = self.get_feature_importance_trends()
        return [name for name, imp in trends.items() if imp < threshold]
