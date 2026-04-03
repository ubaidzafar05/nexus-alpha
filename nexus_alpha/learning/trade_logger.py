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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from nexus_alpha.logging import get_logger

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

    def log_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        realized_pnl: float,
        reward: float | None = None,
        exit_context: str = "",
    ) -> None:
        """Update a trade with exit info, compute reward, and store exit context."""
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            row = conn.execute(
                "SELECT entry_price, quantity, side, timestamp FROM trades WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()

            if not row:
                logger.warning("trade_not_found_for_close", trade_id=trade_id)
                return

            entry_price, quantity, side, open_ts = row
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
            if side == "sell":
                pnl_pct = -pnl_pct

            try:
                open_dt = datetime.fromisoformat(open_ts)
                hold_seconds = (datetime.utcnow() - open_dt).total_seconds()
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
            """, (exit_price, now, realized_pnl, pnl_pct, hold_seconds, reward, exit_context, trade_id))

        logger.info(
            "trade_closed",
            trade_id=trade_id,
            pnl_pct=f"{pnl_pct:.4f}",
            reward=f"{reward:.4f}",
        )

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

    def get_closed_trades(self, limit: int = 10000) -> list[dict[str, Any]]:
        """Get closed trades for offline training."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Get currently open positions."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_training_data(self, min_trades: int = 50) -> dict[str, np.ndarray] | None:
        """
        Extract feature vectors and rewards from closed trades for RL training.
        Returns None if not enough data yet.
        """
        trades = self.get_closed_trades()
        if len(trades) < min_trades:
            logger.info("insufficient_trades_for_training", count=len(trades), required=min_trades)
            return None

        features = []
        rewards = []
        for trade in trades:
            try:
                fv = json.loads(trade["feature_vector"]) if trade["feature_vector"] else None
                if fv is None:
                    continue
                features.append(fv)
                rewards.append(trade["reward"])
            except (json.JSONDecodeError, TypeError):
                continue

        if len(features) < min_trades:
            return None

        return {
            "features": np.array(features, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "n_trades": len(features),
        }

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

    def log_metric(self, name: str, value: float, details: str = "") -> None:
        """Log a learning/performance metric."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO learning_metrics (metric_name, metric_value, details) VALUES (?, ?, ?)",
                (name, value, details),
            )
