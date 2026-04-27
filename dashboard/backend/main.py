import logging

_logger = logging.getLogger("nexus.dashboard")

import json
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="NEXUS-ALPHA Dashboard API")

# Enable CORS for the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
REGISTRY_PATH = BASE_DIR / "data/tournament/registry.json"
TRADES_DB_PATH = BASE_DIR / "data/trade_logs/trades.db"
OHLCV_DIR = BASE_DIR / "data/ohlcv"
CONTROL_STATE_PATH = BASE_DIR / "data/bot_control.json"
LIVE_STATE_PATH = BASE_DIR / "data/live_state.json"


def _read_live_state() -> dict:
    """Read atomic live-state snapshot written by TradingLoop each cycle."""
    if not LIVE_STATE_PATH.exists():
        return {}
    try:
        with open(LIVE_STATE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        _logger.warning("live_state read failed: %s", e)
        return {}


def _heartbeat_age_seconds(state: dict) -> Optional[float]:
    """Seconds since last heartbeat, or None if unknown."""
    hb = state.get("heartbeat")
    if not hb:
        return None
    try:
        dt = datetime.fromisoformat(hb.replace("Z", ""))
        return (datetime.utcnow() - dt).total_seconds()
    except Exception:
        return None

# --- Models ---

class Trade(BaseModel):
    trade_id: str
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    realized_pnl_pct: float
    status: str

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# --- Symbols ---
def normalize_symbol(symbol: str, target: str = "ui") -> str:
    """Standardize symbol formats across DB, Files, and UI."""
    clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
    if target == "db":
        return clean # BTCUSDT
    if target == "file":
        # Handle cases like BTCUSDT -> BTC_USDT (assuming 3-letter base for now or common pairs)
        if clean.endswith("USDT"):
            return f"{clean[:-4]}_{clean[-4:]}"
        return clean
    return f"{clean[:-4]}/{clean[-4:]}" if clean.endswith("USDT") else symbol

# --- Endpoints ---

@app.get("/api/health")
def health_check():
    """System health monitor."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "registry_synced": REGISTRY_PATH.exists(),
        "database_synced": TRADES_DB_PATH.exists()
    }

@app.get("/api/registry")
def get_registry():
    """Fetch current model registry with high-availability fallback."""
    if not REGISTRY_PATH.exists():
        return {
            "champion": {"id": "WAITING_FOR_DATA", "metrics": {"equity_curve": [10000, 10000]}},
            "candidates": [],
            "past_champions": []
        }
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

@app.get("/api/trades")
def get_trades(symbol: Optional[str] = None, limit: int = 100):
    """Fetch trade history from SQLite with normalization support."""
    if not TRADES_DB_PATH.exists():
        return []
    
    try:
        conn = sqlite3.connect(TRADES_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if symbol:
            db_symbol = normalize_symbol(symbol, target="db")
            cursor.execute("SELECT * FROM trades WHERE symbol = ? OR symbol = ? ORDER BY timestamp DESC LIMIT ?", 
                           (symbol, db_symbol, limit))
        else:
            cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
            
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        _logger.error("Database query error: %s", e)
        return []

@app.get("/api/candles/{symbol}")
def get_candles(symbol: str, timeframe: str = "1h"):
    """Stream OHLCV data with strict chronological ordering and normalization."""
    db_symbol = normalize_symbol(symbol, target="db")
    file_symbol = normalize_symbol(symbol, target="file")
    
    # Try multiple naming conventions
    paths = [
        OHLCV_DIR / f"{file_symbol}_{timeframe}.parquet",
        OHLCV_DIR / f"{db_symbol}_{timeframe}.parquet",
        OHLCV_DIR / f"binance_{db_symbol}_{timeframe}.parquet",
        OHLCV_DIR / f"binance_{file_symbol}_{timeframe}.parquet"
    ]
    
    parquet_path = None
    for p in paths:
        if p.exists():
            parquet_path = p
            break
            
    if not parquet_path:
        return {"error": f"No data found for {symbol} ({file_symbol}) {timeframe}"}
    
    try:
        df = pd.read_parquet(parquet_path)
        # STABILIZATION: Ensure strictly chronological order (Crucial for rendering)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        
        # Limit to last 1500 candles for a dense view
        df = df.tail(1500)
        
        candles = []
        for _, row in df.iterrows():
            # Lightweight Charts expects time as UTC Unix timestamp (seconds)
            ts = int(row["timestamp"].timestamp())
            candles.append({
                "time": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0))
            })
        return candles
    except Exception as e:
        _logger.error("Candle processing error: %s", e)
        return {"error": "Internal processing failure"}

@app.get("/api/analytics")
def get_analytics():
    """Calculate key performance metrics."""
    if not TRADES_DB_PATH.exists():
        return {"error": "No trade data"}
        
    conn = sqlite3.connect(TRADES_DB_PATH)
    df = pd.read_sql_query("SELECT realized_pnl_pct, status FROM trades WHERE status='closed'", conn)
    conn.close()
    
    if df.empty:
        return {"win_rate": 0, "profit_factor": 0, "max_drawdown": 0, "total_trades": 0}
        
    pnl = df["realized_pnl_pct"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    
    win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 1.0
    
    # Simple cumulative drawdown
    cum_returns = (1 + pnl / 100).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = float(drawdown.min() * 100)
    
    return {
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_drawdown, 2),
        "total_trades": len(pnl),
        "avg_trade_pnl": round(float(pnl.mean()), 2)
    }

@app.get("/api/v4/telemetry")
def get_v4_telemetry():
    """Fetch deep V4 system metrics from learning database."""
    if not TRADES_DB_PATH.exists():
        return {"error": "No trade data"}
        
    conn = sqlite3.connect(TRADES_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get latest metrics for pruning, slippage, and microstructure
    cursor.execute("""
        SELECT metric_name, metric_value, timestamp 
        FROM learning_metrics 
        WHERE metric_name IN ('pruning_rate_causal', 'avg_slippage_bps', 'tick_vpin_max', 'ofi_l2_max')
        ORDER BY timestamp DESC LIMIT 200
    """)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

@app.get("/api/swarm/genealogy")
def get_swarm_genealogy():
    """Fetch the full swarm hierarchy with lineage and genetic drift metadata."""
    registry_path = BASE_DIR / "data/tournament/swarm_registry.json"
    if not registry_path.exists():
        return {"error": "Swarm state not yet persisted", "swarm": []}
    
    try:
        with open(registry_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to read swarm state: {str(e)}", "swarm": []}

# --- Control Endpoints ---

@app.get("/api/control/status")
def get_control_status():
    """Get the current manual control state of the bot."""
    if not CONTROL_STATE_PATH.exists():
        return {"paused": False, "market_exit_pending": False}
    with open(CONTROL_STATE_PATH, "r") as f:
        return json.load(f)

@app.post("/api/control/pause")
def pause_bot():
    """Trigger manual pause."""
    state = get_control_status()
    state["paused"] = True
    with open(CONTROL_STATE_PATH, "w") as f:
        json.dump(state, f)
    return state

@app.post("/api/control/resume")
def resume_bot():
    """Trigger manual resume."""
    state = get_control_status()
    state["paused"] = False
    with open(CONTROL_STATE_PATH, "w") as f:
        json.dump(state, f)
    return state

@app.post("/api/control/market-exit")
def trigger_market_exit():
    """Trigger immediate market exit for all open positions."""
    state = get_control_status()
    state["market_exit_pending"] = True
    with open(CONTROL_STATE_PATH, "w") as f:
        json.dump(state, f)
    return state

# --- Live-State Endpoints (backed by data/live_state.json) ---

@app.get("/api/bot/status")
def get_bot_status():
    """Bot heartbeat, cycle count, paused state, circuit-breaker level, freshness."""
    state = _read_live_state()
    age = _heartbeat_age_seconds(state)
    control = {}
    if CONTROL_STATE_PATH.exists():
        try:
            with open(CONTROL_STATE_PATH, "r") as f:
                control = json.load(f)
        except Exception:
            control = {}
    return {
        "heartbeat": state.get("heartbeat"),
        "heartbeat_age_s": age,
        "fresh": (age is not None and age < 30),
        "cycle_counter": state.get("cycle_counter", 0),
        "cycle_interval_s": state.get("cycle_interval_s"),
        "paused": bool(state.get("paused") or control.get("paused")),
        "blind_halt": bool(state.get("blind_halt")),
        "cb_level": state.get("cb_level", 0),
        "market_exit_pending": bool(control.get("market_exit_pending")),
    }


@app.get("/api/portfolio")
def get_portfolio():
    """Current NAV, cash, realized PnL, leverage, open positions."""
    state = _read_live_state()
    pf = state.get("portfolio") or {}
    # Fallback to portfolio_state.json if live-state missing
    if not pf:
        pf_path = BASE_DIR / "data/trade_logs/portfolio_state.json"
        if pf_path.exists():
            try:
                with open(pf_path, "r") as f:
                    raw = json.load(f)
                pf = {
                    "nav": raw.get("nav", 0.0),
                    "cash": raw.get("cash", 0.0),
                    "realized_pnl": raw.get("total_realized_pnl", 0.0),
                    "position_count": len(raw.get("positions", []) or []),
                    "leverage": raw.get("leverage", 0.0),
                    "positions": raw.get("positions", []) or [],
                }
            except Exception:
                pass
    return pf or {
        "nav": 0.0, "cash": 0.0, "realized_pnl": 0.0,
        "position_count": 0, "leverage": 0.0, "positions": [],
    }


@app.get("/api/regime")
def get_regime():
    """Current regime classification + changepoint probability."""
    state = _read_live_state()
    return state.get("regime") or {"name": "unknown", "changepoint_probability": 0.0}


@app.get("/api/signals/recent")
def get_recent_signals(limit: int = 30):
    """Rolling buffer of recent fused signals (symbol, direction, confidence, regime)."""
    state = _read_live_state()
    sigs = state.get("recent_signals") or []
    return list(reversed(sigs))[:limit]


@app.get("/api/microstructure")
def get_microstructure():
    """Latest microstructure flags (VPIN, OFI)."""
    state = _read_live_state()
    return state.get("microstructure") or {"vpin_max": 0.0, "ofi_max": 0.0}


@app.get("/api/internal/performance")
def get_performance():
    """Win/Loss counts + aggregate PnL (satisfies legacy UI endpoint)."""
    if not TRADES_DB_PATH.exists():
        return {"wins": 0, "losses": 0, "break_even": 0, "total_pnl_pct": 0.0}
    try:
        conn = sqlite3.connect(TRADES_DB_PATH)
        df = pd.read_sql_query(
            "SELECT realized_pnl_pct FROM trades WHERE status='closed'", conn
        )
        conn.close()
    except Exception as e:
        _logger.error("performance query error: %s", e)
        return {"wins": 0, "losses": 0, "break_even": 0, "total_pnl_pct": 0.0}
    if df.empty:
        return {"wins": 0, "losses": 0, "break_even": 0, "total_pnl_pct": 0.0}
    pnl = df["realized_pnl_pct"]
    return {
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "break_even": int((pnl == 0).sum()),
        "total_pnl_pct": round(float(pnl.sum()), 2),
        "total_trades": int(len(pnl)),
    }


@app.get("/api/analytics/24h")
def get_analytics_24h():
    """Trade counts + PnL restricted to the last 24 hours."""
    if not TRADES_DB_PATH.exists():
        return {"trades_24h": 0, "pnl_24h_pct": 0.0, "wins_24h": 0, "losses_24h": 0}
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    try:
        conn = sqlite3.connect(TRADES_DB_PATH)
        df = pd.read_sql_query(
            "SELECT realized_pnl_pct, timestamp FROM trades "
            "WHERE status='closed' AND timestamp >= ?",
            conn, params=(cutoff,)
        )
        conn.close()
    except Exception as e:
        _logger.error("24h analytics error: %s", e)
        return {"trades_24h": 0, "pnl_24h_pct": 0.0, "wins_24h": 0, "losses_24h": 0}
    if df.empty:
        return {"trades_24h": 0, "pnl_24h_pct": 0.0, "wins_24h": 0, "losses_24h": 0}
    pnl = df["realized_pnl_pct"]
    return {
        "trades_24h": int(len(pnl)),
        "pnl_24h_pct": round(float(pnl.sum()), 2),
        "wins_24h": int((pnl > 0).sum()),
        "losses_24h": int((pnl < 0).sum()),
    }


@app.get("/api/equity-curve")
def get_equity_curve(limit: int = 500):
    """Derive equity curve from trade journal when registry lacks one."""
    if not TRADES_DB_PATH.exists():
        return {"equity_curve": [10000.0], "source": "empty"}
    try:
        conn = sqlite3.connect(TRADES_DB_PATH)
        df = pd.read_sql_query(
            "SELECT realized_pnl_pct FROM trades "
            "WHERE status='closed' ORDER BY timestamp ASC",
            conn
        )
        conn.close()
    except Exception as e:
        _logger.error("equity curve error: %s", e)
        return {"equity_curve": [10000.0], "source": "error"}
    if df.empty:
        return {"equity_curve": [10000.0], "source": "empty"}
    pnl = df["realized_pnl_pct"].fillna(0.0) / 100.0
    curve = (1 + pnl).cumprod() * 10000.0
    curve_list = [10000.0] + [round(float(v), 2) for v in curve.tolist()]
    if len(curve_list) > limit:
        curve_list = curve_list[-limit:]
    return {"equity_curve": curve_list, "source": "trades_db", "n": len(curve_list)}


@app.post("/api/internal/notify_trade")
async def notify_trade(trade: Trade):
    """Internal hook for the bot to notify UI of new trade events."""
    await manager.broadcast({
        "type": "NEW_TRADE",
        "data": trade.dict(),
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"status": "broadcast_sent"}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Background Monitor ---
async def monitor_registry():
    last_modified = 0
    while True:
        if REGISTRY_PATH.exists():
            mtime = REGISTRY_PATH.stat().st_mtime
            if mtime > last_modified:
                last_modified = mtime
                try:
                    with open(REGISTRY_PATH, "r") as f:
                        data = json.load(f)
                        await manager.broadcast({
                            "type": "REGISTRY_UPDATE",
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                except Exception:
                    pass
        await asyncio.sleep(5)

async def monitor_live_state():
    """Watch data/live_state.json and broadcast HEARTBEAT on every change."""
    last_mtime = 0.0
    while True:
        try:
            if LIVE_STATE_PATH.exists():
                mtime = LIVE_STATE_PATH.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    state = _read_live_state()
                    await manager.broadcast({
                        "type": "HEARTBEAT",
                        "data": {
                            "heartbeat": state.get("heartbeat"),
                            "cycle_counter": state.get("cycle_counter"),
                            "paused": state.get("paused"),
                            "blind_halt": state.get("blind_halt"),
                            "cb_level": state.get("cb_level"),
                            "portfolio": state.get("portfolio"),
                            "regime": state.get("regime"),
                            "microstructure": state.get("microstructure"),
                            "metrics": state.get("metrics"),
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    })
        except Exception as e:
            _logger.warning("monitor_live_state error: %s", e)
        await asyncio.sleep(2)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_registry())
    asyncio.create_task(monitor_live_state())
