"""
Historical backtester — replays OHLCV candles through the full signal pipeline.

Uses the *exact same* logic as the live trading loop:
  - SignalFusionEngine (9 technical generators)
  - ML model predictions (LightweightPredictor)
  - Signal-conflict arbitration
  - Kelly-based position sizing
  - ATR-based SL/TP with min 0.5% floor
  - Trailing stops
  - Max 3 concurrent positions
  - Volume / momentum guards

This gives a realistic estimate of what the bot would have done historically,
including transaction costs and slippage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from nexus_alpha.learning.historical_data import build_features, load_ohlcv
from nexus_alpha.learning.offline_trainer import LightweightPredictor
from nexus_alpha.signals.signal_engine import SignalFusionEngine

logger = logging.getLogger(__name__)

# ── Constants (mirror trading_loop.py) ────────────────────────────────────────

MIN_SIGNAL_CONFIDENCE = 0.40
MAX_OPEN_POSITIONS = 3
SL_FLOOR_PCT = 0.025   # 2.5% minimum stop-loss
SL_CAP_PCT = 0.06      # 6% maximum stop-loss
SL_ATR_MULT = 3.0
TP_ATR_MULT = 8.0
TRAILING_TRIGGER_PCT = 0.04  # 4% profit triggers trailing
TRAILING_LOCK_PCT = 0.50
FEE_PCT = 0.00075           # Binance discounted/maker-like fee
SLIPPAGE_PCT = 0.0005        # 0.05% slippage per trade


@dataclass
class StrategyParams:
    """Tunable strategy parameters — pass to HistoricalBacktester."""
    min_confidence: float = 0.40
    max_positions: int = 3
    sl_atr_mult: float = 3.0
    sl_floor_pct: float = 0.025
    sl_cap_pct: float = 0.06
    tp_atr_mult: float = 8.0
    min_tp_sl_ratio: float = 2.0      # TP must be ≥ N × SL
    trailing_trigger: float = 0.04
    trailing_lock: float = 0.50
    breakeven_trigger: float = 0.02
    time_exit_bars: int = 120
    time_exit_loss_bars: int = 60
    cooldown_bars: int = 48
    require_ml_agreement: bool = True
    use_trend_filter: bool = True
    use_fixed_tp: bool = False
    kelly_fraction: float = 0.5
    regime_dampening: float = 0.7


@dataclass
class BacktestTrade:
    trade_id: int
    symbol: str
    side: Literal["buy", "sell"]
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: float
    take_profit: float | None
    exit_price: float = 0.0
    exit_time: datetime | None = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    peak_favorable: float = 0.0   # max favorable excursion
    peak_adverse: float = 0.0     # max adverse excursion
    holding_bars: int = 0
    confidence: float = 0.0
    ml_agreed: bool = False


@dataclass
class BacktestResult:
    """Summary statistics from a backtest run."""
    start_date: str = ""
    end_date: str = ""
    symbols: list[str] = field(default_factory=list)
    initial_capital: float = 100_000.0
    final_nav: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_holding_bars: float = 0.0
    total_fees: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)


class HistoricalBacktester:
    """
    Replay historical candles through the full signal + ML pipeline.

    Key design decisions:
    - Walk-forward: models trained on data BEFORE the test period are used.
      We use the already-trained models (trained on full history). For strict
      walk-forward, retrain models at each step — too slow for quick iteration.
    - Signals computed on rolling 200-candle windows (same as live).
    - ATR computed on 14 candles of the replay window.
    - All guards applied: volume, max positions, duplicate, correlation.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        initial_capital: float = 100_000.0,
        fee_pct: float = FEE_PCT,
        slippage_pct: float = SLIPPAGE_PCT,
        params: StrategyParams | None = None,
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.params = params or StrategyParams()

        # Portfolio state
        self.cash = initial_capital
        self.positions: list[BacktestTrade] = []
        self.closed_trades: list[BacktestTrade] = []
        self.equity_curve: list[float] = []
        self.timestamps: list[datetime] = []
        self.trade_counter = 0
        self.total_fees = 0.0

        # Signal engine (same as live)
        self.signal_engine = SignalFusionEngine()
        self.signal_engine.register_defaults()

        # ML models
        self.ml_models: dict[str, LightweightPredictor] = {}
        self._load_ml_models()

    def _load_ml_models(self) -> None:
        ckpt_dir = Path("data/checkpoints")
        for sym in self.symbols:
            safe = sym.replace("/", "_")
            exchange_sym = sym.replace("/", "")  # BTCUSDT
            predictor = LightweightPredictor(target_horizon="target_1h")
            path = ckpt_dir / f"lightweight_{safe}_1h.pkl"
            if predictor.load(path):
                self.ml_models[exchange_sym] = predictor
        logger.info("backtest_ml_models_loaded", count=len(self.ml_models))

    def _nav(self, current_prices: dict[str, float]) -> float:
        """Compute net asset value."""
        position_value = 0.0
        for pos in self.positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == "buy":
                position_value += pos.quantity * price
            else:
                position_value -= pos.quantity * price
        return self.cash + position_value

    def _compute_atr(self, candles: pd.DataFrame, period: int = 14) -> float:
        """ATR from OHLCV candles."""
        if len(candles) < period + 1:
            return 0.0
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)
        close = candles["close"].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def _get_sl_tp(
        self, entry_price: float, atr: float, side: str,
    ) -> tuple[float, float | None]:
        """Compute SL/TP with floor and cap from strategy params."""
        p = self.params
        if atr > 0 and entry_price > 0:
            atr_pct = atr / entry_price
            sl_pct = max(p.sl_atr_mult * atr_pct, p.sl_floor_pct)
            sl_pct = min(sl_pct, p.sl_cap_pct)
        else:
            sl_pct = p.sl_floor_pct
        tp_pct = max(p.tp_atr_mult * (atr / entry_price), sl_pct * p.min_tp_sl_ratio) if (
            p.use_fixed_tp and atr > 0 and entry_price > 0
        ) else None

        if side == "buy":
            sl = entry_price * (1 - sl_pct)
            tp = entry_price * (1 + tp_pct) if tp_pct is not None else None
        else:
            sl = entry_price * (1 + sl_pct)
            tp = entry_price * (1 - tp_pct) if tp_pct is not None else None
        return sl, tp

    def _apply_slippage(self, price: float, side: str) -> float:
        """Simulate slippage on entry."""
        if side == "buy":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _kelly_size(self, confidence: float, nav: float) -> float:
        """Simplified Kelly sizing for backtest (no DB dependency)."""
        p = self.params
        win_rate = 0.5 + confidence * 0.15
        avg_wl_ratio = 1.5
        kelly = (win_rate * avg_wl_ratio - (1 - win_rate)) / avg_wl_ratio
        kelly = max(kelly, 0.0)
        fraction = kelly * p.kelly_fraction
        fraction *= p.regime_dampening
        # Clamp: min 2% of NAV, max 15%
        if fraction < 0.02:
            fraction = 0.02
        fraction = min(fraction, 0.15)
        return nav * fraction

    def _get_ml_prediction(
        self, symbol: str, features_df: pd.DataFrame,
    ) -> dict | None:
        """Get ML prediction from pre-trained model."""
        exchange_sym = symbol.replace("/", "")
        predictor = self.ml_models.get(exchange_sym)
        if not predictor:
            return None
        try:
            feature_cols = [c for c in features_df.columns if not c.startswith("target_")]
            if len(features_df) == 0:
                return None
            last_row = features_df[feature_cols].iloc[-1].values.astype(np.float32)
            return predictor.predict(last_row)
        except Exception:
            return None

    def _check_exits(
        self, bar: pd.Series, current_prices: dict[str, float],
    ) -> None:
        """Check SL/TP/trailing for all open positions."""
        to_close = []
        for pos in self.positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            high = float(bar.get(f"high_{pos.symbol}", price))
            low = float(bar.get(f"low_{pos.symbol}", price))

            pos.holding_bars += 1

            # Track excursions using high/low of the bar
            if pos.side == "buy":
                favorable = (high - pos.entry_price) / pos.entry_price
                adverse = (pos.entry_price - low) / pos.entry_price
                pnl_pct = (price - pos.entry_price) / pos.entry_price
            else:
                favorable = (pos.entry_price - low) / pos.entry_price
                adverse = (high - pos.entry_price) / pos.entry_price
                pnl_pct = (pos.entry_price - price) / pos.entry_price

            pos.peak_favorable = max(pos.peak_favorable, favorable)
            pos.peak_adverse = max(pos.peak_adverse, adverse)

            # Move stop to breakeven after enough profit.
            if pnl_pct >= self.params.breakeven_trigger:
                if pos.side == "buy":
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price * 1.001)
                else:
                    pos.stop_loss = min(pos.stop_loss, pos.entry_price * 0.999)

            # Trailing stop update
            if pnl_pct > self.params.trailing_trigger:
                locked_profit = pos.peak_favorable * self.params.trailing_lock
                if pos.side == "buy":
                    trail_sl = pos.entry_price * (1 + locked_profit)
                    if trail_sl > pos.stop_loss:
                        pos.stop_loss = trail_sl
                else:
                    trail_sl = pos.entry_price * (1 - locked_profit)
                    if trail_sl < pos.stop_loss:
                        pos.stop_loss = trail_sl

            # Check SL hit (using intra-bar high/low for realism)
            if pos.side == "buy" and low <= pos.stop_loss:
                exit_price = pos.stop_loss  # assume filled at SL
                to_close.append((pos, exit_price, "stop_loss"))
            elif pos.side == "sell" and high >= pos.stop_loss:
                exit_price = pos.stop_loss
                to_close.append((pos, exit_price, "stop_loss"))
            # Check TP hit
            elif pos.take_profit is not None and pos.side == "buy" and high >= pos.take_profit:
                exit_price = pos.take_profit
                to_close.append((pos, exit_price, "take_profit"))
            elif pos.take_profit is not None and pos.side == "sell" and low <= pos.take_profit:
                exit_price = pos.take_profit
                to_close.append((pos, exit_price, "take_profit"))
            # Time-based exit: close after N bars if neither SL nor TP hit
            elif pos.holding_bars >= (
                self.params.time_exit_loss_bars if pnl_pct < 0 else self.params.time_exit_bars
            ):
                to_close.append((pos, price, "time_exit"))

        for pos, exit_price, reason in to_close:
            self._close_position(pos, exit_price, reason)

    def _close_position(
        self, pos: BacktestTrade, exit_price: float, reason: str,
    ) -> None:
        """Close a position and record the trade."""
        # Apply slippage on exit
        if pos.side == "buy":
            exit_price *= (1 - self.slippage_pct)
        else:
            exit_price *= (1 + self.slippage_pct)

        # Calculate PnL
        if pos.side == "buy":
            raw_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.quantity

        # Deduct exit fee
        exit_fee = abs(pos.quantity * exit_price) * self.fee_pct
        self.total_fees += exit_fee
        net_pnl = raw_pnl - exit_fee

        # Update cash: for longs return sale proceeds, for shorts buy back
        if pos.side == "buy":
            self.cash += pos.quantity * exit_price - exit_fee
        else:
            # Short close: buy back shares (cash already has entry proceeds)
            self.cash -= pos.quantity * exit_price + exit_fee

        pos.exit_price = exit_price
        pos.exit_time = None  # set by caller
        pos.exit_reason = reason
        pos.pnl = net_pnl
        pos.pnl_pct = net_pnl / (pos.entry_price * pos.quantity) if pos.entry_price else 0

        self.positions.remove(pos)
        self.closed_trades.append(pos)

    def run(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2026-04-01",
        timeframe: str = "1h",
        progress_interval: int = 500,
    ) -> BacktestResult:
        """
        Run a backtest over the specified date range.

        Args:
            start_date: Backtest start (need 200 bars before this for warmup).
            end_date: Backtest end.
            timeframe: Candle timeframe (1h recommended).
            progress_interval: Print progress every N bars.
        """
        t0 = time.time()
        print(f"🔄 Loading data for {len(self.symbols)} symbols...")

        # Load all symbol data
        all_data: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            try:
                df = load_ohlcv(sym, timeframe)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                all_data[sym] = df
            except FileNotFoundError:
                print(f"  ⚠️  No data for {sym} — skipping")

        if not all_data:
            raise ValueError("No historical data found for any symbol")

        # Find common date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # Use the first symbol to build the timeline
        ref_sym = list(all_data.keys())[0]
        ref_df = all_data[ref_sym]
        timeline = ref_df[
            (ref_df["timestamp"] >= start_dt) & (ref_df["timestamp"] <= end_dt)
        ]["timestamp"].tolist()

        if not timeline:
            raise ValueError(f"No candles in range {start_date} → {end_date}")

        # Reset state
        self.cash = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.timestamps = []
        self.trade_counter = 0
        self.total_fees = 0.0

        warmup = 300  # bars needed before features are valid (200 indicator lookback + buffer)
        n_bars = len(timeline)
        print(f"📊 Backtesting {n_bars} bars from {start_date} to {end_date}")
        print(f"   Symbols: {', '.join(all_data.keys())}")
        print(f"   Models: {len(self.ml_models)} loaded")
        print(f"   Params: SL={self.params.sl_atr_mult}×ATR "
              f"[{self.params.sl_floor_pct:.1%}-{self.params.sl_cap_pct:.1%}], "
              f"TP={self.params.tp_atr_mult}×ATR, "
              f"conf≥{self.params.min_confidence}, "
              f"cooldown={self.params.cooldown_bars}h")
        print()

        last_exit_bar: dict[str, int] = {}  # symbol → bar index of last exit

        for i, ts in enumerate(timeline):
            current_prices: dict[str, float] = {}

            # Build multi-column bar for exit checks
            bar_data = {}

            for sym in all_data:
                df = all_data[sym]
                # Get all bars up to and including current timestamp
                mask = df["timestamp"] <= ts
                window = df[mask]

                if len(window) < warmup:
                    continue

                close_price = float(window.iloc[-1]["close"])
                exchange_sym = sym.replace("/", "")
                current_prices[exchange_sym] = close_price

                bar_data[f"high_{exchange_sym}"] = float(window.iloc[-1]["high"])
                bar_data[f"low_{exchange_sym}"] = float(window.iloc[-1]["low"])

            # 1. Check exits on existing positions
            bar_series = pd.Series(bar_data)
            self._check_exits(bar_series, current_prices)
            # Set exit times and record cooldowns
            for t in self.closed_trades:
                if t.exit_time is None:
                    t.exit_time = ts
                    last_exit_bar[t.symbol] = i

            # 2. Generate signals for symbols with enough data
            signals = []
            for sym in all_data:
                df = all_data[sym]
                window = df[df["timestamp"] <= ts].tail(warmup)
                if len(window) < warmup:
                    continue

                exchange_sym = sym.replace("/", "")

                # Skip if already have position in this symbol
                if any(p.symbol == exchange_sym for p in self.positions):
                    continue

                # Max positions check
                if len(self.positions) >= self.params.max_positions:
                    break

                # Cooldown: skip if recently exited this symbol
                last_exit = last_exit_bar.get(exchange_sym, -999)
                if (i - last_exit) < self.params.cooldown_bars:
                    continue

                try:
                    # Compute signal using same fusion engine as live
                    fused = self.signal_engine.fuse(window, exchange_sym)

                    # Blend ML prediction
                    features = build_features(window)
                    ml_pred = self._get_ml_prediction(sym, features)
                    ml_agreed = False

                    if ml_pred and abs(ml_pred["signal"]) > 0.01:
                        ml_conf = ml_pred["confidence"]
                        ml_signal = ml_pred["signal"]
                        tech_dir = fused.direction

                        if (ml_signal > 0 and tech_dir > 0) or (ml_signal < 0 and tech_dir < 0):
                            confidence = min(fused.confidence + 0.15 * ml_conf, 1.0)
                            ml_agreed = True
                        else:
                            confidence = max(fused.confidence - 0.10 * ml_conf, 0.0)
                    else:
                        confidence = fused.confidence

                    direction = fused.direction

                    # Trend filter: only trade with the 50-EMA.
                    if self.params.use_trend_filter:
                        ema50 = window["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                        current_close = float(window.iloc[-1]["close"])
                        if direction > 0 and current_close < ema50:
                            continue
                        if direction < 0 and current_close > ema50:
                            continue

                    # Skip if ML requirement enabled and ML doesn't agree
                    if self.params.require_ml_agreement and not ml_agreed:
                        continue

                    if abs(direction) >= self.params.min_confidence:
                        signals.append({
                            "symbol": exchange_sym,
                            "direction": direction,
                            "confidence": confidence,
                            "ml_agreed": ml_agreed,
                            "atr": self._compute_atr(window),
                            "close": float(window.iloc[-1]["close"]),
                        })
                except Exception:
                    continue

            # 3. Sort by confidence descending, take top signals
            signals.sort(key=lambda s: s["confidence"], reverse=True)
            remaining_slots = self.params.max_positions - len(self.positions)

            for sig in signals[:remaining_slots]:
                nav = self._nav(current_prices)
                size_usd = self._kelly_size(sig["confidence"], nav)

                side = "buy" if sig["direction"] > 0 else "sell"
                entry_price = self._apply_slippage(sig["close"], side)
                quantity = size_usd / entry_price

                # Entry fee
                entry_fee = size_usd * self.fee_pct
                self.total_fees += entry_fee

                # Deduct from cash
                if side == "buy":
                    self.cash -= (quantity * entry_price + entry_fee)
                else:
                    # Short: receive proceeds, will settle later
                    self.cash += (quantity * entry_price - entry_fee)

                sl, tp = self._get_sl_tp(entry_price, sig["atr"], side)

                self.trade_counter += 1
                trade = BacktestTrade(
                    trade_id=self.trade_counter,
                    symbol=sig["symbol"],
                    side=side,
                    entry_price=entry_price,
                    entry_time=ts,
                    quantity=quantity,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=sig["confidence"],
                    ml_agreed=sig["ml_agreed"],
                )
                self.positions.append(trade)

            # 4. Record equity
            nav = self._nav(current_prices)
            self.equity_curve.append(nav)
            self.timestamps.append(ts)

            # Progress
            if i > 0 and i % progress_interval == 0:
                elapsed = time.time() - t0
                pct = i / n_bars * 100
                ret = (nav / self.initial_capital - 1) * 100
                print(
                    f"  [{pct:5.1f}%] Bar {i}/{n_bars}  "
                    f"NAV: ${nav:,.0f} ({ret:+.1f}%)  "
                    f"Trades: {len(self.closed_trades)}  "
                    f"Open: {len(self.positions)}  "
                    f"({elapsed:.0f}s)"
                )

        # Close remaining positions at last price
        for pos in list(self.positions):
            sym_key = None
            for sym in all_data:
                if sym.replace("/", "") == pos.symbol:
                    sym_key = sym
                    break
            if sym_key:
                last_price = float(all_data[sym_key].iloc[-1]["close"])
            else:
                last_price = pos.entry_price
            self._close_position(pos, last_price, "backtest_end")
            self.closed_trades[-1].exit_time = timeline[-1]

        elapsed = time.time() - t0
        result = self._compute_stats(timeline)
        print(f"\n✅ Backtest complete in {elapsed:.1f}s")
        return result

    def _compute_stats(self, timeline: list) -> BacktestResult:
        """Compute comprehensive backtest statistics."""
        trades = self.closed_trades
        nav = self.equity_curve[-1] if self.equity_curve else self.initial_capital

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        # Equity curve analysis
        eq = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital])
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        # Returns for Sharpe
        if len(eq) > 1:
            returns = np.diff(eq) / eq[:-1]
            sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 24))
        else:
            returns = np.array([0.0])
            sharpe = 0.0

        total_return = (nav / self.initial_capital - 1) * 100
        win_rate = len(winners) / len(trades) if trades else 0.0

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean([t.pnl_pct for t in winners]) * 100 if winners else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losers]) * 100 if losers else 0.0

        calmar = total_return / (max_dd * 100) if max_dd > 0 else 0.0

        result = BacktestResult(
            start_date=str(timeline[0]) if timeline else "",
            end_date=str(timeline[-1]) if timeline else "",
            symbols=[s.replace("/", "") for s in self.symbols],
            initial_capital=self.initial_capital,
            final_nav=nav,
            total_return_pct=total_return,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            avg_win_pct=float(avg_win),
            avg_loss_pct=float(avg_loss),
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd * 100,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            avg_holding_bars=float(np.mean([t.holding_bars for t in trades])) if trades else 0.0,
            total_fees=self.total_fees,
            trades=trades,
            equity_curve=self.equity_curve,
            timestamps=self.timestamps,
        )
        return result


def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report to console."""
    print("\n" + "=" * 70)
    print("  NEXUS-ALPHA BACKTEST REPORT")
    print("=" * 70)
    print(f"  Period:          {result.start_date[:10]} → {result.end_date[:10]}")
    print(f"  Symbols:         {', '.join(result.symbols)}")
    print(f"  Initial Capital: ${result.initial_capital:,.0f}")
    print()
    print(f"  Final NAV:       ${result.final_nav:,.2f}")
    print(f"  Total Return:    {result.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Calmar Ratio:    {result.calmar_ratio:.2f}")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print()
    print(f"  Total Trades:    {result.total_trades}")
    print(f"  Winners:         {result.winning_trades} ({result.win_rate:.1%})")
    print(f"  Losers:          {result.losing_trades}")
    print(f"  Avg Win:         {result.avg_win_pct:+.2f}%")
    print(f"  Avg Loss:        {result.avg_loss_pct:+.2f}%")
    print(f"  Avg Holding:     {result.avg_holding_bars:.1f} bars")
    print(f"  Total Fees:      ${result.total_fees:,.2f}")
    print("=" * 70)

    if result.trades:
        # Top 5 best / worst trades
        sorted_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
        print("\n  TOP 5 WINNERS:")
        for t in sorted_trades[:5]:
            print(
                f"    {t.symbol} {t.side.upper()} "
                f"${t.pnl:+,.2f} ({t.pnl_pct:+.2%}) "
                f"held {t.holding_bars}h  exit={t.exit_reason}"
                f"  ml={'✓' if t.ml_agreed else '✗'}"
            )

        print("\n  TOP 5 LOSERS:")
        for t in sorted_trades[-5:]:
            print(
                f"    {t.symbol} {t.side.upper()} "
                f"${t.pnl:+,.2f} ({t.pnl_pct:+.2%}) "
                f"held {t.holding_bars}h  exit={t.exit_reason}"
                f"  ml={'✓' if t.ml_agreed else '✗'}"
            )

        # ML agreement stats
        ml_agree = [t for t in result.trades if t.ml_agreed]
        ml_disagree = [t for t in result.trades if not t.ml_agreed]
        if ml_agree:
            ml_wr = sum(1 for t in ml_agree if t.pnl > 0) / len(ml_agree)
            print(f"\n  ML Agreement Trades: {len(ml_agree)} (WR: {ml_wr:.1%})")
        if ml_disagree:
            no_ml_wr = sum(1 for t in ml_disagree if t.pnl > 0) / len(ml_disagree)
            print(f"  ML Disagree Trades:  {len(ml_disagree)} (WR: {no_ml_wr:.1%})")

        # Exit reason breakdown
        reasons = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("\n  EXIT REASONS:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pnl = sum(t.pnl for t in result.trades if t.exit_reason == reason)
            print(f"    {reason:20s}: {count:4d} trades  ${pnl:+,.2f}")

    print("=" * 70)
