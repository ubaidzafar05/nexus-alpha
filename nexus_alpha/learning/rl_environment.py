"""
Phase 4: RL Training Environment — gym-style wrapper for offline RL training.

Wraps historical OHLCV data into an environment where the TD3 agent can
learn position sizing and entry/exit timing from years of price action.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nexus_alpha.learning.historical_data import build_features, load_ohlcv
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


class TradingEnvironment:
    """
    Gym-style trading environment for RL training on historical data.

    State: feature vector + portfolio state
    Action: position fraction [-1, 1] (negative = short/reduce, positive = long/increase)
    Reward: risk-adjusted PnL
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        initial_balance: float = 10000.0,
        max_position_pct: float = 0.20,
        trading_fee_pct: float = 0.001,
        data_dir: Path | None = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.max_position_pct = max_position_pct
        self.trading_fee = trading_fee_pct

        # Load and prepare data
        df = load_ohlcv(symbol, timeframe, data_dir=data_dir) if data_dir else load_ohlcv(symbol, timeframe)
        self.prices = df["close"].values.astype(np.float64)
        self.features = build_features(df)
        self._feature_cols = [c for c in self.features.columns if not c.startswith("target_")]
        self._feature_matrix = self.features[self._feature_cols].values.astype(np.float32)

        # Align prices with features (features have warmup dropped)
        offset = len(self.prices) - len(self._feature_matrix)
        self.prices = self.prices[offset:]

        self.n_steps = len(self._feature_matrix)
        self.state_dim = len(self._feature_cols) + 5  # features + portfolio state

        # Episode state
        self._step = 0
        self._balance = initial_balance
        self._position = 0.0  # In units of asset
        self._entry_price = 0.0
        self._total_pnl = 0.0
        self._peak_equity = initial_balance
        self._max_drawdown = 0.0
        self._n_trades = 0

    @property
    def equity(self) -> float:
        return self._balance + self._position * self.prices[self._step]

    def reset(self, start_step: int | None = None) -> np.ndarray:
        """Reset environment. Returns initial state."""
        self._step = start_step or 0
        self._balance = self.initial_balance
        self._position = 0.0
        self._entry_price = 0.0
        self._total_pnl = 0.0
        self._peak_equity = self.initial_balance
        self._max_drawdown = 0.0
        self._n_trades = 0
        return self._get_state()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step.
        action: float in [-1, 1] — target position as fraction of equity.
        Returns: (next_state, reward, done, info)
        """
        action = np.clip(action, -1.0, 1.0)

        current_price = self.prices[self._step]
        current_equity = self.equity

        # Convert action to target position value
        target_position_value = action * current_equity * self.max_position_pct
        target_units = target_position_value / current_price if current_price > 0 else 0.0

        # Execute trade
        delta_units = target_units - self._position
        trade_value = abs(delta_units * current_price)
        fee = trade_value * self.trading_fee

        if abs(delta_units) > 1e-8:
            self._balance -= delta_units * current_price + fee
            self._position = target_units
            if abs(self._position) > 1e-8:
                self._entry_price = current_price
            self._n_trades += 1

        # Advance time
        self._step += 1
        done = self._step >= self.n_steps - 1

        if done:
            # Force close position at end
            if abs(self._position) > 1e-8:
                close_value = self._position * self.prices[self._step]
                self._balance += close_value - abs(close_value) * self.trading_fee
                self._position = 0.0

        # Compute reward
        new_equity = self.equity
        pnl_pct = (new_equity - current_equity) / current_equity if current_equity > 0 else 0.0

        # Track drawdown
        self._peak_equity = max(self._peak_equity, new_equity)
        drawdown = (self._peak_equity - new_equity) / self._peak_equity
        self._max_drawdown = max(self._max_drawdown, drawdown)

        reward = self._compute_reward(pnl_pct, drawdown, fee / current_equity)

        info = {
            "equity": new_equity,
            "position": self._position,
            "balance": self._balance,
            "pnl_pct": pnl_pct,
            "drawdown": drawdown,
            "n_trades": self._n_trades,
        }

        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Combine market features with portfolio state."""
        market_features = self._feature_matrix[self._step]

        current_price = self.prices[self._step]
        equity = self.equity
        position_pct = (self._position * current_price) / equity if equity > 0 else 0.0
        unrealized_pnl = 0.0
        if abs(self._position) > 1e-8 and self._entry_price > 0:
            unrealized_pnl = (current_price - self._entry_price) / self._entry_price

        portfolio_state = np.array([
            position_pct,
            unrealized_pnl,
            self._balance / self.initial_balance,
            equity / self.initial_balance,
            self._max_drawdown,
        ], dtype=np.float32)

        return np.concatenate([market_features, portfolio_state])

    def _compute_reward(self, pnl_pct: float, drawdown: float, fee_pct: float) -> float:
        """Risk-adjusted reward: rewards profit, penalizes drawdown and fees."""
        # Volatility-normalized return
        reward = pnl_pct * 100  # Scale up small percentages

        # Drawdown penalty
        if drawdown > 0.05:
            reward -= (drawdown - 0.05) * 10

        # Fee penalty (encourage fewer but better trades)
        reward -= fee_pct * 50

        return float(np.clip(reward, -5.0, 5.0))


def train_rl_agent(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    n_episodes: int = 100,
    episode_length: int = 720,  # 30 days at 1h
    checkpoint_dir: Path = Path("data/checkpoints"),
) -> dict:
    """
    Train TD3 RL agent on historical data.
    Requires torch (optional research dep).
    """
    try:
        import torch
    except ImportError:
        logger.warning("torch_not_installed", msg="RL training requires torch. Install with: pip install -e '.[research]'")
        return {"error": "torch not available"}

    from nexus_alpha.agents.rl_agent import TD3Agent

    env = TradingEnvironment(symbol=symbol, timeframe=timeframe)
    agent = TD3Agent(state_dim=env.state_dim, action_dim=1)

    # Try to load existing checkpoint
    ckpt_path = checkpoint_dir / "rl_agent_latest.pt"
    if ckpt_path.exists():
        agent.load(ckpt_path)
        logger.info("rl_checkpoint_loaded", path=str(ckpt_path))

    best_reward = -float("inf")
    episode_rewards = []

    for ep in range(n_episodes):
        # Random starting point for diversity
        max_start = env.n_steps - episode_length - 1
        if max_start <= 0:
            start = 0
        else:
            start = np.random.randint(0, max_start)

        state = env.reset(start_step=start)
        episode_reward = 0.0

        for step in range(episode_length):
            if env._step >= env.n_steps - 1:
                break

            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(float(action[0]))

            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) > agent.batch_size:
                agent.train_step()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            agent.save(ckpt_path)

        if (ep + 1) % 10 == 0:
            avg_recent = np.mean(episode_rewards[-10:])
            logger.info(
                "rl_training_progress",
                episode=ep + 1,
                avg_reward=f"{avg_recent:.2f}",
                best=f"{best_reward:.2f}",
                equity=f"{info.get('equity', 0):.0f}",
                trades=info.get("n_trades", 0),
            )

    return {
        "episodes": n_episodes,
        "best_reward": round(best_reward, 2),
        "final_avg_reward": round(np.mean(episode_rewards[-10:]), 2),
        "total_avg_reward": round(np.mean(episode_rewards), 2),
        "checkpoint": str(ckpt_path),
    }
