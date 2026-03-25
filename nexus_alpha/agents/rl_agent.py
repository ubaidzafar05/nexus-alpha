"""
RL Trading Agent — Twin Delayed Deep Deterministic Policy Gradient (TD3).

TD3 advantages over vanilla DDPG:
1. Twin critics prevent Q-value overestimation
2. Delayed policy updates (more stable learning)
3. Target policy smoothing (prevents exploitation of critic errors)

State space: 85-dimensional representation of market + portfolio state
Action space: Continuous [-1, 1] = fraction of max position to hold
Reward: Risk-adjusted return (Sharpe-penalized, with tail risk penalty)
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import random
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from nexus_alpha.agents.tournament import BaseAgent
from nexus_alpha.config import RLAgentConfig
from nexus_alpha.logging import get_logger
from nexus_alpha.types import Signal

logger = get_logger(__name__)


# ─── Neural Network Components ───────────────────────────────────────────────


class Actor(nn.Module):
    """Deterministic policy: maps state → action [-1, 1]."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Q-value network: maps (state, action) → Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


# ─── Replay Buffer ───────────────────────────────────────────────────────────


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 1_000_000):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self._buffer), min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


# ─── TD3 Agent ───────────────────────────────────────────────────────────────


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

    Key innovations over DDPG:
    - Twin Q-networks: take min of Q1, Q2 to reduce overestimation
    - Delayed policy updates: update actor less frequently than critics
    - Target policy smoothing: add clipped noise to target actions
    """

    STATE_COMPONENTS = {
        # Market features
        "price_returns_multi_tf": 5,       # 5 timeframe returns
        "volatility_multi_tf": 5,          # 5 timeframe volatilities
        "order_book_imbalance": 3,         # bid/ask imbalance at 3 depths
        "volume_delta": 3,                 # CVD at 3 timeframes
        "rsi_multi_tf": 3,                 # RSI at 3 timeframes
        "macd_features": 3,               # MACD line, signal, histogram
        "bollinger_position": 1,           # Position within bands [-1, 1]
        # World model features
        "world_model_quantiles": 7,        # 7 quantile predictions
        "world_model_uncertainty": 1,      # Epistemic uncertainty
        "regime_probabilities": 5,         # Probability of each regime
        # Portfolio features
        "current_position": 1,             # Current position [-1, 1]
        "unrealized_pnl_normalized": 1,    # PnL / initial capital
        "time_in_position": 1,             # Decay factor
        "portfolio_heat": 1,               # Total risk exposure
        # Sentiment / external
        "sentiment_score": 1,              # Aggregated sentiment [-1, 1]
        "funding_rate": 1,                 # Perpetual funding rate
        # Cross-asset
        "btc_dominance": 1,
        "dxy_change": 1,
        # Microstructure
        "vpin": 1,                         # Volume-Sync. Prob. Informed Trading
        "kyle_lambda": 1,                  # Price impact coefficient
        "spread_normalized": 1,            # Bid-ask spread / mid
        # Time encoding
        "hour_sin": 1,
        "hour_cos": 1,
        "day_of_week_sin": 1,
        "day_of_week_cos": 1,
        # Derived
        "rolling_sharpe": 1,              # Rolling Sharpe of recent trades
        "drawdown_current": 1,            # Current drawdown from peak
        # Technical
        "atr_normalized": 1,              # ATR / price
        "obv_slope": 1,                   # On-Balance Volume direction
    }

    def __init__(self, config: RLAgentConfig | None = None):
        self.config = config or RLAgentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor (policy)
        self.actor = Actor(self.config.state_dim, self.config.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.actor_lr)

        # Twin critics
        self.critic_1 = Critic(self.config.state_dim, self.config.action_dim).to(self.device)
        self.critic_2 = Critic(self.config.state_dim, self.config.action_dim).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=self.config.critic_lr)
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=self.config.critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.config.replay_buffer_size)

        # Tracking
        self._total_steps = 0
        self._training_steps = 0

        logger.info(
            "td3_agent_initialized",
            state_dim=self.config.state_dim,
            device=str(self.device),
        )

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> float:
        """Select action from policy, optionally with exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()[0]

        if add_noise:
            noise = np.random.normal(0, self.config.noise_std)
            action = np.clip(action + noise, -1.0, 1.0)

        return float(action)

    def compute_reward(
        self,
        pnl: float,
        position_size: float,
        volatility: float,
        max_drawdown: float,
    ) -> float:
        """
        Risk-adjusted reward function.

        Components:
        1. PnL (primary)
        2. Risk penalty for large positions in high volatility
        3. Drawdown penalty (exponential)
        4. Turnover penalty (discourages excessive trading)
        """
        # Volatility-normalized PnL
        vol_adjusted_pnl = pnl / (volatility + 1e-8)

        # Position risk penalty
        risk_penalty = -0.1 * abs(position_size) * volatility

        # Drawdown penalty (exponential — punishes deeper drawdowns much more)
        dd_penalty = -2.0 * (np.exp(max_drawdown * 3) - 1) if max_drawdown > 0.02 else 0.0

        reward = vol_adjusted_pnl + risk_penalty + dd_penalty
        return float(reward)

    def train_step(self) -> dict[str, float] | None:
        """One training step of TD3."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.config.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.FloatTensor(np.array([t.action for t in transitions])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in transitions])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        dones = torch.FloatTensor(np.array([float(t.done) for t in transitions])).unsqueeze(1).to(self.device)

        # ─── Critic Update ────────────────────────────────────────────
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(actions) * self.config.noise_std).clamp(-0.5, 0.5)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            # Take minimum of twin target Q-values (reduces overestimation)
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = rewards + self.config.gamma * (1 - dones) * torch.min(target_q1, target_q2)

        # Update critic 1
        current_q1 = self.critic_1(states, actions)
        critic_1_loss = nn.functional.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Update critic 2
        current_q2 = self.critic_2(states, actions)
        critic_2_loss = nn.functional.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        self._training_steps += 1
        metrics = {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
        }

        # ─── Delayed Policy Update ────────────────────────────────────
        if self._training_steps % self.config.policy_delay == 0:
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            metrics["actor_loss"] = actor_loss.item()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)

        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Polyak averaging for target network update."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "training_steps": self._training_steps,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_1_target.load_state_dict(checkpoint["critic_1_target"])
        self.critic_2_target.load_state_dict(checkpoint["critic_2_target"])
        self._training_steps = checkpoint.get("training_steps", 0)


# ─── Tournament-Compatible RL Agent ──────────────────────────────────────────


class RLTradingAgent(BaseAgent):
    """TD3-based RL agent compatible with the tournament framework."""

    def __init__(self, config: RLAgentConfig | None = None, symbol: str = "BTC/USDT"):
        super().__init__(agent_type="rl-td3")
        self.td3 = TD3Agent(config=config)
        self.symbol = symbol
        self._last_state: np.ndarray | None = None
        self._last_action: float = 0.0

    def generate_signal(self, features: dict[str, np.ndarray]) -> Signal | None:
        """Generate signal by querying the TD3 policy."""
        state = self._build_state_vector(features)
        if state is None:
            return None

        action = self.td3.select_action(state, add_noise=not self.td3.config.batch_size)
        self._last_state = state
        self._last_action = action

        # Only signal if action exceeds threshold
        if abs(action) < 0.1:
            return None

        return Signal(
            signal_id=uuid.uuid4().hex[:12],
            source=self.agent_id,
            symbol=self.symbol,
            direction=action,
            confidence=min(abs(action), 1.0),
            timestamp=datetime.utcnow(),
            timeframe="adaptive",
            metadata={"agent_type": "rl-td3"},
        )

    def update(self, market_data: dict) -> None:
        """Store transition and run a training step."""
        pass  # Training happens externally via td3.train_step()

    def _build_state_vector(self, features: dict[str, np.ndarray]) -> np.ndarray | None:
        """
        Assemble the 85-dimensional state vector from feature dict.
        Missing features are zero-filled.
        """
        state = np.zeros(self.td3.config.state_dim, dtype=np.float32)
        idx = 0
        for component, dim in TD3Agent.STATE_COMPONENTS.items():
            if component in features:
                values = np.asarray(features[component]).flatten()[:dim]
                state[idx : idx + len(values)] = values
            idx += dim
        return state
