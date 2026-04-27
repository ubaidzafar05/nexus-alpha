"""
LOB Simulation Environment for Reinforcement Learning execution.
Stochastically models price (GBM) and order book depth (Poisson).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LOBSimEnv(gym.Env):
    """
    Gymnasium environment that simulates a Limit Order Book.
    
    Observation space:
    - [inventory_remaining, time_remaining, bid_depth_ratio, ask_depth_ratio, spread_bps]
    
    Action space:
    - Discrete(5): Change urgency level from -2 to +2 relative to Almgren-Chriss schedule.
    """

    def __init__(
        self,
        total_quantity: float = 1000.0,
        time_horizon_steps: int = 100,
        sigma: float = 0.01,
        initial_price: float = 50000.0,
    ):
        super().__init__()
        self.total_quantity = total_quantity
        self.time_horizon_steps = time_horizon_steps
        self.sigma = sigma
        self.initial_price = initial_price

        # Continuous state: [inventory, time_index, depth_bid, depth_ask, spread]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1, 1, 10, 10, 100]),
            dtype=np.float32,
        )

        # 5 levels of urgency adjustments: -2, -1, 0, +1, +2
        self.action_space = spaces.Discrete(5)

        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = self.total_quantity
        self.current_price = self.initial_price
        
        # Initial LOB state
        self.bid_depth = np.random.uniform(0.5, 2.0)
        self.ask_depth = np.random.uniform(0.5, 2.0)
        self.spread = np.random.uniform(1, 5)  # basis points

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.inventory / self.total_quantity,
            self.current_step / self.time_horizon_steps,
            self.bid_depth,
            self.ask_depth,
            self.spread,
        ], dtype=np.float32)

    def step(self, action: int):
        # Interpret action as urgency adjustment (-2 to +2)
        urgency_adj = action - 2
        
        # Base quantity from AC (linear TWAP for simplicity in sim)
        base_qty = self.total_quantity / self.time_horizon_steps
        trade_qty = base_qty * (1 + 0.2 * urgency_adj)
        trade_qty = min(trade_qty, self.inventory)
        
        # Price evolution (Geometric Brownian Motion)
        dt = 1.0 / self.time_horizon_steps
        self.current_price *= np.exp(
            (0 - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
        )
        
        # Market Impact (Simulates adverse price movement from our own trade)
        # Impact is high when we consume large portion of available depth
        depth_for_side = self.ask_depth if trade_qty > 0 else self.bid_depth
        impact_factor = (trade_qty / (depth_for_side + 1e-6)) * 0.0001
        
        # Real-world Execution Cost: Mid-price + Impact + Half-Spread
        # If we "cross the spread" (Market Order), we pay Spread/2
        # If we "provide liquidity" (Limit Order), we capture a rebate (negative spread/2)
        # Simulation: High urgency (+2 action) = Market Order. Low urgency (-2 action) = Limit Order.
        is_aggressive = urgency_adj >= 1
        is_passive = urgency_adj <= -1
        
        cost_bps = self.spread / 2.0
        if is_passive:
            cost_bps *= -0.2  # Capture small rebate/spread portion
        elif is_aggressive:
            cost_bps *= 1.5   # Pay full spread + extra for speed
            
        executed_price = self.current_price * (1 + impact_factor + (cost_bps / 10000))
        
        # Update state
        self.inventory -= trade_qty
        self.current_step += 1
        
        # Update LOB stochasticity (Simulates mean-reverting liquidity)
        self.bid_depth = np.clip(self.bid_depth + np.random.normal(0, 0.1), 0.1, 5.0)
        self.ask_depth = np.clip(self.ask_depth + np.random.normal(0, 0.1), 0.1, 5.0)
        self.spread = np.clip(self.spread + np.random.normal(0, 0.5), 0.5, 20.0)

        # REWARD LOGIC:
        # 1. Implementation Shortfall (IS): (Executed Value - Benchmark Value) / Benchmark
        # We want to minimize IS => Maximize -IS
        slippage = (executed_price - self.current_price) / self.current_price
        reward = -slippage * 50

        # 2. Early Finish Bonus (Reward for finishing early in high-liquidity regimes)
        if self.inventory <= 0 and self.current_step < self.time_horizon_steps:
            reward += (self.time_horizon_steps - self.current_step) * 0.5

        # 3. Terminal Penalty (Steep penalty for remaining inventory)
        # Non-linear: The closer to the end, the higher the penalty for each unit
        done = self.current_step >= self.time_horizon_steps or self.inventory <= 0
        if done and self.inventory > 0:
            terminal_penalty = (self.inventory / self.total_quantity) ** 1.5 * 100
            reward -= terminal_penalty
            
        return self._get_obs(), float(reward), done, False, {}
