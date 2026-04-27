"""
World Model — Continually-updating, regime-aware generative model of market dynamics.

Architecture: Temporal Fusion Transformer (TFT) with:
- Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
- Continual learning via gradient episodic memory
- Uncertainty quantification via Monte Carlo Dropout

The World Model predicts the DISTRIBUTION of possible price paths,
their probabilities, and the regime context under which each path is likely.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from nexus_alpha.config import WorldModelConfig
from nexus_alpha.log_config import get_logger
from nexus_alpha.schema_types import MarketRegime, WorldModelOutput

logger = get_logger(__name__)


# ─── Elastic Weight Consolidation ────────────────────────────────────────────


class ElasticWeightConsolidation:
    """
    Preserves critical weights from prior regimes.
    Adds a quadratic penalty to the loss for parameters that moved
    away from their values at the end of previous tasks.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self._fisher: dict[str, torch.Tensor] = {}
        self._optimal_params: dict[str, torch.Tensor] = {}

    def update_fisher(self, data: torch.Tensor, targets: torch.Tensor) -> None:
        """Compute Fisher information matrix after learning a regime."""
        self._optimal_params = {
            n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad
        }
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.eval()
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for batch_x, batch_y in loader:
            self.model.zero_grad()
            output = self.model(batch_x)
            loss = nn.functional.mse_loss(output, batch_y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

        # Normalize
        n_samples = len(dataset)
        self._fisher = {n: f / n_samples for n, f in fisher.items()}

    def compute_penalty(self) -> torch.Tensor:
        """Compute EWC penalty: sum of Fisher-weighted squared parameter deltas."""
        if not self._fisher:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            if n in self._fisher:
                delta = p - self._optimal_params[n]
                penalty += (self._fisher[n] * delta.pow(2)).sum()
        return self.lambda_ewc * penalty


# ─── Episodic Memory Buffer ──────────────────────────────────────────────────


class EpisodicMemoryBuffer:
    """
    Stores representative examples from each regime.
    When learning new data, replays old memories to prevent forgetting.
    """

    def __init__(self, capacity_per_regime: int = 5000):
        self.capacity_per_regime = capacity_per_regime
        self._buffer: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)

    def store(self, features: torch.Tensor, targets: torch.Tensor, regime: str) -> None:
        """Store a batch of (features, targets) for a regime."""
        for i in range(features.shape[0]):
            buf = self._buffer[regime]
            if len(buf) < self.capacity_per_regime:
                buf.append((features[i].detach().clone(), targets[i].detach().clone()))
            else:
                # Reservoir sampling: replace a random existing sample
                idx = np.random.randint(0, len(buf))
                buf[idx] = (features[i].detach().clone(), targets[i].detach().clone())

    def sample_balanced(self, batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample uniformly across all regimes."""
        all_regimes = list(self._buffer.keys())
        if not all_regimes:
            return None

        per_regime = max(1, batch_size // len(all_regimes))
        features_list, targets_list = [], []

        for regime in all_regimes:
            buf = self._buffer[regime]
            if not buf:
                continue
            indices = np.random.choice(len(buf), size=min(per_regime, len(buf)), replace=False)
            for idx in indices:
                features_list.append(buf[idx][0])
                targets_list.append(buf[idx][1])

        if not features_list:
            return None
        return torch.stack(features_list), torch.stack(targets_list)


# ─── Temporal Fusion Transformer (Simplified) ────────────────────────────────


class GatedResidualNetwork(nn.Module):
    """GRN block used in TFT — gated skip connections with ELU activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        output = self.fc2(h)
        gate = self.sigmoid(self.gate(h))
        return self.layer_norm(gate * output + residual)


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over time steps."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + self.dropout(attn_out))


class TFTEncoder(nn.Module):
    """
    Simplified Temporal Fusion Transformer encoder.
    Full TFT has variable selection, static covariate encoders, etc.
    This implements the core architecture for the World Model use case.
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_quantiles: int = 7,
    ):
        super().__init__()
        self.input_projection = GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
        self.temporal_layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": TemporalSelfAttention(hidden_size, num_heads, dropout),
                "grn": GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout),
            })
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(hidden_size, output_quantiles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size) or (batch, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq dim

        h = self.input_projection(x)
        for layer in self.temporal_layers:
            h = layer["attention"](h)
            h = layer["grn"](h)

        # Take last time step output
        out = h[:, -1, :]
        return self.output_head(self.dropout(out))


# ─── World Model ─────────────────────────────────────────────────────────────


class WorldModel:
    """
    A continually-updating, regime-aware generative model of market dynamics.

    The World Model does NOT predict prices.
    It predicts the DISTRIBUTION of possible price paths, their probabilities,
    and the regime context under which each path is likely.
    """

    QUANTILE_INDICES = {0.02: 0, 0.1: 1, 0.25: 2, 0.5: 3, 0.75: 4, 0.9: 5, 0.98: 6}

    def __init__(self, config: WorldModelConfig | None = None):
        self.config = config or WorldModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tft = TFTEncoder(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.attention_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            output_quantiles=len(self.config.output_quantiles),
        ).to(self.device)

        self.ewc = ElasticWeightConsolidation(
            model=self.tft,
            lambda_ewc=self.config.ewc_lambda,
        )

        self.episodic_memory = EpisodicMemoryBuffer(
            capacity_per_regime=self.config.episodic_memory_per_regime,
        )

        self.optimizer = Adam(self.tft.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

        logger.info(
            "world_model_initialized",
            device=str(self.device),
            params=sum(p.numel() for p in self.tft.parameters()),
        )

    def predict_with_uncertainty(
        self,
        features: torch.Tensor,
        regime: MarketRegime,
    ) -> WorldModelOutput:
        """
        Returns full predictive distribution, not a point estimate.
        Every prediction comes with calibrated uncertainty bounds.
        Uses Monte Carlo Dropout for epistemic uncertainty estimation.
        """
        features = features.to(self.device)
        self.tft.train()  # Enable dropout for MC sampling

        predictions = torch.stack([
            self.tft(features) for _ in range(self.config.mc_dropout_samples)
        ])

        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.std(dim=0)

        quantile_preds = {}
        for q, idx in self.QUANTILE_INDICES.items():
            quantile_preds[q] = mean_pred[:, idx].detach().cpu().numpy()

        uncertainty_scalar = epistemic_uncertainty.mean().item()

        return WorldModelOutput(
            quantile_predictions=quantile_preds,
            epistemic_uncertainty=uncertainty_scalar,
            regime_context=regime,
            confidence=max(0.0, 1.0 - uncertainty_scalar),
        )

    def online_update(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        current_regime: MarketRegime,
    ) -> float:
        """
        Continuous online learning with catastrophic forgetting prevention.
        Called every N hours with recent market data.

        Returns the training loss.
        """
        features = features.to(self.device)
        targets = targets.to(self.device)

        # 1. Store current experience in episodic memory
        self.episodic_memory.store(features.cpu(), targets.cpu(), current_regime.value)

        # 2. Sample replay buffer (memories from all regimes)
        replay = self.episodic_memory.sample_balanced(batch_size=256)

        # 3. Build combined training batch
        if replay is not None:
            replay_features, replay_targets = replay
            combined_features = torch.cat([features, replay_features.to(self.device)])
            combined_targets = torch.cat([targets, replay_targets.to(self.device)])
        else:
            combined_features = features
            combined_targets = targets

        # 4. Training step with EWC regularization
        self.tft.train()
        dataset = TensorDataset(combined_features, combined_targets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        total_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            self.optimizer.zero_grad()
            output = self.tft(batch_x)
            prediction_loss = self.criterion(output, batch_y)
            ewc_penalty = self.ewc.compute_penalty()
            loss = prediction_loss + ewc_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tft.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # 5. Update Fisher information matrix for next EWC computation
        self.ewc.update_fisher(combined_features.detach(), combined_targets.detach())

        logger.info(
            "world_model_updated",
            regime=current_regime.value,
            loss=f"{avg_loss:.6f}",
            replay_size=len(combined_features),
        )
        return avg_loss

    def save(self, path: str) -> None:
        torch.save({
            "tft_state": self.tft.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "ewc_fisher": self.ewc._fisher,
            "ewc_params": self.ewc._optimal_params,
        }, path)
        logger.info("world_model_saved", path=path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.tft.load_state_dict(checkpoint["tft_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.ewc._fisher = checkpoint.get("ewc_fisher", {})
        self.ewc._optimal_params = checkpoint.get("ewc_params", {})
        logger.info("world_model_loaded", path=path)
