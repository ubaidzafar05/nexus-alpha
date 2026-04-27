"""
Meta-Learning Controller — MAML-based rapid regime adaptation.

Trains the system to adapt to new market regimes in 10-50 gradient steps
instead of thousands. Instead of training a model FOR each regime,
trains a model that can RAPIDLY BECOME GOOD at any regime.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam

from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeTask:
    """A meta-learning task = a regime's data split into support/query sets."""
    regime: str
    support_features: torch.Tensor  # Few-shot data for inner loop
    support_targets: torch.Tensor
    query_features: torch.Tensor    # Evaluation data for outer loop
    query_targets: torch.Tensor


class MetaLearningController:
    """
    MAML-based meta-learning: trains the system to adapt to new market
    regimes in a small number of gradient steps.

    Algorithm:
    1. Sample a batch of regime tasks
    2. For each task: clone model, do k inner-loop updates on support set
    3. Evaluate adapted model on query set
    4. Backprop query loss through inner-loop to update meta-parameters
    """

    def __init__(
        self,
        base_model: nn.Module,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
    ):
        self.meta_model = base_model
        self.meta_optimizer = Adam(self.meta_model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.criterion = nn.MSELoss()

    def meta_train_step(self, task_batch: list[RegimeTask]) -> float:
        """
        One step of MAML:
        For each task in the batch:
          - Clone model parameters
          - Perform `inner_steps` gradient updates on the support set (inner loop)
          - Compute loss on the query set with the adapted parameters
        Accumulate query losses and update meta-parameters (outer loop).

        Returns the average query loss.
        """
        self.meta_optimizer.zero_grad()
        total_query_loss = torch.tensor(0.0)

        for task in task_batch:
            # Clone model for inner-loop adaptation
            adapted_params = {
                name: param.clone() for name, param in self.meta_model.named_parameters()
            }

            # Inner loop: adapt to this regime's support set
            for _step in range(self.inner_steps):
                support_pred = self._forward_with_params(
                    task.support_features, adapted_params
                )
                support_loss = self.criterion(support_pred, task.support_targets)

                # Compute gradients w.r.t. adapted params
                grads = torch.autograd.grad(
                    support_loss,
                    adapted_params.values(),
                    create_graph=True,  # Need second-order gradients for MAML
                )

                # Update adapted params
                adapted_params = {
                    name: param - self.inner_lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

            # Outer loop: evaluate on query set
            query_pred = self._forward_with_params(task.query_features, adapted_params)
            query_loss = self.criterion(query_pred, task.query_targets)
            total_query_loss = total_query_loss + query_loss

        # Meta update
        avg_query_loss = total_query_loss / len(task_batch)
        avg_query_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), max_norm=1.0)
        self.meta_optimizer.step()

        logger.info(
            "meta_train_step",
            n_tasks=len(task_batch),
            avg_query_loss=f"{avg_query_loss.item():.6f}",
        )

        return avg_query_loss.item()

    def adapt_to_new_regime(
        self,
        support_features: torch.Tensor,
        support_targets: torch.Tensor,
        n_steps: int | None = None,
    ) -> nn.Module:
        """
        Rapidly adapt the meta-model to a new regime using few-shot data.
        Returns a new model specialized for the current regime.
        """
        steps = n_steps or self.inner_steps * 4  # More steps for real adaptation
        adapted_model = copy.deepcopy(self.meta_model)
        optimizer = Adam(adapted_model.parameters(), lr=self.inner_lr)

        adapted_model.train()
        for step in range(steps):
            optimizer.zero_grad()
            pred = adapted_model(support_features)
            loss = self.criterion(pred, support_targets)
            loss.backward()
            optimizer.step()

        logger.info(
            "regime_adaptation_complete",
            n_steps=steps,
            final_loss=f"{loss.item():.6f}",
        )

        return adapted_model

    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Functional forward pass using custom parameters (not model's stored params).
        Required for MAML's inner loop to be differentiable.
        """
        # Load temporary parameters
        original_params = {}
        for name, param in self.meta_model.named_parameters():
            original_params[name] = param.data.clone()
            param.data = params[name]

        output = self.meta_model(x)

        # Restore original parameters
        for name, param in self.meta_model.named_parameters():
            param.data = original_params[name]

        return output
