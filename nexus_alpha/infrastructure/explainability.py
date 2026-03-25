"""
Explainability Engine — Per-Trade SHAP Attribution.

Provides interpretable explanations for every trade decision:
- Feature importance via SHAP values
- Causal chain reconstruction
- LLM-generated natural language explanations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from nexus_alpha.logging import get_logger
from nexus_alpha.types import Signal, TradeExplanation

logger = get_logger(__name__)


@dataclass
class FeatureAttribution:
    """SHAP attribution for a single feature."""
    feature_name: str
    shap_value: float
    feature_value: float
    baseline_value: float


class TreeSHAPExplainer:
    """
    SHAP explainer for tree-based and ensemble models.
    Uses KernelSHAP as fallback for neural models.

    In production, wraps the `shap` library. This implementation
    provides a permutation-based approximation for any model.
    """

    def __init__(self, model_predict_fn: Any = None, feature_names: list[str] | None = None):
        self.predict_fn = model_predict_fn
        self.feature_names = feature_names or []
        self._background_data: np.ndarray | None = None

    def set_background(self, data: np.ndarray) -> None:
        """Set background data for SHAP baseline computation."""
        self._background_data = data

    def explain(self, instance: np.ndarray) -> list[FeatureAttribution]:
        """
        Compute SHAP values for a single prediction instance.
        Uses permutation-based approximation.
        """
        if self.predict_fn is None or self._background_data is None:
            return self._mock_explanations(instance)

        n_features = instance.shape[-1]
        baseline_pred = float(np.mean([
            self.predict_fn(bg.reshape(1, -1)) for bg in self._background_data[:50]
        ]))
        instance_pred = float(self.predict_fn(instance.reshape(1, -1)))

        # Permutation SHAP approximation
        shap_values = np.zeros(n_features)
        n_permutations = min(100, n_features * 10)

        for _ in range(n_permutations):
            perm = np.random.permutation(n_features)
            bg_sample = self._background_data[
                np.random.randint(len(self._background_data))
            ].copy()

            for j in range(n_features):
                feat_idx = perm[j]

                # With feature
                x_with = bg_sample.copy()
                for k in range(j + 1):
                    x_with[perm[k]] = instance[perm[k]]
                pred_with = float(self.predict_fn(x_with.reshape(1, -1)))

                # Without feature
                x_without = bg_sample.copy()
                for k in range(j):
                    x_without[perm[k]] = instance[perm[k]]
                pred_without = float(self.predict_fn(x_without.reshape(1, -1)))

                shap_values[feat_idx] += (pred_with - pred_without) / n_permutations

        attributions = []
        for i in range(n_features):
            name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            attributions.append(FeatureAttribution(
                feature_name=name,
                shap_value=float(shap_values[i]),
                feature_value=float(instance[i]),
                baseline_value=baseline_pred,
            ))

        # Sort by absolute SHAP value
        attributions.sort(key=lambda a: abs(a.shap_value), reverse=True)
        return attributions

    def _mock_explanations(self, instance: np.ndarray) -> list[FeatureAttribution]:
        """Generate placeholder explanations when model is not available."""
        n_features = instance.shape[-1] if instance.ndim > 0 else 1
        return [
            FeatureAttribution(
                feature_name=self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                shap_value=0.0,
                feature_value=float(instance[i]) if i < len(instance) else 0.0,
                baseline_value=0.0,
            )
            for i in range(min(n_features, 10))
        ]


class TradeExplainabilityEngine:
    """
    Generates comprehensive explanations for every trade decision.

    Combines:
    1. SHAP feature attributions (quantitative)
    2. Causal chain reconstruction (logical)
    3. Risk check results (compliance)
    4. LLM natural language summary (human-readable)
    """

    def __init__(self, shap_explainer: TreeSHAPExplainer | None = None):
        self.shap_explainer = shap_explainer or TreeSHAPExplainer()

    def explain_trade(
        self,
        trade_id: str,
        signal: Signal,
        feature_vector: np.ndarray,
        risk_checks_passed: list[str],
        risk_checks_failed: list[str],
        causal_chain: list[str] | None = None,
    ) -> TradeExplanation:
        """Generate a full explanation for a trade."""

        # Step 1: SHAP attribution
        attributions = self.shap_explainer.explain(feature_vector)
        top_features = [(a.feature_name, a.shap_value) for a in attributions[:10]]

        # Step 2: Causal chain
        if causal_chain is None:
            causal_chain = self._reconstruct_causal_chain(signal, attributions)

        # Step 3: Generate natural language explanation
        explanation_text = self._generate_explanation(
            signal=signal,
            top_features=top_features,
            causal_chain=causal_chain,
            risk_passed=risk_checks_passed,
            risk_failed=risk_checks_failed,
        )

        return TradeExplanation(
            trade_id=trade_id,
            signal_source=signal.source,
            top_features=top_features,
            causal_chain=causal_chain,
            risk_checks_passed=risk_checks_passed,
            risk_checks_failed=risk_checks_failed,
            llm_explanation=explanation_text,
        )

    def _reconstruct_causal_chain(
        self, signal: Signal, attributions: list[FeatureAttribution]
    ) -> list[str]:
        """Reconstruct the causal decision chain from features."""
        chain: list[str] = []

        direction_label = "LONG" if signal.direction > 0 else "SHORT"
        chain.append(f"Signal: {signal.source} generated {direction_label} signal "
                     f"(strength={signal.direction:.3f}, confidence={signal.confidence:.2f})")

        if attributions:
            top = attributions[0]
            chain.append(f"Primary driver: {top.feature_name} (SHAP={top.shap_value:+.4f}, "
                        f"value={top.feature_value:.4f})")

        if len(attributions) > 1:
            secondary = attributions[1]
            chain.append(f"Secondary driver: {secondary.feature_name} "
                        f"(SHAP={secondary.shap_value:+.4f})")

        if signal.causal_validated:
            chain.append("Causal validation: PASSED — signal has verified causal relationship")
        else:
            chain.append("Causal validation: NOT PERFORMED")

        return chain

    def _generate_explanation(
        self,
        signal: Signal,
        top_features: list[tuple[str, float]],
        causal_chain: list[str],
        risk_passed: list[str],
        risk_failed: list[str],
    ) -> str:
        """Generate natural language explanation (offline mode, no LLM call)."""
        direction = "buy" if signal.direction > 0 else "sell"
        strength = abs(signal.direction)
        confidence = signal.confidence

        lines = [
            f"Trade Decision: {direction.upper()} {signal.symbol}",
            f"Signal source: {signal.source} (strength: {strength:.2f}, confidence: {confidence:.2f})",
            "",
            "Top contributing factors:",
        ]

        for name, shap_val in top_features[:5]:
            direction_arrow = "↑" if shap_val > 0 else "↓"
            lines.append(f"  {direction_arrow} {name}: {shap_val:+.4f}")

        lines.append("")
        lines.append("Decision chain:")
        for step in causal_chain:
            lines.append(f"  → {step}")

        if risk_failed:
            lines.append("")
            lines.append(f"⚠ Risk checks failed: {', '.join(risk_failed)}")

        return "\n".join(lines)
