"""Phase 1 foundation runner: train/evaluate core intelligence modules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nexus_alpha.config import NexusConfig
from nexus_alpha.core.causal_validator import CausalSignalValidator
from nexus_alpha.core.regime_oracle import RegimeOracle
from nexus_alpha.infrastructure.explainability import TreeSHAPExplainer
from nexus_alpha.intelligence.contracts import (
    CausalVerdict,
    ExplainabilitySummary,
    IntelligenceOutput,
    PredictionBand,
    UncertaintyMetrics,
)
from nexus_alpha.schema_types import MarketRegime


@dataclass(frozen=True)
class FoundationRunArtifacts:
    report_path: Path
    checkpoint_path: Path


class _NumpyWorldModelFallback:
    """Fallback backend when torch/world-model dependencies are unavailable."""

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None

    def train(self, features: np.ndarray, targets: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(features, targets[:, 3], rcond=None)
        self._weights = beta
        residual = targets[:, 3] - features @ beta
        return float(np.mean(residual**2))

    def predict(self, features: np.ndarray) -> dict[float, np.ndarray]:
        if self._weights is None:
            raise RuntimeError("world_model_not_trained")
        median = features @ self._weights
        return {
            0.02: median - 0.03,
            0.1: median - 0.02,
            0.25: median - 0.01,
            0.5: median,
            0.75: median + 0.01,
            0.9: median + 0.02,
            0.98: median + 0.03,
        }

    def save(self, path: Path) -> None:
        payload = {"weights": self._weights.tolist() if self._weights is not None else []}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def uncertainty(self) -> float:
        return 0.02


class IntelligenceFoundationRunner:
    """Reproducible training/eval harness for Phase 1 modules."""

    def __init__(self, config: NexusConfig | None = None):
        self._config = config or NexusConfig()
        self._regime_oracle = RegimeOracle(lookback_window=200)
        self._causal = CausalSignalValidator()
        self._world_backend = self._build_world_backend()

    def run(
        self,
        output_dir: Path,
        symbol: str = "BTCUSDT",
        seed: int = 42,
        n_samples: int = 512,
    ) -> tuple[IntelligenceOutput, FoundationRunArtifacts]:
        np.random.seed(seed)
        if isinstance(self._world_backend, dict):
            self._world_backend["torch"].manual_seed(seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        features, targets, returns = self._build_training_data(
            n_samples=n_samples,
            input_size=self._config.world_model.input_size,
        )
        loss = self._train_world_model(features, targets)
        checkpoint_path = output_dir / "world_model.pt"
        self._save_world_model(checkpoint_path)

        regime_state = self._regime_oracle.update(returns)
        quantiles, uncertainty = self._predict_world_model(features[:1], regime_state.regime)
        causal_result = self._run_causal_validation(features, returns)
        explain_summary = self._run_explainability(features, returns)

        output = IntelligenceOutput(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            prediction=self._to_prediction_band(quantiles),
            uncertainty=UncertaintyMetrics(
                epistemic=uncertainty,
                confidence=max(0.0, 1.0 - uncertainty),
            ),
            regime=regime_state.regime,
            regime_confidence=regime_state.confidence,
            causal=CausalVerdict(
                is_causal=causal_result.is_causal,
                effect_size=causal_result.causal_effect,
                p_value=causal_result.p_value,
                information_coefficient=causal_result.information_coefficient,
                granger_p_value=causal_result.granger_p_value,
            ),
            explainability=explain_summary,
        )
        report_path = output_dir / "intelligence_foundation_report.json"
        report_payload = {
            "output": output.model_dump(mode="json"),
            "metrics": {"world_model_loss": loss, "samples": n_samples, "seed": seed},
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        artifacts = FoundationRunArtifacts(
            report_path=report_path,
            checkpoint_path=checkpoint_path,
        )
        return output, artifacts

    def _build_training_data(
        self,
        n_samples: int,
        input_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.random.normal(0, 1, (n_samples, input_size)).astype(np.float32)
        signal = 0.03 * x[:, 0] + 0.02 * x[:, 1] - 0.015 * x[:, 2]
        noise = np.random.normal(0, 0.01, n_samples).astype(np.float32)
        returns = (signal + noise).astype(np.float32)
        quantiles = np.column_stack([
            returns - 0.03,
            returns - 0.02,
            returns - 0.01,
            returns,
            returns + 0.01,
            returns + 0.02,
            returns + 0.03,
        ]).astype(np.float32)
        return x, quantiles, returns

    def _run_causal_validation(self, features: np.ndarray, returns: np.ndarray):
        index = pd.date_range(
            end=datetime.utcnow(),
            periods=features.shape[0],
            freq="min",
        )
        signal_series = pd.Series(
            features[:, 0],
            index=index,
            name="feature_0_signal",
        )
        forward = pd.Series(returns, index=index, name="forward_return")
        confounders = pd.DataFrame(
            {
                "feature_1": features[:, 1],
                "feature_2": features[:, 2],
            },
            index=index,
        )
        return self._causal.validate_signal(signal_series, forward, confounders=confounders)

    def _run_explainability(
        self,
        features: np.ndarray,
        returns: np.ndarray,
    ) -> ExplainabilitySummary:
        beta, *_ = np.linalg.lstsq(features, returns, rcond=None)

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            return arr @ beta

        explainer = TreeSHAPExplainer(
            model_predict_fn=predict_fn,
            feature_names=[f"feature_{i}" for i in range(features.shape[1])],
        )
        explainer.set_background(features[:200])
        attributions = explainer.explain(features[-1])
        top = [(attr.feature_name, float(attr.shap_value)) for attr in attributions[:5]]
        return ExplainabilitySummary(top_drivers=top)

    def _to_prediction_band(self, quantile_predictions: dict[float, np.ndarray]) -> PredictionBand:
        def _mean(q: float) -> float:
            values = quantile_predictions[q]
            return float(np.mean(values))

        return PredictionBand(
            p02=_mean(0.02),
            p10=_mean(0.1),
            p25=_mean(0.25),
            p50=_mean(0.5),
            p75=_mean(0.75),
            p90=_mean(0.9),
            p98=_mean(0.98),
        )

    def _build_world_backend(self) -> Any:
        try:
            import torch

            from nexus_alpha.core.world_model import WorldModel

            return {"torch": torch, "model": WorldModel(self._config.world_model)}
        except Exception:
            return _NumpyWorldModelFallback()

    def _train_world_model(self, features: np.ndarray, targets: np.ndarray) -> float:
        if isinstance(self._world_backend, _NumpyWorldModelFallback):
            return self._world_backend.train(features, targets)

        backend = self._world_backend
        torch_mod = backend["torch"]
        model = backend["model"]
        return float(
            model.online_update(
                features=torch_mod.tensor(features),
                targets=torch_mod.tensor(targets),
                current_regime=MarketRegime.UNKNOWN,
            )
        )

    def _predict_world_model(
        self,
        features: np.ndarray,
        regime: MarketRegime,
    ) -> tuple[dict[float, np.ndarray], float]:
        if isinstance(self._world_backend, _NumpyWorldModelFallback):
            return self._world_backend.predict(features), self._world_backend.uncertainty()

        backend = self._world_backend
        torch_mod = backend["torch"]
        model = backend["model"]
        output = model.predict_with_uncertainty(torch_mod.tensor(features), regime)
        return output.quantile_predictions, float(output.epistemic_uncertainty)

    def _save_world_model(self, path: Path) -> None:
        if isinstance(self._world_backend, _NumpyWorldModelFallback):
            self._world_backend.save(path)
            return
        self._world_backend["model"].save(str(path))
