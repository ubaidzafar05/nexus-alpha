from __future__ import annotations

import numpy as np

from nexus_alpha.config import NexusConfig
from nexus_alpha.infrastructure.explainability import TreeSHAPExplainer
from nexus_alpha.intelligence.foundation import IntelligenceFoundationRunner
from nexus_alpha.types import MarketRegime


def test_intelligence_foundation_runner_generates_artifacts(tmp_path) -> None:
    config = NexusConfig()
    config.world_model.input_size = 8
    config.world_model.hidden_size = 16
    config.world_model.attention_heads = 2
    config.world_model.num_layers = 1
    config.world_model.mc_dropout_samples = 3
    config.world_model.episodic_memory_per_regime = 128

    runner = IntelligenceFoundationRunner(config)
    output, artifacts = runner.run(output_dir=tmp_path, n_samples=96, seed=7)

    assert artifacts.report_path.exists()
    assert artifacts.checkpoint_path.exists()
    assert output.symbol == "BTCUSDT"
    assert output.regime in set(MarketRegime)
    assert 0.0 <= output.uncertainty.confidence <= 1.0
    assert len(output.explainability.top_drivers) > 0


def test_tree_shap_explainer_produces_ranked_attributions() -> None:
    rng = np.random.default_rng(0)
    background = rng.normal(size=(32, 4))
    weights = np.array([0.4, -0.2, 0.1, 0.05], dtype=np.float64)

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return x @ weights

    explainer = TreeSHAPExplainer(
        model_predict_fn=predict_fn,
        feature_names=["f0", "f1", "f2", "f3"],
    )
    explainer.set_background(background)
    attributions = explainer.explain(background[0])

    assert len(attributions) == 4
    assert abs(attributions[0].shap_value) >= abs(attributions[-1].shap_value)
