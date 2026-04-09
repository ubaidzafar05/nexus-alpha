import os
import shutil
import tempfile
from pathlib import Path

import pytest

from nexus_alpha.learning.offline_trainer import OnlineLearner
from nexus_alpha.learning.offline_trainer import benchmark_learning_targets


# We'll monkeypatch OnlineLearner.retrain_from_journal and benchmark_learning_targets

def test_safe_retrain_promote(monkeypatch, tmp_path):
    # Create dummy candidate file path
    candidate = tmp_path / "candidate.pkl"
    candidate.write_text("dummy")
    promote = tmp_path / "promote.pkl"

    # Replace OnlineLearner.__init__ to avoid actual model loading
    def dummy_init(self, predictor=None, retrain_interval_hours=6, min_new_trades=20, min_total_trades=50, *args, **kwargs):
        self.predictor = predictor
        self.retrain_interval = retrain_interval_hours * 3600
        self.min_new_trades = min_new_trades
        self.min_total_trades = min_total_trades
        self._last_retrain = 0.0
    monkeypatch.setattr(OnlineLearner, "__init__", dummy_init)

    # Mock OnlineLearner.retrain_from_journal to return an accepted stats dict
    def mock_retrain(self, tl):
        return {"updated": True, "n_trades": 100, "val_direction_accuracy": 0.62, "val_balanced_accuracy": 0.55}

    monkeypatch.setattr(OnlineLearner, "retrain_from_journal", mock_retrain)

    # Mock benchmark_learning_targets to show improvement
    def mock_benchmark(trade_logger, min_trades=30, **kwargs):
        return {
            "variants": {
                "binary_chronological": {"best_model": "random_forest", "baseline_balanced_accuracy": 0.45, "models": {"random_forest": {"balanced_accuracy": 0.51}}}
            }
        }

    monkeypatch.setattr("nexus_alpha.learning.offline_trainer.benchmark_learning_targets", mock_benchmark)

    # Run safe_retrain main
    from infra.self_healing.safe_retrain import main

    ret = main(candidate_path=str(candidate), promote_path=str(promote), min_trades=30)
    assert ret == 0
    assert promote.exists()


def test_safe_retrain_reject(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.pkl"
    candidate.write_text("dummy")
    promote = tmp_path / "promote.pkl"

    # Replace OnlineLearner.__init__ to avoid actual model loading
    def dummy_init(self, predictor=None, retrain_interval_hours=6, min_new_trades=20, min_total_trades=50, *args, **kwargs):
        self.predictor = predictor
        self.retrain_interval = retrain_interval_hours * 3600
        self.min_new_trades = min_new_trades
        self.min_total_trades = min_total_trades
        self._last_retrain = 0.0
    monkeypatch.setattr(OnlineLearner, "__init__", dummy_init)

    def mock_retrain(self, tl):
        return {"updated": False, "n_trades": 20, "val_direction_accuracy": 0.4, "val_balanced_accuracy": 0.35}

    monkeypatch.setattr(OnlineLearner, "retrain_from_journal", mock_retrain)

    def mock_benchmark(trade_logger, min_trades=30, **kwargs):
        return None

    monkeypatch.setattr("nexus_alpha.learning.offline_trainer.benchmark_learning_targets", mock_benchmark)

    from infra.self_healing.safe_retrain import main

    ret = main(candidate_path=str(candidate), promote_path=str(promote), min_trades=30)
    assert ret != 0
    assert not promote.exists()
