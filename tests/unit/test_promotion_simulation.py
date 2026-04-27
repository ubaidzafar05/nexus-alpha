import tempfile
import os
from types import SimpleNamespace

import pytest

import importlib.util
import os

# Import safe_retrain from infra/self_healing as a module
spec = importlib.util.spec_from_file_location('safe_retrain', os.path.join(os.path.dirname(__file__), '..', '..', 'infra', 'self_healing', 'safe_retrain.py'))
sr_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr_mod)

from nexus_alpha.learning.offline_trainer import OnlineLearner


class DummyTradeLogger:
    def __init__(self):
        pass


def test_promotion_simulation(monkeypatch, tmp_path):
    # Patch TradeLogger to avoid DB/files
    monkeypatch.setattr(sr_mod, 'TradeLogger', lambda: DummyTradeLogger())

    # Patch TelegramAlerts to avoid network calls
    monkeypatch.setattr(sr_mod, 'TelegramAlerts', SimpleNamespace(from_env=lambda: SimpleNamespace(is_configured=False)))

    # Patch OnlineLearner.__init__ to avoid loading pickles and set minimal state
    def fake_init(self, model_path=None):
        self.model_path = model_path
        # store placeholder predictor attribute used elsewhere
        self.predictor = None
        return None
    monkeypatch.setattr(OnlineLearner, '__init__', fake_init)

    # Patch OnlineLearner.retrain_from_journal to simulate an accepted retrain
    def fake_retrain(self, tl):
        return {
            'updated': True,
            'n_trades': 50,
            'val_balanced_accuracy': 0.6,
            'val_direction_accuracy': 0.7,
        }

    monkeypatch.setattr(OnlineLearner, 'retrain_from_journal', fake_retrain)

    # Patch benchmark_learning_targets to return an improving benchmark
    def fake_bench(tl, min_trades=30):
        return {
            'variants': {
                'v1': {
                    'baseline_balanced_accuracy': 0.5,
                    'best_model': 'm1',
                    'models': {'m1': {'balanced_accuracy': 0.55}},
                }
            }
        }

    monkeypatch.setattr(sr_mod, 'benchmark_learning_targets', fake_bench)

    candidate = tmp_path / 'candidate.pkl'
    promote = tmp_path / 'promoted.pkl'
    # create dummy candidate file to simulate saved model
    candidate.write_text('dummy')

    rc = sr_mod.main(candidate_path=str(candidate), promote_path=str(promote), min_trades=30)
    assert rc == 0
    assert promote.exists()
    assert promote.read_text() == 'dummy'
