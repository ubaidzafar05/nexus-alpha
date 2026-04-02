"""Unit-test fixtures — keep test isolation clean."""
from __future__ import annotations

from pathlib import Path

import pytest

_PORTFOLIO_STATE = Path("data/trade_logs/portfolio_state.json")


@pytest.fixture(autouse=True)
def _clean_portfolio_state():
    """Remove any portfolio state file before/after each test to prevent cross-test pollution."""
    _PORTFOLIO_STATE.unlink(missing_ok=True)
    yield
    _PORTFOLIO_STATE.unlink(missing_ok=True)
