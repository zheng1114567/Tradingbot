"""Shared test fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture
def sample_ticker() -> str:
    return "000001.SZ"


@pytest.fixture
def sample_trade_date() -> str:
    return "2026-07-10"


@pytest.fixture
def base_state(sample_ticker, sample_trade_date):
    """Minimal workflow state for agent node tests."""
    return {
        "company_of_interest": sample_ticker,
        "trade_date": sample_trade_date,
        "tier1_data": {},
        "tier2_data": {},
        "sender": "test",
    }
