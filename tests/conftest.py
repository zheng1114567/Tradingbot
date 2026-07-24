"""Shared test fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_vendor_throttle_sleep(monkeypatch):
    """Keep tests deterministic, offline, and fast."""
    from advanced_trading_agent.data_agent import vendor_throttle

    monkeypatch.setenv("ATA_DISABLE_LLM", "1")
    vendor_throttle.reset_vendor_throttle()
    monkeypatch.setattr(vendor_throttle.time, "sleep", lambda _seconds: None)
    yield
    vendor_throttle.reset_vendor_throttle()


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
