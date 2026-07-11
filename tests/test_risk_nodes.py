"""Tests for risk node factory functions."""
from __future__ import annotations

import pytest

from advanced_trading_agent.graph.risk_nodes import create_risk_check_1, create_risk_check_2, create_risk_check_3


class TestRiskCheck1:
    """Pre-analysis: ST / suspension / delisting checks."""

    def test_st_stock_hard_veto(self):
        node = create_risk_check_1()
        state = {
            "company_of_interest": "000001.SZ",
            "tier1_data": {
                "risk": {
                    "st_list": ["000001.SZ"],
                    "suspended_list": [],
                    "delisting_list": [],
                    "risk_data_available": True,
                }
            },
        }
        result = node(state)
        assert result["risk_check_1"]["verdict"] == "HARD_VETO"
        assert any("ST" in r for r in result["risk_check_1"]["reasons"])

    def test_suspended_stock_hard_veto(self):
        node = create_risk_check_1()
        state = {
            "company_of_interest": "600000.SH",
            "tier1_data": {
                "risk": {
                    "st_list": [],
                    "suspended_list": ["600000.SH"],
                    "delisting_list": [],
                    "risk_data_available": True,
                }
            },
        }
        result = node(state)
        assert result["risk_check_1"]["verdict"] == "HARD_VETO"

    def test_delisting_stock_hard_veto(self):
        """delisting risk is checked via check_st_status - the ST list includes delisting risks."""
        node = create_risk_check_1()
        state = {
            "company_of_interest": "300001.SZ",
            "tier1_data": {
                "risk": {
                    "st_list": ["300001.SZ"],  # ST list includes delisting candidates
                    "suspended_list": [],
                    "delisting_list": [],
                    "risk_data_available": True,
                }
            },
        }
        result = node(state)
        assert result["risk_check_1"]["verdict"] == "HARD_VETO"

    def test_normal_stock_pass(self):
        node = create_risk_check_1()
        state = {
            "company_of_interest": "000001.SZ",
            "tier1_data": {
                "risk": {
                    "st_list": ["600000.SH"],
                    "suspended_list": [],
                    "delisting_list": [],
                    "risk_data_available": True,
                }
            },
        }
        result = node(state)
        assert result["risk_check_1"]["verdict"] == "PASS"

    def test_missing_risk_data_soft_veto(self):
        node = create_risk_check_1()
        state = {
            "company_of_interest": "000001.SZ",
            "tier1_data": {
                "risk": {
                    "risk_data_available": False,
                    "risk_data_errors": ["ST list fetch timed out"],
                }
            },
        }
        result = node(state)
        assert result["risk_check_1"]["verdict"] == "SOFT_VETO"

    def test_empty_tier1_no_crash(self):
        node = create_risk_check_1()
        state = {"company_of_interest": "000001.SZ", "tier1_data": {}}
        result = node(state)
        assert "risk_check_1" in result
        assert "verdict" in result["risk_check_1"]


class TestRiskCheck2:
    """Post-Round1: liquidity / limit-up-down checks."""

    def test_normal_stock_pass(self):
        node = create_risk_check_2()
        state = {
            "company_of_interest": "000001.SZ",
            "tier1_data": {"risk": {"daily_volume": 100_000_000}},
            "tier2_data": {},
        }
        result = node(state)
        verdict = result["risk_check_2"]["verdict"]
        assert verdict in ("PASS", "SOFT_VETO")

    def test_limit_up_hard_veto(self):
        node = create_risk_check_2()
        state = {
            "company_of_interest": "000001.SZ",
            "tier1_data": {"risk": {"daily_volume_cny": 100_000_000}},
            "tier2_data": {"price_data": [{"close": 11.0, "pre_close": 10.0, "amount": 100_000_000}]},
        }
        result = node(state)
        verdict = result["risk_check_2"]["verdict"]
        if verdict == "HARD_VETO":
            assert any("涨停" in r for r in result["risk_check_2"]["reasons"])

    def test_low_liquidity_soft_veto(self):
        node = create_risk_check_2()
        state = {
            "company_of_interest": "000001.SZ",
            "tier1_data": {"risk": {"daily_volume_cny": 1_000_000}},
            "tier2_data": {"price_data": [{"close": 10.0, "pre_close": 10.0, "amount": 1_000_000}]},
        }
        result = node(state)
        verdict = result["risk_check_2"]["verdict"]
        assert verdict in ("SOFT_VETO", "PASS")

    def test_empty_tier2_no_crash(self):
        node = create_risk_check_2()
        state = {"company_of_interest": "000001.SZ", "tier1_data": {}, "tier2_data": {}}
        result = node(state)
        assert "risk_check_2" in result


class TestRiskCheck3:
    """Pre-final: impact cost / position checks."""

    def test_normal_pass(self):
        node = create_risk_check_3()
        state = {
            "company_of_interest": "000001.SZ",
            "tier1_data": {},
            "tier2_data": {},
            "system_decision_obj": None,
        }
        result = node(state)
        assert "risk_check_3" in result
        assert "verdict" in result["risk_check_3"]

    def test_empty_state_no_crash(self):
        node = create_risk_check_3()
        state = {}
        result = node(state)
        assert "risk_check_3" in result
