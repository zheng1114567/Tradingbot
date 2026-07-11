"""Tests for agent contract utilities."""
from __future__ import annotations

from advanced_trading_agent.agents.contract import basic_self_check, build_agent_update, build_node_audit_update


class TestBuildAgentUpdate:
    def test_basic_output_structure(self):
        state = {"company_of_interest": "000001.SZ"}
        result = build_agent_update(
            state,
            sender="Market Agent",
            report_key="market_report",
            report="market report text",
            report_obj_key="market_report_obj",
            report_obj={"state": "normal"},
        )
        assert result["sender"] == "Market Agent"
        assert result["market_report"] == "market report text"
        assert result["market_report_obj"] == {"state": "normal"}

    def test_with_evidence(self):
        state = {}
        result = build_agent_update(
            state,
            sender="Event Agent",
            report_key="event_report",
            report="event text",
            report_obj_key="event_report_obj",
            report_obj={},
            evidence=["ev1", "ev2"],
        )
        assert result["sender"] == "Event Agent"
        assert "agent_evidence" in result
        assert "Event Agent" in result["agent_evidence"]

    def test_with_tool_calls(self):
        state = {}
        result = build_agent_update(
            state,
            sender="Analysis Agent",
            report_key="analysis_report",
            report="analysis",
            report_obj_key="analysis_report_obj",
            report_obj={},
            tool_calls=[{"tool": "get_factor_data", "args": {}}],
        )
        assert "agent_tool_calls" in result
        assert "Analysis Agent" in result["agent_tool_calls"]

    def test_with_self_check(self):
        state = {}
        sc = basic_self_check(evidence=["ev1"], passed_rules=["rule1"], confidence="high")
        result = build_agent_update(
            state,
            sender="Backtest Agent",
            report_key="backtest_report",
            report="backtest",
            report_obj_key="backtest_report_obj",
            report_obj={},
            self_check=sc,
        )
        assert "agent_self_checks" in result
        assert "Backtest Agent" in result["agent_self_checks"]


class TestBuildNodeAuditUpdate:
    def test_includes_all_audit_fields(self):
        result = build_node_audit_update(
            sender="Risk Check 1",
            risk_check_1={"verdict": "PASS"},
            evidence=["e1"],
            self_check=basic_self_check(evidence=["e1"], passed_rules=["r1"], confidence="PASS"),
        )
        assert "sender" in result
        assert result["risk_check_1"] == {"verdict": "PASS"}
        assert "agent_evidence" in result
        assert "agent_self_checks" in result

    def test_system_state_included(self):
        result = build_node_audit_update(
            sender="Risk Check 1",
            risk_check_1={"verdict": "HARD_VETO"},
            system_state="vetoed",
        )
        assert result.get("system_state") == "vetoed"

    def test_sender_always_present(self):
        result = build_node_audit_update(sender="Test")
        assert result["sender"] == "Test"
        assert "agent_evidence" in result
        assert "agent_self_checks" in result


class TestBasicSelfCheck:
    def test_all_passed(self):
        sc = basic_self_check(
            evidence=["e1", "e2"],
            passed_rules=["rule1", "rule2"],
            confidence="high",
        )
        assert sc["evidence_count"] == 2
        assert sc["passed_rules"] == ["rule1", "rule2"]
        assert sc["warnings"] == []
        assert sc["confidence"] == "high"
        assert sc["needs_review"] is False

    def test_with_warnings(self):
        sc = basic_self_check(
            evidence=["e1"],
            passed_rules=["rule1"],
            warnings=["data stale"],
            confidence="medium",
        )
        assert sc["warnings"] == ["data stale"]
        assert sc["needs_review"] is True

    def test_empty_inputs(self):
        sc = basic_self_check(evidence=[], confidence=0)
        assert sc["evidence_count"] == 0
        assert sc["passed_rules"] == []
        assert sc["warnings"] == []
        assert sc["confidence"] == 0
        assert sc["needs_review"] is False

    def test_default_values(self):
        sc = basic_self_check(evidence=["e1"])
        assert sc["passed_rules"] == []
        assert sc["warnings"] == []
        assert sc["confidence"] is None
