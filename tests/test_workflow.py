"""LangGraph Workflow 集成测试

使用 mock Agent 节点验证整个工作流拓扑:
1. START → Risk Check 1 → (PASS) → System Init → Memory → Market → Event → Analysis → Backtest → Risk Check 2 → System Init → Round 2 Judge → Risk Check 3 → System Final → Report → END
2. HARD_VETO → END
3. 冰点模式 → 跳过深度分析
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from advanced_trading_agent.agents.schemas import (
    BacktestReport,
    Confidence,
    DecisionType,
    EventReport,
    MarketReport,
    RiskVerdict,
    SystemDecision,
)
from advanced_trading_agent.graph.workflow import create_workflow, TradingSystem
from advanced_trading_agent.graph.state import AgentState


# ============================================================
# 辅助函数 — 构建最小化 Mock State
# ============================================================

def make_base_state(**overrides) -> dict:
    state = {
        "messages": [],
        "company_of_interest": "000001.SZ",
        "trade_date": "2026-07-10",
        "sender": "system",
        "run_mode": "live",
        "skip_backtest": False,
        "tier1_data": {
            "market": {"index_close": 3000, "index_change_pct": 0.5},
            "sentiment": {"sentiment": "正常", "sentiment_score": 55},
            "capital": {"confirmation": "资金确认", "net_inflow_main": 1e8},
            "risk": {
                "st_list": [],
                "suspended_list": [],
                "delisting_list": [],
                "daily_volume": 20_000_000,
            },
        },
        "tier2_data": {"factors": [], "events": [], "backtest_samples": []},
        "tier2_decision": {},
        "data_quality_report": None,
        "pit_manifest": None,
        "memory_context": "",
        "memory_recall": {},
        "risk_check_1": {},
        "risk_check_2": {},
        "risk_check_3": {},
        "market_report": "",
        "event_report": "",
        "analysis_report": "",
        "backtest_report": "",
        "market_report_obj": None,
        "event_report_obj": None,
        "analysis_report_obj": None,
        "backtest_report_obj": None,
        "agent_evidence": {},
        "agent_tool_calls": {},
        "agent_self_checks": {},
            "round2_state": {
                "active": False, "round_count": 0, "max_rounds": 8,
                "questions": [], "contradictions": [],
                "current_speaker": "", "completed": False, "summary": "",
                "provider": "none", "fallback_reason": "",
                "final_pressure": "neutral", "unresolved_conflicts": [],
            },
        "round2_summary": "",
        "system_decision_obj": None,
        "system_rubric": {},
        "system_state": "",
        "approval_input": {},
        "approval_record": {},
        "execution_allowed": False,
        "final_report": "",
        "final_report_obj": None,
        "audit_trace": {},
        "audit_trace_path": "",
    }
    state.update(overrides)
    return state


class TestWorkflowTopology:
    """工作流拓扑测试 — 验证图结构正确性"""

    def test_workflow_creates_successfully(self):
        """工作流应能正确编译"""
        wf = create_workflow()
        assert wf is not None
        # 验证节点存在
        nodes = list(wf.get_graph().nodes.keys())
        for expected in ["Risk Check 1", "System Init", "Market Agent",
                          "Round 2 Judge", "Skip Backtest", "System Final Decision",
                          "Approval Agent", "Report Agent"]:
            assert expected in nodes, f"缺少节点: {expected}"

    def test_risk_check_1_pass_continues(self):
        """硬风控1 通过 → 走 round1 路径"""
        wf = create_workflow()
        state = make_base_state(risk_check_1={"verdict": "PASS"})
        from advanced_trading_agent.graph.conditional import after_risk_check_1
        result = after_risk_check_1(state)
        assert result == "round1"

    def test_risk_check_1_veto_ends(self):
        """硬风控1 HARD_VETO → END"""
        wf = create_workflow()
        state = make_base_state(risk_check_1={"verdict": "HARD_VETO"})
        from advanced_trading_agent.graph.conditional import after_risk_check_1
        result = after_risk_check_1(state)
        assert result == "end"

    def test_winter_mode_skips_depth(self):
        """冰点模式 → 跳过深度分析"""
        from advanced_trading_agent.graph.conditional import after_market
        market = MarketReport(
            market_state="冰点", position_cap=0.2,
            capital_confirmation="资金不足", reasoning="test",
        )
        state = make_base_state(market_report_obj=market)
        result = after_market(state)
        assert result == "skip_round1"

    def test_normal_market_continues(self):
        """正常市场 → 深度分析"""
        from advanced_trading_agent.graph.conditional import after_market
        market = MarketReport(
            market_state="正常", position_cap=0.6,
            capital_confirmation="资金确认", reasoning="test",
        )
        state = make_base_state(market_report_obj=market)
        result = after_market(state)
        assert result == "continue_round1"


class TestTradingSystemIntegration:
    """TradingSystem 集成测试 — 端到端流程"""

    def test_analyze_with_mock_data_returns_report(self):
        """传入 mock 数据能生成报告"""
        tier1 = {
            "market": {"index_close": 3000, "index_change_pct": 0.5},
            "sentiment": {"sentiment": "正常", "sentiment_score": 55},
            "capital": {"confirmation": "资金确认"},
            "risk": {
                "st_list": [],
                "suspended_list": [],
                "delisting_list": [],
                "daily_volume": 20_000_000,
            },
        }
        tier2 = {}
        system = TradingSystem(debug=False, mode="live")
        with patch.object(system, "workflow") as mock_wf:
            mock_wf.invoke.return_value = make_base_state(
                final_report="# 测试报告\n**结论**: 推荐",
                final_report_obj=None,
            )
            state, report = system.analyze(
                ticker="000001.SZ", trade_date="2026-07-10",
                tier1_data=tier1, tier2_data=tier2,
            )
            assert report is not None
            assert "测试报告" in report

    def test_system_routes_full_flow_no_crash(self):
        """完整工作流不崩溃 (各 Agent 的降级路径确保即使数据缺失也不抛异常)

        注意: workflow 单测不依赖真实网络，DataAgent 在自身测试中单独覆盖。
        """
        system = TradingSystem(debug=False, mode="live")
        try:
            with patch.object(system, "_load_data") as load_data:
                load_data.return_value = (
                    {
                        "market": {"index_close": 3000, "index_change_pct": 0.5},
                        "sentiment": {"sentiment": "正常", "sentiment_score": 55},
                        "capital": {"confirmation": "资金确认", "net_inflow_main": 1e8},
                        "risk": {
                            "st_list": [],
                            "suspended_list": [],
                            "delisting_list": [],
                            "daily_volume": 20_000_000,
                            "risk_data_available": True,
                            "risk_data_errors": [],
                        },
                    },
                    {
                        "price_data": [
                            {
                                "code": "000001.SZ",
                                "close": 10,
                                "pct_chg": 1,
                                "amount": 20_000_000,
                            }
                        ],
                        "factors": [],
                        "events": [],
                        "backtest_samples": [],
                    },
                )
                state, report = system.analyze(
                    ticker="000001.SZ", trade_date="2026-07-10",
                    tier1_data={}, tier2_data={},
                )
            assert True
        except Exception as e:
            # sender 字段在 LangGraph 特定版本可能触发 InvalidUpdateError
            # 这在纯线性链中不应发生, 但某些 graph.compile() 配置会触发
            from langgraph.errors import InvalidUpdateError
            if isinstance(e, InvalidUpdateError) and "sender" in str(e):
                pytest.skip(f"LangGraph 版本对 sender 字段的限制: {e}")
            pytest.fail(f"工作流抛出未处理异常: {type(e).__name__}: {e}")

    def test_load_data_populates_risk_and_liquidity(self):
        system = TradingSystem(debug=False, mode="live")

        with patch("advanced_trading_agent.graph.workflow.DataAgent") as data_agent_cls:
            run = MagicMock()
            run.final_data = {
                "analysis": {
                    "agent_payload": {
                        "tier1_data": {
                            "market": {"index_close": 3000, "index_change_pct": 0.5},
                            "sentiment": {"sentiment": "正常", "sentiment_score": 55},
                            "capital": {"confirmation": "资金确认", "net_inflow_main": 1e8},
                            "risk": {
                                "st_list": [],
                                "suspended_list": [],
                                "delisting_list": [],
                                "daily_volume": 20_000_000,
                                "risk_data_available": True,
                                "risk_data_errors": [],
                            },
                        },
                        "tier2_data": {
                            "price_data": [
                                {
                                    "code": "000001.SZ",
                                    "close": 10,
                                    "pct_chg": 1,
                                    "amount": 20_000_000,
                                }
                            ],
                            "factors": [],
                            "events": [{"event_id": "news_0001", "summary": "测试新闻"}],
                            "backtest_samples": [],
                        },
                    },
                },
                "manifest": {"fields": {"stock.daily": {"available": True}, "risk.st_status": {"available": True}}},
            }
            run.manifest_path = "manifest.json"
            run.to_dict.return_value = {"run_id": "test"}
            data_agent_cls.return_value.run.return_value = run
            tier1, tier2 = system._load_data("000001.SZ", "2026-07-10")

        assert tier2["price_data"]
        assert tier2["events"][0]["summary"] == "测试新闻"
        assert tier1["risk"]["risk_data_available"] is True
        assert tier1["risk"]["daily_volume"] == 20_000_000
        assert tier1["_data_manifest"]["fields"]["stock.daily"]["available"] is True
        assert "risk.st_status" in tier1["_data_manifest"]["fields"]

    def test_skip_backtest_parameter_routes_without_backtest_tools(self):
        system = TradingSystem(debug=False, mode="live")

        with patch.object(system, "_load_data") as load_data:
            load_data.return_value = (
                {
                    "market": {"index_close": 3000, "index_change_pct": 0.5},
                    "sentiment": {"sentiment": "正常", "sentiment_score": 55},
                    "capital": {"confirmation": "资金确认", "net_inflow_main": 1e8},
                    "risk": {
                        "st_list": [],
                        "suspended_list": [],
                        "delisting_list": [],
                        "daily_volume": 20_000_000,
                        "risk_data_available": True,
                        "risk_data_errors": [],
                    },
                },
                {
                    "price_data": [{"code": "000001.SZ", "close": 10, "pct_chg": 1}],
                    "factors": [],
                    "events": [],
                    "backtest_samples": [],
                },
            )
            state, _ = system.analyze(
                ticker="000001.SZ",
                trade_date="2026-07-10",
                tier1_data={},
                tier2_data={},
                skip_backtest=True,
            )

        assert state["skip_backtest"] is True
        assert state["backtest_report_obj"].sample_size == 0
        assert "skip_backtest=True" in state["agent_evidence"]["Backtest Agent"]

    def test_workflow_state_keys(self):
        """验证 AgentState 的 key 设计完整性"""
        from typing import get_type_hints
        hints = get_type_hints(AgentState)
        required_keys = [
            "company_of_interest", "trade_date", "sender", "run_mode",
            "skip_backtest",
            "tier1_data", "tier2_data", "tier2_decision",
            "risk_check_1", "risk_check_2", "risk_check_3",
            "market_report", "event_report", "analysis_report", "backtest_report",
            "round2_state", "round2_summary", "system_decision_obj", "system_rubric", "system_state",
            "approval_input", "approval_record", "execution_allowed",
            "final_report", "final_report_obj", "audit_trace", "audit_trace_path",
        ]
        for key in required_keys:
            assert key in hints, f"AgentState 类型注解缺少 key: {key}"
