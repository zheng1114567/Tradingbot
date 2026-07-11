"""Agent 单元测试 — 覆盖全部 5 个 Agent 的 LLM 路径和降级路径

测试策略:
- Mock LLMClient.chat() 返回预定义的 Pydantic 对象
- Mock Tool 类返回受控数据
- 测试正常路径 (LLM 成功)
- 测试降级路径 (LLM 异常 → 规则兜底)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from advanced_trading_agent.config import config
from advanced_trading_agent.agents.schemas import (
    AnalysisReport,
    BacktestReport,
    Confidence,
    DecisionType,
    EventReport,
    FinalReport,
    MarketReport,
    MemoryRecall,
    RiskVerdict,
    StockRanking,
    SystemDecision,
    SystemRubric,
)
from advanced_trading_agent.agents.market_agent import create_market_agent
from advanced_trading_agent.agents.event_agent import create_event_agent
from advanced_trading_agent.agents.analysis_agent import create_analysis_agent
from advanced_trading_agent.agents.backtest_agent import create_backtest_agent
from advanced_trading_agent.agents.system_agent import create_system_agent
from advanced_trading_agent.agents.approval_agent import create_approval_agent
from advanced_trading_agent.agents.report_agent import create_report_agent


# ============================================================
# Mock LLM
# ============================================================

class MockLLM:
    """Mock LLMClient — chat 方法返回预设 Pydantic 对象"""

    def __init__(self, return_value: Any = None, raise_error: bool = False):
        self._return = return_value
        self._raise = raise_error
        self.last_kwargs = None

    def chat(self, messages, response_format=None, temperature=None, max_tokens=4096):
        self.last_kwargs = {
            "messages": messages,
            "response_format": response_format,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._raise:
            raise RuntimeError("Mock LLM error")
        if self._return is not None:
            return self._return
        if response_format:
            return _default_for(response_format)
        return "mock response"

    @property
    def client(self):
        return "mock"


_DEFAULTS: dict = {}


def _default_for(model_class):
    """为 Pydantic 类生成默认实例的深拷贝"""
    if model_class not in _DEFAULTS:
        _DEFAULTS[model_class] = {
            MarketReport: MarketReport(
                market_state="正常", position_cap=0.6,
                capital_confirmation="资金确认", reasoning="mock",
            ),
            EventReport: EventReport(
                event_id="mock_e1", event_type="政策", direction="利好",
                confidence=0.6, transmission_path="mock", evidence_level="权威媒体",
                pricing_status="未定价", chain_quality="direct", reasoning="mock",
            ),
            AnalysisReport: AnalysisReport(
                factor_explanation="mock", reasoning="mock",
                stock_rankings=[
                    StockRanking(code="000001.SZ", name="测试股",
                                 composite_score=7.5, main_driver="mock因子")
                ],
            ),
            BacktestReport: BacktestReport(
                sample_size=30, win_rate=0.55, avg_excess_return=0.02,
                confidence=Confidence.MEDIUM, reasoning="mock",
            ),
            SystemDecision: SystemDecision(
                decision=DecisionType.RECOMMEND, position=0.1,
                alpha_source=["mock"], horizon_days=5,
                reasons=["mock支持"], objections=[],
                risk_verdict=RiskVerdict.PASS, reasoning="mock",
            ),
        }[model_class]
    return _DEFAULTS[model_class].model_copy(deep=True)


# ============================================================
# 共享 Fixtures
# ============================================================

@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_llm_fail():
    return MockLLM(raise_error=True)


@pytest.fixture
def base_state():
    """最小化 state，所有 Agent 都能跑"""
    return {
        "company_of_interest": "000001.SZ",
        "trade_date": "2026-07-10",
        "tier1_data": {
            "market": {"index_close": 3000, "index_change_pct": 0.5},
            "sentiment": {"sentiment": "正常", "sentiment_score": 55},
            "capital": {"confirmation": "资金确认", "net_inflow_main": 1e8},
            "risk": {"st_list": [], "suspended_list": [], "delisting_list": []},
        },
        "tier2_data": {
            "factors": [],
            "events": [],
            "backtest_samples": [],
        },
    }


# ============================================================
# Market Agent 测试
# ============================================================

class TestMarketAgent:
    """Market Agent — 市场温度分析"""

    def test_happy_path(self, mock_llm, base_state):
        node = create_market_agent(mock_llm)
        result = node(base_state)
        assert "market_report" in result
        assert "market_report_obj" in result
        assert isinstance(result["market_report_obj"], MarketReport)
        assert result["sender"] == "Market Agent"

    def test_llm_fallback_to_rules(self, mock_llm_fail, base_state):
        node = create_market_agent(mock_llm_fail)
        result = node(base_state)
        obj = result["market_report_obj"]
        assert isinstance(obj, MarketReport)
        assert obj.market_state == "正常"
        assert obj.position_cap == 0.6

    def test_market_frozen_caps_position(self, mock_llm, base_state):
        state = {**base_state, "tier1_data": {
            **base_state["tier1_data"],
            "sentiment": {"sentiment": "冰点", "sentiment_score": 10},
        }}
        node = create_market_agent(mock_llm)
        result = node(state)
        obj = result["market_report_obj"]
        assert obj.position_cap == 0.2

    def test_market_high_caps_position(self, mock_llm, base_state):
        state = {**base_state, "tier1_data": {
            **base_state["tier1_data"],
            "sentiment": {"sentiment": "高潮", "sentiment_score": 90},
        }}
        node = create_market_agent(mock_llm)
        result = node(state)
        obj = result["market_report_obj"]
        assert obj.position_cap == 0.3

    def test_too_high_capped_by_rules(self, mock_llm, base_state):
        llm = MockLLM(return_value=MarketReport(
            market_state="低迷", position_cap=0.99,
            capital_confirmation="资金背离", reasoning="test",
        ))
        node = create_market_agent(llm)
        result = node(base_state)
        obj = result["market_report_obj"]
        assert obj.position_cap == 0.60


# ============================================================
# Event Agent 测试
# ============================================================

class TestEventAgent:
    """Event Agent — 事件分析"""

    def test_happy_path(self, mock_llm, base_state):
        node = create_event_agent(mock_llm)
        result = node(base_state)
        assert "event_report" in result
        assert "event_report_obj" in result
        assert isinstance(result["event_report_obj"], EventReport)
        assert result["sender"] == "Event Agent"

    def test_llm_fallback(self, mock_llm_fail, base_state):
        node = create_event_agent(mock_llm_fail)
        result = node(base_state)
        obj = result["event_report_obj"]
        assert isinstance(obj, EventReport)
        assert obj.chain_quality == "weak"

    def test_with_tier2_events(self, mock_llm, base_state):
        state = {**base_state, "tier2_data": {
            "events": [{"event_id": "e1", "event_type": "政策", "summary": "降准",
                        "direction": "利好", "confidence": 0.8}],
        }}
        node = create_event_agent(mock_llm)
        result = node(state)
        obj = result["event_report_obj"]
        assert isinstance(obj, EventReport)


# ============================================================
# Analysis Agent 测试
# ============================================================

class TestAnalysisAgent:
    """Analysis Agent — 因子分析"""

    def test_happy_path(self, mock_llm, base_state):
        node = create_analysis_agent(mock_llm)
        result = node(base_state)
        assert "analysis_report" in result
        assert "analysis_report_obj" in result
        assert isinstance(result["analysis_report_obj"], AnalysisReport)
        assert result["sender"] == "Analysis Agent"

    def test_llm_fallback(self, mock_llm_fail, base_state):
        class MockTools:
            def get_factor_data(self, code: str, top_n: int = 20):
                return []

        node = create_analysis_agent(mock_llm_fail, tools=MockTools())
        result = node(base_state)
        obj = result["analysis_report_obj"]
        assert isinstance(obj, AnalysisReport)
        assert len(obj.stock_rankings) == 0

    def test_deterministic_ranking_used(self, mock_llm, base_state):
        llm = MockLLM(return_value=AnalysisReport(
            factor_explanation="test", reasoning="test",
            stock_rankings=[StockRanking(code="X", name="LLM虚构",
                                         composite_score=9.9, main_driver="虚构")],
        ))
        node = create_analysis_agent(llm)
        result = node(base_state)
        obj = result["analysis_report_obj"]
        # 无因子数据时, 确定性排序为空, LLM 排序被保留
        assert len(obj.stock_rankings) == 1

    def test_negative_factor_score_is_clamped(self, mock_llm, base_state):
        class MockTools:
            def get_factor_data(self, code: str, top_n: int = 20):
                return [{
                    "code": code,
                    "name": "负分样本",
                    "composite_score": -0.5,
                    "quality_score": 1,
                    "growth_score": 2,
                }]

        node = create_analysis_agent(mock_llm, tools=MockTools())
        result = node(base_state)
        obj = result["analysis_report_obj"]

        assert obj.stock_rankings[0].composite_score == 0.0


# ============================================================
# Backtest Agent 测试
# ============================================================

class TestBacktestAgent:
    """Backtest Agent — 回测验证"""

    def test_happy_path(self, mock_llm, base_state):
        node = create_backtest_agent(mock_llm)
        result = node(base_state)
        assert "backtest_report" in result
        assert "backtest_report_obj" in result
        assert isinstance(result["backtest_report_obj"], BacktestReport)
        assert result["sender"] == "Backtest Agent"

    def test_llm_fallback(self, mock_llm_fail, base_state):
        node = create_backtest_agent(mock_llm_fail)
        result = node(base_state)
        obj = result["backtest_report_obj"]
        assert isinstance(obj, BacktestReport)
        assert obj.confidence == Confidence.LOW

    def test_with_backtest_samples(self, mock_llm, base_state):
        state = {**base_state, "tier2_data": {
            "backtest_samples": [{
                "sample_size": 50, "win_rate": 0.6,
                "avg_excess_return": 0.03, "confidence": "high",
            }],
        }}
        node = create_backtest_agent(mock_llm)
        result = node(state)
        obj = result["backtest_report_obj"]
        assert isinstance(obj, BacktestReport)

    def test_confidence_capped_when_low_samples(self, mock_llm, base_state):
        llm = MockLLM(return_value=BacktestReport(
            sample_size=5, win_rate=0.6, avg_excess_return=0.02,
            confidence=Confidence.HIGH, reasoning="mock",
        ))
        node = create_backtest_agent(llm)
        result = node(base_state)
        obj = result["backtest_report_obj"]
        assert obj.confidence == Confidence.MEDIUM


# ============================================================
# System Agent 测试
# ============================================================

class TestSystemAgent:
    """System Agent — 初始化+裁定"""

    def test_init_node(self, mock_llm, base_state):
        sa = create_system_agent(mock_llm)
        result = sa["init"](base_state)
        assert "data_quality_report" in result
        assert "system_state" in result
        assert result["system_state"] == "running"

    def test_init_node_winter_mode(self, mock_llm, base_state):
        state = {**base_state, "tier1_data": {
            **base_state["tier1_data"],
            "sentiment": {"sentiment": "冰点", "sentiment_score": 8},
        }}
        sa = create_system_agent(mock_llm)
        result = sa["init"](state)
        assert result["tier1_data"].get("winter_mode") is True

    def test_round2_judge_no_contradiction(self, mock_llm, base_state):
        sa = create_system_agent(mock_llm)
        result = sa["round2_judge"](base_state)
        assert result["round2_state"]["active"] is False
        assert result["round2_state"]["completed"] is True

    def test_round2_judge_with_contradiction(self, mock_llm, base_state):
        state = {
            **base_state,
            "market_report_obj": MarketReport(
                market_state="正常", position_cap=0.6,
                capital_confirmation="资金背离", reasoning="test",
            ),
            "event_report_obj": EventReport(
                event_id="e1", event_type="政策", direction="利好",
                confidence=0.8, transmission_path="直接受益",
                evidence_level="权威媒体", pricing_status="未定价",
                chain_quality="direct", reasoning="test",
            ),
            "backtest_report_obj": BacktestReport(
                sample_size=30, win_rate=0.6, avg_excess_return=0.02,
                confidence=Confidence.MEDIUM, reasoning="test",
            ),
        }
        sa = create_system_agent(mock_llm)
        result = sa["round2_judge"](state)
        assert result["round2_state"]["active"] is True
        assert len(result["round2_state"]["contradictions"]) > 0

    def test_final_decision(self, mock_llm, base_state):
        sa = create_system_agent(mock_llm)
        state = {
            **base_state,
            "risk_check_3": {"verdict": "PASS", "reasons": []},
            "market_report_obj": _default_for(MarketReport),
        }
        result = sa["final"](state)
        assert "system_decision_obj" in result
        assert "system_rubric" in result
        assert isinstance(result["system_decision_obj"], SystemDecision)
        assert isinstance(SystemRubric(**result["system_rubric"]), SystemRubric)
        assert result["system_state"] == "completed"

    def test_final_rubric_downgrades_weak_recommendation(self, mock_llm, base_state):
        llm = MockLLM(return_value=SystemDecision(
            decision=DecisionType.RECOMMEND,
            position=0.1,
            alpha_source=["mock"],
            horizon_days=5,
            reasons=["LLM wants to recommend"],
            objections=[],
            risk_verdict=RiskVerdict.PASS,
            reasoning="mock",
        ))
        sa = create_system_agent(llm)
        state = {
            **base_state,
            "risk_check_3": {"verdict": "PASS", "reasons": []},
        }
        result = sa["final"](state)
        decision = result["system_decision_obj"]
        assert decision.decision == DecisionType.WATCH
        assert result["system_rubric"]["recommendation_floor"] == DecisionType.WATCH.value
        assert "结构化rubric限制" in decision.reasons[0]

    def test_final_round2_downgrade_pressure_blocks_recommendation(self, base_state):
        llm = MockLLM(return_value=SystemDecision(
            decision=DecisionType.RECOMMEND,
            position=0.1,
            alpha_source=["mock"],
            horizon_days=5,
            reasons=["LLM wants to recommend"],
            objections=[],
            risk_verdict=RiskVerdict.PASS,
            reasoning="mock",
        ))
        sa = create_system_agent(llm)
        state = {
            **base_state,
            "market_report_obj": _default_for(MarketReport),
            "event_report_obj": _default_for(EventReport),
            "analysis_report_obj": _default_for(AnalysisReport),
            "backtest_report_obj": _default_for(BacktestReport),
            "risk_check_3": {"verdict": "PASS", "reasons": []},
            "round2_state": {
                "active": True,
                "round_count": 8,
                "max_rounds": 8,
                "questions": [],
                "contradictions": ["Market/Event conflict"],
                "current_speaker": "AutoGenRoundtable",
                "completed": True,
                "summary": "System_Moderator: final_pressure=downgrade",
                "provider": "autogen",
                "fallback_reason": "",
                "final_pressure": "downgrade",
                "unresolved_conflicts": ["Market/Event conflict"],
            },
            "round2_summary": "System_Moderator: final_pressure=downgrade",
        }

        result = sa["final"](state)
        decision = result["system_decision_obj"]

        assert decision.decision == DecisionType.WATCH
        assert "Round 2 Moderator 要求降级" in decision.objections
        assert result["system_rubric"]["recommendation_floor"] == DecisionType.WATCH.value

    def test_final_decision_llm_fallback(self, mock_llm_fail, base_state):
        sa = create_system_agent(mock_llm_fail)
        state = {
            **base_state,
            "risk_check_3": {"verdict": "PASS", "reasons": []},
        }
        result = sa["final"](state)
        decision = result["system_decision_obj"]
        assert decision.decision == DecisionType.WATCH
        assert "LLM 不可用" in decision.reasons[0]

    def test_final_hard_veto_override(self, mock_llm, base_state):
        sa = create_system_agent(mock_llm)
        state = {
            **base_state,
            "risk_check_3": {"verdict": "HARD_VETO", "reasons": ["测试拒绝"]},
            "market_report_obj": _default_for(MarketReport),
        }
        result = sa["final"](state)
        decision = result["system_decision_obj"]
        assert decision.decision == DecisionType.REJECT
        assert decision.risk_verdict == RiskVerdict.HARD_VETO

    def test_final_risk_check_1_soft_veto_blocks_recommendation(self, base_state):
        llm = MockLLM(return_value=SystemDecision(
            decision=DecisionType.RECOMMEND,
            position=0.1,
            alpha_source=["mock"],
            horizon_days=5,
            reasons=["LLM wants to recommend"],
            objections=[],
            risk_verdict=RiskVerdict.PASS,
            reasoning="mock",
        ))
        sa = create_system_agent(llm)
        state = {
            **base_state,
            "market_report_obj": _default_for(MarketReport),
            "event_report_obj": _default_for(EventReport),
            "analysis_report_obj": _default_for(AnalysisReport),
            "backtest_report_obj": _default_for(BacktestReport),
            "risk_check_1": {
                "verdict": "SOFT_VETO",
                "reasons": ["风险基础数据缺失，无法确认 ST/停牌/退市状态"],
            },
            "risk_check_2": {"verdict": "PASS", "reasons": []},
            "risk_check_3": {"verdict": "PASS", "reasons": []},
        }

        result = sa["final"](state)
        decision = result["system_decision_obj"]

        assert decision.decision == DecisionType.WATCH
        assert decision.risk_verdict == RiskVerdict.SOFT_VETO
        assert "风险基础数据缺失，无法确认 ST/停牌/退市状态" in decision.objections
        assert result["system_rubric"]["recommendation_floor"] == DecisionType.WATCH.value

    def test_final_position_and_sector_risk_caps_recommendation(self, base_state):
        llm = MockLLM(return_value=SystemDecision(
            decision=DecisionType.RECOMMEND,
            position=0.2,
            alpha_source=["mock"],
            horizon_days=5,
            reasons=["LLM wants to recommend"],
            objections=[],
            risk_verdict=RiskVerdict.PASS,
            reasoning="mock",
        ))
        sa = create_system_agent(llm)
        state = {
            **base_state,
            "market_report_obj": _default_for(MarketReport),
            "event_report_obj": _default_for(EventReport),
            "analysis_report_obj": _default_for(AnalysisReport),
            "backtest_report_obj": _default_for(BacktestReport),
            "risk_check_1": {"verdict": "PASS", "reasons": []},
            "risk_check_2": {"verdict": "PASS", "reasons": []},
            "risk_check_3": {"verdict": "PASS", "reasons": []},
            "tier1_data": {
                **base_state["tier1_data"],
                "risk": {
                    "current_position": 0.58,
                    "current_sector_position": 0.28,
                    "proposed_position": 0.2,
                    "daily_volume": 100_000_000,
                },
            },
        }

        result = sa["final"](state)
        decision = result["system_decision_obj"]

        assert decision.decision == DecisionType.WATCH
        assert decision.risk_verdict == RiskVerdict.SOFT_VETO
        assert decision.position == 0
        assert any("总仓位" in item for item in decision.objections)
        assert any("单板块仓位" in item for item in decision.objections)


# ============================================================
# Report Agent 测试
# ============================================================

class TestReportAgent:
    """Report Agent — 报告输出"""

    def test_with_decision(self, base_state):
        node = create_report_agent()
        state = {
            **base_state,
            "system_decision_obj": SystemDecision(
                decision=DecisionType.RECOMMEND, position=0.1,
                alpha_source=["因子"], horizon_days=5,
                reasons=["好"], objections=[],
                risk_verdict=RiskVerdict.PASS, reasoning="test",
            ),
        }
        result = node(state)
        assert "final_report" in result
        assert "final_report_obj" in result
        report = result["final_report_obj"]
        assert isinstance(report, FinalReport)
        assert report.decision == DecisionType.RECOMMEND

    def test_without_decision(self, base_state):
        node = create_report_agent()
        result = node(base_state)
        report = result["final_report_obj"]
        assert report.decision == DecisionType.REJECT

    def test_persists_audit_trace(self, base_state, tmp_path):
        original_results_dir = config.get("results_dir")
        config.update({"results_dir": str(tmp_path)})
        try:
            node = create_report_agent()
            state = {
                **base_state,
                "approval_record": {"action": "pending_human_review"},
                "execution_allowed": False,
                "risk_check_1": {"verdict": "PASS"},
                "agent_evidence": {"Market Agent": ["sentiment=正常"]},
                "system_decision_obj": SystemDecision(
                    decision=DecisionType.WATCH,
                    position=0,
                    alpha_source=[],
                    horizon_days=5,
                    reasons=["观察"],
                    objections=[],
                    risk_verdict=RiskVerdict.PASS,
                    reasoning="test",
                ),
            }
            result = node(state)
        finally:
            config.update({"results_dir": original_results_dir})

        assert result["audit_trace_path"]
        audit_path = Path(result["audit_trace_path"])
        assert audit_path.exists()
        audit_text = audit_path.read_text(encoding="utf-8")
        assert "pending_human_review" in audit_text
        assert "Market Agent" in audit_text

    def test_report_includes_round2_audit_section(self, base_state, tmp_path):
        original_results_dir = config.get("results_dir")
        config.update({"results_dir": str(tmp_path)})
        try:
            node = create_report_agent()
            state = {
                **base_state,
                "round2_state": {
                    "active": True,
                    "round_count": 8,
                    "max_rounds": 8,
                    "questions": [],
                    "contradictions": ["Market/Event conflict"],
                    "current_speaker": "AutoGenRoundtable",
                    "completed": True,
                    "summary": "System_Moderator: final_pressure=downgrade",
                    "provider": "autogen",
                    "fallback_reason": "",
                    "final_pressure": "downgrade",
                    "unresolved_conflicts": ["Market/Event conflict"],
                },
                "round2_summary": "System_Moderator: final_pressure=downgrade",
                "system_decision_obj": SystemDecision(
                    decision=DecisionType.WATCH,
                    position=0,
                    alpha_source=[],
                    horizon_days=5,
                    reasons=["观察"],
                    objections=["Round 2 Moderator 要求降级"],
                    risk_verdict=RiskVerdict.PASS,
                    reasoning="test",
                ),
            }
            result = node(state)
        finally:
            config.update({"results_dir": original_results_dir})

        assert "## Round 2 圆桌审计" in result["final_report"]
        assert "Provider: autogen" in result["final_report"]
        assert "Final Pressure: downgrade" in result["final_report"]


class TestApprovalAgent:
    """Approval Agent — 人工审批记录"""

    def test_default_pending_review_blocks_execution(self, base_state):
        node = create_approval_agent()
        state = {
            **base_state,
            "system_decision_obj": SystemDecision(
                decision=DecisionType.RECOMMEND, position=0.1,
                alpha_source=["因子"], horizon_days=5,
                reasons=["好"], objections=[],
                risk_verdict=RiskVerdict.PASS, reasoning="test",
            ),
        }
        result = node(state)
        assert result["approval_record"]["action"] == "pending_human_review"
        assert result["execution_allowed"] is False
        assert result["sender"] == "Approval Agent"

    def test_approved_recommend_allows_execution(self, base_state):
        node = create_approval_agent()
        state = {
            **base_state,
            "approval_input": {"action": "approve", "reviewer": "tester"},
            "system_decision_obj": SystemDecision(
                decision=DecisionType.RECOMMEND, position=0.1,
                alpha_source=["因子"], horizon_days=5,
                reasons=["好"], objections=[],
                risk_verdict=RiskVerdict.PASS, reasoning="test",
            ),
        }
        result = node(state)
        assert result["approval_record"]["action"] == "approved"
        assert result["execution_allowed"] is True
