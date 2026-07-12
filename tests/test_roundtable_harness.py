"""Roundtable harness behavior tests."""

from advanced_trading_agent.roundtable import RoundtableHarness
from advanced_trading_agent.graph.workflow import _create_round2_subgraph


def test_roundtable_harness_builds_dataaware_agent_contexts():
    state = {
        "company_of_interest": "000001.SZ",
        "trade_date": "2026-07-10",
        "tier1_data": {
            "market": {"index_close": 3000, "index_change_pct": -0.8},
            "sentiment": {"sentiment": "正常", "sentiment_score": 55},
            "capital": {"confirmation": "资金背离", "net_inflow_main": -120000000},
            "risk": {"risk_data_available": True, "daily_volume": 20000000},
            "sector": {"matched_sector": "银行", "match_confidence": 0.8},
        },
        "tier2_data": {
            "events": [
                {
                    "event_id": "news_0001",
                    "summary": "平安银行零售业务保持稳定。",
                    "evidence_text": "平安银行零售业务保持稳定。",
                    "confidence": 0.72,
                }
            ],
            "factors": [
                {
                    "code": "000001.SZ",
                    "name": "平安银行",
                    "composite_score": 7.8,
                    "quality_score": 8.0,
                }
            ],
            "price_data": [{"trade_date": "2026-07-10", "close": 10, "pct_chg": -0.5}],
            "data_quality": {"daily_consistency": {"confidence_score": 0.7}},
        },
        "pit_manifest": {"fields": {"stock.daily": {"available": True}}},
        "market_report": "Market: 资金背离。",
        "event_report": "Event: 零售业务稳定，偏利好。",
        "analysis_report": "Analysis: 因子评分较高。",
        "backtest_report": "Backtest: 样本不足。",
    }

    context = RoundtableHarness().build_context(state, ["Market/Event conflict"])

    assert context.data_brief.ticker == "000001.SZ"
    assert "资金背离" in context.agent_contexts["Market"].evidence_text
    assert "平安银行零售业务保持稳定" in context.agent_contexts["Event"].evidence_text
    assert "composite_score" in context.agent_contexts["Analysis"].evidence_text
    assert "confidence_score" in context.agent_contexts["Backtest"].evidence_text
    assert "stock.daily" in context.shared_evidence_text
    assert "只能引用自己的 AgentContext" in context.agent_contexts["Market"].system_message
    assert "DATA_AGENT_BRIEF" in context.task


def test_roundtable_fallback_produces_valid_state(monkeypatch):
    class FailingLLM:
        def chat(self, *args, **kwargs):
            raise RuntimeError("offline")

    def fail_autogen(self, state):
        raise RuntimeError("autogen offline")

    monkeypatch.setattr(
        "advanced_trading_agent.graph.workflow.AutoGenRoundtable.run",
        fail_autogen,
    )
    graph = _create_round2_subgraph(FailingLLM())
    state = {
        "company_of_interest": "000001.SZ",
        "trade_date": "2026-07-10",
        "tier1_data": {
            "capital": {"confirmation": "资金背离", "net_inflow_main": -120000000},
            "sentiment": {"sentiment": "正常", "sentiment_score": 55},
            "risk": {"risk_data_available": True},
        },
        "tier2_data": {
            "events": [{"summary": "平安银行零售业务保持稳定。", "confidence": 0.72}],
            "factors": [{"code": "000001.SZ", "composite_score": 7.8}],
            "data_quality": {"daily_consistency": {"confidence_score": 0.7}},
        },
        "market_report": "Market: 资金背离。",
        "event_report": "Event: 偏利好。",
        "analysis_report": "Analysis: 高分。",
        "backtest_report": "Backtest: 数据不足。",
        "round2_state": {
            "active": True,
            "round_count": 0,
            "max_rounds": 1,
            "contradiction_records": [{"id": "ct_0", "description": "Market:资金背离 ↔ Event:利好", "agents_involved": [], "detection_method": "pattern", "severity": "medium"}],
            "current_speaker": "",
            "completed": False,
            "summary": "",
            "provider": "none",
            "fallback_reason": "",
            "final_pressure": "neutral",
            "unresolved_conflicts": [],
        },
    }

    result = graph.invoke(state)

    round2 = result["round2_state"]
    assert round2["completed"] is True
    assert round2["final_pressure"] in ("neutral", "downgrade", "upgrade")
