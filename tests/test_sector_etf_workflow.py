from __future__ import annotations

from advanced_trading_agent.agents.conversation_memory import ConversationMemoryStore
from advanced_trading_agent.graph.sector_etf_workflow import SectorETFTradingSystem


class FakeSelector:
    def explain_sector(self, sector_name, trade_date):
        return {
            "sector_name": sector_name,
            "status": "matched",
            "verdict": "暂不适合",
            "score": 4.8,
            "reasons": ["板块动量=2.0", "候选宽度=0"],
            "risks": ["板块内强势样本少，可能只是个别成分股扰动。"],
            "primary_etf": {"code": "512480.SH", "name": "半导体ETF"},
            "candidate": {"raw": {"news": []}},
        }


def fake_roundtable(question, **kwargs):
    return {
        "provider": "fake_roundtable",
        "answer": f"{kwargs['sector_name']}不适合：宽度不足。",
        "final_pressure": "downgrade",
        "roundtable": {"round_history": [{"round": 1}]},
        "evidence": kwargs["explanation"],
    }


def test_sector_etf_workflow_runs_langgraph_roundtable_and_memory(tmp_path):
    memory = ConversationMemoryStore(path=str(tmp_path / "conversation.jsonl"))
    system = SectorETFTradingSystem(
        selector=FakeSelector(),
        roundtable_fn=fake_roundtable,
        memory_store=memory,
    )

    state, report = system.analyze(
        "半导体",
        question="半导体板块为什么不好？",
        trade_date="2026-07-15",
        use_autogen=False,
    )

    assert state["sector_evidence"]["sector_name"] == "半导体"
    assert state["roundtable_result"]["provider"] == "fake_roundtable"
    assert "板块ETF LangGraph 决策报告" in report
    assert "半导体不适合" in report
    rows = memory.load()
    assert rows[0]["target"] == "半导体"
    assert rows[0]["evidence"]["provider"] == "fake_roundtable"


def test_sector_etf_workflow_can_skip_memory(tmp_path):
    memory = ConversationMemoryStore(path=str(tmp_path / "conversation.jsonl"))
    system = SectorETFTradingSystem(
        selector=FakeSelector(),
        roundtable_fn=fake_roundtable,
        memory_store=memory,
    )

    _state, _report = system.analyze(
        "半导体",
        question="半导体板块为什么不好？",
        trade_date="2026-07-15",
        use_autogen=False,
        store_memory=False,
    )

    assert memory.load() == []
