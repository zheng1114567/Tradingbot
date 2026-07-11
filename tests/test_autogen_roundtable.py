"""AutoGen roundtable adapter tests."""

from advanced_trading_agent.roundtable import AutoGenRoundtable, RoundtableResult


def test_autogen_result_pressure_parsing():
    result = AutoGenRoundtable._to_result(
        ["Market/Event conflict"],
        [
            {"source": "Market_Agent", "content": "资金不足，建议降级。"},
            {"source": "System_Moderator", "content": "final_pressure=downgrade"},
        ],
    )

    assert isinstance(result, RoundtableResult)
    assert result.final_pressure == "downgrade"
    assert result.unresolved_conflicts == ["Market/Event conflict"]
    assert result.questions[0]["answers"]


def test_autogen_task_contains_agent_reports():
    state = {
        "market_report": "market",
        "event_report": "event",
        "analysis_report": "analysis",
        "backtest_report": "backtest",
    }

    task = AutoGenRoundtable._build_task(state, ["conflict"])

    assert "conflict" in task
    assert "每个 Agent 只能基于自身 system message" in task
    assert "market" not in task
    assert "backtest" not in task


def test_autogen_agent_report_is_scoped_and_truncated():
    state = {"market_report": "m" * 1300}

    report = AutoGenRoundtable._agent_report(state, "market_report")

    assert report == "m" * 1200
    assert AutoGenRoundtable._agent_report({}, "missing") == "暂无该 Agent 报告"
