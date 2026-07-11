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
        "market_report": "PRIVATE_MARKET_REPORT",
        "event_report": "PRIVATE_EVENT_REPORT",
        "analysis_report": "PRIVATE_ANALYSIS_REPORT",
        "backtest_report": "PRIVATE_BACKTEST_REPORT",
    }

    task = AutoGenRoundtable._build_task(state, ["conflict"])

    assert "conflict" in task
    assert "DATA_AGENT_BRIEF" in task
    assert "每个 Agent 必须基于自己的 AgentContext" in task
    assert "PRIVATE_MARKET_REPORT" not in task
    assert "PRIVATE_BACKTEST_REPORT" not in task


def test_autogen_agent_report_is_scoped_and_truncated():
    state = {"market_report": "m" * 1300}

    report = AutoGenRoundtable._agent_report(state, "market_report")

    assert report == "m" * 1200
    assert AutoGenRoundtable._agent_report({}, "missing") == "暂无该 Agent 报告"
