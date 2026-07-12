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


