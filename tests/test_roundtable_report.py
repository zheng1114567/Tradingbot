from __future__ import annotations

from advanced_trading_agent.core.audit import format_roundtable_visualization


def test_roundtable_visualization_handles_debate_engine_history():
    text = format_roundtable_visualization(
        {
            "provider": "debate_engine",
            "completed": True,
            "round_count": 1,
            "final_pressure": "downgrade",
            "contradiction_records": [{"description": "Market/Event conflict"}],
            "round_history": [
                {
                    "round_number": 0,
                    "turns": [
                        {
                            "agent_name": "Risk",
                            "stance": {
                                "pressure": "downgrade",
                                "confidence": 0.8,
                                "reasoning": "流动性不足",
                                "evidence_ids": ["ev_risk_001"],
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert "Risk" in text
    assert "流动性不足" in text
    assert "Final pressure" in text


def test_roundtable_visualization_handles_autogen_questions():
    text = format_roundtable_visualization(
        {
            "provider": "autogen",
            "completed": True,
            "round_count": 1,
            "contradiction_records": [{"description": "Market/Event conflict"}],
            "questions": [
                {
                    "answers": [
                        {
                            "target_agent": "HotMoney_Agent",
                            "answer": "短线过热，建议降级",
                            "evidence": "a_share_signals.hot_money.signal=overheated",
                        }
                    ]
                }
            ],
        }
    )

    assert "HotMoney_Agent" in text
    assert "短线过热" in text
