from __future__ import annotations

import numpy as np

from advanced_trading_agent.data_agent.scan_cleaning import json_safe_value
from advanced_trading_agent.roundtable import RoundtableHarness


def test_json_safe_value_preserves_numpy_bool():
    assert json_safe_value(np.bool_(True)) is True
    assert json_safe_value(np.bool_(False)) is False


def test_evidence_board_preserves_zero_values_and_risk_context():
    state = {
        "company_of_interest": "000001.SZ",
        "trade_date": "2026-07-10",
        "tier1_data": {
            "market": {"index_change_pct": 0},
            "sentiment": {"sentiment": "正常"},
            "capital": {"confirmation": "资金背离"},
            "risk": {"risk_data_available": False, "risk_data_errors": ["risk api down"]},
        },
        "tier2_data": {
            "factors": [{"name": "x", "composite_score": 0}],
            "backtest_samples": [],
            "data_quality": {"daily_consistency": {"status": "conflict"}},
            "a_share_signals": {
                "hot_money": {"signal": "confirmed", "score": 0, "board_count": 0, "warnings": ["warn"]}
            },
        },
        "risk_check_2": {"verdict": "HARD_VETO", "reasons": ["limit up"]},
    }

    board = RoundtableHarness().build_evidence_board(state)
    fields = {item.field_path: item.value for item in board}

    assert fields["tier1_data.market.index_change_pct"] == "0"
    assert fields["tier2_data.factors[0].composite_score"] == "0"
    assert fields["a_share_signals.hot_money.score"] == "0"
    assert any("risk_data_errors" in item.field_path for item in board)


def test_hot_money_context_builds_system_message_and_tools():
    context = RoundtableHarness().build_context(
        {
            "company_of_interest": "000001.SZ",
            "trade_date": "2026-07-10",
            "tier1_data": {
                "market": {"index_change_pct": 0},
                "sentiment": {"sentiment": "正常"},
                "capital": {"confirmation": "资金确认"},
                "risk": {"risk_data_available": True},
            },
            "tier2_data": {
                "a_share_signals": {
                    "hot_money": {
                        "signal": "confirmed",
                        "score": 0,
                        "board_count": 0,
                        "data_status": "available",
                    }
                }
            },
        },
        ["hot-money signal needs review"],
    )

    hot_money = context.agent_contexts["HotMoney"]
    assert "HotMoney Specialist" in hot_money.system_message
    assert "get_limit_up_tiers" in hot_money.allowed_tool_names

