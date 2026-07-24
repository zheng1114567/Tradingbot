from __future__ import annotations

from advanced_trading_agent.data_agent.request import DataAgentRequest


def test_data_agent_request_rolls_weekend_trade_date_to_previous_weekday():
    request = DataAgentRequest(ticker="000001.SZ", trade_date="2026-07-11")

    assert request.normalized_trade_date() == "2026-07-10"
    assert request.normalized_end_date() == "20260710"
