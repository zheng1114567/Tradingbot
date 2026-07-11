"""Standalone DataAgent tests."""

import json

from advanced_trading_agent.data_agent.data_agent import DataAgent, DataAgentRequest


def test_data_agent_persists_layered_trace(tmp_path):
    def fake_route(method, **kwargs):
        if method == "get_daily":
            return [
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260709",
                    "open": 10.0,
                    "high": 10.6,
                    "low": 9.8,
                    "close": 10.4,
                    "pre_close": 10.0,
                    "pct_chg": 4.0,
                    "vol": 1000,
                    "amount": 10400,
                },
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260710",
                    "open": 10.4,
                    "high": 11.5,
                    "low": 10.4,
                    "close": 11.44,
                    "pre_close": 10.4,
                    "pct_chg": 10.0,
                    "vol": 1200,
                    "amount": 13728,
                },
            ]
        if method == "get_capital_flow":
            return [{"ts_code": kwargs["code"], "trade_date": "20260710", "net_mf_amount": 88.0}]
        raise AssertionError(f"unexpected method: {method}")

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path)).run(
        DataAgentRequest(
            ticker="000001.SZ",
            trade_date="2026-07-10",
            start_date="20260701",
            end_date="20260710",
        )
    )

    payload = result.to_dict()
    assert set(payload["artifacts"]) == {"input", "raw", "cleaned", "analysis", "final"}
    for artifact in payload["artifacts"].values():
        assert str(tmp_path) in artifact["path"]

    final_path = result.artifacts["final"].path
    final_payload = json.loads(open(final_path, encoding="utf-8").read())
    assert final_payload["input"]["request"]["ticker"] == "000001.SZ"
    assert final_payload["cleaned"]["daily"]["record_count"] == 2
    assert final_payload["analysis"]["summary"]["latest"]["code"] == "000001.SZ"
    assert final_payload["manifest"]["fields"]["stock.daily"]["available"] is True


def test_data_agent_react_planner_persists_plan(tmp_path):
    calls = []

    def fake_route(method, **kwargs):
        calls.append((method, kwargs))
        if method == "get_daily":
            return [
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260710",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.9,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "pct_chg": 2.0,
                    "vol": 1000,
                    "amount": 10200,
                }
            ]
        raise AssertionError(f"unexpected method: {method}")

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path)).run(
        DataAgentRequest(
            ticker="000001.SZ",
            trade_date="2026-07-10",
            include_capital_flow=False,
            include_factors=False,
            use_react_planner=True,
        )
    )

    assert "planner" in result.artifacts
    assert result.plan is not None
    assert result.plan["required_methods"] == ["get_daily"]
    assert result.request["end_date"] == "20260710"
    assert calls[0][0] == "get_daily"

    final_payload = json.loads(open(result.response_path, encoding="utf-8").read())
    assert final_payload["planner"]["trace"][0]["action"] == "inspect_request"
    assert final_payload["input"]["planner"]["skipped_methods"] == [
        "get_capital_flow",
        "compute_factors",
    ]
