"""Tests for the standalone DataAgent CLI."""
from __future__ import annotations

import json

from advanced_trading_agent.data_agent import cli


class FakeRun:
    def to_dict(self):
        return {
            "run_id": "run-1",
            "response_path": "out/response.json",
            "manifest_path": "out/manifest.json",
            "final_data": {
                "cleaned": {
                    "daily": {"record_count": 2},
                    "news": {"record_count": 1},
                },
                "analysis": {
                    "events": {
                        "record_count": 1,
                        "filter": {"mode": "llm", "used_llm": True},
                    },
                    "data_quality": {
                        "daily_consistency": {"status": "single_source"},
                    },
                },
                "vendor_health": {
                    "attempt_count": 3,
                    "vendors": {"akshare": {"success_count": 2}},
                },
            },
        }


class FakeDataAgent:
    calls = []

    def __init__(self, *, results_dir=None):
        self.results_dir = results_dir

    def run(self, request):
        self.__class__.calls.append({"results_dir": self.results_dir, "request": request})
        return FakeRun()


def test_data_agent_cli_runs_and_prints_summary(monkeypatch, capsys):
    FakeDataAgent.calls = []
    monkeypatch.setattr(cli, "DataAgent", FakeDataAgent)

    exit_code = cli.main([
        "--ticker",
        "000001.SZ",
        "--date",
        "2026-07-10",
        "--start-date",
        "20260701",
        "--end-date",
        "20260710",
        "--output-dir",
        "data/results",
        "--react-planner",
        "--news-keyword",
        "Ping An",
        "--no-llm-news-filter",
        "--no-market",
    ])

    assert exit_code == 0
    request = FakeDataAgent.calls[0]["request"]
    assert FakeDataAgent.calls[0]["results_dir"] == "data/results"
    assert request.ticker == "000001.SZ"
    assert request.trade_date == "2026-07-10"
    assert request.start_date == "20260701"
    assert request.end_date == "20260710"
    assert request.output_dir == "data/results"
    assert request.use_react_planner is True
    assert request.news_keyword == "Ping An"
    assert request.use_llm_news_filter is False
    assert request.include_market is False

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-1"
    assert payload["daily_records"] == 2
    assert payload["news_filter"]["used_llm"] is True
    assert payload["daily_consistency"]["status"] == "single_source"
    assert payload["vendor_health"]["attempt_count"] == 3


def test_data_agent_cli_json_mode_prints_full_payload(monkeypatch, capsys):
    FakeDataAgent.calls = []
    monkeypatch.setattr(cli, "DataAgent", FakeDataAgent)

    exit_code = cli.main(["--ticker", "000001.SZ", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["final_data"]["vendor_health"]["attempt_count"] == 3
