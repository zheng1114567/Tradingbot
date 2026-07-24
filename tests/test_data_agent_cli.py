"""Tests for the standalone DataAgent CLI."""
from __future__ import annotations

import json

from advanced_trading_agent.data_agent import cli


class FakeRun:
    def to_dict(self):
        payload = {
            "run_id": "run-1",
            "response_path": "out/response.json",
            "manifest_path": "out/manifest.json",
            "final_data": {
                "cleaned": {
                    "daily": {"record_count": 2},
                    "news": {"record_count": 1},
                },
                "analysis": {
                    "sector": {
                        "status": "matched",
                        "matched_sector": "银行",
                        "match_confidence": 0.9,
                        "top_sectors": [{"sector_name": "银行"}],
                    },
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
        payload["artifacts"] = {"news_events": {"path": "out/news_events.json"}}
        return payload


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
        "--sector-keyword",
        "Bank",
        "--no-llm-news-filter",
        "--no-news-full-text",
        "--no-market",
        "--no-sector-context",
        "--sector-top-n",
        "7",
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
    assert request.sector_keyword == "Bank"
    assert request.use_llm_news_filter is False
    assert request.fetch_news_full_text is False
    assert request.include_market is False
    assert request.include_sector_context is False
    assert request.sector_top_n == 7

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-1"
    assert payload["daily_records"] == 2
    assert payload["news_events_path"] == "out/news_events.json"
    assert payload["news_filter"]["used_llm"] is True
    assert payload["sector"]["matched_sector"] == "银行"
    assert payload["sector"]["top_sector_count"] == 1
    assert payload["daily_consistency"]["status"] == "single_source"
    assert payload["vendor_health"]["attempt_count"] == 3


def test_data_agent_cli_json_mode_prints_full_payload(monkeypatch, capsys):
    FakeDataAgent.calls = []
    monkeypatch.setattr(cli, "DataAgent", FakeDataAgent)

    exit_code = cli.main(["--ticker", "000001.SZ", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["final_data"]["vendor_health"]["attempt_count"] == 3

class FakeFullRun:
    def to_dict(self):
        return {
            "stage": "full_analysis",
            "analysis_mode": "rules",
            "ticker": "000001.SZ",
            "trade_date": "2026-07-10",
            "data_agent": {
                "run_id": "run-full",
                "response_path": "out/response.json",
                "manifest_path": "out/manifest.json",
                "collection_summary": {
                    "categories_with_data": 5,
                    "categories_failed": 0,
                    "categories_empty": 1,
                },
                "errors": [],
            },
            "analysis": {
                "final_report_path": "out/report.md",
                "audit_trace_path": "out/audit.json",
                "execution_allowed": False,
                "round2_state": {"final_pressure": "downgrade"},
            },
        }


def test_data_agent_cli_analyze_mode_runs_full_pipeline(monkeypatch, capsys):
    calls = []

    def fake_run_full_analysis(**kwargs):
        calls.append(kwargs)
        return FakeFullRun()

    import advanced_trading_agent.pipeline as pipeline

    monkeypatch.setattr(pipeline, "run_full_analysis", fake_run_full_analysis)

    exit_code = cli.main([
        "--ticker", "000001.SZ",
        "--date", "2026-07-10",
        "--start-date", "20260701",
        "--end-date", "20260710",
        "--sector-keyword", "银行",
        "--analyze",
        "--skip-backtest",
        "--lookback-days", "45",
        "--store-memory",
    ])

    assert exit_code == 0
    assert calls == [{
        "ticker": "000001.SZ",
        "trade_date": "2026-07-10",
        "start_date": "20260701",
        "end_date": "20260710",
        "output_dir": None,
        "use_react_planner": False,
        "news_keyword": None,
        "sector_keyword": "银行",
        "use_llm_news_filter": True,
        "fetch_news_full_text": True,
        "skip_backtest": True,
            "analysis_mode": "rules",
            "lookback_days": 45,
            "store_memory": True,
    }]
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "full_analysis"
    assert payload["analysis_mode"] == "rules"
    assert payload["run_id"] == "run-full"
    assert payload["report_path"] == "out/report.md"
    assert payload["final_pressure"] == "downgrade"
    assert payload["collection"]["categories_with_data"] == 5
