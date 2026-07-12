from __future__ import annotations

from advanced_trading_agent.data_agent.data_agent import DataAgentArtifact, DataAgentRun
from advanced_trading_agent.data_agent.reports import format_dataagent_report, format_scan_report
from advanced_trading_agent.data_agent.scanner import ScanBundle, ScanResult


def _bundle() -> ScanBundle:
    result = ScanResult(
        ticker="000001.SZ",
        name="平安银行",
        source="hot_sector+limit_up",
        sector="银行",
        score=8.5,
        reason="涨停池; 板块共振",
    )
    return ScanBundle(
        trade_date="2026-07-10",
        results=[result],
        shared_raw={
            "market": [{"close": 3500}],
            "sector_context": [{"sector_name": "银行"}],
            "risk": {"st_status": [], "suspended": [], "delisting": []},
        },
        ticker_data={
            "000001.SZ": {
                "daily": [{"close": 10.5}],
                "capital_flow": [{"net_inflow_main": 88}],
                "news": [{"title": "经营动态"}],
            }
        },
        route_trace=[
            {"method": "get_daily", "vendor": "local_cache", "status": "success"},
            {"method": "get_news", "vendor": "akshare", "status": "error"},
        ],
    )


def _run() -> DataAgentRun:
    return DataAgentRun(
        run_id="run-1",
        request={"ticker": "000001.SZ", "trade_date": "2026-07-10"},
        artifacts={"final": DataAgentArtifact(stage="final", path="out/response.json")},
        manifest_path="out/manifest.json",
        response_path="out/response.json",
        final_data={
            "cleaned": {"daily": {"record_count": 1}},
            "analysis": {
                "factors": {"record_count": 1},
                "events": {"record_count": 1},
            },
            "agent_payload": {
                "tier1_data": {
                    "market": {"index_close": 3500, "index_change_pct": 1.2},
                    "sentiment": {"sentiment": "温热", "sentiment_score": 65},
                    "capital": {"confirmation": "资金确认", "net_inflow_main": 88},
                    "sector": {"status": "matched", "matched_sector": "银行", "match_confidence": 0.9},
                    "risk": {"risk_data_available": True, "is_limit_up": True, "is_limit_down": False},
                },
                "tier2_data": {
                    "price_data": [{"close": 10.5}],
                    "factors": [{"momentum_score": 0.1}],
                    "events": [{"summary": "经营动态"}],
                    "sector_context": {"status": "matched"},
                    "data_quality": {"daily_consistency": {"status": "single_source", "confidence_score": 0.7}},
                },
            },
        },
        collection_summary={
            "categories_with_data": 4,
            "total_categories": 8,
            "categories_empty": 4,
            "categories_failed": 0,
        },
    )


def test_format_scan_report_includes_candidates_and_health():
    report = format_scan_report(_bundle(), llm_summary="市场偏热", model="deepseek-chat")

    assert "# Scan Report" in report
    assert "000001.SZ" in report
    assert "市场偏热" in report
    assert "Data Ready" in report
    assert "get_daily" in report


def test_format_dataagent_report_includes_tiered_data():
    report = format_dataagent_report([_run()], bundle=_bundle(), model="deepseek-chat")

    assert "# DataAgent Layered Report" in report
    assert "Tier 1 Default Context" in report
    assert "Tier 2 On-Demand Data" in report
    assert "tier1+tier2" in report
    assert "single_source" in report
