from __future__ import annotations

from advanced_trading_agent.data_agent.etf_watchlist import (
    SectorCandidatePayload,
    WatchlistETFCandidate,
    build_watchlist_report,
)
from advanced_trading_agent.data_agent.scanner import ScanResult
from advanced_trading_agent.data_agent.sector_etf import SectorETFSelector
from advanced_trading_agent.graph.sector_etf_workflow import SectorETFWatchlistSystem


def test_selector_returns_roundtable_candidates_and_exclusions():
    def fake_route(method, **kwargs):
        if method == "get_sector":
            return [
                {"sector_name": "半导体", "change_pct": 3.2, "strength_score": 4.0},
                {"sector_name": "冷门主题", "change_pct": 2.0, "strength_score": 3.0},
            ]
        if method == "get_etf_spot":
            return [
                {"code": "512480.SH", "name": "半导体ETF", "amount": 800_000_000},
                {"code": "510300.SH", "name": "沪深300ETF", "amount": 900_000_000},
            ]
        if method == "get_news":
            return [{"title": "半导体国产替代事件"}] if kwargs.get("sector") == "半导体" else []
        raise AssertionError(method)

    scan_results = [
        ScanResult("688001.SH", "芯片A", "hot_sector", "半导体", 8.0, "热板", extra={"all_sectors": ["半导体"]}),
        ScanResult("688002.SH", "芯片B", "hot_sector", "半导体", 7.0, "热板", extra={"all_sectors": ["半导体"]}),
        ScanResult("688003.SH", "芯片C", "limit_up", "半导体", 6.0, "涨停", extra={"all_sectors": ["半导体"]}),
    ]

    selector = SectorETFSelector(route_fn=fake_route, top_sectors=4, top_etfs_per_sector=5)
    selection = selector.select_with_exclusions("2026-07-15", scan_results=scan_results)

    assert [c.sector_name for c in selection.candidates] == ["半导体"]
    assert selection.candidates[0].primary_etf.code == "512480.SH"
    assert selection.excluded[0].sector == "冷门主题"
    assert selection.excluded[0].excluded_reason == "no_tradable_etf"


def test_watchlist_report_requires_primary_etf_and_limits_active_count():
    def fake_route(method, **kwargs):
        if method == "get_sector":
            return [{"sector_name": "半导体", "change_pct": 4.0, "strength_score": 5.0}]
        if method == "get_etf_spot":
            return [
                {"code": "512480.SH", "name": "半导体ETF", "amount": 800_000_000},
                {"code": "159995.SZ", "name": "芯片ETF", "amount": 700_000_000},
                {"code": "588200.SH", "name": "科创芯片ETF", "amount": 600_000_000},
            ]
        if method == "get_news":
            return [{"title": "事件"}]
        raise AssertionError(method)

    selector = SectorETFSelector(route_fn=fake_route, top_sectors=6, top_etfs_per_sector=5)
    selection = selector.select_with_exclusions(
        "2026-07-15",
        max_roundtable_sectors=6,
        scan_results=[
            ScanResult("688001.SH", "芯片A", "hot_sector", "半导体", 8.0, "热板", extra={"all_sectors": ["半导体"]}),
            ScanResult("688002.SH", "芯片B", "limit_up", "半导体", 7.0, "涨停", extra={"all_sectors": ["半导体"]}),
            ScanResult("688003.SH", "芯片C", "short_term_signal", "半导体", 6.0, "趋势", extra={"all_sectors": ["半导体"]}),
        ],
    )
    report = build_watchlist_report(
        trade_date="2026-07-15",
        candidates=selection.watchlist_payloads(),
        excluded=selection.excluded,
    )

    assert all(decision.primary_etf.code for decision in report.decisions)
    assert all(len(decision.backup_etfs) <= 2 for decision in report.decisions)
    assert len(report.decisions) <= 3
    assert sum(1 for decision in report.decisions if decision.status == "active") <= 3
    assert sum(decision.watchlist_weight_hint for decision in report.decisions) <= 0.60
    assert report.approval["execution_allowed"] is False
    assert report.roundtable_summary["backtest_used"] is False
    agents = {item["agent"] for item in report.roundtable_summary["agent_outputs"]}
    assert agents == {"Market", "Event", "Analysis", "Risk"}
    assert report.roundtable_summary["dialogue_records"][0]["speaker"] == "Moderator"
    assert any(turn["speaker"] == "Risk" for turn in report.roundtable_summary["dialogue_records"])
    assert report.roundtable_summary["round_history"][0]["turn_count"] >= 6


def test_watchlist_report_keeps_only_top_three_final_decisions():
    candidates = [_watchlist_candidate(f"主题{i}", f"51248{i}.SH", 12 - i) for i in range(6)]
    report = build_watchlist_report(
        trade_date="2026-07-15",
        candidates=candidates,
        excluded=[],
    )

    assert [decision.sector for decision in report.decisions] == ["主题0", "主题1", "主题2"]
    assert report.roundtable_summary["roundtable_candidate_count"] == 6
    assert report.roundtable_summary["decision_count"] == 3
    assert report.roundtable_summary["max_final_decisions"] == 3
    assert len(report.roundtable_summary["agent_outputs"]) == 24
    assert len(report.roundtable_summary["round_history"]) == 6
    assert all("turns" in item for item in report.roundtable_summary["round_history"])


def test_batch_watchlist_workflow_outputs_json_contract(tmp_path):
    class FakeSelector:
        def select_with_exclusions(self, trade_date, max_roundtable_sectors=8, **kwargs):
            selector = SectorETFSelector(route_fn=fake_route, top_sectors=2, top_etfs_per_sector=3)
            return selector.select_with_exclusions(
                trade_date,
                max_roundtable_sectors=max_roundtable_sectors,
                scan_results=[
                    ScanResult("688001.SH", "芯片A", "hot_sector", "半导体", 8.0, "热板", extra={"all_sectors": ["半导体"]}),
                    ScanResult("688002.SH", "芯片B", "limit_up", "半导体", 7.0, "涨停", extra={"all_sectors": ["半导体"]}),
                    ScanResult("688003.SH", "芯片C", "short_term_signal", "半导体", 6.0, "趋势", extra={"all_sectors": ["半导体"]}),
                ],
            )

    def fake_route(method, **kwargs):
        if method == "get_sector":
            return [{"sector_name": "半导体", "change_pct": 3.2, "strength_score": 4.0}]
        if method == "get_etf_spot":
            return [{"code": "512480.SH", "name": "半导体ETF", "amount": 800_000_000}]
        if method == "get_news":
            return [{"title": "半导体国产替代事件"}]
        raise AssertionError(method)

    from advanced_trading_agent.agents.conversation_memory import ConversationMemoryStore

    system = SectorETFWatchlistSystem(
        selector=FakeSelector(),
        memory_store=ConversationMemoryStore(path=str(tmp_path / "memory.jsonl")),
    )
    state, markdown = system.analyze(trade_date="2026-07-15", store_memory=True)

    report = state["watchlist_report"]
    assert report["scope"] == "a_share_sector_etf_watchlist"
    assert report["decisions"][0]["primary_etf"]["code"] == "512480.SH"
    assert "支持理由" in markdown
    assert "圆桌输出" in markdown
    assert "圆桌对话记录" in markdown
    assert report["roundtable_summary"]["backtest_used"] is False
    assert len(report["roundtable_summary"]["agent_outputs"]) == 4
    assert len(report["roundtable_summary"]["dialogue_records"]) >= 6
    assert report["roundtable_summary"]["round_history"][0]["sector"] == "半导体"
    assert report["roundtable_summary"]["timings"]["rules_roundtable_seconds"] >= 0


def _watchlist_candidate(sector: str, code: str, score: float) -> SectorCandidatePayload:
    return SectorCandidatePayload(
        sector_name=sector,
        pre_score=score,
        momentum_score=6.0,
        breadth_score=2.0,
        event_score=2.0,
        evidence={
            "momentum": [f"{sector}动量强"],
            "breadth": [f"{sector}宽度够"],
            "events": [f"{sector}事件催化"],
        },
        support_evidence=[f"{sector}综合证据"],
        raw_etf_candidates=[
            WatchlistETFCandidate(
                code=code,
                name=f"{sector}ETF",
                match_score=5.0,
                liquidity_score=3.0,
                tracking_purity_score=2.0,
                total_score=10.0,
                reason=f"{sector}首选ETF",
            )
        ],
    )
