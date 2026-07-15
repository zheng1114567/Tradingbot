from __future__ import annotations

from advanced_trading_agent.data_agent.etf_watchlist import build_watchlist_report
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
            return [{"sector_name": f"主题{i}", "change_pct": 4.0, "strength_score": 5.0} for i in range(6)]
        if method == "get_etf_spot":
            return [
                {"code": f"51248{i}.SH", "name": f"主题{i}ETF", "amount": 800_000_000}
                for i in range(6)
            ]
        if method == "get_news":
            return [{"title": "事件"}]
        raise AssertionError(method)

    selector = SectorETFSelector(route_fn=fake_route, top_sectors=6, top_etfs_per_sector=5)
    selection = selector.select_with_exclusions("2026-07-15", max_roundtable_sectors=6)
    report = build_watchlist_report(
        trade_date="2026-07-15",
        candidates=selection.watchlist_payloads(),
        excluded=selection.excluded,
    )

    assert all(decision.primary_etf.code for decision in report.decisions)
    assert all(len(decision.backup_etfs) <= 2 for decision in report.decisions)
    assert sum(1 for decision in report.decisions if decision.status == "active") <= 4
    assert sum(decision.watchlist_weight_hint for decision in report.decisions) <= 0.60
    assert report.approval["execution_allowed"] is False


def test_batch_watchlist_workflow_outputs_json_contract(tmp_path):
    class FakeSelector:
        def select_with_exclusions(self, trade_date, max_roundtable_sectors=8, **kwargs):
            selector = SectorETFSelector(route_fn=fake_route, top_sectors=2, top_etfs_per_sector=3)
            return selector.select_with_exclusions(trade_date, max_roundtable_sectors=max_roundtable_sectors, scan_results=[ScanResult("688001.SH", "芯片A", "hot_sector", "半导体", 8.0, "热板", extra={"all_sectors": ["半导体"]}), ScanResult("688002.SH", "芯片B", "limit_up", "半导体", 7.0, "涨停", extra={"all_sectors": ["半导体"]}), ScanResult("688003.SH", "芯片C", "short_term_signal", "半导体", 6.0, "趋势", extra={"all_sectors": ["半导体"]})])

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
