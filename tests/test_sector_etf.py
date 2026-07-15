from __future__ import annotations

from advanced_trading_agent.data_agent.scanner import ScanResult
from advanced_trading_agent.data_agent.sector_etf import SectorETFSelector


def test_sector_selector_defaults_to_saved_cache_without_auto_refresh():
    selector = SectorETFSelector(route_fn=lambda *_args, **_kwargs: [])

    assert selector.scanner._auto_refresh_cache is False


def test_sector_selector_can_opt_in_to_cache_refresh():
    selector = SectorETFSelector(
        route_fn=lambda *_args, **_kwargs: [],
        auto_refresh_cache=True,
    )

    assert selector.scanner._auto_refresh_cache is True


def test_sector_selector_maps_hot_sector_to_liquid_etf():
    calls: list[tuple[str, dict]] = []

    def fake_route(method, **kwargs):
        calls.append((method, kwargs))
        if method == "get_sector":
            return [
                {"sector_name": "半导体", "change_pct": 3.2, "strength_score": 4.0},
                {"sector_name": "银行", "change_pct": 0.5, "strength_score": 1.0},
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
        ScanResult("688001.SH", "芯片A", "hot_sector+limit_up", "半导体", 9.0, "涨停池", extra={"all_sectors": ["半导体"]}),
        ScanResult("688002.SH", "芯片B", "hot_sector", "半导体", 8.0, "热板", extra={"all_sectors": ["半导体"]}),
        ScanResult("688003.SH", "芯片C", "short_term_signal", "半导体", 7.0, "趋势", extra={"all_sectors": ["半导体"]}),
    ]

    selector = SectorETFSelector(route_fn=fake_route, top_sectors=3)
    candidates = selector.select("2026-07-15", sector_query="半导体", scan_results=scan_results)

    assert candidates[0].sector_name == "半导体"
    assert candidates[0].primary_etf is not None
    assert candidates[0].primary_etf.code == "512480.SH"
    assert candidates[0].score >= 6
    assert any(call[0] == "get_etf_spot" for call in calls)


def test_sector_selector_explains_why_sector_is_not_good_without_etf_or_events():
    def fake_route(method, **kwargs):
        if method == "get_sector":
            return [{"sector_name": "冷门主题", "change_pct": 0.1, "strength_score": 0.2}]
        if method == "get_etf_spot":
            return [{"code": "510300.SH", "name": "沪深300ETF", "amount": 900_000_000}]
        if method == "get_news":
            return []
        raise AssertionError(method)

    selector = SectorETFSelector(route_fn=fake_route, top_sectors=3)
    explanation = selector.explain_sector("冷门主题", "2026-07-15")

    assert explanation["verdict"] == "暂不适合"
    assert any("未匹配到可交易ETF" in risk for risk in explanation["risks"])
    assert any("缺少新闻" in risk for risk in explanation["risks"])
