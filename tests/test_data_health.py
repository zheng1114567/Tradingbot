from __future__ import annotations

from advanced_trading_agent.data_agent.data_health import build_daily_health_report
from advanced_trading_agent.data_agent.data_health import refresh_etf_cache, run_data_source_health


def test_daily_health_report_flags_duplicate_dates():
    records = [
        {"trade_date": "2026-07-10", "open": 10, "high": 11, "low": 9, "close": 10.5},
        {"trade_date": "2026-07-10", "open": 10, "high": 11, "low": 9, "close": 10.6},
    ]

    report = build_daily_health_report(records, start_date="20260710", end_date="20260710")

    assert report["status"] == "warning"
    assert any(issue["code"] == "duplicate_dates" for issue in report["issues"])


def test_daily_health_report_includes_cache_entry():
    cache_entry = {"status": "cache_hit", "source": "baostock"}
    report = build_daily_health_report(
        [{"date": "2026-07-10", "close": 10.5}],
        cache_entry=cache_entry,
    )

    assert report["status"] == "ok"
    assert report["cache"] == cache_entry


def test_data_source_health_reports_required_probe_statuses():
    def fake_route(method, _route_trace=None, **kwargs):
        if _route_trace is not None:
            _route_trace.append({"method": method, "vendor": "fake", "status": "success"})
        if method == "get_daily":
            return [{"code": kwargs["code"], "close": 10}]
        if method == "get_sector":
            return [{"sector_name": "半导体"}]
        if method == "get_limit_up_tiers":
            return {"stocks": [{"code": "000001"}]}
        if method == "get_news":
            return [{"title": kwargs.get("keyword", "news")}]
        if method == "get_etf_spot":
            return [{"code": "512480.SH", "amount": 50_000_000}]
        raise AssertionError(method)

    report = run_data_source_health("2026-07-10", route_fn=fake_route)

    assert report["overall_status"] == "ok"
    assert report["probes"]["a_share_daily"]["status"] == "ok"
    assert report["probes"]["ticker_news"]["status"] == "ok"
    assert report["probes"]["sector_news"]["status"] == "ok"
    assert report["probes"]["etf_spot"]["success_vendor"] == "fake"


def test_refresh_etf_cache_selects_liquid_etfs_and_fetches_daily():
    calls = []

    def fake_route(method, _route_trace=None, **kwargs):
        calls.append((method, kwargs))
        if _route_trace is not None:
            _route_trace.append({"method": method, "vendor": "fake", "status": "success"})
        if method == "get_etf_spot":
            return [
                {"code": "512480.SH", "amount": 50_000_000},
                {"code": "159995.SZ", "amount": 80_000_000},
            ]
        if method == "get_etf_daily":
            return [{"code": kwargs["code"], "trade_date": "2026-07-10", "close": 1.0}]
        raise AssertionError(method)

    result = refresh_etf_cache("2026-07-10", daily_limit=1, route_fn=fake_route)

    assert result["spot"]["status"] == "ok"
    assert result["daily"]["requested"] == 1
    assert result["daily"]["success_count"] == 1
    assert calls[1] == (
        "get_etf_daily",
        {"code": "159995.SZ", "start_date": "2026-07-10", "end_date": "2026-07-10"},
    )
