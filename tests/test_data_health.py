from __future__ import annotations

from advanced_trading_agent.data_agent.data_health import build_daily_health_report


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
