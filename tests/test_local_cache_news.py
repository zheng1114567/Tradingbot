from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from advanced_trading_agent.data_agent.build_cache import ensure_candidate_daily_cache
from advanced_trading_agent.data_agent.local_cache import (
    get_cached_daily,
    get_cached_market_breadth,
    get_cached_news,
    get_cached_northbound_top10,
    get_cached_risk_snapshot,
)


def test_get_cached_news_reads_trade_date_file(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    news_dir = cache_root / "news" / "2026-07-10"
    news_dir.mkdir(parents=True)
    payload = [{"title": "平安银行经营动态", "source": "sina"}]
    (news_dir / "000001_SZ.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache._CACHE_DIR", cache_root)

    records = get_cached_news("000001.SZ", trade_date="2026-07-10")
    assert records == payload


def test_get_cached_risk_snapshot_reads_date_file(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    cache_root.mkdir(parents=True)
    payload = {"st_status": ["000001"], "suspended": [], "delisting": []}
    (cache_root / "risk_2026-07-10.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache._CACHE_DIR", cache_root)

    snapshot = get_cached_risk_snapshot("2026-07-10")
    assert snapshot == payload


def test_get_cached_market_breadth_uses_previous_close_when_pct_missing(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    daily_dir = cache_root / "daily"
    daily_dir.mkdir(parents=True)
    df = pd.DataFrame(
        [
            {"datetime": "2026-07-09 15:00:00", "close": 10.0, "code": "000001.SZ"},
            {"datetime": "2026-07-10 15:00:00", "close": 10.5, "code": "000001.SZ"},
            {"datetime": "2026-07-09 15:00:00", "close": 20.0, "code": "000002.SZ"},
            {"datetime": "2026-07-10 15:00:00", "close": 19.5, "code": "000002.SZ"},
        ]
    )
    df[df["code"] == "000001.SZ"].to_parquet(daily_dir / "000001_SZ.parquet", index=False)
    df[df["code"] == "000002.SZ"].to_parquet(daily_dir / "000002_SZ.parquet", index=False)

    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache._CACHE_DIR", cache_root)

    breadth = get_cached_market_breadth("2026-07-10")
    assert breadth["advance_count"] == 1
    assert breadth["decline_count"] == 1
    assert breadth["sample_size"] == 2


def test_get_cached_northbound_top10_reads_date_file(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    cache_root.mkdir(parents=True)
    payload = [{"code": "000001", "name": "平安银行", "net_buy": 100_000_000}]
    (cache_root / "northbound_top10_2026-07-10.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache._CACHE_DIR", cache_root)

    assert get_cached_northbound_top10("2026-07-10") == payload


def test_get_cached_daily_date_filter_includes_intraday_timestamp(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    daily_dir = cache_root / "daily"
    daily_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"datetime": "2026-07-09 15:00:00", "close": 10.0},
            {"datetime": "2026-07-10 15:00:00", "close": 10.5},
        ]
    ).to_parquet(daily_dir / "000001_SZ.parquet", index=False)

    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache._CACHE_DIR", cache_root)

    records = get_cached_daily("000001.SZ", start_date="20260710", end_date="20260710")
    assert len(records) == 1
    assert records[0]["close"] == 10.5


def test_ensure_candidate_daily_cache_uses_existing_project_cache(tmp_path, monkeypatch):
    daily_dir = tmp_path / "local_cache" / "daily"
    daily_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-07-10", "close": 10.5},
        ]
    ).to_parquet(daily_dir / "000001_SZ.parquet", index=False)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("should not fetch when project cache covers date")

    monkeypatch.setattr("advanced_trading_agent.data_agent.build_cache._fetch_candidate_daily", fail_fetch)

    statuses = ensure_candidate_daily_cache(
        ["000001.SZ"],
        "2026-07-10",
        output_dir=str(tmp_path),
        min_sleep_seconds=0,
        max_sleep_seconds=0,
    )

    assert statuses[0]["status"] == "cache_hit"
    assert statuses[0]["source"] == "project_cache"


def test_ensure_candidate_daily_cache_fetches_missing_candidate(tmp_path, monkeypatch):
    rows = [
        {"trade_date": "2026-07-10", "close": 10.5, "data_source": "fake_vendor"},
    ]

    monkeypatch.setattr(
        "advanced_trading_agent.data_agent.build_cache._fetch_candidate_daily",
        lambda *args, **kwargs: rows,
    )
    monkeypatch.setattr(
        "advanced_trading_agent.data_agent.build_cache.Path.home",
        lambda: tmp_path / "isolated_home",
    )

    statuses = ensure_candidate_daily_cache(
        ["000001.SZ"],
        "2026-07-10",
        output_dir=str(tmp_path),
        min_sleep_seconds=0,
        max_sleep_seconds=0,
    )

    target = tmp_path / "local_cache" / "daily" / "000001_SZ.parquet"
    assert target.exists()
    assert statuses[0]["status"] == "fetched"
    assert statuses[0]["source"] == "fake_vendor"
