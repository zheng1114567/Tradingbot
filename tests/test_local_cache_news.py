from __future__ import annotations

import json

import pandas as pd

from advanced_trading_agent.data_agent import build_cache
from advanced_trading_agent.data_agent.build_cache import ensure_candidate_daily_cache
from advanced_trading_agent.data_agent.local_cache import (
    get_cached_daily,
    get_cached_market_breadth,
    get_cached_news,
    get_cached_sector_news,
    get_cached_northbound_top10,
    get_cached_risk_snapshot,
    save_cached_news,
    save_cached_sector_news,
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


def test_get_cached_daily_cache_only_does_not_repair_online(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    daily_dir = cache_root / "daily"
    daily_dir.mkdir(parents=True)
    pd.DataFrame([{"trade_date": "2026-07-10", "close": 10.5}]).to_parquet(
        daily_dir / "000001_SZ.parquet", index=False
    )
    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache._CACHE_DIR", cache_root)

    def fail_repair(*args, **kwargs):
        raise AssertionError("cache-only read must not repair online")

    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache.LocalCache.ensure_daily_data", fail_repair)

    records = get_cached_daily(
        "000001.SZ",
        start_date="20260710",
        end_date="20260710",
        allow_online_repair=False,
    )
    assert records[0]["_cache_status"] == "cache_only"


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


def test_cache_via_baostock_refetches_existing_file_when_date_missing(tmp_path, monkeypatch):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    pd.DataFrame([{"date": "2026-07-10", "close": 10.0}]).to_parquet(
        daily_dir / "000001_SZ.parquet",
        index=False,
    )

    class FakeResult:
        error_code = "0"
        fields = ["date", "code", "close"]

        def __init__(self):
            self._rows = [["2026-07-15", "sz.000001", "10.5"]]
            self._idx = -1

        def next(self):
            self._idx += 1
            return self._idx < len(self._rows)

        def get_row_data(self):
            return self._rows[self._idx]

    class FakeBaoStock:
        def login(self):
            return None

        def logout(self):
            return None

        def query_history_k_data_plus(self, *args, **kwargs):
            return FakeResult()

    monkeypatch.setitem(__import__("sys").modules, "baostock", FakeBaoStock())
    monkeypatch.setattr(build_cache, "_vendor_jitter", lambda *args, **kwargs: None)

    cached = build_cache._cache_via_baostock(["000001.SZ"], daily_dir, "2026-07-15", days_back=10)

    assert cached == 1
    refreshed = pd.read_parquet(daily_dir / "000001_SZ.parquet")
    assert "2026-07-15" in set(refreshed["date"].astype(str))


def test_fetch_news_records_prefers_eastmoney_ticker_news(monkeypatch):
    calls = []

    def fake_get_vendor_impl(method, vendor):
        assert method == "get_news"

        def eastmoney_impl(**kwargs):
            calls.append(("eastmoney", kwargs))
            return [{"title": "??????", "source": "eastmoney", "url": "u1"}]

        def fallback_impl(**kwargs):
            raise AssertionError("fallback should not be called when eastmoney returns records")

        return {"eastmoney": eastmoney_impl, "eastmoney_global": fallback_impl, "sina": fallback_impl}.get(vendor)

    monkeypatch.setattr(build_cache, "ensure_default_vendor_registration", lambda: None)
    monkeypatch.setattr(build_cache, "get_vendor_impl", fake_get_vendor_impl)

    records = build_cache._fetch_news_records("000001.SZ", keyword="????", limit=5)

    assert records == [{"title": "??????", "source": "eastmoney", "url": "u1"}]
    assert calls == [("eastmoney", {"code": "000001.SZ", "limit": 5})]


def test_fetch_sector_news_records_uses_sector_keyword_sources(monkeypatch):
    calls = []

    def fake_get_vendor_impl(method, vendor):
        assert method == "get_news"

        def eastmoney_global_impl(**kwargs):
            calls.append(("eastmoney_global", kwargs))
            return [{"title": "半导体政策催化", "source": "eastmoney_global", "url": "u1"}]

        def cls_impl(**kwargs):
            calls.append(("cls", kwargs))
            return [{"title": "半导体快讯", "source": "cls", "url": "u2"}]

        return {"eastmoney_global": eastmoney_global_impl, "cls": cls_impl}.get(vendor)

    monkeypatch.setattr(build_cache, "ensure_default_vendor_registration", lambda: None)
    monkeypatch.setattr(build_cache, "get_vendor_impl", fake_get_vendor_impl)

    records = build_cache._fetch_sector_news_records("半导体", limit=5)

    assert [record["url"] for record in records] == ["u1", "u2"]
    assert all(record["sector_name"] == "半导体" for record in records)
    assert all(record["news_scope"] == "sector" for record in records)
    assert calls == [
        ("eastmoney_global", {"sector": "半导体", "keyword": "半导体", "limit": 5}),
        ("cls", {"sector": "半导体", "keyword": "半导体", "limit": 4}),
    ]


def test_cache_news_for_sector_prefers_direct_sector_news(tmp_path, monkeypatch):
    monkeypatch.setattr(
        build_cache,
        "_fetch_sector_news_records",
        lambda sector_name, **kwargs: [{"title": "direct", "url": "u1", "source": "cls"}],
    )

    def fail_ticker_fetch(*args, **kwargs):
        raise AssertionError("direct sector news should avoid constituent fallback when records exist")

    monkeypatch.setattr(build_cache, "_fetch_news_records", fail_ticker_fetch)

    path = build_cache.cache_news_for_sector(
        "半导体",
        "2026-07-10",
        output_dir=str(tmp_path),
        constituent_tickers=["000001.SZ"],
        force=True,
    )

    records = json.loads(path.read_text("utf-8"))
    assert records == [{"title": "direct", "url": "u1", "source": "cls"}]


def test_cache_hot_sector_constituents_fetches_by_sector_not_by_all_tickers(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    cache_root.mkdir(parents=True)
    trade_date = "2026-07-10"
    (cache_root / f"sector_ranking_{trade_date}.json").write_text(
        json.dumps(
            [
                {"sector_name": "半导体", "rank": 1},
                {"sector_name": "机器人", "rank": 2},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_get_vendor_impl(method, vendor):
        assert method == "get_sector_constituents"
        if vendor != "akshare":
            return None

        def impl(sector_name, trade_date=None):
            calls.append((sector_name, trade_date))
            return [{"code": f"{len(calls):06d}.SZ", "sector": sector_name}]

        return impl

    monkeypatch.setattr(build_cache, "ensure_default_vendor_registration", lambda: None)
    monkeypatch.setattr(build_cache, "get_vendor_impl", fake_get_vendor_impl)
    monkeypatch.setattr(build_cache, "_vendor_jitter", lambda *args, **kwargs: None)

    result = build_cache._cache_hot_sector_constituents(cache_root, trade_date)

    assert result == {"半导体": ["000001.SZ"], "机器人": ["000002.SZ"]}
    assert calls == [("半导体", trade_date), ("机器人", trade_date)]


def test_cache_batch_news_force_overwrites_existing_news(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    news_dir = cache_root / "news" / "2026-07-10"
    news_dir.mkdir(parents=True)
    (cache_root / "short_term_signals_2026-07-10.json").write_text(
        json.dumps({"ranked_tickers": ["000001.SZ"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    target = news_dir / "000001_SZ.json"
    target.write_text(json.dumps([{"title": "old"}], ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        build_cache,
        "_fetch_news_records",
        lambda ticker, **kwargs: [{"title": "new", "source": "cls"}],
    )
    monkeypatch.setattr(build_cache, "_vendor_jitter", lambda *args, **kwargs: None)

    build_cache._cache_batch_news(cache_root, "2026-07-10", force=True)

    assert json.loads(target.read_text("utf-8")) == [{"title": "new", "source": "cls"}]


def test_cache_batch_news_writes_status_when_no_news_targets(tmp_path):
    cache_root = tmp_path / "local_cache"

    count = build_cache._cache_batch_news(cache_root, "2026-07-10", force=False)

    assert count == 0
    assert (cache_root / "news" / "2026-07-10" / "_cache_status.json").exists()
    assert (cache_root / "sector_news" / "2026-07-10" / "_cache_status.json").exists()
    assert build_cache._cache_gaps(cache_root, "2026-07-10", require_news=True) == [
        "board_index",
        "sector_ranking",
        "dragon_tiger",
        "limit_up",
        "risk_snapshot",
        "daily_parquet",
    ]


def test_cache_gaps_requires_ticker_and_sector_news_scopes(tmp_path):
    cache_root = tmp_path / "local_cache"
    news_dir = cache_root / "news" / "2026-07-10"
    news_dir.mkdir(parents=True)
    (news_dir / "_cache_status.json").write_text("{}", encoding="utf-8")

    gaps = build_cache._cache_gaps(cache_root, "2026-07-10", require_news=True)

    assert "news" not in gaps
    assert "sector_news" in gaps


def test_cache_ready_returns_true_only_when_required_cache_exists(tmp_path):
    cache_root = tmp_path / "local_cache"
    cache_root.mkdir(parents=True)
    trade_date = "2026-07-10"

    assert build_cache._cache_ready(cache_root, trade_date) is False

    (cache_root / "board_index.json").write_text("{}", encoding="utf-8")
    (cache_root / f"sector_ranking_{trade_date}.json").write_text("[]", encoding="utf-8")
    (cache_root / f"dragon_tiger_{trade_date}.json").write_text("[]", encoding="utf-8")
    (cache_root / f"limit_up_{trade_date}.json").write_text('{"stocks":[]}', encoding="utf-8")
    (cache_root / f"risk_{trade_date}.json").write_text("{}", encoding="utf-8")
    daily_dir = cache_root / "daily"
    daily_dir.mkdir()
    pd.DataFrame([{"date": "2026-07-09", "close": 10.0}]).to_parquet(
        daily_dir / "000001_SZ.parquet",
        index=False,
    )

    assert build_cache._cache_ready(cache_root, trade_date) is False

    pd.DataFrame([{"date": trade_date, "close": 10.5}]).to_parquet(
        daily_dir / "000001_SZ.parquet",
        index=False,
    )
    assert build_cache._cache_ready(cache_root, trade_date) is True


def test_save_cached_news_merges_and_deduplicates(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    news_dir = cache_root / "news" / "2026-07-10"
    news_dir.mkdir(parents=True)
    target = news_dir / "000001_SZ.json"
    target.write_text(
        json.dumps([{"title": "old", "url": "https://example.com/old"}], ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache._CACHE_DIR", cache_root)

    save_cached_news(
        "000001.SZ",
        [
            {"title": "old duplicate", "url": "https://example.com/old"},
            {"title": "new", "url": "https://example.com/new", "source": "eastmoney"},
        ],
        trade_date="2026-07-10",
    )

    records = json.loads(target.read_text("utf-8"))
    assert [record["url"] for record in records] == [
        "https://example.com/old",
        "https://example.com/new",
    ]


def test_save_cached_sector_news_merges_and_deduplicates(tmp_path, monkeypatch):
    cache_root = tmp_path / "local_cache"
    sector_dir = cache_root / "sector_news" / "2026-07-10"
    sector_dir.mkdir(parents=True)
    target = sector_dir / "半导体.json"
    target.write_text(
        json.dumps([{"title": "old", "url": "https://example.com/old"}], ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache._CACHE_DIR", cache_root)

    save_cached_sector_news(
        "半导体",
        [
            {"title": "old duplicate", "url": "https://example.com/old"},
            {"title": "new", "url": "https://example.com/new", "source": "cls"},
        ],
        trade_date="2026-07-10",
    )

    records = get_cached_sector_news("半导体", trade_date="2026-07-10")
    assert [record["url"] for record in records] == [
        "https://example.com/old",
        "https://example.com/new",
    ]
    assert records[1]["news_scope"] == "sector"
    assert records[1]["sector_name"] == "半导体"
