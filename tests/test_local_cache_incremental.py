from __future__ import annotations

import pandas as pd

from advanced_trading_agent.data_agent.cache_manifest import CacheManifest
from advanced_trading_agent.data_agent.local_cache import LocalCache


def test_ensure_daily_data_cache_hit_does_not_fetch(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    pd.DataFrame(
        [
            {"date": "2026-07-09", "close": 10.0, "data_source": "baostock"},
            {"date": "2026-07-10", "close": 10.5, "data_source": "baostock"},
        ]
    ).to_parquet(daily_dir / "000001_SZ.parquet", index=False)

    cache = LocalCache(cache_dir=tmp_path)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("covered cache should not fetch")

    cache._fetch_daily_baostock = fail_fetch  # type: ignore[method-assign]

    records = cache.ensure_daily_data("000001.SZ", "20260710", "20260710")

    assert len(records) == 1
    assert records[0]["close"] == 10.5
    assert records[0]["_cache_status"] == "cache_hit"
    entry = CacheManifest(tmp_path).get_daily("000001.SZ")
    assert entry is not None
    assert entry.status == "cache_hit"


def test_ensure_daily_data_repairs_missing_range(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    pd.DataFrame(
        [{"date": "2026-07-09", "close": 10.0, "data_source": "baostock"}]
    ).to_parquet(daily_dir / "000001_SZ.parquet", index=False)

    cache = LocalCache(cache_dir=tmp_path)
    calls = []

    def fetch(ticker, start_date, end_date):
        calls.append((ticker, start_date, end_date))
        return pd.DataFrame(
            [{"date": "2026-07-10", "close": 10.5, "data_source": "baostock", "code": ticker}]
        )

    cache._fetch_daily_baostock = fetch  # type: ignore[method-assign]

    records = cache.ensure_daily_data("000001.SZ", "20260709", "20260710")

    assert calls == [("000001.SZ", "2026-07-09", "2026-07-10")]
    assert len(records) == 2
    assert records[-1]["close"] == 10.5
    assert records[-1]["_cache_status"] == "vendor_fetch"
    persisted = pd.read_parquet(daily_dir / "000001_SZ.parquet")
    assert len(persisted) == 2
    entry = CacheManifest(tmp_path).get_daily("000001.SZ")
    assert entry is not None
    assert entry.status == "vendor_fetch"
    assert entry.end_date == "2026-07-10"
