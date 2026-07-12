from __future__ import annotations

import json

from advanced_trading_agent.data_agent.cache_manifest import CacheManifest


def test_cache_manifest_persists_daily_entry(tmp_path):
    cache_dir = tmp_path / "local_cache"
    cache_dir.mkdir()
    data_path = cache_dir / "daily" / "000001_SZ.parquet"
    data_path.parent.mkdir()
    data_path.write_bytes(b"daily-data")

    manifest = CacheManifest(cache_dir)
    manifest.update_daily(
        ticker="000001.SZ",
        path=data_path,
        start_date="2026-07-01",
        end_date="2026-07-10",
        source="baostock",
        row_count=8,
        status="vendor_fetch",
    )

    reloaded = CacheManifest(cache_dir)
    entry = reloaded.get_daily("000001.SZ")
    assert entry is not None
    assert entry.start_date == "2026-07-01"
    assert entry.end_date == "2026-07-10"
    assert entry.row_count == 8
    assert entry.status == "vendor_fetch"
    assert entry.checksum

    payload = json.loads((cache_dir / "cache_manifest.json").read_text("utf-8"))
    assert "daily:000001.SZ:1d" in payload["entries"]
