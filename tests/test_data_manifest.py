"""Data manifest tests."""

import json

from advanced_trading_agent.data_agent.manifest import DataManifest


def test_manifest_records_fields_and_soft_vetoes(tmp_path):
    manifest = DataManifest(ticker="../000001.SZ", trade_date="2026/07/10")
    manifest.add_field(
        "stock.daily",
        available=False,
        source="vendor_router:get_daily",
        vendor_chain=["tushare", "akshare"],
        error="NO_DATA_AVAILABLE",
    )

    path = manifest.save(results_dir=str(tmp_path))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent.name == "manifests"
    assert "stock.daily unavailable" in payload["soft_veto_reasons"]
    assert payload["fields"]["stock.daily"]["source"] == "vendor_router:get_daily"
