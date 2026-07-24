from __future__ import annotations

from advanced_trading_agent.data_agent.cleaner import DataCleaner


def test_clean_daily_does_not_forward_fill_across_codes():
    rows = [
        {"code": "000001.SZ", "trade_date": "2026-07-09", "close": 10.0},
        {"code": "600000.SH", "trade_date": "2026-07-09", "close": 20.0},
        {"code": "000001.SZ", "trade_date": "2026-07-10", "close": 11.0},
        {"code": "600000.SH", "trade_date": "2026-07-10", "close": 19.0},
    ]

    df = DataCleaner.clean_daily(rows)
    by_code_date = {
        (row["code"], row["trade_date"].date().isoformat()): row
        for row in df.to_dict("records")
    }

    assert by_code_date[("000001.SZ", "2026-07-10")]["pre_close"] == 10.0
    assert by_code_date[("600000.SH", "2026-07-10")]["pre_close"] == 20.0
    assert by_code_date[("600000.SH", "2026-07-09")]["pre_close"] != 10.0


def test_standardize_columns_merges_duplicate_vendor_fields_with_first_non_empty():
    rows = [
        {
            "code": "000001.SZ",
            "trade_date": "2026-07-10",
            "close": 10.0,
            "volume": None,
            "vol": 12345,
        }
    ]

    df = DataCleaner.clean_daily(rows)

    assert df.loc[0, "volume"] == 12345
