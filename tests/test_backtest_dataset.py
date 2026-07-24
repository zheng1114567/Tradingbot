from __future__ import annotations

import pandas as pd

from advanced_trading_agent.backtest.data_qa import BacktestDatasetBuilder


def test_backtest_dataset_builder_marks_insufficient_cache(monkeypatch):
    def fake_daily(code, start_date=None, end_date=None, allow_online_repair=True):
        return [
            {"code": code, "trade_date": day.isoformat(), "open": 1.0, "close": 1.1}
            for day in pd.date_range("2026-07-01", periods=5, freq="B")
        ]

    monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache.get_cached_daily", fake_daily)

    dataset = BacktestDatasetBuilder(lookback_days=60, min_coverage_days=30).build(
        [{"code": "000001.SZ", "date": "2026-07-10"}],
        end_date="2026-07-10",
    )

    assert "000001.SZ" in dataset.prices_by_code
    assert dataset.coverage["status"] == "insufficient_cache"
    assert dataset.coverage["codes"]["000001.SZ"]["trade_day_count"] == 5

