"""Portfolio observation-pool backtest tests."""
from __future__ import annotations

import pandas as pd

from advanced_trading_agent.backtest.portfolio import ObservationPortfolioBacktester


def test_portfolio_backtest_enters_next_day_and_records_trade():
    dates = pd.date_range("2026-07-10", periods=8, freq="B")
    signals = pd.DataFrame({
        "signal_date": [dates[0]],
        "code": ["000001.SZ"],
        "decision": ["推荐"],
        "score": [9.0],
        "alpha_source": ["factor"],
    })
    prices = pd.DataFrame({
        "trade_date": list(dates),
        "code": ["000001.SZ"] * len(dates),
        "open": [10, 11, 12, 13, 14, 15, 16, 17],
        "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5],
    })

    result = ObservationPortfolioBacktester(holding_days=3).run(signals, prices)

    assert result.summary["trade_count"] == 1
    assert result.trades.iloc[0]["entry_date"] == dates[1].date().isoformat()
    assert result.trades.iloc[0]["actual_exit_date"] == dates[4].date().isoformat()
    assert result.summary["total_return"] > 0


def test_portfolio_backtest_respects_max_positions():
    dates = pd.date_range("2026-07-10", periods=8, freq="B")
    signals = pd.DataFrame({
        "signal_date": [dates[0], dates[0]],
        "code": ["000001.SZ", "000002.SZ"],
        "decision": ["推荐", "推荐"],
        "score": [1.0, 9.0],
    })
    prices = pd.DataFrame([
        {"trade_date": day, "code": code, "open": 10.0, "close": 10.5}
        for day in dates
        for code in ["000001.SZ", "000002.SZ"]
    ])

    result = ObservationPortfolioBacktester(max_positions=1, holding_days=2).run(signals, prices)

    assert result.summary["trade_count"] == 1
    assert result.trades.iloc[0]["code"] == "000002.SZ"

