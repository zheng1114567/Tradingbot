"""回测引擎测试"""
import pytest
import pandas as pd
from datetime import date, timedelta
from advanced_trading_agent.backtest.engine import BacktestEngine, BacktestResult
from advanced_trading_agent.backtest.metrics import PerformanceMetrics


@pytest.fixture
def sample_price_data():
    """生成模拟行情数据"""
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    data = {
        "trade_date": dates,
        "open": [10 + i * 0.1 for i in range(30)],
        "close": [10 + i * 0.1 + 0.05 for i in range(30)],
        "high": [10 + i * 0.1 + 0.2 for i in range(30)],
        "low": [10 + i * 0.1 - 0.1 for i in range(30)],
        "volume": [1e7 + i * 1e5 for i in range(30)],
        "amount": [1e8 + i * 1e6 for i in range(30)],
        "pct_chg": [0.5 + i * 0.1 for i in range(30)],
        "is_limit_up": [False] * 30,
        "is_limit_down": [False] * 30,
    }
    df = pd.DataFrame(data)
    return {"000001.SZ": df}


class TestBacktestEngine:
    """测试回测引擎"""

    def test_run_single(self, sample_price_data):
        engine = BacktestEngine()
        df = sample_price_data["000001.SZ"]
        result = engine.run_single(
            price_df=df,
            entry_date=date(2026, 1, 5),
            code="000001.SZ",
            decision="推荐",
            alpha_source=["测试"],
        )
        assert result.code == "000001.SZ"
        assert result.tradable is not False
        assert result.holding_days > 0

    def test_run_batch(self, sample_price_data):
        engine = BacktestEngine()
        signals = [
            {"code": "000001.SZ", "date": "2026-01-05", "decision": "推荐"},
            {"code": "000001.SZ", "date": "2026-01-06", "decision": "观察"},
        ]
        results = engine.run_batch(sample_price_data, signals)
        assert len(results) == 2
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_limit_up_blocks_trade(self):
        """涨停时买入不可成交"""
        df = pd.DataFrame({
            "trade_date": pd.date_range("2026-01-01", periods=5, freq="B"),
            "open": [10.0] * 5,
            "close": [11.0] + [10.5] * 4,
            "high": [11.0] * 5,
            "low": [9.9] * 5,
            "volume": [1e7] * 5,
            "amount": [1e8] * 5,
            "pct_chg": [10.0, 0.0, 0.0, 0.0, 0.0],
            "is_limit_up": [False, True, False, False, False],
            "is_limit_down": [False] * 5,
        })
        engine = BacktestEngine()
        result = engine.run_single(df, date(2026, 1, 1), "000001.SZ")
        assert not result.tradable


class TestMetrics:
    """测试绩效指标"""

    def test_win_rate(self):
        results = [
            BacktestResult(run_date=date.today(), target_date=date.today(),
                           code="A", decision="推荐", returns={5: 0.01}),
            BacktestResult(run_date=date.today(), target_date=date.today(),
                           code="B", decision="推荐", returns={5: -0.01}),
            BacktestResult(run_date=date.today(), target_date=date.today(),
                           code="C", decision="推荐", returns={5: 0.02}),
        ]
        wr = PerformanceMetrics.win_rate(results, 5)
        assert wr == 2 / 3

    def test_summary(self):
        results = [
            BacktestResult(run_date=date.today(), target_date=date.today(),
                           code="A", decision="推荐", tradable=True,
                           returns={1: 0.01, 3: 0.02, 5: 0.03, 10: 0.05}),
        ]
        summary = PerformanceMetrics.summary(results)
        assert summary["total_signals"] == 1
        assert summary["tradable_ratio"] == 1.0
        assert summary["avg_return_5d"] == 0.03
