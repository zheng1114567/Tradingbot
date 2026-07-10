"""回测引擎边缘测试 — 补充 test_backtest.py 未覆盖的场景"""
import pytest
import pandas as pd
from datetime import date, timedelta
from advanced_trading_agent.backtest.engine import BacktestEngine, BacktestResult
from advanced_trading_agent.backtest.metrics import PerformanceMetrics


class TestBacktestEdgeCases:
    """回测引擎边缘情况"""

    def test_empty_dataframe(self):
        """空 DataFrame 应返回不可成交结果"""
        engine = BacktestEngine()
        df = pd.DataFrame()
        result = engine.run_single(df, date(2026, 1, 5), "000001.SZ")
        assert result.tradable is False

    def test_entry_date_not_in_data(self):
        """买入日期不在数据中应返回不可成交"""
        engine = BacktestEngine()
        df = pd.DataFrame({
            "trade_date": pd.date_range("2026-01-05", periods=5, freq="B"),
            "open": [10.0] * 5,
            "close": [10.5] * 5,
            "high": [11.0] * 5,
            "low": [9.9] * 5,
            "volume": [1e7] * 5,
            "amount": [1e8] * 5,
            "pct_chg": [0.5] * 5,
        })
        result = engine.run_single(df, date(2025, 12, 30), "000001.SZ")
        assert result.tradable is False

    def test_limit_down_sell_blocked(self):
        """跌停时卖出不可成交，应顺延"""
        df = pd.DataFrame({
            "trade_date": pd.date_range("2026-01-01", periods=10, freq="B"),
            "open": [10.0] * 10,
            "close": [10.5] * 10,
            "high": [11.0] * 10,
            "low": [9.9] * 10,
            "volume": [1e7] * 10,
            "amount": [1e8] * 10,
            "pct_chg": [0.5] * 10,
            "is_limit_up": [False] * 10,
            "is_limit_down": [True, False, False, False, False, False, False, False, False, False],
        })
        engine = BacktestEngine()
        result = engine.run_single(df, date(2026, 1, 1), "000001.SZ")
        # 买入日没涨停，应可买入
        assert result.tradable is not False

    def test_no_limit_columns(self):
        """没有 is_limit_up/down 列时应正常处理"""
        df = pd.DataFrame({
            "trade_date": pd.date_range("2026-01-01", periods=10, freq="B"),
            "open": [10.0] * 10,
            "close": [10.5] * 10,
            "volume": [1e7] * 10,
            "amount": [1e8] * 10,
        })
        engine = BacktestEngine()
        result = engine.run_single(df, date(2026, 1, 5), "000001.SZ")
        assert result.tradable is not False

    def test_missing_volume_amount(self):
        """缺失 volume/amount 列应导致不可成交"""
        df = pd.DataFrame({
            "trade_date": pd.date_range("2026-01-01", periods=10, freq="B"),
            "close": [10.5] * 10,
        })
        engine = BacktestEngine()
        result = engine.run_single(df, date(2026, 1, 5), "000001.SZ")
        assert result.tradable is False  # 无成交额数据不可成交

    def test_limit_up_and_down_tradable(self):
        """测试非涨停日的正常成交"""
        df = pd.DataFrame({
            "trade_date": pd.date_range("2026-01-01", periods=10, freq="B"),
            "open": [10.0] * 10,
            "close": [10.5] * 10,
            "volume": [1e7] * 10,
            "amount": [1e8] * 10,
            "is_limit_up": [False, False, False, True, False, False, False, False, False, False],
            "is_limit_down": [False] * 10,
        })
        engine = BacktestEngine()
        # Jan 5 (Mon) 不是涨停日
        result = engine.run_single(df, date(2026, 1, 5), "000001.SZ")
        assert result.tradable is not False
        assert result.entry_price is not None

    def test_benchmark_default(self):
        """默认基准应为沪深300"""
        engine = BacktestEngine()
        assert engine.benchmark == "000300.SH"

    def test_calc_trade_cost_buy(self):
        """买入成本应包含佣金+滑点"""
        engine = BacktestEngine()
        cost = engine.calc_trade_cost(is_buy=True)
        assert cost == 6  # 3 bp commission + 3 bp slippage

    def test_calc_trade_cost_sell(self):
        """卖出成本应包含佣金+印花税+滑点"""
        engine = BacktestEngine()
        cost = engine.calc_trade_cost(is_buy=False)
        assert cost == 16  # 3 bp commission + 10 bp stamp tax + 3 bp slippage


class TestMetricsEdgeCases:
    """绩效指标边缘情况"""

    def test_empty_results(self):
        """空结果列表应返回安全默认值"""
        assert PerformanceMetrics.win_rate([], 5) == 0.0
        assert PerformanceMetrics.avg_return([], 5) == 0.0
        assert PerformanceMetrics.tradable_ratio([]) == 0.0
        assert PerformanceMetrics.max_drawdown([]) == 0.0
        assert PerformanceMetrics.sharpe_ratio([], 5) == 0.0

    def test_single_result(self):
        """单结果不崩"""
        r = BacktestResult(
            run_date=date.today(), target_date=date.today(),
            code="A", decision="推荐", tradable=True,
            returns={5: 0.01}, excess_returns={5: 0.005},
        )
        assert PerformanceMetrics.avg_return([r], 5) == 0.01
        assert PerformanceMetrics.win_rate([r], 5) == 1.0
        assert PerformanceMetrics.tradable_ratio([r]) == 1.0
