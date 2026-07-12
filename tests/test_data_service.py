"""数据 Service 测试"""
import pytest
import pandas as pd
from advanced_trading_agent.data_agent.schema import (
    MarketSchema, SentimentSchema, CapitalSchema,
    FactorSchema, EventSchema, PointInTime,
    DecisionType, MarketSentiment, CapitalConfirmation,
)
from advanced_trading_agent.data_agent.cleaner import DataCleaner
from advanced_trading_agent.data_agent.factors import FactorCalculator
from advanced_trading_agent.data_agent.vendor_router import (
    DataVendor, route_to_vendor, register_vendor_impl,
    VendorRateLimitError, VendorNotConfiguredError,
)
from datetime import date


class TestSchemas:
    """测试 Pydantic Schema"""

    def test_market_schema(self):
        pit = PointInTime(as_of_date=date.today())
        s = MarketSchema(
            pit=pit, index_close=3000, index_change_pct=-0.5,
            advance_count=1000, decline_count=2000,
            limit_up_count=50, limit_down_count=20,
            total_volume_cny=8e11,
        )
        assert s.advance_count == 1000
        assert s.total_volume_cny == 8e11

    def test_event_schema_weak_chain(self):
        pit = PointInTime(as_of_date=date.today())
        e = EventSchema(
            pit=pit, event_id="test", event_type="情绪",
            summary="test", direction="中性", confidence=0.3,
            transmission_path="弱关联", direct_beneficiaries=[],
            evidence_level="社交传闻", pricing_status="未定价",
            chain_quality="weak",
        )
        assert e.chain_quality == "weak"


class TestCleaner:
    """测试数据清洗"""

    def test_limit_up_detection(self):
        df = pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [11.0],
            "pre_close": [10.0],
        })
        result = DataCleaner.detect_limit_up_down(df)
        assert result.iloc[0]["is_limit_up"]  # +10% = 涨停

    def test_clean_daily_normalizes_free_vendor_fields(self):
        data = [
            {
                "date": "2026-07-10",
                "code": "sh.600000",
                "open": "10.0",
                "high": "10.5",
                "low": "9.8",
                "close": "10.2",
                "preclose": "10.0",
                "pctChg": "2.0",
                "volume": "1000",
                "amount": "10200",
                "turn": "0.5",
            }
        ]
        result = DataCleaner.clean_daily(data)
        assert "trade_date" in result.columns
        assert "pre_close" in result.columns
        assert "pct_chg" in result.columns
        assert "turnover_rate" in result.columns
        assert result.iloc[0]["pct_chg"] == 2.0

    def test_clean_daily_handles_duplicate_normalized_columns(self):
        data = [
            {
                "trade_date": "2026-07-10",
                "code": "603259.SH",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "vol": 1000,
                "volume": 1000,
                "amount": 10500,
            }
        ]

        result = DataCleaner.clean_daily(data)

        assert list(result.columns).count("volume") == 1
        assert result.iloc[0]["volume"] == 1000


class TestFactors:
    """测试因子公式单位."""

    def test_volatility_uses_decimal_returns(self):
        df = pd.DataFrame({"pct_chg": [1.0] * 20 + [2.0]})
        result = FactorCalculator.volatility(df.copy(), periods=20)
        expected = (df["pct_chg"] / 100).rolling(20).std().iloc[-1] * (252 ** 0.5)
        assert result.iloc[-1]["volatility"] == pytest.approx(expected)
        assert result.iloc[-1]["volatility"] < 1

    def test_turnover_and_amihud_use_decimal_scale(self):
        df = pd.DataFrame({
            "pct_chg": [1.0] * 21,
            "amount": [1000.0] * 21,
            "turnover_rate": [2.5] * 21,
        })
        result = FactorCalculator.run_all(df.copy())
        assert result.iloc[-1]["turnover"] == pytest.approx(0.025)
        assert result.iloc[-1]["amihud"] == pytest.approx(0.00001)


class TestVendorRouter:
    """测试供应商路由"""

    def test_route_no_data(self):
        """测试无数据时返回哨兵"""
        def mock_impl(code):
            return None
        register_vendor_impl("test_method", "akshare", mock_impl)
        result = route_to_vendor("test_method", code="000001")
        assert "NO_DATA_AVAILABLE" in str(result)

    def test_register_and_fallback(self):
        results = []

        def vendor_a(code):
            results.append("a")
            raise VendorRateLimitError("rate limit", vendor="a")

        def vendor_b(code):
            results.append("b")
            return {"code": code, "price": 10}

        register_vendor_impl("test_fallback", "a", vendor_a)
        register_vendor_impl("test_fallback", "b", vendor_b)
        # 手动测试
        from advanced_trading_agent.data_agent.vendor_router import get_vendor_chain
        from advanced_trading_agent.data_agent.vendor_router import _VENDOR_IMPLEMENTATIONS
        impls = _VENDOR_IMPLEMENTATIONS.get("test_fallback", {})
        assert "a" in impls
        assert "b" in impls
