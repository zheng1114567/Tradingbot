"""数据 Service 测试"""
import pytest
import pandas as pd
from advanced_trading_agent.data_service.schema import (
    MarketSchema, SentimentSchema, CapitalSchema,
    FactorSchema, EventSchema, PointInTime,
    DecisionType, MarketSentiment, CapitalConfirmation,
)
from advanced_trading_agent.data_service.cleaner import DataCleaner
from advanced_trading_agent.data_service.factors import FactorCalculator
from advanced_trading_agent.data_service.vendor_router import (
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


class TestVendorRouter:
    """测试供应商路由"""

    def test_route_no_data(self):
        """测试无数据时返回哨兵"""
        def mock_impl(code):
            return None
        register_vendor_impl("test_method", "tushare", mock_impl)
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
        from advanced_trading_agent.data_service.vendor_router import get_vendor_chain
        from advanced_trading_agent.data_service.vendor_router import _VENDOR_IMPLEMENTATIONS
        impls = _VENDOR_IMPLEMENTATIONS.get("test_fallback", {})
        assert "a" in impls
        assert "b" in impls
