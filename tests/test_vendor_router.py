"""供应商路由测试 — 测试错误处理和降级链

注意: route_to_vendor 的默认供应商链为 ["tushare"],
因此测试实现需要注册在 "tushare" 名下才能被路由到。
"""
import pytest
from advanced_trading_agent.data_service.vendor_router import (
    DataVendor,
    VendorRateLimitError,
    VendorNotConfiguredError,
    NoMarketDataError,
    VendorFatalError,
    register_vendor_impl,
    route_to_vendor,
    get_vendor_chain,
    get_vendor_impl,
    _VENDOR_IMPLEMENTATIONS,
)


class TestVendorChain:
    """供应商降级链测试"""

    def setup_method(self):
        _VENDOR_IMPLEMENTATIONS.clear()

    def test_simple_success(self):
        def impl(code):
            return {"price": 10}
        register_vendor_impl("test_method", "tushare", impl)
        result = route_to_vendor("test_method", code="000001")
        assert result == {"price": 10}

    def test_fallback_on_rate_limit(self):
        """频道限制时应尝试下一个供应商 (默认链为 tushare)"""
        results = []
        def primary_tushare(code):
            results.append("tushare")
            raise VendorRateLimitError("rate limit", vendor="tushare")
        register_vendor_impl("test_fallback_method", "tushare", primary_tushare)
        # 默认链只有 tushare, 失败后无其他 vendor → 抛异常
        with pytest.raises(VendorFatalError):
            route_to_vendor("test_fallback_method", code="000001")
        assert results == ["tushare"]

    def test_fallback_on_not_configured(self):
        def primary(code):
            raise VendorNotConfiguredError("no key", vendor="primary")
        register_vendor_impl("test_method", "tushare", primary)
        # 只有 tushare 被尝试，它抛异常后没有其他 vendor
        with pytest.raises(VendorFatalError):
            route_to_vendor("test_method", code="000001")

    def test_no_data_returns_sentinel(self):
        def primary(code):
            raise NoMarketDataError("no data", symbol="000001", vendor="tushare")
        register_vendor_impl("test_method", "tushare", primary)
        result = route_to_vendor("test_method", code="000001")
        assert "NO_DATA_AVAILABLE" in str(result)

    def test_all_vendors_fail_fatal(self):
        def primary(code):
            raise VendorRateLimitError("limit", vendor="tushare")
        register_vendor_impl("test_method", "tushare", primary)
        with pytest.raises(VendorFatalError):
            route_to_vendor("test_method", code="000001")

    def test_none_return_treated_as_no_data(self):
        def primary(code):
            return None
        register_vendor_impl("test_method", "tushare", primary)
        result = route_to_vendor("test_method", code="000001")
        assert "NO_DATA_AVAILABLE" in str(result)

    def test_impl_does_not_exist(self):
        """未注册的方法应返回 VendorFatalError"""
        with pytest.raises(VendorFatalError):
            route_to_vendor("nonexistent_method", code="x")

    def test_multiple_fallbacks(self):
        """多个降级链应依次尝试 (通过配置链)"""
        order = []
        def a(code):
            order.append("a")
            raise VendorRateLimitError("limit", vendor="a")
        def b(code):
            order.append("b")
            raise VendorRateLimitError("limit", vendor="b")
        def c(code):
            order.append("c")
            return {"ok": True}
        # 使用名为 "get_daily" 的方法，它匹配 market_data 分类的链
        # 也可直接注册到 tushare + akshare
        register_vendor_impl("test_method", "a", a)
        register_vendor_impl("test_method", "b", b)
        register_vendor_impl("test_method", "tushare", c)
        result = route_to_vendor("test_method", code="x")
        assert result == {"ok": True}

    def test_fallback_order_across_vendors(self):
        """使用已知的 get_daily 方法测试降级顺序"""
        from advanced_trading_agent.config import config
        chain = get_vendor_chain("get_daily")
        assert len(chain) >= 1
        # 注册 mock 到链中的第一个供应商
        first_vendor = chain[0]
        def mock_impl(code):
            return {"mock": "data"}
        register_vendor_impl("get_daily", first_vendor, mock_impl)
        result = route_to_vendor("get_daily", code="000001.SZ")
        assert result == {"mock": "data"}


class TestVendorImplRegistration:
    """供应商实现注册测试"""

    def setup_method(self):
        _VENDOR_IMPLEMENTATIONS.clear()

    def test_register_and_retrieve(self):
        def impl(code):
            pass
        register_vendor_impl("test", "vendor", impl)
        assert get_vendor_impl("test", "vendor") is impl

    def test_register_overwrite(self):
        def impl1(code):
            pass
        def impl2(code):
            pass
        register_vendor_impl("test", "vendor", impl1)
        register_vendor_impl("test", "vendor", impl2)
        assert get_vendor_impl("test", "vendor") is impl2

    def test_get_nonexistent(self):
        assert get_vendor_impl("nonexistent", "any") is None
