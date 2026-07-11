"""供应商路由测试 — 测试免费数据源错误处理和降级链."""
import pytest
from advanced_trading_agent.data_agent.vendor_router import (
    DataVendor,
    VendorRateLimitError,
    VendorNotConfiguredError,
    NoMarketDataError,
    VendorFatalError,
    register_vendor_impl,
    route_to_vendor,
    get_vendor_chain,
    get_vendor_impl,
    ensure_default_vendor_registration,
    _VENDOR_IMPLEMENTATIONS,
)
from advanced_trading_agent.data_agent import vendor_router


class TestVendorChain:
    """供应商降级链测试"""

    def setup_method(self):
        _VENDOR_IMPLEMENTATIONS.clear()
        vendor_router._DEFAULT_VENDOR_REGISTRATION_ATTEMPTED = False

    def test_simple_success(self):
        def impl(code):
            return {"price": 10}
        register_vendor_impl("test_method", "akshare", impl)
        result = route_to_vendor("test_method", code="000001")
        assert result == {"price": 10}

    def test_fallback_on_rate_limit(self):
        """频道限制时应尝试下一个供应商 (未知方法默认链为 akshare)"""
        results = []
        def primary_akshare(code):
            results.append("akshare")
            raise VendorRateLimitError("rate limit", vendor="akshare")
        register_vendor_impl("test_fallback_method", "akshare", primary_akshare)
        # 默认链只有 akshare, 失败后无其他 vendor → 抛异常
        with pytest.raises(VendorFatalError):
            route_to_vendor("test_fallback_method", code="000001")
        assert results == ["akshare"]

    def test_fallback_on_not_configured(self):
        def primary(code):
            raise VendorNotConfiguredError("no key", vendor="primary")
        register_vendor_impl("test_method", "akshare", primary)
        # 只有 akshare 被尝试，它抛异常后没有其他 vendor
        with pytest.raises(VendorFatalError):
            route_to_vendor("test_method", code="000001")

    def test_no_data_returns_sentinel(self):
        def primary(code):
            raise NoMarketDataError("no data", symbol="000001", vendor="akshare")
        register_vendor_impl("test_method", "akshare", primary)
        result = route_to_vendor("test_method", code="000001")
        assert "NO_DATA_AVAILABLE" in str(result)

    def test_all_vendors_fail_fatal(self):
        def primary(code):
            raise VendorRateLimitError("limit", vendor="akshare")
        register_vendor_impl("test_method", "akshare", primary)
        with pytest.raises(VendorFatalError):
            route_to_vendor("test_method", code="000001")

    def test_none_return_treated_as_no_data(self):
        def primary(code):
            return None
        register_vendor_impl("test_method", "akshare", primary)
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
        register_vendor_impl("get_daily", "akshare", a)
        register_vendor_impl("get_daily", "baostock", b)
        register_vendor_impl("get_daily", "baostock_backup", c)
        original = get_vendor_chain("get_daily")
        from advanced_trading_agent.config import config
        config.update({"data_vendors": {"market_data": "akshare,baostock,baostock_backup"}})
        try:
            result = route_to_vendor("get_daily", code="x")
            assert result == {"ok": True}
        finally:
            config.update({"data_vendors": {"market_data": ",".join(original)}})

    def test_route_trace_records_attempts(self):
        trace = []

        def a(code):
            raise VendorRateLimitError("limit", vendor="akshare")

        def b(code):
            return [{"code": code}]

        register_vendor_impl("get_daily", "akshare", a)
        register_vendor_impl("get_daily", "baostock", b)

        result = route_to_vendor("get_daily", code="000001.SZ", _route_trace=trace)

        assert result == [{"code": "000001.SZ"}]
        assert trace[0]["vendor"] == "akshare"
        assert trace[0]["status"] == "rate_limited"
        assert trace[1]["vendor"] == "baostock"
        assert trace[1]["status"] == "success"
        assert trace[1]["record_count"] == 1

    def test_fallback_order_across_vendors(self):
        """使用已知的 get_daily 方法测试降级顺序"""
        chain = get_vendor_chain("get_daily")
        assert chain == ["akshare", "baostock"]
        # 注册 mock 到链中的第一个供应商
        first_vendor = chain[0]
        def mock_impl(code):
            return {"mock": "data"}
        register_vendor_impl("get_daily", first_vendor, mock_impl)
        result = route_to_vendor("get_daily", code="000001.SZ")
        assert result == {"mock": "data"}

    def test_configured_chains_are_free_only(self):
        free_vendors = {"akshare", "baostock", "eastmoney", "sina"}
        for method in [
            "get_daily",
            "get_capital_flow",
            "get_news",
            "get_sector",
            "get_factors",
            "get_st_status",
            "get_suspended",
            "get_delisting",
        ]:
            chain = get_vendor_chain(method)
            assert chain
            assert set(chain) <= free_vendors

    def test_news_and_sector_have_free_fallbacks(self):
        assert get_vendor_chain("get_news") == ["akshare", "sina"]
        assert get_vendor_chain("get_sector") == ["akshare", "eastmoney"]

    def test_default_vendor_registration_covers_tool_methods(self):
        ensure_default_vendor_registration()

        missing = []
        for method in [
            "get_daily",
            "get_capital_flow",
            "get_news",
            "get_sector",
            "get_st_status",
            "get_suspended",
            "get_delisting",
            "get_northbound_flow",
            "get_limit_up_tiers",
            "get_dragon_tiger",
            "get_margin",
            "get_factors",
            "check_crowding",
            "find_similar",
        ]:
            registered = set(_VENDOR_IMPLEMENTATIONS.get(method, {}))
            expected = set(get_vendor_chain(method))
            if not registered.intersection(expected):
                missing.append((method, sorted(expected), sorted(registered)))

        assert missing == []


class TestVendorImplRegistration:
    """供应商实现注册测试"""

    def setup_method(self):
        _VENDOR_IMPLEMENTATIONS.clear()
        vendor_router._DEFAULT_VENDOR_REGISTRATION_ATTEMPTED = False

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
