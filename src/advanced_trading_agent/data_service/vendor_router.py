"""
数据供应商路由 — 借鉴 TradingAgents' interface.py + config.py

核心设计:
- 按工具粒度配置数据源 (e.g. "market_data": "tushare,akshare")
- 有序降级链: 第一个挂了自动试下一个
- 错误类型分级: 决定是否终止运行

错误类型:
- VendorRateLimitError: 频率限制, 尝试下一个
- VendorNotConfiguredError: 未配置, 尝试下一个
- NoMarketDataError: 数据不存在, 返回 NO_DATA 哨兵
- VendorFatalError: 致命错误, 终止运行
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

from ..config import config

logger = logging.getLogger(__name__)


# ============================================================
# 自定义异常
# ============================================================

class VendorError(Exception):
    """供应商异常基类"""
    def __init__(self, message: str, vendor: str = "", method: str = ""):
        self.vendor = vendor
        self.method = method
        super().__init__(message)


class VendorRateLimitError(VendorError):
    """频率限制 — 尝试下一个供应商"""
    pass


class VendorNotConfiguredError(VendorError):
    """未配置 (如缺少 token) — 尝试下一个"""
    pass


class NoMarketDataError(VendorError):
    """数据不存在 — 返回 NO_DATA 哨兵"""
    def __init__(self, message: str, symbol: str = "", detail: str = "",
                 vendor: str = "", method: str = ""):
        self.symbol = symbol
        self.detail = detail
        super().__init__(message, vendor=vendor, method=method)


class VendorFatalError(VendorError):
    """致命错误 — 终止运行"""
    pass


# ============================================================
# 供应商注册
# ============================================================

class DataVendor(str, Enum):
    TUSHARE = "tushare"
    AKSHARE = "akshare"


# 工具分类
TOOL_CATEGORIES = {
    "market_data": {
        "description": "行情数据 (日K, 分钟K)",
        "tools": ["get_daily", "get_minute"],
    },
    "fundamental_data": {
        "description": "基本面数据",
        "tools": ["get_financial", "get_balance", "get_income", "get_cashflow"],
    },
    "news_data": {
        "description": "新闻数据",
        "tools": ["get_news", "get_hot_news"],
    },
    "capital_flow": {
        "description": "资金流向",
        "tools": ["get_capital_flow", "get_moneyflow"],
    },
    "a_share_specific": {
        "description": "A股特有数据",
        "tools": [
            "get_northbound_flow",     # 北向资金
            "get_limit_up_tiers",      # 涨停梯队
            "get_sector",              # 板块数据
            "get_dragon_tiger",        # 龙虎榜
            "get_margin",              # 融资融券
        ],
    },
    "analysis": {
        "description": "分析数据",
        "tools": [
            "get_factors",             # 因子数据
            "check_crowding",          # 因子拥挤度
            "find_similar",            # 相似历史情境
        ],
    },
}


def get_vendor_for_tool(method: str) -> str:
    """获取某个工具方法配置的供应商"""
    # 先找所属分类
    for category, info in TOOL_CATEGORIES.items():
        if method in info["tools"]:
            vendor_config = config.get("data_vendors", {}).get(category, "")
            primary = [v.strip() for v in vendor_config.split(",") if v.strip()]
            if primary:
                return primary[0]
            return DataVendor.TUSHARE.value
    return DataVendor.TUSHARE.value


def get_vendor_chain(method: str) -> list[str]:
    """获取工具方法的供应商降级链"""
    for category, info in TOOL_CATEGORIES.items():
        if method in info["tools"]:
            vendor_config = config.get("data_vendors", {}).get(category, "")
            chain = [v.strip() for v in vendor_config.split(",") if v.strip()]
            return chain if chain else [DataVendor.TUSHARE.value]
    return [DataVendor.TUSHARE.value]


# ============================================================
# 供应商实现注册
# ============================================================

# type: dict[str, dict[str, Callable]]
# 结构: {method_name: {vendor_name: impl_function}}
_VENDOR_IMPLEMENTATIONS: dict[str, dict[str, Callable]] = {}


def register_vendor_impl(method: str, vendor: str, impl: Callable) -> None:
    """注册供应商实现"""
    if method not in _VENDOR_IMPLEMENTATIONS:
        _VENDOR_IMPLEMENTATIONS[method] = {}
    _VENDOR_IMPLEMENTATIONS[method][vendor] = impl


def get_vendor_impl(method: str, vendor: str) -> Callable | None:
    """获取供应商实现"""
    return _VENDOR_IMPLEMENTATIONS.get(method, {}).get(vendor)


# ============================================================
# 路由执行
# ============================================================

def route_to_vendor(method: str, *args, **kwargs) -> Any:
    """路由到供应商并按降级链执行

    流程:
    1. 获取方法的降级链
    2. 依次尝试每个供应商
    3. 成功 -> 返回结果
    4. 频率限制/未配置 -> 尝试下一个
    5. 无数据 -> 返回 NO_DATA 哨兵
    6. 所有供应商失败 -> 抛异常
    """
    chain = get_vendor_chain(method)

    last_no_data: NoMarketDataError | None = None
    first_error: Exception | None = None

    for vendor in chain:
        impl = get_vendor_impl(method, vendor)
        if impl is None:
            logger.warning("No implementation for %s/%s", vendor, method)
            continue

        try:
            result = impl(*args, **kwargs)
            if result is not None:
                return result
            # None 返回值视为无数据
            no_data = NoMarketDataError(
                f"{vendor}/{method} returned None",
                vendor=vendor, method=method
            )
            if last_no_data is None:
                last_no_data = no_data
            continue
        except VendorRateLimitError as e:
            logger.warning("Vendor %s rate-limited for %s; trying next", vendor, method)
            if first_error is None:
                first_error = e
            continue
        except VendorNotConfiguredError as e:
            logger.warning("Vendor %s not configured for %s; trying next", vendor, method)
            if first_error is None:
                first_error = e
            continue
        except NoMarketDataError as e:
            logger.warning("Vendor %s has no data for %s: %s", vendor, method, e)
            if last_no_data is None:
                last_no_data = e
            continue
        except Exception as e:
            logger.warning("Vendor %s failed for %s: %s", vendor, method, e)
            if first_error is None:
                first_error = e
            continue

    # 所有供应商都失败
    if last_no_data is not None:
        symbol = kwargs.get("code", kwargs.get("symbol", "unknown"))
        detail = last_no_data.detail or ""
        return (
            f"NO_DATA_AVAILABLE: No data for '{symbol}' from any vendor{'. ' + detail if detail else ''}. "
            f"Do not estimate or fabricate values."
        )

    if first_error is not None:
        raise VendorFatalError(
            f"All vendors failed for {method}: {first_error}",
            method=method,
        )

    raise VendorFatalError(f"No vendor available for {method}")
