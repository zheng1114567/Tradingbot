"""
数据供应商路由 — 免费数据源优先

核心设计:
- 按工具粒度配置数据源 (e.g. "market_data": "akshare,baostock")
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
import threading
import time
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
    MOOTDX = "mootdx"
    BAOSTOCK = "baostock"
    EASTMONEY = "eastmoney"
    SINA = "sina"
    CLS = "cls"
    TENCENT = "tencent"
    LOCAL_CACHE = "local_cache"


# 工具分类
TOOL_CATEGORIES = {
    "market_data": {
        "description": "行情数据 (日K, 分钟K)",
        "tools": ["get_daily", "get_minute", "get_market_breadth"],
    },
    "fundamental_data": {
        "description": "基本面数据",
        "tools": ["get_financial", "get_balance", "get_income", "get_cashflow", "get_snapshot"],
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
            "get_northbound_top10",    # 北向十大成交
            "get_limit_up_tiers",      # 涨停梯队
            "get_sector",              # 板块数据
            "get_sector_constituents", # 板块成分股
            "get_dragon_tiger",        # 龙虎榜
            "get_margin",              # 融资融券
        ],
    },
    "risk_data": {
        "description": "风险基础数据",
        "tools": ["get_suspended", "get_st_status", "get_delisting"],
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
            return ""
    return ""


def get_vendor_chain(method: str) -> list[str]:
    """获取工具方法的供应商降级链

    优先从 config 读取，然后追加所有已注册但 config 中未列出的供应商。
    """
    chain: list[str] = []
    for category, info in TOOL_CATEGORIES.items():
        if method in info["tools"]:
            vendor_config = config.get("data_vendors", {}).get(category, "")
            chain = [v.strip() for v in vendor_config.split(",") if v.strip()]
            break

    if not chain:
        chain = []

    # Append registered vendors not already in the chain
    registered = list(_VENDOR_IMPLEMENTATIONS.get(method, {}).keys())
    for vendor in registered:
        if vendor not in chain:
            chain.append(vendor)

    return chain


# ============================================================
# 供应商实现注册
# ============================================================

# type: dict[str, dict[str, Callable]]
# 结构: {method_name: {vendor_name: impl_function}}
_VENDOR_IMPLEMENTATIONS: dict[str, dict[str, Callable]] = {}
_DEFAULT_VENDOR_REGISTRATION_ATTEMPTED = False
_VENDOR_LOCK = threading.Lock()


def register_vendor_impl(method: str, vendor: str, impl: Callable) -> None:
    """注册供应商实现 (thread-safe)"""
    with _VENDOR_LOCK:
        if method not in _VENDOR_IMPLEMENTATIONS:
            _VENDOR_IMPLEMENTATIONS[method] = {}
        _VENDOR_IMPLEMENTATIONS[method][vendor] = impl


def get_vendor_impl(method: str, vendor: str) -> Callable | None:
    """获取供应商实现"""
    return _VENDOR_IMPLEMENTATIONS.get(method, {}).get(vendor)


def ensure_default_vendor_registration() -> None:
    """Lazily register built-in adapters for direct tool-node calls."""
    global _DEFAULT_VENDOR_REGISTRATION_ATTEMPTED
    if _DEFAULT_VENDOR_REGISTRATION_ATTEMPTED:
        return
    _DEFAULT_VENDOR_REGISTRATION_ATTEMPTED = True
    existing = {
        method: dict(vendors)
        for method, vendors in _VENDOR_IMPLEMENTATIONS.items()
    }
    try:
        from .collector import register_all_vendors

        register_all_vendors()
        for method, vendors in existing.items():
            _VENDOR_IMPLEMENTATIONS.setdefault(method, {}).update(vendors)
    except Exception as exc:
        logger.warning("Default vendor registration failed: %s", exc)


def _record_route_attempt(
    route_trace: list[dict[str, Any]] | None,
    *,
    method: str,
    vendor: str,
    status: str,
    elapsed_ms: float | None = None,
    record_count: int | None = None,
    error: str | None = None,
) -> None:
    """Append one vendor attempt to the optional auditable route trace."""
    if route_trace is None:
        return
    attempt: dict[str, Any] = {
        "method": method,
        "vendor": vendor,
        "status": status,
    }
    if elapsed_ms is not None:
        attempt["elapsed_ms"] = round(elapsed_ms, 3)
    if record_count is not None:
        attempt["record_count"] = record_count
    if error:
        attempt["error"] = error
    route_trace.append(attempt)


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
    route_trace = kwargs.pop("_route_trace", None)
    if method in {tool for info in TOOL_CATEGORIES.values() for tool in info["tools"]}:
        ensure_default_vendor_registration()
    chain = get_vendor_chain(method)

    last_no_data: NoMarketDataError | None = None
    first_error: Exception | None = None

    for vendor in chain:
        impl = get_vendor_impl(method, vendor)
        if impl is None:
            logger.warning("No implementation for %s/%s", vendor, method)
            _record_route_attempt(
                route_trace,
                method=method,
                vendor=vendor,
                status="missing_impl",
            )
            continue

        try:
            start = time.perf_counter()
            result = impl(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Treat None and empty list/dict as "no data" → try next vendor
            is_empty = result is None or (isinstance(result, (list, dict)) and len(result) == 0)
            if not is_empty:
                _record_route_attempt(
                    route_trace,
                    method=method,
                    vendor=vendor,
                    status="success",
                    elapsed_ms=elapsed_ms,
                    record_count=len(result) if isinstance(result, list) else None,
                )
                return result
            # None or empty container → treat as no data
            no_data = NoMarketDataError(
                f"{vendor}/{method} returned empty",
                vendor=vendor, method=method
            )
            _record_route_attempt(
                route_trace,
                method=method,
                vendor=vendor,
                status="no_data",
                elapsed_ms=elapsed_ms,
                error=str(no_data),
            )
            if last_no_data is None:
                last_no_data = no_data
            continue
        except VendorRateLimitError as e:
            logger.warning("Vendor %s rate-limited for %s; trying next", vendor, method)
            _record_route_attempt(
                route_trace,
                method=method,
                vendor=vendor,
                status="rate_limited",
                error=str(e),
            )
            if first_error is None:
                first_error = e
            continue
        except VendorNotConfiguredError as e:
            logger.warning("Vendor %s not configured for %s; trying next", vendor, method)
            _record_route_attempt(
                route_trace,
                method=method,
                vendor=vendor,
                status="not_configured",
                error=str(e),
            )
            if first_error is None:
                first_error = e
            continue
        except NoMarketDataError as e:
            logger.warning("Vendor %s has no data for %s: %s", vendor, method, e)
            _record_route_attempt(
                route_trace,
                method=method,
                vendor=vendor,
                status="no_data",
                error=str(e),
            )
            if last_no_data is None:
                last_no_data = e
            continue
        except VendorFatalError:
            raise
        except Exception as e:
            logger.warning("Vendor %s failed for %s: %s", vendor, method, e)
            _record_route_attempt(
                route_trace,
                method=method,
                vendor=vendor,
                status="error",
                error=str(e),
            )
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
