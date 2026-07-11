"""
数据采集 — tushare + akshare 双供应商

通过 VendorRouter 统一路由, 支持降级链。
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from .vendor_router import (
    DataVendor,
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
    register_vendor_impl,
)

logger = logging.getLogger(__name__)


# ============================================================
# 供应商实现
# ============================================================

def _get_tushare():
    """延迟加载 tushare (API key 可能未配置)"""
    import os
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise VendorNotConfiguredError("TUSHARE_TOKEN not set", vendor="tushare")
    try:
        import tushare as ts
        ts.set_token(token)
        return ts
    except ImportError:
        raise VendorNotConfiguredError("tushare not installed (pip install tushare)", vendor="tushare")


def _get_akshare():
    """延迟加载 akshare"""
    try:
        import akshare as ak  # noqa
        return ak
    except ImportError:
        raise VendorNotConfiguredError("akshare not installed (pip install akshare)", vendor="akshare")


# ============================================================
# 行情数据
# ============================================================

def get_daily_tushare(code: str, start_date: str | None = None,
                      end_date: str | None = None) -> list[dict[str, Any]]:
    """从 tushare 获取日K数据"""
    pro = _get_tushare().pro_api()
    end = end_date or date.today().strftime("%Y%m%d")
    start = start_date or (date.today() - timedelta(days=365)).strftime("%Y%m%d")
    try:
        df = pro.daily(ts_code=code, start_date=start, end_date=end)
        if df is None or df.empty:
            raise NoMarketDataError(f"No daily data for {code}", symbol=code, vendor="tushare")
        return df.to_dict("records")
    except Exception as e:
        if "over频次" in str(e) or "次数" in str(e):
            raise VendorRateLimitError(str(e), vendor="tushare") from e
        raise


def get_daily_akshare(code: str, start_date: str | None = None,
                      end_date: str | None = None) -> list[dict[str, Any]]:
    """从 akshare 获取日K数据 (A股)"""
    ak = _get_akshare()
    end = end_date or date.today().strftime("%Y%m%d")
    start = start_date or (date.today() - timedelta(days=365)).strftime("%Y%m%d")
    try:
        # akshare 需要去掉后缀 .SZ / .SH
        symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df is None or df.empty:
            raise NoMarketDataError(f"No daily data for {code}", symbol=code, vendor="akshare")
        return df.to_dict("records")
    except Exception as e:
        logger.warning("akshare get_daily failed for %s: %s", code, e)
        raise NoMarketDataError(str(e), symbol=code, vendor="akshare") from e


# ============================================================
# 资金流向
# ============================================================

def get_capital_flow_tushare(code: str, start_date: str | None = None,
                              end_date: str | None = None) -> list[dict[str, Any]]:
    """从 tushare 获取个股资金流"""
    pro = _get_tushare().pro_api()
    end = end_date or date.today().strftime("%Y%m%d")
    start = start_date or (date.today() - timedelta(days=30)).strftime("%Y%m%d")
    try:
        df = pro.moneyflow(ts_code=code, start_date=start, end_date=end)
        if df is None or df.empty:
            raise NoMarketDataError(f"No capital flow for {code}", symbol=code, vendor="tushare")
        return df.to_dict("records")
    except Exception as e:
        if "over频次" in str(e):
            raise VendorRateLimitError(str(e), vendor="tushare") from e
        raise


def get_capital_flow_akshare(code: str, start_date: str | None = None,
                              end_date: str | None = None) -> list[dict[str, Any]]:
    """从 akshare 获取个股资金流 (A股)"""
    ak = _get_akshare()
    symbol = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
    try:
        df = ak.stock_individual_fund_flow(stock=symbol, market="sh" if symbol.startswith("6") else "sz")
        if df is None or df.empty:
            raise NoMarketDataError(f"No capital flow for {code}", symbol=code, vendor="akshare")
        return df.to_dict("records")
    except Exception as e:
        logger.warning("akshare capital_flow failed for %s: %s", code, e)
        raise NoMarketDataError(str(e), symbol=code, vendor="akshare") from e


# ============================================================
# 新闻数据
# ============================================================

def get_news_akshare(sector: str | None = None,
                     keyword: str | None = None) -> list[dict[str, Any]]:
    """从 akshare 获取财经新闻"""
    ak = _get_akshare()
    try:
        df = ak.stock_info_global()
        if df is not None and not df.empty:
            records = df.to_dict("records")
            if keyword:
                records = [r for r in records if keyword in str(r)]
            return records
    except Exception as e:
        logger.warning("akshare news failed: %s", e)
    return []


# ============================================================
# 板块数据
# ============================================================

def get_sector_tushare() -> list[dict[str, Any]]:
    """从 tushare 获取板块数据"""
    pro = _get_tushare().pro_api()
    try:
        df = pro.ths_member()
        if df is not None and not df.empty:
            return df.to_dict("records")
    except Exception as e:
        logger.warning("tushare sector failed: %s", e)
    return []


def get_sector_akshare() -> list[dict[str, Any]]:
    """从 akshare 获取板块数据"""
    ak = _get_akshare()
    try:
        df = ak.stock_board_concept_name_em()
        if df is not None and not df.empty:
            return df.to_dict("records")
    except Exception as e:
        logger.warning("akshare sector failed: %s", e)
    return []


# ============================================================
# 基本面数据
# ============================================================

def get_financial_tushare(code: str) -> list[dict[str, Any]]:
    """从 tushare 获取财务数据"""
    pro = _get_tushare().pro_api()
    try:
        df = pro.fina_indicator(ts_code=code)
        if df is not None and not df.empty:
            return df.to_dict("records")
    except Exception as e:
        if "over频次" in str(e):
            raise VendorRateLimitError(str(e), vendor="tushare") from e
        logger.warning("tushare financial failed for %s: %s", code, e)
    raise NoMarketDataError(f"No financial data for {code}", symbol=code, vendor="tushare")


# ============================================================
# 风险数据
# ============================================================

def get_suspended_tushare() -> list[dict[str, Any]]:
    """获取停牌股票列表"""
    pro = _get_tushare().pro_api()
    try:
        today = date.today().strftime("%Y%m%d")
        df = pro.suspend(suspend_date=today)
        if df is not None and not df.empty:
            return df["ts_code"].tolist()
    except Exception as e:
        logger.warning("tushare suspend failed: %s", e)
    return []


def get_st_status_tushare() -> list[str]:
    """获取 ST 股票列表"""
    pro = _get_tushare().pro_api()
    try:
        today = date.today().strftime("%Y%m%d")
        df = pro.namechange(change_date=today)
        if df is not None and not df.empty:
            st_codes = df[df["name"].str.contains("ST|*ST", na=False)]["ts_code"].tolist()
            return st_codes
    except Exception as e:
        logger.warning("tushare namechange failed: %s", e)
    return []


# ============================================================
# A股特有数据 (北向资金/涨停梯队/龙虎榜/融资融券)
# ============================================================

def get_northbound_flow_tushare(trade_date: str | None = None) -> dict[str, Any]:
    """获取北向资金流向 (A股特有)"""
    pro = _get_tushare().pro_api()
    td = trade_date or date.today().strftime("%Y%m%d")
    try:
        df = pro.moneyflow_hsgt(start_date=td, end_date=td)
        if df is not None and not df.empty:
            return df.to_dict("records")[0]
    except Exception as e:
        logger.warning("tushare northbound failed: %s", e)
    return {"net_inflow": 0, "note": "北向资金数据不可用"}


def get_northbound_flow_akshare(trade_date: str | None = None) -> dict[str, Any]:
    """从 akshare 获取北向资金"""
    import akshare as ak
    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df is not None and not df.empty:
            return df.to_dict("records")[0]
    except Exception as e:
        logger.warning("akshare northbound failed: %s", e)
    return {"net_inflow": 0}


def get_limit_up_tiers_akshare(trade_date: str | None = None) -> dict[str, int]:
    """涨停梯队分析 — 使用 akshare 东方财富涨停板数据

    返回首板/二板/三板及以上数量。
    """
    td = trade_date or date.today().strftime("%Y%m%d")
    td = td.replace("-", "")
    try:
        import akshare as ak
        df = ak.stock_zt_pool_em(date=td)
        if df is None or df.empty:
            logger.info("涨停板数据为空 (交易日: %s)", td)
            return {"first_board": 0, "second_board": 0, "third_plus": 0}

        # 从 "封板资金"、"涨停统计" 等列判断连板数
        # akshare 返回的列中包含 "连板数" 或 "board" 相关字段
        records = df.to_dict("records")
        first_board = 0
        second_board = 0
        third_plus = 0
        for r in records:
            board_count = r.get("连板数", r.get("连续涨停", 0))
            if board_count == 0:
                # 很多情况下连板数为 0 也是首板
                first_board += 1
            elif board_count == 1:
                first_board += 1
            elif board_count == 2:
                second_board += 1
            else:
                third_plus += 1

        return {"first_board": first_board, "second_board": second_board, "third_plus": third_plus}
    except ImportError:
        logger.warning("akshare not installed, using default limit-up data")
        return {"first_board": 0, "second_board": 0, "third_plus": 0}
    except Exception as e:
        logger.warning("涨停板数据获取失败: %s", e)
        return {"first_board": 0, "second_board": 0, "third_plus": 0}


def get_sector_tushare_full(top_n: int = 10) -> list[dict[str, Any]]:
    """获取板块完整数据 (含涨跌幅排名)"""
    try:
        import tushare as ts
        pro = _get_tushare().pro_api()
        df = pro.ths_index()
        if df is not None and not df.empty:
            return df.head(top_n).to_dict("records")
    except Exception as e:
        logger.warning("tushare sector full failed: %s", e)
    return []


def get_factors_computed(code: str = "", sector: str = "") -> list[dict[str, Any]]:
    """因子数据 — 通过 FactorCalculator 从行情数据实时计算

    1. 获取近 365 天的日K
    2. 用 FactorCalculator 计算六因子
    3. 返回最新一期的因子值
    """
    if not code:
        return []

    from .cleaner import DataCleaner
    from .factors import FactorCalculator

    try:
        # 通过路由获取日K (优先 tushare, 降级 akshare)
        from .vendor_router import route_to_vendor
        daily = route_to_vendor("get_daily", code=code)
        if isinstance(daily, str) or not daily:
            return []

        df = DataCleaner.clean_daily(daily)
        if df.empty:
            return []

        # 计算因子
        df = FactorCalculator.run_all(df)
        if df.empty:
            return []

        # 取最新一行
        latest = df.iloc[-1].to_dict()
        result = [{
            "code": code,
            "name": latest.get("name", ""),
            "sector": sector,
            "quality_score": latest.get("roe"),
            "growth_score": latest.get("revenue_growth"),
            "valuation_score": latest.get("pe_quantile", latest.get("pb_quantile")),
            "momentum_score": latest.get("momentum_20d"),
            "volatility_score": latest.get("volatility"),
            "liquidity_score": latest.get("amihud"),
            "composite_score": latest.get("composite_score"),
            "factor_warning": None,
        }]
        return result
    except Exception as e:
        logger.warning("因子计算失败 %s: %s", code, e)
        return []


def check_crowding_stub(sector: str = "") -> dict[str, Any]:
    """因子拥挤度 (stub)"""
    return {"is_crowded": False, "warnings": []}


def find_similar_stub(sentiment: str = "", sector: str = "",
                       event_type: str = "") -> dict[str, Any]:
    """相似历史情境 (stub)"""
    return {"sample_size": 0, "win_rate": 0, "avg_excess_return": 0, "confidence": "low"}


def get_dragon_tiger_tushare(trade_date: str | None = None) -> list[dict[str, Any]]:
    """龙虎榜数据 (A股特有)"""
    pro = _get_tushare().pro_api()
    td = trade_date or date.today().strftime("%Y%m%d")
    try:
        df = pro.lhb(start_date=td, end_date=td)
        if df is not None and not df.empty:
            return df.to_dict("records")[:20]
    except Exception as e:
        logger.warning("tushare dragon_tiger failed: %s", e)
    return []


def get_margin_tushare(trade_date: str | None = None) -> list[dict[str, Any]]:
    """融资融券数据 (A股特有)"""
    pro = _get_tushare().pro_api()
    td = trade_date or date.today().strftime("%Y%m%d")
    try:
        df = pro.margin(start_date=td, end_date=td)
        if df is not None and not df.empty:
            return df.to_dict("records")[:20]
    except Exception as e:
        logger.warning("tushare margin failed: %s", e)
    return []


# ============================================================
# 注册供应商实现
# ============================================================

def register_all_vendors():
    """注册所有供应商实现到路由系统"""
    # 行情
    register_vendor_impl("get_daily", "tushare", get_daily_tushare)
    register_vendor_impl("get_daily", "akshare", get_daily_akshare)
    # 资金流
    register_vendor_impl("get_capital_flow", "tushare", get_capital_flow_tushare)
    register_vendor_impl("get_capital_flow", "akshare", get_capital_flow_akshare)
    # 新闻
    register_vendor_impl("get_news", "akshare", get_news_akshare)
    # 板块
    register_vendor_impl("get_sector", "tushare", get_sector_tushare_full)
    register_vendor_impl("get_sector", "akshare", get_sector_akshare)
    # 财务
    register_vendor_impl("get_financial", "tushare", get_financial_tushare)
    # 风控
    register_vendor_impl("get_suspended", "tushare", get_suspended_tushare)
    register_vendor_impl("get_st_status", "tushare", get_st_status_tushare)
    # A股特有: 北向资金
    register_vendor_impl("get_northbound_flow", "tushare", get_northbound_flow_tushare)
    register_vendor_impl("get_northbound_flow", "akshare", get_northbound_flow_akshare)
    # A股特有: 涨停梯队
    register_vendor_impl("get_limit_up_tiers", "tushare", get_limit_up_tiers_akshare)
    register_vendor_impl("get_limit_up_tiers", "akshare", get_limit_up_tiers_akshare)
    # A股特有: 龙虎榜
    register_vendor_impl("get_dragon_tiger", "tushare", get_dragon_tiger_tushare)
    # A股特有: 融资融券
    register_vendor_impl("get_margin", "tushare", get_margin_tushare)
    # 分析数据
    register_vendor_impl("get_factors", "tushare", get_factors_computed)
    register_vendor_impl("get_factors", "akshare", get_factors_computed)
    register_vendor_impl("check_crowding", "tushare", check_crowding_stub)
    register_vendor_impl("find_similar", "tushare", find_similar_stub)


# 自动注册
register_all_vendors()
