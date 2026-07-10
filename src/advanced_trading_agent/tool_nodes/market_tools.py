"""
Market Agent 工具箱 — A 股市场分析专用

工具清单:
1. get_market_sentiment() — 市场情绪 (涨跌比/涨停梯度/炸板率)
2. get_sector_rotation() — 板块轮动分析
3. get_northbound_flow() — 北向资金流向 (A股特有)
4. get_capital_flow() — 主力资金流向
5. get_limit_up_tiers() — 涨停梯队 (A股特有)
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from ..core.cache_manager import CacheManager
from ..data_service.vendor_router import route_to_vendor

logger = logging.getLogger(__name__)


class MarketTools:
    """Market Agent 的工具箱 — 每个方法可以作为 LangChain Tool"""

    def __init__(self):
        self.cache = CacheManager()

    def get_market_sentiment(self, trade_date: str | None = None) -> dict[str, Any]:
        """获取市场情绪: 涨跌比, 涨停梯度, 炸板率

        A 股特有: 涨停梯度分析 (首板/二板/三板以上数量)
        """
        cache_key = f"market_sentiment:{trade_date or date.today()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        trade_date = trade_date or str(date.today())
        try:
            data = route_to_vendor("get_daily", code="000001.SH",
                                   start_date=trade_date, end_date=trade_date)
            # 涨停梯队需要额外计算
            result = {
                "sentiment": data.get("sentiment", "正常") if isinstance(data, dict) else "正常",
                "advance_count": data.get("advance_count", 0),
                "decline_count": data.get("decline_count", 0),
                "limit_up_count": data.get("limit_up_count", 0),
                "limit_down_count": data.get("limit_down_count", 0),
                "timestamp": datetime.now().isoformat(),
            }
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("市场情绪获取失败: %s", e)
            return {"sentiment": "正常", "error": str(e)}

    def get_northbound_flow(self, trade_date: str | None = None) -> dict[str, Any]:
        """获取北向资金流向 (A股特有)

        北向资金 = 沪股通 + 深股通
        数据在每日 17:30 后可用
        """
        cache_key = f"northbound:{trade_date or date.today()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            data = route_to_vendor("get_northbound_flow",
                                   trade_date=trade_date or str(date.today()))
            result = data if isinstance(data, dict) else {"net_inflow": 0}
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("北向资金获取失败: %s", e)
            return {"net_inflow": 0, "error": str(e)}

    def get_capital_flow(self, sector: str | None = None,
                          trade_date: str | None = None) -> dict[str, Any]:
        """获取主力资金流向

        Args:
            sector: 板块名称 (可选)
            trade_date: 交易日
        """
        cache_key = f"capital_flow:{sector or 'all'}:{trade_date or date.today()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            data = route_to_vendor("get_capital_flow",
                                   code=sector or "",
                                   trade_date=trade_date or str(date.today()))
            result = data if isinstance(data, dict) else {"net_inflow_main": 0}
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("资金流向获取失败: %s", e)
            return {"net_inflow_main": 0, "error": str(e)}

    def get_sector_rotation(self, top_n: int = 10) -> list[dict[str, Any]]:
        """获取板块轮动 (涨跌幅排名)

        A 股特有: 申万一级行业轮动分析
        """
        cache_key = f"sector_rotation:{date.today()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached[:top_n]

        try:
            data = route_to_vendor("get_sector", top_n=top_n)
            result = data[:top_n] if isinstance(data, list) else []
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("板块轮动获取失败: %s", e)
            return []

    def get_limit_up_tiers(self, trade_date: str | None = None) -> dict[str, int]:
        """获取涨停梯队分析 (A股特有)

        返回:
            first_board: 首板数量
            second_board: 二板数量
            third_plus: 三板及以上数量
        """
        cache_key = f"limit_up_tiers:{trade_date or date.today()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            data = route_to_vendor("get_limit_up_tiers",
                                   trade_date=trade_date or str(date.today()))
            result = data if isinstance(data, dict) else {
                "first_board": 0, "second_board": 0, "third_plus": 0
            }
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("涨停梯队获取失败: %s", e)
            return {"first_board": 0, "second_board": 0, "third_plus": 0}


# LangChain Tool 包装函数
_tools_instance = MarketTools()

def get_market_sentiment(**kwargs) -> str:
    """获取市场情绪数据"""
    result = _tools_instance.get_market_sentiment(kwargs.get("trade_date"))
    return str(result)

def get_northbound_flow(**kwargs) -> str:
    """获取北向资金流向"""
    result = _tools_instance.get_northbound_flow(kwargs.get("trade_date"))
    return str(result)

def get_capital_flow(**kwargs) -> str:
    """获取主力资金流向"""
    result = _tools_instance.get_capital_flow(
        kwargs.get("sector"), kwargs.get("trade_date")
    )
    return str(result)

def get_sector_rotation(**kwargs) -> str:
    """获取板块轮动"""
    result = _tools_instance.get_sector_rotation(kwargs.get("top_n", 10))
    return str(result)

def get_limit_up_tiers(**kwargs) -> str:
    """获取涨停梯队"""
    result = _tools_instance.get_limit_up_tiers(kwargs.get("trade_date"))
    return str(result)
