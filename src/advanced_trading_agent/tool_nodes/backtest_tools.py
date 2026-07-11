"""
Backtest Agent 工具箱 — 回测验证专用

工具清单:
1. run_backtest() — 运行回测
2. find_similar_history() — 查找相似历史情境
3. get_statistical_significance() — 统计显著性检验
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from langchain_core.tools import tool

from ..backtest.engine import BacktestEngine
from ..core.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class BacktestTools:
    """Backtest Agent 工具箱"""

    def __init__(self):
        self.cache = CacheManager()
        self.engine = BacktestEngine()

    def run_backtest(self, code: str, entry_date: str,
                      holding_days: list[int] | None = None) -> dict[str, Any]:
        """运行回测并返回结果摘要"""
        cache_key = f"backtest:{code}:{entry_date}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            from ..data_service.vendor_router import route_to_vendor
            price_data = route_to_vendor("get_daily", code=code)
            if isinstance(price_data, str) and "NO_DATA" in price_data:
                return {"sample_size": 0, "error": price_data}

            import pandas as pd
            df = pd.DataFrame(price_data) if isinstance(price_data, list) else pd.DataFrame()

            if df.empty:
                return {"sample_size": 0, "error": "无价格数据"}

            result = self.engine.run_single(
                price_df=df,
                entry_date=date.fromisoformat(entry_date),
                code=code,
            )
            summary = {
                "tradable": result.tradable,
                "holding_days": result.holding_days,
                "returns": result.returns,
                "max_drawdown": result.max_drawdown,
                "cost_bps": result.cost_bps,
            }
            self.cache.set(cache_key, summary)
            return summary
        except Exception as e:
            logger.warning("回测失败 %s: %s", code, e)
            return {"sample_size": 0, "error": str(e)}

    def find_similar_history(self, conditions: dict[str, Any]) -> dict[str, Any]:
        """查找相似历史情境

        Args:
            conditions: {sentiment, sector, event_type, ...}
        """
        sentiment = conditions.get("sentiment", "正常")
        sector = conditions.get("sector", "")
        event_type = conditions.get("event_type", "")

        cache_key = f"similar:{sentiment}:{sector}:{event_type}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        from ..data_service.vendor_router import route_to_vendor
        data = route_to_vendor("find_similar",
                                sentiment=sentiment,
                                sector=sector,
                                event_type=event_type)
        result = data if isinstance(data, dict) else {
            "sample_size": 0,
            "win_rate": 0,
            "avg_excess_return": 0,
            "confidence": "low",
        }
        self.cache.set(cache_key, result)
        return result


# 函数接口
_tools = BacktestTools()

@tool
def run_backtest(code: str, entry_date: str | None = None) -> str:
    """运行回测"""
    result = _tools.run_backtest(
        code,
        entry_date or str(date.today()),
    )
    return str(result)

@tool
def find_similar(sentiment: str = "正常", sector: str = "", event_type: str = "") -> str:
    """查找相似历史情境"""
    result = _tools.find_similar_history({
        "sentiment": sentiment,
        "sector": sector,
        "event_type": event_type,
    })
    return str(result)
