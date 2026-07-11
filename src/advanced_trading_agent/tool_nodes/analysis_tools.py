"""
Analysis Agent 工具箱 — 因子评分和个股排序

Analysis Agent 不计算因子 (因子由 DataService 确定性计算),
但可以通过工具获取因子数据并进行比较分析。

工具清单:
1. get_factor_data() — 获取预计算因子
2. rank_stocks_by_factor() — 按因子排序
3. check_factor_crowding() — 因子拥挤度
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from ..core.cache_manager import CacheManager
from ..data_service.vendor_router import route_to_vendor

logger = logging.getLogger(__name__)


class AnalysisTools:
    """Analysis Agent 工具箱"""

    def __init__(self):
        self.cache = CacheManager()

    def get_factor_data(self, code: str | None = None,
                         sector: str | None = None,
                         top_n: int = 20) -> list[dict[str, Any]]:
        """获取个股因子数据

        因子由 DataService 的 FactorCalculator 预先算好,
        这里只做查询和过滤。
        """
        cache_key = f"factors:{code or sector or 'all'}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached[:top_n]

        try:
            data = route_to_vendor(
                "get_factors",
                code=code or "",
                sector=sector or "",
            )
            result = data[:top_n] if isinstance(data, list) else []
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("因子数据获取失败: %s", e)
            return []

    def rank_stocks(self, sector: str,
                     sort_by: str = "composite_score",
                     top_n: int = 10) -> list[dict[str, Any]]:
        """按因子排序板块内个股"""
        factors = self.get_factor_data(sector=sector, top_n=50)
        if not factors:
            return []
        sorted_factors = sorted(
            factors,
            key=lambda x: x.get(sort_by, 0) or 0,
            reverse=True,
        )
        return sorted_factors[:top_n]

    def check_crowding(self, sector: str) -> dict[str, Any]:
        """检查板块因子拥挤度

        拥挤信号:
        - 换手率过高
        - 集中度上升
        - 估值分位偏高
        """
        try:
            data = route_to_vendor("check_crowding", sector=sector)
            return data if isinstance(data, dict) else {
                "is_crowded": False, "warnings": []
            }
        except Exception as e:
            logger.warning("因子拥挤度检查失败: %s", e)
            return {"is_crowded": False, "warnings": []}


# 函数接口
_tools = AnalysisTools()

@tool
def get_factors(code: str | None = None, sector: str | None = None, top_n: int = 20) -> str:
    """获取个股因子数据"""
    result = _tools.get_factor_data(code, sector, top_n)
    return str(result)

@tool
def rank_stocks(sector: str, sort_by: str = "composite_score", top_n: int = 10) -> str:
    """按因子排序个股"""
    result = _tools.rank_stocks(sector, sort_by, top_n)
    return str(result)

@tool
def check_crowding(sector: str) -> str:
    """检查板块拥挤度"""
    result = _tools.check_crowding(sector)
    return str(result)
