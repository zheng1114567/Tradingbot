"""
Event Agent 工具箱 — 事件分析专用

工具清单:
1. search_cailianshe_news() — 搜索财联社新闻
2. search_eastmoney_news() — 搜索东方财富新闻
3. get_company_announcements() — 公司公告
4. get_calendar_events() — 财经日历事件
5. analyze_transmission_chain() — 事件传导链分析 (LLM辅助)
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from langchain_core.tools import tool

from ..core.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class EventTools:
    """Event Agent 工具箱"""

    def __init__(self):
        self.cache = CacheManager()

    def search_cailianshe_news(self, keyword: str,
                                 days_back: int = 3) -> list[dict[str, Any]]:
        """搜索财经新闻 (akshare 全球财经资讯)

        使用 akshare 的 stock_info_global() 获取全球财经资讯,
        通过关键词过滤相关新闻。
        """
        cache_key = f"news:{keyword}:{days_back}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            import akshare as ak
            df = ak.stock_info_global()
            if df is not None and not df.empty:
                records = df.to_dict("records")
                # 关键词过滤
                filtered = [
                    r for r in records
                    if keyword.lower() in str(r).lower()
                ]
                result = filtered[:20]
                self.cache.set(cache_key, result)
                return result
        except ImportError:
            logger.warning("akshare not installed")
        except Exception as e:
            logger.warning("新闻搜索失败: %s", e)
        return []

    def search_eastmoney_news(self, keyword: str,
                               days_back: int = 3) -> list[dict[str, Any]]:
        """搜索 A 股个股新闻资讯 (akshare stock_info_global)

        与 search_news 使用相同数据源, 提供别名方便后续扩展。
        """
        return self.search_cailianshe_news(keyword, days_back)

    def get_company_announcements(self, code: str,
                                   days_back: int = 30) -> list[dict[str, Any]]:
        """获取公司公告 (A股特有: 巨潮资讯/交易所公告)"""
        cache_key = f"announcements:{code}:{days_back}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            import akshare as ak
            df = ak.stock_zh_a_notice(code=code)
            if df is not None and not df.empty:
                result = df.head(20).to_dict("records")
                self.cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.warning("公告获取失败 %s: %s", code, e)
        return []

    def get_calendar_events(self,
                             start_date: str | None = None,
                             end_date: str | None = None) -> list[dict[str, Any]]:
        """获取财经日历事件

        包括: 宏观经济数据公布, 央行决议, 重要会议等
        当前使用 akshare stock_info_global, 后续接入专业财经日历 API。
        """
        cache_key = f"calendar:{start_date}:{end_date}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            import akshare as ak
            df = ak.stock_info_global()
            if df is not None and not df.empty:
                result = df.head(20).to_dict("records")
                self.cache.set(cache_key, result)
                return result
        except Exception as e:
            logger.warning("财经日历获取失败: %s", e)
        return []

    def detect_entity(self, text: str) -> list[dict[str, str]]:
        """从文本中提取 A 股相关实体

        简单的关键词 → 板块/上市公司映射
        """
        # A 股常见主题映射 (后续可扩展为知识图谱)
        THEME_MAP = {
            "新能源": ["新能源汽车", "光伏", "风电", "锂电池"],
            "AI": ["人工智能", "大模型", "算力", "芯片"],
            "消费": ["白酒", "食品饮料", "家电", "医美"],
            "医药": ["创新药", "医疗器械", "CXO", "中药"],
            "金融": ["银行", "券商", "保险", "互联网金融"],
            "地产": ["房地产开发", "物业管理", "建材"],
            "周期": ["煤炭", "钢铁", "有色", "化工", "石油"],
            "科技": ["半导体", "消费电子", "软件", "通信"],
        }

        detected = []
        for theme, keywords in THEME_MAP.items():
            for kw in keywords:
                if kw in text:
                    detected.append({"theme": theme, "keyword": kw})
        return detected[:5]


# 函数接口 (用于 LangChain Tool)
_tools = EventTools()

@tool
def search_news(keyword: str, days_back: int = 3) -> str:
    """搜索 A 股财经新闻"""
    results = _tools.search_cailianshe_news(keyword, days_back)
    results.extend(_tools.search_eastmoney_news(keyword, days_back))
    return str(results[:10])

@tool
def get_announcements(code: str, days_back: int = 30) -> str:
    """获取公司公告"""
    results = _tools.get_company_announcements(code, days_back)
    return str(results[:10])

@tool
def get_calendar(start_date: str | None = None, end_date: str | None = None) -> str:
    """获取财经日历"""
    results = _tools.get_calendar_events(start_date, end_date)
    return str(results[:10])
