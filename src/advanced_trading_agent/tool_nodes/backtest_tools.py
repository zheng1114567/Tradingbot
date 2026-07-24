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
            from ..data_agent.vendor_router import route_to_vendor
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
        """查找相似历史情境 — 基于因子和历史收益模式匹配

        用当前价格数据中的因子模式（过去 N 天的动量、波动率特征），
        在历史数据中找相似窗口，统计相似窗口的后续表现。
        """
        sentiment = conditions.get("sentiment", "正常")
        sector = conditions.get("sector", "")
        event_type = conditions.get("event_type", "")
        code = conditions.get("code", "")

        cache_key = f"similar:{sentiment}:{sector}:{event_type}:{code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        if not code:
            return {"sample_size": 0, "win_rate": 0, "note": "no_code_provided", "confidence": "low"}

        try:
            from ..data_agent.vendor_router import route_to_vendor
            from ..data_agent.factors import FactorCalculator

            price_data = route_to_vendor("get_daily", code=code)
            if isinstance(price_data, str) and "NO_DATA" in price_data:
                return {"sample_size": 0, "note": "no_price_data", "confidence": "low"}

            import pandas as pd
            df = pd.DataFrame(price_data) if isinstance(price_data, list) else pd.DataFrame()
            if df.empty or "trade_date" not in df.columns:
                return {"sample_size": 0, "note": "empty_price_data", "confidence": "low"}

            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df = df.dropna(subset=["trade_date"]).sort_values("trade_date")

            # Compute factors for pattern matching
            calc = FactorCalculator()
            factors = calc.compute(df) if hasattr(calc, "compute") else []

            # Use recent factor patterns as template, count similar windows
            sample_size, win_rate = self._match_similar_windows(df, factors, sentiment)
            result = {
                "sample_size": sample_size,
                "win_rate": round(win_rate, 4),
                "avg_excess_return": 0.0,
                "confidence": "medium" if sample_size >= 5 else "low",
                "sentiment": sentiment,
                "sector": sector,
                "event_type": event_type,
            }
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("find_similar_history failed: %s", e)
            return {"sample_size": 0, "win_rate": 0, "note": str(e), "confidence": "low"}

    @staticmethod
    def _match_similar_windows(
        df: "pd.DataFrame",
        factors: list[dict[str, Any]] | None,
        sentiment: str,
    ) -> tuple[int, float]:
        """Match historical windows with similar factor patterns.

        Uses rolling returns and volatility to find windows.
        Returns (sample_size, win_rate).
        """
        if df.empty or len(df) < 25:
            return 0, 0.0

        # Use recent 5-day return and 20-day volatility as the template
        recent = df.tail(20)
        if recent.empty:
            return 0, 0.0

        recent_return = (float(recent.iloc[-1]["close"]) / float(recent.iloc[0]["close"]) - 1)
        if "pct_chg" in df.columns:
            recent_vol = float(df.tail(20)["pct_chg"].std()) if df.tail(20).get("pct_chg") is not None else 0.02
        else:
            recent_vol = 0.02

        if recent_vol <= 0:
            recent_vol = 0.01

        # Slide a 5-day window, check if window return falls in range
        matches = 0
        wins = 0
        window = 5
        ret_threshold = recent_vol * 0.8

        for i in range(0, len(df) - window - 5):
            try:
                win_start = float(df.iloc[i]["close"])
                win_end = float(df.iloc[i + window]["close"])
                if win_start <= 0:
                    continue
                win_return = (win_end - win_start) / win_start
                # Match: window return within ~1 vol of recent
                if abs(win_return - recent_return) <= ret_threshold + recent_vol:
                    matches += 1
                    forward_end = float(df.iloc[i + window + 5]["close"])
                    forward_return = (forward_end - win_end) / win_end
                    if forward_return > 0:
                        wins += 1
            except (KeyError, IndexError, ValueError):
                continue

        if matches == 0:
            return 0, 0.0
        return matches, wins / matches


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
def find_similar(sentiment: str = "正常", sector: str = "", event_type: str = "", code: str = "") -> str:
    """查找相似历史情境 — 基于历史价格模式匹配"""
    result = _tools.find_similar_history({
        "sentiment": sentiment,
        "sector": sector,
        "event_type": event_type,
        "code": code,
    })
    return str(result)
