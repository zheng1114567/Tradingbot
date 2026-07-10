"""
Market Agent — 市场温度分析师

职责:
- 判断市场情绪: 冰点/正常/高潮
- 计算仓位上限
- 判断板块资金是否持续
- 检测价格/资金背离

借鉴 TradingAgents 的 market_analyst.py + sentiment_analyst.py 模式
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from ..llm.client import LLMClient
from .schemas import MarketReport

logger = logging.getLogger(__name__)

# 情绪 → 仓位映射规则
SENTIMENT_POSITION_MAP = {
    "冰点": 0.2,
    "低迷": 0.4,
    "正常": 0.6,
    "温热": 0.5,
    "高潮": 0.3,
}


def create_market_agent(llm: LLMClient):
    """创建 Market Agent 节点函数

    借鉴 TradingAgents: 先规则判断, 再 LLM 增强
    """

    def market_node(state: dict[str, Any]) -> dict[str, Any]:
        trade_date = state.get("trade_date", str(date.today()))
        ticker = state.get("company_of_interest", "")

        # 从 state 读取 Tier 1 数据
        tier1_data = state.get("tier1_data", {})
        market_data = tier1_data.get("market", {})
        sentiment_data = tier1_data.get("sentiment", {})
        capital_data = tier1_data.get("capital", {})

        # 确定性规则: 情绪 → 仓位
        sentiment_level = sentiment_data.get("sentiment", "正常")
        if isinstance(sentiment_level, str):
            position_cap = SENTIMENT_POSITION_MAP.get(sentiment_level, 0.5)
        else:
            position_cap = 0.5

        # LLM 增强分析
        prompt = f"""你是 Market Agent, 负责判断当前市场环境和资金状态。

## 市场数据
- 大盘收盘: {market_data.get('index_close', 'N/A')}
- 涨跌幅: {market_data.get('index_change_pct', 'N/A')}%
- 上涨/下跌: {market_data.get('advance_count', 'N/A')}/{market_data.get('decline_count', 'N/A')}
- 涨停/跌停: {market_data.get('limit_up_count', 'N/A')}/{market_data.get('limit_down_count', 'N/A')}
- 两市成交额: {market_data.get('total_volume_cny', 'N/A')}

## 情绪数据
- 情绪档位: {sentiment_level}
- 情绪评分: {sentiment_data.get('sentiment_score', 'N/A')}

## 资金数据
- {ticker} 主力净流入: {capital_data.get('net_inflow_main', 'N/A')}
- 连续流入天数: {capital_data.get('consecutive_inflow_days', 0)}

请输出结构化的市场分析报告。"""

        try:
            report = llm.chat(
                messages=[
                    ("system", "你是一个专业的 A 股市场分析师。请基于数据输出结构化报告。"),
                    ("human", prompt),
                ],
                response_format=MarketReport,
            )
        except Exception as e:
            logger.warning("LLM market analysis failed, using rule-based: %s", e)
            report = MarketReport(
                market_state=sentiment_level,
                position_cap=position_cap,
                capital_confirmation=capital_data.get("confirmation", "未知"),
                sector_preference=[],
                reasoning=f"规则判断: 情绪 {sentiment_level}, 仓位上限 {position_cap:.0%}",
            )

        return {
            "market_report": report.model_dump() if hasattr(report, "model_dump") else str(report),
            "market_report_obj": report,
        }

    return market_node
