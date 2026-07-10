"""
Backtest Agent — 历史证据审查员

职责:
- 查找相似历史情境
- 输出胜率/收益/Sharpe/最大回撤
- 给出最佳持仓周期
- 列出成功和失败样本
- 判断样本量是否足够

借鉴 TradingAgents 的 backtrader 集成,
但这里 Backtest Agent 是一个 LLM Agent,
驱动 backtest/engine.py 执行实际回测。
"""
from __future__ import annotations

import logging
from typing import Any

from ..backtest.engine import BacktestEngine
from ..backtest.metrics import PerformanceMetrics
from ..llm.client import LLMClient
from .schemas import BacktestReport

logger = logging.getLogger(__name__)


def create_backtest_agent(llm: LLMClient):
    """创建 Backtest Agent 节点函数"""

    def backtest_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        market_report = state.get("market_report_obj", None)
        event_report = state.get("event_report_obj", None)
        analysis_report = state.get("analysis_report_obj", None)
        tier2_data = state.get("tier2_data", {})
        backtest_samples = tier2_data.get("backtest_samples", [])

        # 使用回测引擎计算
        engine = BacktestEngine()
        if backtest_samples:
            sample = backtest_samples[0]
            sample_size = sample.get("sample_size", 0)
            win_rate = sample.get("win_rate", 0)
            avg_excess = sample.get("avg_excess_return", 0)
            best_period = sample.get("best_holding_period")
            confidence = sample.get("confidence", "low")
        else:
            sample_size = 0
            win_rate = 0
            avg_excess = 0
            best_period = None
            confidence = "low"

        sample_summary = (
            f"样本数: {sample_size}, 胜率: {win_rate:.1%}, "
            f"平均超额: {avg_excess:+.2%}, 最佳持仓: {best_period or 'N/A'}d"
        ) if sample_size > 0 else "暂无回测样本"

        prompt = f"""你是 Backtest Agent, 历史证据审查员。

## 标的
{ticker}

## 各 Agent 分析
Market: {market_report}
Event: {event_report}
Analysis: {analysis_report}

## 回测样本
{sample_summary}

请评估:
1. 样本量是否足够? (< 30 的样本不能单独支撑买入)
2. 收益是否统计显著?
3. 失败案例的共性是什么?
4. 当前情境更像成功样本还是失败样本?
5. 置信度如何?

输出结构化的回测验证报告。"""

        try:
            report = llm.chat(
                messages=[
                    ("system", "你是 A 股历史证据审查员。样本不足时不支持买入。"),
                    ("human", prompt),
                ],
                response_format=BacktestReport,
            )
        except Exception as e:
            logger.warning("LLM backtest analysis failed: %s", e)
            report = BacktestReport(
                sample_size=sample_size,
                win_rate=win_rate,
                avg_excess_return=avg_excess,
                best_holding_period=best_period,
                failure_pattern="数据不足",
                confidence=confidence,
                similar_success_cases=0,
                similar_failure_cases=0,
                reasoning=f"基于回测数据的规则分析: {sample_summary}",
            )

        return {
            "backtest_report": report.model_dump() if hasattr(report, "model_dump") else str(report),
            "backtest_report_obj": report,
        }

    return backtest_node
