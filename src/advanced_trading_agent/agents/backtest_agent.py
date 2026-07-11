"""
Backtest Agent — 历史证据审查员

模式: 先跑回测引擎 (确定性) → LLM 解释结果

职责:
1. 跑回测引擎获取统计结果
2. 判断样本量是否足够 (< 30 不能支撑买入)
3. 分析失败案例共性
4. 给出置信度

借鉴 TradingAgents 的 deferred reflection 模式:
- Phase A: 运行结束存 pending
- Phase B: 下次运行拉真实收益反思
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from ..backtest.engine import BacktestEngine
from ..llm.client import LLMClient
from ..tool_nodes.backtest_tools import BacktestTools
from .schemas import BacktestReport, Confidence

logger = logging.getLogger(__name__)

MIN_SAMPLE_SIZE = 30  # 最小样本量阈值


def create_backtest_agent(llm: LLMClient):
    """创建 Backtest Agent 节点函数"""

    def backtest_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", str(date.today()))
        tier2 = state.get("tier2_data", {})
        backtest_samples = tier2.get("backtest_samples", [])

        # 先用工具跑回测
        tools = BacktestTools()
        bt_result = tools.run_backtest(ticker, trade_date)

        # 从 Tier 2 回测样本获取统计
        sample_size = 0
        win_rate = 0
        avg_excess = 0
        best_period = None
        confidence = "low"

        if backtest_samples:
            s = backtest_samples[0]
            sample_size = s.get("sample_size", 0)
            win_rate = s.get("win_rate", 0)
            avg_excess = s.get("avg_excess_return", 0)
            best_period = s.get("best_holding_period")
            confidence = s.get("confidence", "low")
        elif bt_result:
            sample_size = 1  # 只有当前样本
            win_rate = 0
            avg_excess = 0

        # 确定性规则校验
        enough_samples = sample_size >= MIN_SAMPLE_SIZE

        summary_parts = [
            f"样本数: {sample_size} {'(充足)' if enough_samples else '(不足)'}",
            f"胜率: {win_rate:.1%}",
            f"超额收益: {avg_excess:+.2%}",
        ]
        if best_period:
            summary_parts.append(f"最佳持仓: {best_period}d")

        sample_summary = " | ".join(summary_parts)

        prompt = f"""你是 Backtest Agent, 历史证据审查员。

## 标的
{ticker}

## 回测统计
{sample_summary}

## 约束规则
- 样本 < {MIN_SAMPLE_SIZE}: 置信度最多 medium
- 胜率 < 50%: 不能推荐, 只能观察
- 超额收益 < 0: 拒绝

请评估当前标的是否有足够的历史证据支撑买入。
输出结构化回测验证报告。"""

        try:
            report = llm.chat(
                messages=[
                    ("system",
                     "你是 A 股历史证据审查员。样本不足时不能支撑买入。"
                     "必须同时考虑成功样本和失败样本。"),
                    ("human", prompt),
                ],
                response_format=BacktestReport,
            )
            # 规则覆盖 LLM (置信度强制降级)
            if not enough_samples and report.confidence == Confidence.HIGH:
                report.confidence = Confidence.MEDIUM
        except Exception as e:
            logger.warning("LLM backtest analysis failed, using rules: %s", e)
            report = BacktestReport(
                sample_size=sample_size,
                win_rate=win_rate,
                avg_excess_return=avg_excess,
                best_holding_period=best_period,
                failure_pattern="数据不足以分析失败共性",
                confidence=Confidence.LOW if not enough_samples else Confidence.MEDIUM,
                reasoning=f"基于回测数据的规则分析: {sample_summary}",
            )

        return {
            "backtest_report": report.to_markdown() if hasattr(report, 'to_markdown') else str(report),
            "backtest_report_obj": report,
        }

    return backtest_node
