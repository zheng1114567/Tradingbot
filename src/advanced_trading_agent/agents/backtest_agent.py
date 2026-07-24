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

import pandas as pd

from ..backtest.engine import BacktestEngine
from ..llm.client import LLMClient
from ..tool_nodes.backtest_tools import BacktestTools
from ..tool_nodes.registry import get_agent_tools
from .contract import (
    basic_self_check,
    build_agent_update,
    build_react_agent,
    run_react_agent,
)
from .schemas import BacktestReport, Confidence
from .specs import get_agent_skill

logger = logging.getLogger(__name__)

MIN_SAMPLE_SIZE = 30  # 最小样本量阈值


def _backtest_from_price_data(
    price_data: Any,
    *,
    ticker: str,
    trade_date: str,
) -> dict[str, Any] | None:
    """Run deterministic backtest from DataAgent price_data when available."""
    if not isinstance(price_data, list) or len(price_data) < 2:
        return None
    df = pd.DataFrame(price_data)
    if df.empty or "trade_date" not in df.columns or "close" not in df.columns:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
    if df.empty:
        return None
    result = BacktestEngine().run_single(
        price_df=df,
        entry_date=date.fromisoformat(trade_date),
        code=ticker,
    )
    valid_returns = [v for v in result.returns.values() if v is not None]
    valid_excess = [v for v in result.excess_returns.values() if v is not None]
    return {
        "sample_size": len(valid_returns),
        "win_rate": sum(1 for v in valid_returns if v > 0) / len(valid_returns) if valid_returns else 0,
        "avg_excess_return": sum(valid_excess) / len(valid_excess) if valid_excess else 0,
        "best_holding_period": result.holding_days or None,
        "tradable": result.tradable,
        "returns": result.returns,
        "error": None if valid_returns else "no executable holding-period returns",
    }


def create_backtest_agent(llm: LLMClient, tools: BacktestTools | None = None):
    """创建 Backtest Agent 节点函数"""
    skill = get_agent_skill("backtest")
    react_agent = build_react_agent(
        llm=llm,
        tools=get_agent_tools("backtest"),
        system_prompt=skill.react_prompt,
        response_format=BacktestReport,
    )

    def backtest_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", str(date.today()))
        tier2 = state.get("tier2_data", {})
        backtest_samples = tier2.get("backtest_samples", [])

        # 先用工具跑回测
        backtest_tools = tools or BacktestTools()
        bt_result = backtest_tools.run_backtest(ticker, trade_date)
        local_bt_result = _backtest_from_price_data(
            tier2.get("price_data", []),
            ticker=ticker,
            trade_date=trade_date,
        )
        tool_calls = [
            {
                "tool": "run_backtest",
                "args": {"ticker": ticker, "trade_date": trade_date},
                "records": 1 if bt_result else 0,
            },
            {
                "tool": "backtest_from_tier2_price_data",
                "args": {"ticker": ticker, "trade_date": trade_date},
                "records": 1 if local_bt_result else 0,
            }
        ]

        # 从 Tier 2 回测样本获取统计
        sample_size = 0
        win_rate = 0
        avg_excess = 0
        best_period = None

        if backtest_samples:
            s = backtest_samples[0]
            sample_size = s.get("sample_size", 0)
            win_rate = s.get("win_rate", 0)
            avg_excess = s.get("avg_excess_return", 0)
            best_period = s.get("best_holding_period")
        elif local_bt_result and not local_bt_result.get("error"):
            sample_size = int(local_bt_result.get("sample_size", 0) or 0)
            win_rate = float(local_bt_result.get("win_rate", 0) or 0)
            avg_excess = float(local_bt_result.get("avg_excess_return", 0) or 0)
            best_period = local_bt_result.get("best_holding_period")
        elif bt_result and not bt_result.get("error"):
            sample_size = int(bt_result.get("sample_size", 1) or 1)
            win_rate = float(bt_result.get("win_rate", 0) or 0)
            avg_excess = float(bt_result.get("avg_excess_return", 0) or 0)
            best_period = bt_result.get("best_holding_period")

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

        report, react_trace = run_react_agent(react_agent, prompt)
        if react_trace:
            tool_calls.extend(react_trace)

        try:
            if report is None:
                report = llm.chat(
                    messages=[
                        ("system", skill.fallback_system_prompt),
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

        evidence = [
            f"sample_size={report.sample_size}",
            f"win_rate={report.win_rate:.1%}",
            f"avg_excess_return={report.avg_excess_return:+.2%}",
            f"best_holding_period={report.best_holding_period or 'N/A'}",
            f"confidence={report.confidence.value}",
        ]
        warnings = []
        if report.sample_size < MIN_SAMPLE_SIZE:
            warnings.append(f"样本不足: {report.sample_size} < {MIN_SAMPLE_SIZE}")
        if report.win_rate < 0.5:
            warnings.append(f"胜率不足: {report.win_rate:.1%}")
        if report.avg_excess_return < 0:
            warnings.append(f"超额收益为负: {report.avg_excess_return:+.2%}")

        return build_agent_update(
            state,
            sender="Backtest Agent",
            report_key="backtest_report",
            report=report.to_markdown()
            if hasattr(report, "to_markdown")
            else str(report),
            report_obj_key="backtest_report_obj",
            report_obj=report,
            evidence=evidence,
            tool_calls=tool_calls,
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=["sample_size_confidence_cap_enforced"],
                warnings=warnings,
                confidence=report.confidence.value,
            ),
        )

    return backtest_node
