"""
Report Agent — 报告输出层

职责:
- 从 System Decision 格式化输出
- 保留反对意见和风控结果
- 生成复盘字段

借鉴 TradingAgents' reporting.py 的报告树模式,
但输出内容按设计方案调整为"明日观察池"格式。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..data_service.schema import DecisionType
from .schemas import FinalReport, SystemDecision

logger = logging.getLogger(__name__)


def create_report_agent(llm=None):
    """创建 Report Agent 节点函数 (不需要 LLM, 纯格式化)"""

    def report_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        # 读取 System Agent 裁定
        system_decision: SystemDecision | None = state.get("system_decision_obj")
        if system_decision is None:
            system_decision = SystemDecision(
                decision=DecisionType.REJECT,
                alpha_source=[],
                horizon_days=5,
                reasons=["无系统裁定"],
                objections=[],
                reasoning="未完成分析",
            )

        # 生成最终报告
        report = FinalReport(
            decision=system_decision.decision,
            position=system_decision.position,
            alpha_source=system_decision.alpha_source,
            horizon_days=system_decision.horizon_days,
            reasons=system_decision.reasons,
            objections=system_decision.objections,
            invalid_conditions=system_decision.invalid_conditions,
            risk_result=system_decision.risk_verdict,
            code=ticker,
            name="",
            generated_at=datetime.now().isoformat(),
        )

        # 生成 Markdown 报告
        md = _format_report_markdown(report, trade_date)
        state["final_report"] = md
        state["final_report_obj"] = report

        # 保存报告到文件
        results_dir = Path("data/results") / ticker
        results_dir.mkdir(parents=True, exist_ok=True)
        report_path = results_dir / f"report_{trade_date}.md"
        report_path.write_text(md, encoding="utf-8")

        return {
            "final_report": md,
            "final_report_obj": report,
        }

    return report_node


def _format_report_markdown(report: FinalReport, trade_date: str) -> str:
    """格式化为 Markdown 报告"""
    decision_label = {
        DecisionType.RECOMMEND: "推荐",
        DecisionType.WATCH: "观察",
        DecisionType.REJECT: "拒绝",
    }.get(report.decision, report.decision.value if hasattr(report.decision, 'value') else str(report.decision))

    lines = [
        "# 明日观察池报告",
        "",
        f"**生成时间**: {report.generated_at}",
        f"**目标日期**: {trade_date}",
        f"**标的**: {report.code} {report.name or ''}",
        "",
        "---",
        "",
        f"## 结论: {decision_label}",
        "",
    ]

    if report.position is not None:
        lines.append(f"**建议仓位**: {report.position:.0%}")
        lines.append("")

    if report.alpha_source:
        lines.extend(["**Alpha 来源**:"] + [f"- {s}" for s in report.alpha_source])
        lines.append("")

    if report.reasons:
        lines.extend(["**支持理由**:"] + [f"- {r}" for r in report.reasons])
        lines.append("")

    if report.objections:
        lines.extend(["**反对意见**:"] + [f"- {o}" for o in report.objections])
        lines.append("")

    if report.invalid_conditions:
        lines.extend(["**失效条件**:"] + [f"- {c}" for c in report.invalid_conditions])
        lines.append("")

    if report.risk_result:
        lines.append(f"**风控结果**: {report.risk_result}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 复盘字段",
        f"- 评估周期: {report.horizon_days} 个交易日",
        f"- 主要 Alpha: {'/'.join(report.alpha_source) if report.alpha_source else '无'}",
        "- 需追踪: 1/3/5/10 日超额收益",
    ])

    return "\n".join(lines)
