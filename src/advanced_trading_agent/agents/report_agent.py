"""
Report Agent — 报告输出层

纯格式化, 不需要 LLM。
输出 "明日观察池" 格式的 Markdown 报告。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import config
from .schemas import DecisionType, FinalReport, SystemDecision

logger = logging.getLogger(__name__)


_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_path_part(value: str, fallback: str) -> str:
    safe = _SAFE_PATH_PART.sub("_", value).strip("._")
    return safe or fallback


def create_report_agent(llm=None):
    """创建 Report Agent 节点函数 (不需要 LLM)"""

    def report_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        decision: SystemDecision | None = state.get("system_decision_obj")
        if decision is None:
            decision = SystemDecision(
                decision=DecisionType.REJECT,
                alpha_source=[],
                horizon_days=5,
                reasons=["无系统裁定"],
                objections=[],
                reasoning="未完成分析",
            )

        report = FinalReport(
            code=ticker,
            name="",
            trade_date=trade_date,
            decision=decision.decision,
            position=decision.position,
            alpha_source=decision.alpha_source,
            horizon_days=decision.horizon_days,
            reasons=decision.reasons,
            objections=decision.objections,
            invalid_conditions=decision.invalid_conditions,
            risk_verdict=decision.risk_verdict.value if hasattr(decision.risk_verdict, 'value') else str(decision.risk_verdict),
        )

        # 生成 Markdown
        md = report.to_markdown()

        # 保存文件
        results_dir = Path(config.get("results_dir", "data/results"))
        safe_ticker = _safe_path_part(ticker.replace(".", "_"), "unknown_ticker")
        safe_trade_date = _safe_path_part(trade_date, "unknown_date")
        save_dir = results_dir / safe_ticker
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / f"report_{safe_trade_date}.md"
        report_path.write_text(md, encoding="utf-8")

        # 写入 Memory
        store = None
        try:
            from .memory_agent import MemoryStore
            store = MemoryStore()
            store.store_decision(ticker, trade_date, decision)
        except Exception as e:
            logger.warning("Memory store failed: %s", e)

        return {
            "final_report": md,
            "final_report_obj": report,
        }

    return report_node
