"""
Report Agent — 报告输出层

纯格式化, 不需要 LLM。
输出 "明日观察池" 格式的 Markdown 报告。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import config
from .contract import basic_self_check, build_node_audit_update
from .schemas import DecisionType, FinalReport, SystemDecision

logger = logging.getLogger(__name__)


_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_path_part(value: str, fallback: str) -> str:
    safe = _SAFE_PATH_PART.sub("_", value).strip("._")
    return safe or fallback


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_audit_trace(
    state: dict[str, Any],
    *,
    report: FinalReport,
    report_path: Path,
) -> dict[str, Any]:
    decision = state.get("system_decision_obj")
    return {
        "generated_at": datetime.now().isoformat(),
        "ticker": state.get("company_of_interest", report.code),
        "trade_date": state.get("trade_date", report.trade_date),
        "run_mode": state.get("run_mode", ""),
        "report_path": str(report_path),
        "pit_manifest": _json_safe(state.get("pit_manifest")),
        "approval_record": _json_safe(state.get("approval_record", {})),
        "execution_allowed": bool(state.get("execution_allowed", False)),
        "risk_checks": {
            "risk_check_1": _json_safe(state.get("risk_check_1", {})),
            "risk_check_2": _json_safe(state.get("risk_check_2", {})),
            "risk_check_3": _json_safe(state.get("risk_check_3", {})),
        },
        "agent_evidence": _json_safe(state.get("agent_evidence", {})),
        "agent_tool_calls": _json_safe(state.get("agent_tool_calls", {})),
        "agent_self_checks": _json_safe(state.get("agent_self_checks", {})),
        "round2_state": _json_safe(state.get("round2_state", {})),
        "round2_summary": state.get("round2_summary", ""),
        "system_rubric": _json_safe(state.get("system_rubric", {})),
        "system_decision": _json_safe(decision),
        "final_report": _json_safe(report),
    }


def _round2_markdown(state: dict[str, Any]) -> str:
    round2 = state.get("round2_state", {}) or {}
    if not round2.get("contradictions") and not state.get("round2_summary"):
        return ""

    lines = [
        "",
        "---",
        "## Round 2 圆桌审计",
        f"- Provider: {round2.get('provider', 'none')}",
        f"- Final Pressure: {round2.get('final_pressure', 'neutral')}",
    ]
    if round2.get("fallback_reason"):
        lines.append(f"- Fallback Reason: {round2['fallback_reason']}")
    if round2.get("contradictions"):
        lines.append("- 矛盾点: " + "; ".join(str(c) for c in round2["contradictions"]))
    summary = state.get("round2_summary", "") or round2.get("summary", "")
    if summary:
        lines.extend(["", "```text", str(summary)[:3000], "```"])
    return "\n".join(lines)


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
        md = report.to_markdown() + _round2_markdown(state)

        # 保存文件
        results_dir = Path(config.get("results_dir", "data/results"))
        safe_ticker = _safe_path_part(ticker.replace(".", "_"), "unknown_ticker")
        safe_trade_date = _safe_path_part(trade_date, "unknown_date")
        save_dir = results_dir / safe_ticker
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / f"report_{safe_trade_date}.md"
        report_path.write_text(md, encoding="utf-8")
        audit_trace = _build_audit_trace(state, report=report, report_path=report_path)
        audit_path = save_dir / f"audit_{safe_trade_date}.json"
        audit_path.write_text(
            json.dumps(audit_trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 写入 Memory
        store = None
        try:
            from .memory_agent import MemoryStore
            store = MemoryStore()
            store.store_decision(ticker, trade_date, decision)
        except Exception as e:
            logger.warning("Memory store failed: %s", e)

        evidence = [
            f"report_path={report_path}",
            f"audit_path={audit_path}",
            f"decision={report.decision.value}",
            f"reasons={len(report.reasons)}",
            f"objections={len(report.objections)}",
        ]

        return build_node_audit_update(
            sender="Report Agent",
            final_report=md,
            final_report_obj=report,
            audit_trace=audit_trace,
            audit_trace_path=str(audit_path),
            evidence=evidence,
            tool_calls=[{
                "tool": "write_text",
                "args": {"path": str(report_path)},
                "records": 1,
            }, {
                "tool": "write_text",
                "args": {"path": str(audit_path)},
                "records": 1,
            }],
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=[
                    "final_report_rendered",
                    "audit_trace_persisted",
                    "memory_store_attempted",
                ],
                warnings=[],
                confidence=report.decision.value,
            ),
        )

    return report_node
