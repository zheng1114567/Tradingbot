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
from ..core.atomic_write import atomic_write_json, atomic_write_text
from ..core.audit import format_audit_trail, format_data_collection_summary, format_roundtable_visualization
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
    if not round2.get("contradiction_records") and not state.get("round2_summary"):
        return ""
    return format_roundtable_visualization(round2)


def _risk_veto_markdown(state: dict[str, Any]) -> str:
    """Render risk check vetoes with clear reasons."""
    lines: list[str] = []
    for check_name in ("risk_check_1", "risk_check_2", "risk_check_3"):
        check = state.get(check_name, {}) or {}
        verdict = check.get("verdict", "")
        if isinstance(verdict, str):
            verdict_val = verdict
        elif hasattr(verdict, "value"):
            verdict_val = verdict.value
        else:
            verdict_val = str(verdict)

        if "VETO" in verdict_val.upper():
            reasons = check.get("reasons", [])
            if isinstance(reasons, str):
                reasons = [reasons]
            lines.append(f"### {check_name}: VETO")
            lines.append("")
            for r in reasons:
                lines.append(f"- **否决原因**: {r}")
            lines.append("")
    return "\n".join(lines)


def _data_collection_markdown(state: dict[str, Any]) -> str:
    """Render data collection summary from tier1 metadata."""
    tier1 = state.get("tier1_data", {}) or {}
    data_manifest = tier1.get("_data_manifest") or {}
    collection_summary = tier1.get("_collection_summary") or {}
    if not collection_summary:
        return ""
    return format_data_collection_summary(collection_summary)


def _audit_trail_markdown(state: dict[str, Any]) -> str:
    """Render audit trail from state."""
    audit_events = state.get("audit_trail", [])
    if not audit_events:
        tier1 = state.get("tier1_data", {}) or {}
        audit_events = tier1.get("_audit_trail", [])
    if not audit_events:
        return ""
    return format_audit_trail(audit_events)


def _failure_markdown(state: dict[str, Any]) -> str:
    """Render any failures recorded in the state."""
    errors = state.get("errors", [])
    if not errors:
        tier1 = state.get("tier1_data", {}) or {}
        errors = tier1.get("_errors", [])
    if not errors:
        return ""
    lines = ["## Failures & Warnings", ""]
    for err in errors:
        stage = err.get("stage", "unknown")
        error_msg = err.get("error", str(err))
        lines.append(f"- **{stage}**: {error_msg}")
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

        # 生成 Markdown (含完整审计链)
        sections = [
            report.to_markdown(),
            _data_collection_markdown(state),
            _risk_veto_markdown(state),
            _round2_markdown(state),
            _failure_markdown(state),
            _audit_trail_markdown(state),
        ]
        md = "\n\n---\n\n".join(s for s in sections if s)

        # 保存文件 (atomic writes)
        results_dir = Path(config.get("results_dir"))
        safe_ticker = _safe_path_part(ticker.replace(".", "_"), "unknown_ticker")
        safe_trade_date = _safe_path_part(trade_date, "unknown_date")
        save_dir = results_dir / safe_ticker
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / f"report_{safe_trade_date}.md"
        atomic_write_text(report_path, md)
        audit_trace = _build_audit_trace(state, report=report, report_path=report_path)
        audit_path = save_dir / f"audit_{safe_trade_date}.json"
        atomic_write_json(audit_path, audit_trace)

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
