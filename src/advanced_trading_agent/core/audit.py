"""Audit trail — structured event log woven through the analysis pipeline.

Each event has:
  - timestamp: ISO 8601 UTC
  - stage: which pipeline stage (data, market_agent, risk_check_1, roundtable, etc.)
  - level: info | warning | veto | error
  - message: human-readable one-liner
  - detail: optional structured payload (vendor used, record counts, veto reasons, etc.)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def audit_event(
    stage: str,
    message: str,
    *,
    level: str = "info",
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "level": level,
        "message": message,
        "detail": detail or {},
    }


def format_audit_trail(events: list[dict[str, Any]]) -> str:
    """Render the audit trail as a readable Markdown block."""
    if not events:
        return "_No audit events recorded._"

    level_icons = {"info": "i", "warning": "!", "veto": "X", "error": "!!"}
    lines = ["## Audit Trail", ""]
    for ev in events:
        icon = level_icons.get(ev.get("level", "info"), "?")
        stage = ev.get("stage", "?")
        msg = ev.get("message", "")
        detail = ev.get("detail", {})
        detail_str = ""
        if detail:
            items = [f"{k}={v}" for k, v in detail.items()]
            detail_str = f"  `({', '.join(items)})`"
        lines.append(f"- `[{icon}]` **{stage}**: {msg}{detail_str}")
    return "\n".join(lines)


def build_data_collection_summary(
    raw_payload: dict[str, Any],
    vendor_health: dict[str, Any],
    route_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize what data was collected and from which vendors."""
    categories: dict[str, dict[str, Any]] = {}

    category_map = {
        "daily": "日线行情",
        "capital_flow": "资金流向",
        "news": "新闻资讯",
        "sector": "板块数据",
        "factors": "因子数据",
        "risk_st": "ST风险",
        "risk_suspended": "停牌风险",
        "risk_delisting": "退市风险",
    }

    for key, label in category_map.items():
        if key == "sector":
            data = raw_payload.get("sector_context")
        elif key == "factors":
            data = raw_payload.get("factors")
            if data is None:
                categories[key] = {"label": label, "status": "computed", "count": 0, "vendor": "FactorCalculator"}
                continue
        elif key.startswith("risk_"):
            risk = raw_payload.get("risk", {})
            sub_key = {"risk_st": "st_status", "risk_suspended": "suspended", "risk_delisting": "delisting"}.get(key, key)
            data = risk.get(sub_key) if isinstance(risk, dict) else None
        else:
            data = raw_payload.get(key)
        if data is None:
            categories[key] = {"label": label, "status": "not_requested", "count": 0, "vendor": None}
            continue

        if isinstance(data, dict) and "error" in data:
            categories[key] = {"label": label, "status": "error", "count": 0, "vendor": data.get("vendor"), "error": data.get("error")}
            continue

        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data)
        else:
            count = 1 if data else 0

        vendor = _find_vendor_for_method(route_trace, key)
        categories[key] = {"label": label, "status": "ok" if count > 0 else "empty", "count": count, "vendor": vendor}

    with_data = sum(1 for c in categories.values() if c["status"] == "ok")
    computed = sum(1 for c in categories.values() if c["status"] == "computed")

    return {
        "categories": categories,
        "vendor_health": vendor_health,
        "total_categories": len(categories),
        "categories_with_data": with_data,
        "categories_computed": computed,
        "categories_failed": sum(1 for c in categories.values() if c["status"] == "error"),
        "categories_empty": sum(1 for c in categories.values() if c["status"] == "empty"),
    }


def format_data_collection_summary(summary: dict[str, Any]) -> str:
    """Render the data collection summary as readable Markdown."""
    lines = [
        "## Data Collection Summary",
        "",
        f"- **Categories collected**: {summary.get('categories_with_data', 0)}/{summary.get('total_categories', 0)}",
        f"- **Failed**: {summary.get('categories_failed', 0)}",
        f"- **Empty**: {summary.get('categories_empty', 0)}",
        "",
        "| Category | Status | Records | Vendor |",
        "|----------|--------|---------|--------|",
    ]
    for key, cat in summary.get("categories", {}).items():
        status_icon = {"ok": "OK", "empty": "EMPTY", "error": "ERROR", "not_requested": "SKIPPED", "computed": "COMPUTED"}.get(cat["status"], "?")
        vendor = cat.get("vendor") or "-"
        lines.append(f"| {cat['label']} | {status_icon} | {cat['count']} | {vendor} |")

    return "\n".join(lines)


def _find_vendor_for_method(route_trace: list[dict[str, Any]], method: str) -> str | None:
    for entry in route_trace:
        if entry.get("method") == method and entry.get("status") in {"ok", "success"}:
            return entry.get("vendor")
    return None


def format_roundtable_visualization(round2_state: dict[str, Any]) -> str:
    """Render roundtable debate records as a structured, readable Markdown block."""
    if not round2_state:
        return "_No roundtable debate was held._"

    lines = ["## Roundtable Debate Record", ""]

    provider = round2_state.get("provider", "unknown")
    fallback = round2_state.get("fallback_reason", "")
    completed = round2_state.get("completed", False)
    round_count = round2_state.get("round_count", 0)
    round_history = round2_state.get("round_history", [])

    lines.append(f"- **Provider**: {provider}")
    lines.append(f"- **Rounds**: {round_count}")
    lines.append(f"- **Completed**: {completed}")
    if fallback:
        lines.append(f"- **Fallback reason**: {fallback}")
    lines.append("")

    contradiction_records = round2_state.get("contradiction_records", [])
    if contradiction_records:
        lines.append("### Contradictions Identified")
        lines.append("")
        for i, c in enumerate(contradiction_records, 1):
            desc = c.get("description", str(c)) if isinstance(c, dict) else str(c)
            lines.append(f"{i}. {desc}")
        lines.append("")

    if round_history:
        lines.append("### Debate Rounds")
        lines.append("")
        for round_data in round_history:
            rn = round_data.get("round_number", "?")
            lines.append(f"#### Round {rn + 1}")
            turns = round_data.get("turns", [])
            messages = round_data.get("messages", [])
            for turn in turns:
                agent = turn.get("agent_name", "?")
                stance = turn.get("stance", {})
                pressure = stance.get("pressure", "neutral")
                confidence = stance.get("confidence", 0)
                reasoning = (stance.get("reasoning", "") or "")[:200]
                lines.append(f"- **{agent}**: pressure={pressure}, conf={confidence:.1f}")
                if reasoning:
                    lines.append(f"  - {reasoning}")
            for message in messages:
                source = message.get("source", "?")
                content = str(message.get("content", ""))[:240].replace("\n", " ")
                if source and content:
                    lines.append(f"- **{source}**: {content}")
            lines.append("")
            if turns:
                lines.append("| Agent | Position | Evidence Cited | Impact |")
                lines.append("|-------|----------|----------------|--------|")
                for turn in turns:
                    agent_name = turn.get("agent_name", "?")
                    stance = turn.get("stance", {})
                    reasoning = (stance.get("reasoning", "") or "")[:200].replace("\n", " ")
                    evidence = ", ".join(stance.get("evidence_ids", []) or [])
                    impact = str(stance.get("pressure", "neutral")).upper()
                    lines.append(f"| **{agent_name}** | {reasoning} | {evidence or '-'} | {impact} |")
                lines.append("")

    questions = round2_state.get("questions", [])
    if questions:
        lines.append("### Agent Answers")
        lines.append("")
        lines.append("| Agent | Position | Evidence Cited | Impact |")
        lines.append("|-------|----------|----------------|--------|")
        for item in questions:
            for answer in item.get("answers", []):
                agent_name = answer.get("target_agent", "?")
                answer_text = answer.get("answer", "")[:200].replace("\n", " ")
                evidence = (answer.get("evidence", "") or "")[:100].replace("\n", " ")
                impact = _extract_impact(answer_text)
                lines.append(f"| **{agent_name}** | {answer_text} | {evidence or '-'} | {impact} |")
        lines.append("")

    summary = round2_state.get("summary", "")
    if summary:
        lines.append("### Final Resolution")
        lines.append("")
        lines.append(summary)
        lines.append("")

    final_pressure = round2_state.get("final_pressure", "")
    unresolved = round2_state.get("unresolved_conflicts", [])
    if final_pressure:
        lines.append(f"- **Final pressure on decision**: **{final_pressure}**")
    if unresolved:
        lines.append(f"- **Unresolved conflicts**: {len(unresolved) if isinstance(unresolved, list) else 1}")
    lines.append("")

    return "\n".join(lines)


def _extract_impact(text: str) -> str:
    """Extract impact direction from agent answer text."""
    lowered = text.lower()
    if any(w in lowered for w in ["upgrade", "升级", "上调", "看多"]):
        return "UPGRADE"
    if any(w in lowered for w in ["downgrade", "降级", "下调", "看空"]):
        return "DOWNGRADE"
    return "NEUTRAL"
