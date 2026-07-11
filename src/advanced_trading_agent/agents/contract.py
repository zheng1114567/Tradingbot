"""Shared helpers for agent node outputs."""
from __future__ import annotations

from typing import Any


def build_agent_update(
    state: dict[str, Any],
    *,
    sender: str,
    report_key: str,
    report: str,
    report_obj_key: str,
    report_obj: Any,
    evidence: list[str] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    self_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an incremental LangGraph state update for an agent node."""
    return {
        "sender": sender,
        report_key: report,
        report_obj_key: report_obj,
        "agent_evidence": {sender: evidence or []},
        "agent_tool_calls": {sender: tool_calls or []},
        "agent_self_checks": {sender: self_check or {}},
    }


def build_node_audit_update(
    *,
    sender: str,
    evidence: list[str] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    self_check: dict[str, Any] | None = None,
    **updates: Any,
) -> dict[str, Any]:
    """Attach a standard audit envelope to any LangGraph node update."""
    return {
        "sender": sender,
        **updates,
        "agent_evidence": {sender: evidence or []},
        "agent_tool_calls": {sender: tool_calls or []},
        "agent_self_checks": {sender: self_check or {}},
    }


def basic_self_check(
    *,
    evidence: list[str],
    passed_rules: list[str] | None = None,
    warnings: list[str] | None = None,
    confidence: str | float | None = None,
) -> dict[str, Any]:
    """Create a compact, auditable self-check instead of exposing raw CoT."""
    warnings = warnings or []
    return {
        "evidence_count": len([item for item in evidence if item]),
        "passed_rules": passed_rules or [],
        "warnings": warnings,
        "confidence": confidence,
        "needs_review": bool(warnings),
    }
