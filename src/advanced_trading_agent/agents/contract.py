"""Shared helpers for agent node outputs."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def build_react_agent(
    *,
    llm: Any,
    tools: Any,
    system_prompt: str,
    response_format: type,
) -> Any | None:
    """Build a LangGraph prebuilt ReAct agent."""
    if not hasattr(llm, "as_langchain_chat_model"):
        return None
    try:
        return create_agent(
            model=llm.as_langchain_chat_model(),
            tools=tools,
            system_prompt=system_prompt,
            response_format=response_format,
        )
    except Exception as exc:
        logger.warning("Unable to create react agent, falling back: %s", exc)
        return None


def run_react_agent(
    agent: Any | None,
    user_content: str,
) -> tuple[Any | None, list[dict[str, Any]]]:
    """Run a prebuilt agent and extract tool traces, or fall back cleanly."""
    if agent is None:
        return None, []
    try:
        result = agent.invoke({"messages": [HumanMessage(content=user_content)]})
    except Exception as exc:
        logger.warning("Prebuilt agent failed, falling back: %s", exc)
        return None, []
    return result.get("structured_response"), extract_react_tool_calls(
        result.get("messages", [])
    )


def extract_react_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract tool call trace from a LangGraph create_react_agent result messages."""
    trace: list[dict[str, Any]] = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            trace.append(
                {
                    "tool": call.get("name", ""),
                    "args": call.get("args", {}),
                    "id": call.get("id", ""),
                }
            )
        if getattr(msg, "type", None) == "tool":
            trace.append(
                {
                    "tool": getattr(msg, "name", ""),
                    "observation": str(getattr(msg, "content", ""))[:1000],
                    "id": getattr(msg, "tool_call_id", ""),
                }
            )
    return trace


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
