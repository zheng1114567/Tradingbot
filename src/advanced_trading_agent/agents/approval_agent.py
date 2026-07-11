"""Approval Agent — records the human approval gate."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .contract import basic_self_check, build_node_audit_update
from .schemas import DecisionType, SystemDecision


def create_approval_agent():
    """Create a non-blocking approval gate node.

    The first product version still generates research reports automatically,
    but execution is explicitly marked as not approved unless an upstream caller
    injects an approval action into state["approval_input"].
    """

    def approval_node(state: dict[str, Any]) -> dict[str, Any]:
        decision: SystemDecision | None = state.get("system_decision_obj")
        approval_input = state.get("approval_input") or {}
        requested_action = str(approval_input.get("action", "")).lower()

        if requested_action in {"approve", "approved", "通过"}:
            action = "approved"
            execution_allowed = decision is not None and decision.decision == DecisionType.RECOMMEND
        elif requested_action in {"reject", "rejected", "拒绝"}:
            action = "rejected"
            execution_allowed = False
        elif requested_action in {"reduce", "reduced", "降仓"}:
            action = "reduced_position"
            execution_allowed = decision is not None and decision.decision == DecisionType.RECOMMEND
        else:
            action = "pending_human_review"
            execution_allowed = False

        approval_record = {
            "action": action,
            "execution_allowed": execution_allowed,
            "reviewer": approval_input.get("reviewer", "human_required"),
            "comment": approval_input.get("comment", ""),
            "reviewed_at": approval_input.get("reviewed_at") or datetime.now().isoformat(),
            "decision": decision.decision.value if decision is not None else "无系统裁定",
        }

        evidence = [
            f"approval_action={approval_record['action']}",
            f"execution_allowed={approval_record['execution_allowed']}",
            f"decision={approval_record['decision']}",
        ]
        warnings = []
        if not execution_allowed:
            warnings.append("未获人工审批，不允许执行交易")

        return build_node_audit_update(
            sender="Approval Agent",
            approval_record=approval_record,
            execution_allowed=execution_allowed,
            evidence=evidence,
            tool_calls=[],
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=["human_approval_gate_recorded"],
                warnings=warnings,
                confidence=approval_record["action"],
            ),
        )

    return approval_node
