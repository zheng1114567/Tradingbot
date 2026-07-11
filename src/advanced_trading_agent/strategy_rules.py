"""Strategy rulebook versioning helpers.

The rulebook is intentionally small: it captures the deterministic thresholds
that constrain agent output so post-trade review can compare signals generated
under the same rule set.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import config
from .core.atomic_write import atomic_write_jsonl


@dataclass(frozen=True)
class StrategyRulebook:
    """Immutable snapshot of strategy rules used for one decision."""

    version: str
    generated_at: str
    rules: dict[str, Any]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "rules": self.rules,
            "fingerprint": self.fingerprint,
        }


@dataclass
class StrategyChangeProposal:
    """Human-audited proposal for a future rulebook change."""

    proposal_id: str
    alpha_source: str
    rule_version: str
    recommendation: str
    proposed_action: str
    reasons: list[str]
    metrics: dict[str, Any]
    status: str = "pending_human_review"
    reviewer: str = "human_required"
    reviewer_comment: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reviewed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "alpha_source": self.alpha_source,
            "rule_version": self.rule_version,
            "recommendation": self.recommendation,
            "proposed_action": self.proposed_action,
            "reasons": self.reasons,
            "metrics": self.metrics,
            "status": self.status,
            "reviewer": self.reviewer,
            "reviewer_comment": self.reviewer_comment,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
        }


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def strategy_audit_queue_path(path: str | None = None) -> Path:
    """Return the JSONL path for pending strategy change proposals."""
    configured = path or config.get("strategy_audit_queue_path", "")
    if configured:
        target = Path(configured).expanduser()
    else:
        target = Path(config.get("results_dir", "data/results")) / "strategy_audit_queue.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def current_rulebook() -> StrategyRulebook:
    """Return the current deterministic rule snapshot."""
    strategy_cfg = config.get("strategy_rules", {})
    version = str(strategy_cfg.get("version", "rules-v1"))
    rules = {
        "risk_config": config.get("risk_config", {}),
        "backtest_config": config.get("backtest_config", {}),
        "review_config": config.get("review_config", {}),
        "rubric_thresholds": strategy_cfg.get("rubric_thresholds", {}),
        "memory_policy": strategy_cfg.get("memory_policy", {}),
    }
    payload = {"version": version, "rules": rules}
    return StrategyRulebook(
        version=version,
        generated_at=datetime.now().isoformat(),
        rules=rules,
        fingerprint=_stable_hash(payload),
    )


def proposal_from_performance(performance: Any) -> StrategyChangeProposal | None:
    """Convert weak review performance into a human-review proposal."""
    recommendation = str(getattr(performance, "recommendation", ""))
    if recommendation not in {"DOWNWEIGHT", "PAUSE"}:
        return None

    proposed_action = "pause_alpha_source" if recommendation == "PAUSE" else "downweight_alpha_source"
    metrics = {
        "sample_size": getattr(performance, "sample_size", 0),
        "tradable_count": getattr(performance, "tradable_count", 0),
        "hit_rate": getattr(performance, "hit_rate", 0.0),
        "avg_return": getattr(performance, "avg_return", 0.0),
        "avg_excess_return": getattr(performance, "avg_excess_return", 0.0),
        "max_drawdown": getattr(performance, "max_drawdown", 0.0),
    }
    payload = {
        "alpha_source": getattr(performance, "alpha_source", "UNKNOWN"),
        "rule_version": getattr(performance, "rule_version", "UNKNOWN"),
        "recommendation": recommendation,
        "metrics": metrics,
    }
    return StrategyChangeProposal(
        proposal_id=f"proposal-{_stable_hash(payload)}",
        alpha_source=str(payload["alpha_source"]),
        rule_version=str(payload["rule_version"]),
        recommendation=recommendation,
        proposed_action=proposed_action,
        reasons=list(getattr(performance, "reasons", []) or []),
        metrics=metrics,
    )


def load_strategy_proposals(path: str | None = None) -> list[dict[str, Any]]:
    """Load the strategy audit queue."""
    queue_path = strategy_audit_queue_path(path)
    if not queue_path.exists():
        return []
    proposals: list[dict[str, Any]] = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            proposals.append(parsed)
    return proposals


def write_strategy_proposals(proposals: list[dict[str, Any]], path: str | None = None) -> str:
    """Persist the strategy audit queue as JSONL (atomic write)."""
    queue_path = strategy_audit_queue_path(path)
    atomic_write_jsonl(queue_path, proposals)
    return str(queue_path)


def enqueue_strategy_proposals(
    performances: list[Any],
    *,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """Append new pending proposals without duplicating existing IDs."""
    existing = load_strategy_proposals(path)
    seen = {str(item.get("proposal_id", "")) for item in existing}
    created: list[dict[str, Any]] = []
    for performance in performances:
        proposal = proposal_from_performance(performance)
        if proposal is None or proposal.proposal_id in seen:
            continue
        item = proposal.to_dict()
        existing.append(item)
        created.append(item)
        seen.add(proposal.proposal_id)
    if created:
        write_strategy_proposals(existing, path)
    return created


def review_strategy_proposal(
    proposal_id: str,
    *,
    action: str,
    reviewer: str,
    comment: str = "",
    path: str | None = None,
) -> dict[str, Any]:
    """Approve or reject a queued strategy proposal.

    Approval records intent only. It does not mutate live strategy config.
    """
    normalized = action.strip().lower()
    if normalized in {"approve", "approved", "通过"}:
        status = "approved"
    elif normalized in {"reject", "rejected", "拒绝"}:
        status = "rejected"
    else:
        raise ValueError(f"Unsupported audit action: {action}")

    proposals = load_strategy_proposals(path)
    for item in proposals:
        if item.get("proposal_id") != proposal_id:
            continue
        item["status"] = status
        item["reviewer"] = reviewer
        item["reviewer_comment"] = comment
        item["reviewed_at"] = datetime.now().isoformat()
        write_strategy_proposals(proposals, path)
        return item
    raise KeyError(f"Unknown strategy proposal: {proposal_id}")
