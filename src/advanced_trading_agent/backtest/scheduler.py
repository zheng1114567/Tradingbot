"""Daily review scheduler helpers.

This module intentionally does not run a daemon. It provides a deterministic
function that can be called by cron, Windows Task Scheduler, or the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..agents.memory_agent import MemoryStore
from ..config import config
from ..core.atomic_write import atomic_write_text
from ..strategy_rules import current_rulebook, enqueue_strategy_proposals
from .review import ReviewEngine


@dataclass
class DailyReviewResult:
    """Result of one daily review run."""

    resolved_count: int
    pending_count: int
    summary_path: str
    report: str
    rulebook: dict[str, Any]
    proposal_count: int
    proposals: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_count": self.resolved_count,
            "pending_count": self.pending_count,
            "summary_path": self.summary_path,
            "report": self.report,
            "rulebook": self.rulebook,
            "proposal_count": self.proposal_count,
            "proposals": self.proposals,
        }


def run_daily_review(
    *,
    price_file: str | None = None,
    as_of: str | None = None,
    store: MemoryStore | None = None,
    reviewer: ReviewEngine | None = None,
    audit_queue_path: str | None = None,
) -> DailyReviewResult:
    """Resolve due memory entries and write the daily review summary."""
    store = store or MemoryStore()
    reviewer = reviewer or ReviewEngine()
    rulebook = current_rulebook().to_dict()

    if price_file:
        price_df = pd.read_csv(price_file)
        if "trade_date" in price_df.columns:
            price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])

        def price_loader(ticker: str, signal_date: str):
            if "code" not in price_df.columns:
                return price_df
            return price_df[price_df["code"] == ticker].copy()

        resolved = reviewer.resolve_due(store, price_loader=price_loader, as_of=as_of)
    else:
        resolved = []

    entries = store.load_entries()
    pending_count = sum(1 for entry in entries if entry.get("pending"))
    summary = reviewer.summarize_entries(entries)
    proposals = enqueue_strategy_proposals(summary, path=audit_queue_path)
    report = reviewer.format_summary(summary)
    summary_path = Path(config.get("results_dir")) / "daily_review_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(summary_path, report)
    return DailyReviewResult(
        resolved_count=len(resolved),
        pending_count=pending_count,
        summary_path=str(summary_path),
        report=report,
        rulebook=rulebook,
        proposal_count=len(proposals),
        proposals=proposals,
    )
