"""Conversation-only memory for analyst Q&A.

This memory is intentionally separate from post-trade review memory.  It
stores user questions, retrieved evidence, and final answers so follow-up
questions can reference prior dialogue without pretending to validate trading
performance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import config
from ..core.atomic_write import atomic_write_jsonl


@dataclass
class ConversationEntry:
    question: str
    answer: str
    trade_date: str
    target_type: str
    target: str
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationMemoryStore:
    """Small JSONL-backed memory for Q&A context only."""

    def __init__(self, path: str | None = None) -> None:
        configured = path or config.get("conversation_memory_path", "")
        if not configured:
            configured = str(Path(config.get("results_dir")) / "conversation_memory.jsonl")
        self.path = Path(configured).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: ConversationEntry) -> None:
        entries = self.load()
        entries.append(entry.to_dict())
        atomic_write_jsonl(self.path, entries)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows

    def recall(self, target: str = "", limit: int = 5) -> str:
        entries = self.load()
        if target:
            entries = [
                item for item in entries
                if target in str(item.get("target", "")) or target in str(item.get("question", ""))
            ]
        selected = entries[-limit:]
        if not selected:
            return ""
        lines = ["历史对话记忆（仅用于上下文，不代表回测验证）:"]
        for item in selected:
            lines.append(
                f"- {item.get('created_at', '')[:19]} "
                f"{item.get('target_type', '')}:{item.get('target', '')} "
                f"Q={item.get('question', '')[:80]} A={item.get('answer', '')[:120]}"
            )
        return "\n".join(lines)
