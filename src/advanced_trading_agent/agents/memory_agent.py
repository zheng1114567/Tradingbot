"""
Memory Agent - human-readable memory with a structured JSONL index.

Design:
- Markdown remains the primary review surface.
- JSONL is the machine-readable index for deterministic recall and updates.
- Entries are stored as pending, then resolved later when outcomes are known.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..config import config
from ..core.atomic_write import atomic_write_jsonl, atomic_write_text
from ..llm.client import LLMClient
from ..strategy_rules import current_rulebook
from .contract import basic_self_check, build_node_audit_update
from .schemas import MemoryRecall, SystemDecision

logger = logging.getLogger(__name__)

_SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"

OutcomeLoader = Callable[[dict[str, Any]], dict[str, Any] | None]


class MemoryStore:
    """Lightweight Memory store backed by Markdown plus JSONL."""

    _lock = threading.Lock()

    def __init__(self, log_path: str | None = None, index_path: str | None = None):
        path = log_path or config.get("memory_log_path", "")
        self._log_path = Path(path).expanduser() if path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

        if index_path:
            index = index_path
        elif log_path and self._log_path:
            index = str(self._log_path.with_suffix(".jsonl"))
        else:
            index = config.get("memory_index_path", "")
        self._index_path = Path(index).expanduser() if index else None
        if self._index_path:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)

    def store_decision(self, ticker: str, trade_date: str, decision: SystemDecision) -> None:
        """Phase A: store a pending decision for later reflection."""
        entry = {
            "date": trade_date,
            "ticker": ticker,
            "decision": decision.decision.value,
            "pending": True,
            "status": "pending",
            "decision_data": decision.model_dump(mode="json"),
            "rulebook": current_rulebook().to_dict(),
            "outcome": {},
            "reflection": {},
            "updated_at": datetime.now().isoformat(),
        }

        with self._lock:
            entries = self._load_index_entries()
            if any(self._index_key(e) == self._index_key(entry) for e in entries):
                return

            entries.append(entry)
            self._write_index_entries(entries)
            self._write_markdown_entries(entries)

    def load_entries(self) -> list[dict[str, Any]]:
        """Load memory records, preferring JSONL while preserving Markdown fallback."""
        entries = self._load_index_entries()
        if entries:
            return entries
        return self._load_markdown_entries()

    def resolve_pending(
        self,
        *,
        outcomes: dict[str, dict[str, Any]] | None = None,
        outcome_loader: OutcomeLoader | None = None,
        as_of: str | None = None,
        llm: LLMClient | None = None,
    ) -> list[dict[str, Any]]:
        """Phase B: resolve pending decisions and write reflections.

        outcomes accepts keys in either format:
        - "{date}|{ticker}", e.g. "2026-07-10|000001.SZ"
        - "{ticker}", useful for tests or batch injection.
        """
        with self._lock:
            entries = self.load_entries()
            if not entries:
                return []

            resolved: list[dict[str, Any]] = []
            for entry in entries:
                if not entry.get("pending"):
                    continue

                outcome = self._find_outcome(entry, outcomes or {}, outcome_loader)
                if outcome is None:
                    continue

                reflection = self._build_reflection(
                    entry=entry,
                    outcome=outcome,
                    as_of=as_of or datetime.now().date().isoformat(),
                    llm=llm,
                )
                entry["pending"] = False
                entry["status"] = "resolved"
                entry["outcome"] = outcome
                entry["reflection"] = reflection
                entry["updated_at"] = datetime.now().isoformat()
                resolved.append(entry)

            if resolved:
                self._write_index_entries(entries)
                self._write_markdown_entries(entries)

        return resolved

    @staticmethod
    def _index_key(entry: dict[str, Any]) -> tuple[str, str]:
        return str(entry.get("date", "")), str(entry.get("ticker", ""))

    def _load_index_entries(self) -> list[dict[str, Any]]:
        if not self._index_path or not self._index_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed memory index line")
                continue
            if isinstance(parsed, dict):
                entries.append(self._normalize_entry(parsed))
        return entries

    def _write_index_entries(self, entries: list[dict[str, Any]]) -> None:
        if not self._index_path:
            return
        atomic_write_jsonl(self._index_path, [self._normalize_entry(e) for e in entries])

    def _load_markdown_entries(self) -> list[dict[str, Any]]:
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        entries: list[dict[str, Any]] = []
        for block in text.split(_SEPARATOR):
            block = block.strip()
            if not block:
                continue
            entry = self._parse_block(block)
            if entry is not None:
                entries.append(self._normalize_entry(entry))
        return entries

    def _write_markdown_entries(self, entries: list[dict[str, Any]]) -> None:
        if not self._log_path:
            return
        blocks = [self._format_markdown_entry(self._normalize_entry(entry)) for entry in entries]
        atomic_write_text(self._log_path, _SEPARATOR.join(blocks) + (_SEPARATOR if blocks else ""))

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
        pending = bool(entry.get("pending", entry.get("status") != "resolved"))
        status = "pending" if pending else "resolved"
        return {
            "date": str(entry.get("date", "")),
            "ticker": str(entry.get("ticker", "")),
            "decision": str(entry.get("decision", "")),
            "pending": pending,
            "status": status,
            "decision_data": entry.get("decision_data") or {},
            "rulebook": entry.get("rulebook") or {},
            "outcome": entry.get("outcome") or {},
            "reflection": entry.get("reflection") or {},
            "updated_at": str(entry.get("updated_at", "")),
        }

    @staticmethod
    def _parse_block(block: str) -> dict[str, Any] | None:
        lines = block.split("\n")
        tag = lines[0].strip()
        if not tag.startswith("["):
            return None
        fields = [f.strip() for f in tag[1:-1].split("|")]
        body = "\n".join(lines[1:]).strip()
        decision_body = body
        outcome_body = ""
        reflection_body = ""

        if "DECISION:" in decision_body:
            decision_body = decision_body.split("DECISION:", 1)[1].strip()
        if "\nOUTCOME:" in decision_body:
            decision_body, outcome_body = decision_body.split("\nOUTCOME:", 1)
        if "\nREFLECTION:" in outcome_body:
            outcome_body, reflection_body = outcome_body.split("\nREFLECTION:", 1)

        try:
            decision_data = json.loads(decision_body.strip())
        except json.JSONDecodeError:
            decision_data = {}

        try:
            outcome = json.loads(outcome_body.strip()) if outcome_body.strip() else {}
        except json.JSONDecodeError:
            outcome = {}

        try:
            reflection = json.loads(reflection_body.strip()) if reflection_body.strip() else {}
        except json.JSONDecodeError:
            reflection = {}

        pending = "pending" in tag
        return {
            "date": fields[0] if len(fields) > 0 else "",
            "ticker": fields[1] if len(fields) > 1 else "",
            "decision": fields[2] if len(fields) > 2 else "",
            "pending": pending,
            "status": "pending" if pending else "resolved",
            "decision_data": decision_data,
            "rulebook": {},
            "outcome": outcome,
            "reflection": reflection,
        }

    @staticmethod
    def _find_outcome(
        entry: dict[str, Any],
        outcomes: dict[str, dict[str, Any]],
        outcome_loader: OutcomeLoader | None,
    ) -> dict[str, Any] | None:
        if outcome_loader is not None:
            return outcome_loader(entry)
        date_key = f"{entry.get('date', '')}|{entry.get('ticker', '')}"
        ticker_key = str(entry.get("ticker", ""))
        return outcomes.get(date_key) or outcomes.get(ticker_key)

    @staticmethod
    def _hit_decision(decision: str, outcome: dict[str, Any]) -> bool:
        excess = float(outcome.get("excess_return", 0) or 0)
        absolute = float(outcome.get("absolute_return", 0) or 0)
        if decision == "推荐":
            return excess > 0
        if decision == "拒绝":
            return excess <= 0 or absolute <= 0
        return abs(excess) <= 0.03

    def _build_reflection(
        self,
        *,
        entry: dict[str, Any],
        outcome: dict[str, Any],
        as_of: str,
        llm: LLMClient | None,
    ) -> dict[str, Any]:
        decision = str(entry.get("decision", ""))
        hit = self._hit_decision(decision, outcome)
        base = {
            "as_of": as_of,
            "hit": hit,
            "actual_horizon_days": outcome.get("horizon_days"),
            "absolute_return": outcome.get("absolute_return"),
            "excess_return": outcome.get("excess_return"),
            "lesson": self._deterministic_lesson(decision, hit, outcome),
        }

        if llm is None:
            return base

        prompt = (
            "请基于交易决策与事后收益做简洁复盘，只输出 JSON 字符串字段 lesson。"
            f"\n决策: {entry.get('decision_data', {})}"
            f"\n结果: {outcome}"
            f"\n是否命中: {hit}"
        )
        try:
            content = llm.chat([
                ("system", "你是交易系统复盘员，只做事后归因，不改写历史决策。"),
                ("human", prompt),
            ])
            parsed = json.loads(str(content))
            if isinstance(parsed, dict) and parsed.get("lesson"):
                base["lesson"] = str(parsed["lesson"])
        except Exception as e:
            logger.warning("LLM reflection failed, using deterministic lesson: %s", e)
        return base

    @staticmethod
    def _deterministic_lesson(decision: str, hit: bool, outcome: dict[str, Any]) -> str:
        excess = float(outcome.get("excess_return", 0) or 0)
        if hit:
            return f"{decision} 决策命中，事后超额收益 {excess:+.2%}。"
        return f"{decision} 决策未命中，事后超额收益 {excess:+.2%}，后续同类信号需降权。"

    @staticmethod
    def _format_markdown_entry(entry: dict[str, Any]) -> str:
        status = "pending" if entry.get("pending") else "resolved"
        block = (
            f"[{entry.get('date', '')} | {entry.get('ticker', '')} | "
            f"{entry.get('decision', '')} | {status}]\n\n"
            f"DECISION:\n{json.dumps(entry.get('decision_data', {}), ensure_ascii=False, indent=2)}"
        )
        if entry.get("rulebook"):
            block += f"\n\nRULEBOOK:\n{json.dumps(entry.get('rulebook', {}), ensure_ascii=False, indent=2)}"
        if not entry.get("pending"):
            block += (
                f"\n\nOUTCOME:\n{json.dumps(entry.get('outcome', {}), ensure_ascii=False, indent=2)}"
                f"\n\nREFLECTION:\n{json.dumps(entry.get('reflection', {}), ensure_ascii=False, indent=2)}"
            )
        return block

    @staticmethod
    def _format_resolved_entry(entry: dict[str, Any]) -> str:
        return MemoryStore._format_markdown_entry({**entry, "pending": False, "status": "resolved"})

    def get_context(self, ticker: str, n_same: int = 5) -> str:
        """Return resolved historical context for System Agent."""
        entries = [e for e in self.load_entries() if not e.get("pending")]
        same = [e for e in reversed(entries) if e["ticker"] == ticker][:n_same]
        if not same:
            return ""
        parts = [f"Past analyses of {ticker}:"]
        for e in same:
            d = e.get("decision_data", {})
            reasons = d.get("reasons", [])
            if isinstance(reasons, list):
                reasons = "; ".join(reasons)
            reflection = e.get("reflection", {})
            lesson = reflection.get("lesson", "")
            suffix = f" | lesson: {lesson}" if lesson else ""
            parts.append(f"[{e['date']}] {e['decision']}: {reasons}{suffix}")
        return "\n".join(parts)


def create_memory_agent(llm: LLMClient):
    """Create Memory Agent node for System Agent context injection."""

    def memory_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")

        store = MemoryStore()
        context = store.get_context(ticker)
        entries = store.load_entries()

        same_ticker = [e for e in entries if e["ticker"] == ticker and not e.get("pending")]
        success_cases = [e for e in same_ticker if e["decision"] == "推荐"]
        failure_cases = [e for e in same_ticker if e["decision"] in ("拒绝", "观察")]

        recall = MemoryRecall(
            success_cases=[
                {"date": e["date"], "reasons": e.get("decision_data", {}).get("reasons", [])}
                for e in success_cases[:3]
            ],
            failure_cases=[
                {"date": e["date"], "reasons": e.get("decision_data", {}).get("objections", [])}
                for e in failure_cases[:3]
            ],
            agent_accuracy={},
            historical_warnings=[],
            reasoning=f"召回 {len(success_cases)} 成功, {len(failure_cases)} 失败",
        )

        evidence = [
            f"entries={len(entries)}",
            f"same_ticker_resolved={len(same_ticker)}",
            f"success_cases={len(success_cases)}",
            f"failure_cases={len(failure_cases)}",
        ]

        return build_node_audit_update(
            sender="Memory Agent",
            memory_context=context,
            memory_recall=recall.model_dump(),
            evidence=evidence,
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=["resolved_entries_only_recalled", "jsonl_index_preferred"],
                warnings=[] if context else ["无可用历史记忆"],
                confidence=len(same_ticker),
            ),
        )

    return memory_node
