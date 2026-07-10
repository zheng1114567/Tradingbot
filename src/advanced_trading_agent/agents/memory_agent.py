"""
Memory Agent — 经验库与复盘层

设计要点:
- 不参与交易裁定, 只提供历史上下文
- 直接注入 System Agent (不是所有 Agent)
- 每次运行保存完整记录 (pending)
- 下次运行时拉取真实收益 → LLM 反思

存储格式 (当前): Markdown 文件
后续升级: PostgreSQL + ChromaDB (按设计方案)

借鉴 TradingAgents' memory.py 的 deferred reflection 模式:
- Phase A: store_decision() — 结束立即存 pending
- Phase B: resolve_pending() — 下次运行时拉真实收益 → 反思
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import config
from ..llm.client import LLMClient
from .schemas import MemoryRecall, SystemDecision

logger = logging.getLogger(__name__)

_SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"


class MemoryStore:
    """轻量 Memory 存储 — Markdown 文件"""

    def __init__(self, log_path: str | None = None):
        path = log_path or config.get("memory_log_path", "")
        self._log_path = Path(path).expanduser() if path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def store_decision(self, ticker: str, trade_date: str,
                        decision: SystemDecision) -> None:
        """Phase A: 存储决策 (pending)"""
        if not self._log_path:
            return
        if self._log_path.exists():
            raw = self._log_path.read_text(encoding="utf-8")
            if f"[{trade_date} | {ticker} |" in raw:
                return

        entry = (
            f"[{trade_date} | {ticker} | {decision.decision.value} | pending]\n\n"
            f"DECISION:\n{decision.model_dump_json(indent=2)}\n"
            f"{_SEPARATOR}"
        )
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def load_entries(self) -> list[dict]:
        """加载所有记录"""
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        entries = []
        for block in text.split(_SEPARATOR):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            tag = lines[0].strip()
            if not tag.startswith("["):
                continue
            fields = [f.strip() for f in tag[1:-1].split("|")]
            entry = {
                "date": fields[0] if len(fields) > 0 else "",
                "ticker": fields[1] if len(fields) > 1 else "",
                "decision": fields[2] if len(fields) > 2 else "",
                "pending": "pending" in tag,
            }
            body = "\n".join(lines[1:]).strip()
            if body.startswith("DECISION:"):
                body = body[8:].strip()
            try:
                entry["decision_data"] = json.loads(body)
            except json.JSONDecodeError:
                entry["decision_data"] = {}
            entries.append(entry)
        return entries

    def get_context(self, ticker: str, n_same: int = 5) -> str:
        """获取历史上下文 (注入 System Agent)"""
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
            parts.append(f"[{e['date']}] {e['decision']}: {reasons}")
        return "\n".join(parts)


def create_memory_agent(llm: LLMClient):
    """创建 Memory Agent 节点函数

    Memory Agent 在 System Agent 内部被调用, 不是独立节点。
    这个函数返回两种模式:
    1. 作为 graph 节点: 为 System Agent 提供上下文
    2. 作为工具: 在 System Agent 裁定后存储 decision
    """

    def memory_node(state: dict[str, Any]) -> dict[str, Any]:
        """提供历史上下文给 System Agent"""
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

        return {
            "memory_context": context,
            "memory_recall": recall.model_dump(),
        }

    return memory_node
