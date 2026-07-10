"""
Memory Agent — 经验库与复盘层

职责:
- 保存完整运行记录
- 召回相似成功/失败案例
- 统计 Agent 准确率
- 识别冲突记忆

借鉴 TradingAgents' memory.py 的延迟反思模式:
- Phase A: 运行结束立即存 pending
- Phase B: 下次运行时拉取真实收益再反思

当前为轻量实现 (Markdown 文件 + 向量检索占位),
后续可升级到 PostgreSQL + ChromaDB (按设计方案).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import config
from ..llm.client import LLMClient
from .schemas import MemoryRecall, SystemDecision

logger = logging.getLogger(__name__)

# 分隔符
_SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"


class MemoryStore:
    """轻量 Memory 存储 — Markdown 文件

    后续可替换为 PostgreSQL + ChromaDB (按设计方案)
    """

    def __init__(self, log_path: str | None = None):
        path = log_path or config.get("memory_log_path", "")
        self._log_path = Path(path).expanduser() if path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def store_decision(self, ticker: str, trade_date: str,
                        decision: SystemDecision) -> None:
        """存储决策记录 (Phase A: pending)"""
        if not self._log_path:
            return

        # Idempotency check
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
            if not lines:
                continue

            tag = lines[0].strip()
            if not tag.startswith("[") or not tag.endswith("]"):
                continue
            fields = [f.strip() for f in tag[1:-1].split("|")]

            entry = {
                "date": fields[0] if len(fields) > 0 else "",
                "ticker": fields[1] if len(fields) > 1 else "",
                "decision": fields[2] if len(fields) > 2 else "",
                "pending": "pending" in tag,
            }

            # 解析 JSON body
            body = "\n".join(lines[1:]).strip()
            if body.startswith("DECISION:"):
                body = body[8:].strip()
            try:
                entry["decision_data"] = json.loads(body)
            except json.JSONDecodeError:
                entry["decision_data"] = {}

            entries.append(entry)
        return entries

    def get_pending(self) -> list[dict]:
        """获取未处理记录"""
        return [e for e in self.load_entries() if e.get("pending")]

    def get_context(self, ticker: str, n_same: int = 5) -> str:
        """获取历史上下文 (用于 Agent prompt 注入)"""
        entries = [e for e in self.load_entries() if not e.get("pending")]
        same = [e for e in reversed(entries) if e["ticker"] == ticker][:n_same]
        if not same:
            return ""
        parts = [f"Past analyses of {ticker}:"]
        for e in same:
            d = e.get("decision_data", {})
            parts.append(
                f"[{e['date']}] {e['decision']}: "
                f"{' '.join(d.get('reasons', []))}"
            )
        return "\n".join(parts)


def create_memory_agent(llm: LLMClient):
    """创建 Memory Agent 节点函数"""

    def memory_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")

        # 加载历史记录
        store = MemoryStore()
        entries = store.load_entries()
        context = store.get_context(ticker)

        # 统计 Agent 准确率 (占位, 后续完善)
        same_ticker = [e for e in entries if e["ticker"] == ticker and not e.get("pending")]
        success_cases = [e for e in same_ticker if e["decision"] == "推荐"]
        failure_cases = [e for e in same_ticker if e["decision"] in ("拒绝", "观察")]

        recall = MemoryRecall(
            success_cases=[
                {
                    "date": e["date"],
                    "reasons": e.get("decision_data", {}).get("reasons", []),
                }
                for e in success_cases[:5]
            ],
            failure_cases=[
                {
                    "date": e["date"],
                    "reasons": e.get("decision_data", {}).get("objections", []),
                }
                for e in failure_cases[:5]
            ],
            agent_accuracy={},
            historical_warnings=[],
            reasoning=f"召回 {len(success_cases)} 成功案例, {len(failure_cases)} 失败案例",
        )

        return {
            "memory_context": context,
            "memory_recall_obj": recall,
        }

    return memory_node
