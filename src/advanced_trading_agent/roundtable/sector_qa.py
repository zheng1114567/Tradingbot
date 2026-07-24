"""Sector Q&A adapter for the sector-first ETF path.

Uses deterministic sector evidence (no roundtable).
The old AutoGen single-sector roundtable has been removed.
"""

from __future__ import annotations

from typing import Any

from ..data_agent.sector_etf import SectorETFSelector


def answer_sector_question(
    question: str,
    *,
    sector_name: str,
    trade_date: str,
    selector: SectorETFSelector | None = None,
    explanation: dict[str, Any] | None = None,
    memory_context: str = "",
) -> dict[str, Any]:
    """Answer a sector question using deterministic sector evidence."""
    selector = selector or SectorETFSelector()
    explanation = explanation or selector.explain_sector(sector_name, trade_date)
    return _build_answer(question, explanation, memory_context=memory_context)


def _build_answer(question: str, explanation: dict[str, Any], *, memory_context: str = "") -> dict[str, Any]:
    primary = explanation.get("primary_etf") or {}
    lines = [
        f"问题：{question}",
        f"结论：{explanation.get('verdict', '数据不足')}",
    ]
    if explanation.get("score") is not None:
        lines.append(f"板块评分：{explanation.get('score')}")
    if primary:
        lines.append(f"可映射ETF：{primary.get('code')} {primary.get('name')}")
    lines.append("主要理由：")
    for reason in explanation.get("reasons", [])[:6]:
        lines.append(f"- {reason}")
    risks = explanation.get("risks", [])
    if risks:
        lines.append("风险/为什么不好：")
        for risk in risks[:6]:
            lines.append(f"- {risk}")
    result = {
        "answer": "\n".join(lines),
        "final_pressure": "downgrade" if explanation.get("verdict") == "暂不适合" else "neutral",
        "provider": "deterministic",
        "evidence": explanation,
    }
    if memory_context:
        result["memory_context_used"] = True
    return result
