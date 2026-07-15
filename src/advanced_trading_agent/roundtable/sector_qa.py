"""Sector Q&A roundtable adapter.

The production path uses AutoGen when configured.  The deterministic fallback
keeps CLI Q&A useful when API keys or AutoGen packages are unavailable.
"""

from __future__ import annotations

import json
from typing import Any

from ..data_agent.sector_etf import SectorETFSelector
from .autogen_roundtable import AutoGenRoundtable


def answer_sector_question_with_roundtable(
    question: str,
    *,
    sector_name: str,
    trade_date: str,
    selector: SectorETFSelector | None = None,
    explanation: dict[str, Any] | None = None,
    memory_context: str = "",
    use_autogen: bool = True,
) -> dict[str, Any]:
    """Answer a sector question using sector evidence and a roundtable state."""
    selector = selector or SectorETFSelector()
    explanation = explanation or selector.explain_sector(sector_name, trade_date)
    state = _build_sector_roundtable_state(
        question=question,
        sector_name=sector_name,
        trade_date=trade_date,
        explanation=explanation,
        memory_context=memory_context,
    )
    if use_autogen:
        try:
            result = AutoGenRoundtable().run(state)
            if result.summary:
                return {
                    "provider": result.provider,
                    "answer": result.summary,
                    "final_pressure": result.final_pressure,
                    "roundtable": {
                        "questions": result.questions,
                        "moderator_output": result.moderator_output,
                        "round_history": result.round_history,
                    },
                    "evidence": explanation,
                }
        except Exception as exc:
            fallback = _fallback_sector_answer(question, explanation)
            fallback["provider"] = "deterministic_fallback"
            fallback["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return fallback
    fallback = _fallback_sector_answer(question, explanation)
    fallback["provider"] = "deterministic"
    return fallback


def _build_sector_roundtable_state(
    *,
    question: str,
    sector_name: str,
    trade_date: str,
    explanation: dict[str, Any],
    memory_context: str,
) -> dict[str, Any]:
    candidate = explanation.get("candidate") or {}
    primary_etf = explanation.get("primary_etf") or {}
    tier1 = {
        "market": {},
        "sentiment": {},
        "capital": {},
        "risk": {"sector_risks": explanation.get("risks", [])},
        "sector": {
            "status": explanation.get("status"),
            "matched_sector": explanation.get("sector_name", sector_name),
            "score": explanation.get("score", 0),
            "primary_etf": primary_etf,
        },
    }
    tier2 = {
        "sector_context": candidate,
        "events": (candidate.get("raw") or {}).get("news", []),
        "factors": [],
        "price_data": [],
        "data_summary": {
            "question": question,
            "verdict": explanation.get("verdict"),
            "reasons": explanation.get("reasons", []),
        },
        "data_quality": {
            "status": "sector_qa",
            "note": "Conversation Q&A evidence; not a post-trade review.",
        },
    }
    contradiction = {
        "id": "ct_sector_user_question",
        "description": (
            f"用户询问“{sector_name}”为什么不好或是否值得买ETF；"
            "圆桌必须基于板块动量、事件、ETF流动性和风险给出理由。"
        ),
        "agents_involved": ["Market", "Event", "Analysis", "Backtest"],
        "detection_method": "pattern",
        "severity": "medium",
        "evidence_pair": ("ev_sector_score", "ev_sector_risk"),
    }
    return {
        "company_of_interest": f"SECTOR:{sector_name}",
        "trade_date": trade_date,
        "tier1_data": tier1,
        "tier2_data": tier2,
        "memory_context": memory_context,
        "market_report": f"板块评分 {explanation.get('score', 0)}，结论 {explanation.get('verdict')}。",
        "event_report": "事件证据：" + json.dumps(explanation.get("reasons", []), ensure_ascii=False),
        "analysis_report": "ETF匹配：" + json.dumps(primary_etf, ensure_ascii=False),
        "backtest_report": "板块ETF问答未做事后回测，不能把历史命中率作为支持项。",
        "round2_state": {
            "active": True,
            "round_count": 0,
            "max_rounds": 1,
            "completed": False,
            "contradiction_records": [contradiction],
            "evidence_board": [],
            "round_history": [],
            "unresolved_conflicts": [],
            "final_pressure": "neutral",
            "summary": "",
            "provider": "autogen",
            "fallback_reason": "",
            "current_speaker": "",
            "moderator_output": None,
        },
    }


def _fallback_sector_answer(question: str, explanation: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "answer": "\n".join(lines),
        "final_pressure": "downgrade" if explanation.get("verdict") == "暂不适合" else "neutral",
        "roundtable": {},
        "evidence": explanation,
    }
