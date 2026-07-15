"""LangGraph pipeline for sector-first ETF decisions.

This is the strategy-level workflow for the new direction:
select a sector, map it to tradable ETFs, run an AutoGen-backed roundtable,
store conversation memory, then render an auditable report.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents.conversation_memory import ConversationEntry, ConversationMemoryStore
from ..data_agent.sector_etf import SectorETFSelector
from ..roundtable.sector_qa import answer_sector_question_with_roundtable


RoundtableFn = Callable[..., dict[str, Any]]


class SectorETFState(TypedDict, total=False):
    """State carried through the sector ETF LangGraph pipeline."""

    messages: list[Any]
    question: str
    sector_name: str
    trade_date: str
    use_autogen: bool
    store_memory: bool
    memory_context: str
    sector_evidence: dict[str, Any]
    roundtable_result: dict[str, Any]
    final_report: str
    errors: list[str]


def create_sector_etf_workflow(
    *,
    selector: SectorETFSelector | None = None,
    roundtable_fn: RoundtableFn | None = None,
    memory_store: ConversationMemoryStore | None = None,
) -> Any:
    """Create the compiled LangGraph workflow for sector ETF decisions."""
    selector = selector or SectorETFSelector()
    roundtable_fn = roundtable_fn or answer_sector_question_with_roundtable
    memory_store = memory_store or ConversationMemoryStore()

    def select_sector_node(state: SectorETFState) -> dict[str, Any]:
        sector = state.get("sector_name", "")
        trade_date = state.get("trade_date") or date.today().isoformat()
        explanation = selector.explain_sector(sector, trade_date)
        return {
            "sector_evidence": explanation,
            "errors": [] if explanation.get("status") != "not_found" else list(state.get("errors", [])) + ["sector_not_found"],
        }

    def recall_memory_node(state: SectorETFState) -> dict[str, Any]:
        context = memory_store.recall(state.get("sector_name", ""), limit=5)
        return {"memory_context": context}

    def roundtable_node(state: SectorETFState) -> dict[str, Any]:
        result = roundtable_fn(
            state.get("question") or f"{state.get('sector_name', '')}板块是否适合买ETF？",
            sector_name=state.get("sector_name", ""),
            trade_date=state.get("trade_date") or date.today().isoformat(),
            selector=selector,
            explanation=state.get("sector_evidence"),
            memory_context=state.get("memory_context", ""),
            use_autogen=bool(state.get("use_autogen", True)),
        )
        return {"roundtable_result": result}

    def store_memory_node(state: SectorETFState) -> dict[str, Any]:
        if not state.get("store_memory", True):
            return {}
        result = state.get("roundtable_result", {}) or {}
        answer = str(result.get("answer", ""))
        if not answer:
            return {}
        memory_store.append(ConversationEntry(
            question=state.get("question", ""),
            answer=answer,
            trade_date=state.get("trade_date") or date.today().isoformat(),
            target_type="sector",
            target=state.get("sector_name", ""),
            evidence={
                "provider": result.get("provider"),
                "final_pressure": result.get("final_pressure"),
                "sector_evidence": state.get("sector_evidence", {}),
            },
        ))
        return {}

    def report_node(state: SectorETFState) -> dict[str, Any]:
        evidence = state.get("sector_evidence", {}) or {}
        result = state.get("roundtable_result", {}) or {}
        primary = evidence.get("primary_etf") or {}
        lines = [
            "# 板块ETF LangGraph 决策报告",
            "",
            f"**交易日期**: {state.get('trade_date', '')}",
            f"**板块**: {evidence.get('sector_name') or state.get('sector_name', '')}",
            f"**结论**: {evidence.get('verdict', '数据不足')}",
        ]
        if evidence.get("score") is not None:
            lines.append(f"**评分**: {evidence.get('score')}")
        if primary:
            lines.append(f"**首选ETF**: {primary.get('code')} {primary.get('name')}")
        lines.extend([
            f"**圆桌引擎**: {result.get('provider', 'unknown')}",
            "",
            "## 圆桌结论",
            "",
            str(result.get("answer", "")).strip() or "无圆桌输出。",
            "",
            "## 证据摘要",
            "",
        ])
        for reason in evidence.get("reasons", [])[:8]:
            lines.append(f"- {reason}")
        risks = evidence.get("risks", [])
        if risks:
            lines.extend(["", "## 风险/为什么不好", ""])
            for risk in risks[:8]:
                lines.append(f"- {risk}")
        return {"final_report": "\n".join(lines)}

    workflow = StateGraph(SectorETFState)
    workflow.add_node("Select Sector ETF", select_sector_node)
    workflow.add_node("Recall Conversation Memory", recall_memory_node)
    workflow.add_node("AutoGen Roundtable", roundtable_node)
    workflow.add_node("Store Conversation Memory", store_memory_node)
    workflow.add_node("Report", report_node)

    workflow.add_edge(START, "Select Sector ETF")
    workflow.add_edge("Select Sector ETF", "Recall Conversation Memory")
    workflow.add_edge("Recall Conversation Memory", "AutoGen Roundtable")
    workflow.add_edge("AutoGen Roundtable", "Store Conversation Memory")
    workflow.add_edge("Store Conversation Memory", "Report")
    workflow.add_edge("Report", END)
    return workflow.compile()


class SectorETFTradingSystem:
    """Public entrypoint for the sector ETF LangGraph workflow."""

    def __init__(
        self,
        *,
        selector: SectorETFSelector | None = None,
        roundtable_fn: RoundtableFn | None = None,
        memory_store: ConversationMemoryStore | None = None,
    ) -> None:
        self.workflow = create_sector_etf_workflow(
            selector=selector,
            roundtable_fn=roundtable_fn,
            memory_store=memory_store,
        )

    def analyze(
        self,
        sector_name: str,
        *,
        question: str | None = None,
        trade_date: str | None = None,
        use_autogen: bool = True,
        store_memory: bool = True,
    ) -> tuple[dict[str, Any], str]:
        td = trade_date or date.today().isoformat()
        init_state: SectorETFState = {
            "messages": [("human", question or f"{sector_name}板块是否适合买ETF？")],
            "question": question or f"{sector_name}板块是否适合买ETF？",
            "sector_name": sector_name,
            "trade_date": td,
            "use_autogen": use_autogen,
            "store_memory": store_memory,
            "errors": [],
        }
        final_state = self.workflow.invoke(init_state)
        return final_state, str(final_state.get("final_report", ""))
