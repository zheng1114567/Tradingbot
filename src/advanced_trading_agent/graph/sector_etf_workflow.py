"""LangGraph pipeline for sector-first ETF decisions.

This is the strategy-level workflow for the new direction:
select a sector, map it to tradable ETFs, run an AutoGen-backed roundtable,
store conversation memory, then render an auditable report.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents.conversation_memory import ConversationEntry, ConversationMemoryStore
from ..data_agent.etf_watchlist import (
    DailyETFWatchlistReport,
    ETFWatchlistLimits,
    build_watchlist_report,
    render_watchlist_markdown,
)
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
    timings: dict[str, float]
    errors: list[str]


class SectorETFWatchlistState(TypedDict, total=False):
    """State for the batch ETF observation-pool workflow."""

    messages: list[Any]
    trade_date: str
    max_roundtable_sectors: int
    store_memory: bool
    selection: dict[str, Any]
    watchlist_report: dict[str, Any]
    final_report: str
    timings: dict[str, float]
    errors: list[str]


def create_sector_etf_workflow(
    *,
    selector: SectorETFSelector | None = None,
    roundtable_fn: RoundtableFn | None = None,
    memory_store: ConversationMemoryStore | None = None,
    refresh_cache: bool = False,
) -> Any:
    """Create the compiled LangGraph workflow for sector ETF decisions."""
    selector = selector or SectorETFSelector(auto_refresh_cache=refresh_cache)
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
        refresh_cache: bool = False,
    ) -> None:
        self.workflow = create_sector_etf_workflow(
            selector=selector,
            roundtable_fn=roundtable_fn,
            memory_store=memory_store,
            refresh_cache=refresh_cache,
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


def create_sector_etf_watchlist_workflow(
    *,
    selector: SectorETFSelector | None = None,
    memory_store: ConversationMemoryStore | None = None,
    limits: ETFWatchlistLimits | None = None,
    refresh_cache: bool = False,
) -> Any:
    """Create the batch sector ETF observation-pool workflow.

    The roundtable step is JSON-first: every final sector decision must include
    a primary ETF, support reasons, objections, and approval-gated weight hints.
    """
    limits = limits or ETFWatchlistLimits()
    selector = selector or SectorETFSelector(
        top_sectors=limits.max_roundtable_sectors,
        auto_refresh_cache=refresh_cache,
    )
    memory_store = memory_store or ConversationMemoryStore()

    def select_batch_node(state: SectorETFWatchlistState) -> dict[str, Any]:
        started = time.perf_counter()
        trade_date = state.get("trade_date") or date.today().isoformat()
        selection = selector.select_with_exclusions(
            trade_date,
            max_roundtable_sectors=state.get("max_roundtable_sectors", limits.max_roundtable_sectors),
        )
        timings = dict(state.get("timings", {}) or {})
        timings["select_and_process_seconds"] = round(time.perf_counter() - started, 3)
        selection_payload = selection.to_dict()
        selection_payload["timings"] = timings
        return {"selection": selection_payload, "timings": timings}

    def roundtable_json_node(state: SectorETFWatchlistState) -> dict[str, Any]:
        started = time.perf_counter()
        selection_raw = state.get("selection", {}) or {}
        from ..data_agent.etf_watchlist import ExcludedSectorCandidate, SectorCandidatePayload

        candidates = [
            SectorCandidatePayload(**item)
            for item in selection_raw.get("candidates", [])
            if isinstance(item, dict)
        ]
        excluded = [
            ExcludedSectorCandidate(**item)
            for item in selection_raw.get("excluded", [])
            if isinstance(item, dict)
        ]
        report = build_watchlist_report(
            trade_date=state.get("trade_date") or date.today().isoformat(),
            candidates=candidates,
            excluded=excluded,
            limits=limits,
        )
        timings = dict(selection_raw.get("timings", {}) or state.get("timings", {}) or {})
        timings["rules_roundtable_seconds"] = round(time.perf_counter() - started, 3)
        timings["total_pre_render_seconds"] = round(sum(timings.values()), 3)
        payload = report.model_dump(mode="json")
        payload["roundtable_summary"]["timings"] = timings
        return {"watchlist_report": payload, "timings": timings}

    def store_memory_node(state: SectorETFWatchlistState) -> dict[str, Any]:
        if not state.get("store_memory", True):
            return {}
        report = DailyETFWatchlistReport(**(state.get("watchlist_report") or {}))
        summary = ", ".join(
            f"{d.sector}:{d.status}:{d.primary_etf.code}"
            for d in report.decisions[:8]
        )
        memory_store.append(ConversationEntry(
            question="每日板块 ETF 观察池",
            answer=summary,
            trade_date=report.trade_date,
            target_type="sector_etf_watchlist",
            target="batch",
            evidence={
                "run_id": report.run_id,
                "decision_count": len(report.decisions),
                "excluded_count": len(report.excluded_sector_candidates),
            },
        ))
        return {}

    def report_node(state: SectorETFWatchlistState) -> dict[str, Any]:
        report = DailyETFWatchlistReport(**(state.get("watchlist_report") or {}))
        return {"final_report": render_watchlist_markdown(report)}

    workflow = StateGraph(SectorETFWatchlistState)
    workflow.add_node("Select Sector ETF Basket", select_batch_node)
    workflow.add_node("JSON Roundtable Decision", roundtable_json_node)
    workflow.add_node("Store Watchlist Memory", store_memory_node)
    workflow.add_node("Render Report", report_node)

    workflow.add_edge(START, "Select Sector ETF Basket")
    workflow.add_edge("Select Sector ETF Basket", "JSON Roundtable Decision")
    workflow.add_edge("JSON Roundtable Decision", "Store Watchlist Memory")
    workflow.add_edge("Store Watchlist Memory", "Render Report")
    workflow.add_edge("Render Report", END)
    return workflow.compile()


class SectorETFWatchlistSystem:
    """Batch sector ETF observation-pool entrypoint."""

    def __init__(
        self,
        *,
        selector: SectorETFSelector | None = None,
        memory_store: ConversationMemoryStore | None = None,
        limits: ETFWatchlistLimits | None = None,
        refresh_cache: bool = False,
    ) -> None:
        self.limits = limits or ETFWatchlistLimits()
        self.workflow = create_sector_etf_watchlist_workflow(
            selector=selector,
            memory_store=memory_store,
            limits=self.limits,
            refresh_cache=refresh_cache,
        )

    def analyze(
        self,
        *,
        trade_date: str | None = None,
        max_roundtable_sectors: int | None = None,
        store_memory: bool = True,
    ) -> tuple[dict[str, Any], str]:
        td = trade_date or date.today().isoformat()
        init_state: SectorETFWatchlistState = {
            "messages": [("human", f"生成 {td} A股板块 ETF 观察池")],
            "trade_date": td,
            "max_roundtable_sectors": max_roundtable_sectors or self.limits.max_roundtable_sectors,
            "store_memory": store_memory,
            "timings": {},
            "errors": [],
        }
        final_state = self.workflow.invoke(init_state)
        return final_state, str(final_state.get("final_report", ""))
