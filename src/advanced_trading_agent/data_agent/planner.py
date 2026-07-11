"""Auditable planner for DataAgent.

The planner keeps DataAgent decisions reproducible. It records concise
decision summaries, concrete next actions, and user clarification questions
without exposing raw chain-of-thought text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel


def _model_dump(obj: Any) -> dict[str, Any]:
    """Serialize a Pydantic model or dataclass to dict."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj) if isinstance(obj, dict) else {"value": str(obj)}

from .vendor_router import get_vendor_chain


@dataclass(frozen=True)
class DataAgentPlan:
    """Execution plan generated before DataAgent collection starts."""

    objective: str
    required_methods: list[str]
    skipped_methods: list[str]
    adjusted_request: dict[str, Any]
    trace: list[dict[str, Any]]
    next_actions: list[dict[str, Any]]
    clarification_questions: list[dict[str, Any]]
    decision_summary: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataAgentPlanner:
    """Plan DataAgent runs with a small deterministic ReAct loop."""

    def plan(self, request: Any) -> tuple[Any, DataAgentPlan]:
        trace: list[dict[str, Any]] = []
        objective = self._objective(request)
        decision_summary: list[str] = []
        next_actions: list[dict[str, Any]] = []
        clarification_questions = self.clarification_questions(request)

        trace.append({
            "reason": "Inspect request flags and defaults before data collection.",
            "action": "inspect_request",
            "observation": {
                "ticker": request.ticker,
                "trade_date": request.normalized_trade_date(),
                "start_date": request.start_date,
                "end_date": request.normalized_end_date(),
                "include_capital_flow": request.include_capital_flow,
                "include_news": getattr(request, "include_news", True),
                "use_llm_news_filter": getattr(request, "use_llm_news_filter", True),
                "include_factors": request.include_factors,
                "include_sector_context": getattr(request, "include_sector_context", True),
                "sector_keyword": getattr(request, "sector_keyword", None),
            },
        })
        decision_summary.append(
            f"Use trade_date={request.normalized_trade_date()} as the point-in-time anchor."
        )

        required_methods = ["get_daily"]
        skipped_methods: list[str] = []
        next_actions.append(self._next_action(
            "collect_daily_bars",
            "get_daily",
            "Daily bars are the base input for cleaning, factors, and close-data analysis.",
        ))

        trace.append({
            "reason": "Daily bars are the base input for cleaning and factor analysis.",
            "action": "require_method",
            "args": {"method": "get_daily", "vendor_chain": get_vendor_chain("get_daily")},
            "observation": "required",
        })

        if getattr(request, "include_market", True):
            required_methods.append("get_daily:index")
            next_actions.append(self._next_action(
                "collect_market_index",
                "get_daily:index",
                "Market index data is needed for tier-1 market context.",
            ))
            trace.append({
                "reason": "Market index data is needed by Market Agent tier-1 context.",
                "action": "require_method",
                "args": {"method": "get_daily", "code": "000001.SH"},
                "observation": "required",
            })

        if getattr(request, "include_sector_context", True):
            required_methods.append("get_sector")
            next_actions.append(self._next_action(
                "collect_sector_context",
                "get_sector",
                "Sector context gives downstream agents peer and theme context.",
            ))
            trace.append({
                "reason": "Sector context gives downstream agents market-wide peer and theme context.",
                "action": "require_method",
                "args": {
                    "method": "get_sector",
                    "vendor_chain": get_vendor_chain("get_sector"),
                    "sector_keyword": getattr(request, "sector_keyword", None),
                },
                "observation": "required",
            })
        else:
            skipped_methods.append("get_sector")
            trace.append({
                "reason": "Sector context was not requested.",
                "action": "skip_method",
                "args": {"method": "get_sector"},
                "observation": "skipped",
            })

        if request.include_capital_flow:
            required_methods.append("get_capital_flow")
            next_actions.append(self._next_action(
                "collect_capital_flow",
                "get_capital_flow",
                "Capital flow helps confirm or challenge price movement.",
            ))
            trace.append({
                "reason": "Capital flow was requested and should be collected when available.",
                "action": "require_method",
                "args": {
                    "method": "get_capital_flow",
                    "vendor_chain": get_vendor_chain("get_capital_flow"),
                },
                "observation": "required",
            })
        else:
            skipped_methods.append("get_capital_flow")
            trace.append({
                "reason": "Capital flow was not requested.",
                "action": "skip_method",
                "args": {"method": "get_capital_flow"},
                "observation": "skipped",
            })

        if getattr(request, "include_news", True):
            required_methods.append("get_news")
            if getattr(request, "use_llm_news_filter", True):
                required_methods.append("filter_news:llm")
            next_actions.append(self._next_action(
                "collect_and_filter_news",
                "get_news",
                "News events feed EventAgent and should be filtered for ticker relevance.",
            ))
            trace.append({
                "reason": "News events are needed by Event Agent tier-2 context and should be filtered for relevance.",
                "action": "require_method",
                "args": {
                    "method": "get_news",
                    "vendor_chain": get_vendor_chain("get_news"),
                    "llm_filter": getattr(request, "use_llm_news_filter", True),
                },
                "observation": "required",
            })
        else:
            skipped_methods.append("get_news")
            trace.append({
                "reason": "News events were not requested.",
                "action": "skip_method",
                "args": {"method": "get_news"},
                "observation": "skipped",
            })

        if request.include_factors:
            required_methods.append("compute_factors")
            next_actions.append(self._next_action(
                "compute_factors",
                "compute_factors",
                "Factors can be computed locally after daily bars are cleaned.",
                depends_on=["get_daily"],
            ))
            trace.append({
                "reason": "Factors can be computed locally from cleaned daily bars.",
                "action": "require_method",
                "args": {"method": "compute_factors", "depends_on": "get_daily"},
                "observation": "required",
            })
        else:
            skipped_methods.append("compute_factors")
            trace.append({
                "reason": "Factor analysis was not requested.",
                "action": "skip_method",
                "args": {"method": "compute_factors"},
                "observation": "skipped",
            })

        if getattr(request, "include_risk", True):
            required_methods.extend(["get_st_status", "get_suspended", "get_delisting"])
            next_actions.append(self._next_action(
                "collect_risk_lists",
                "get_st_status,get_suspended,get_delisting",
                "Risk gates must be available before downstream agents can recommend.",
            ))
            trace.append({
                "reason": "Risk gates need ST, suspended, and delisting lists before agents recommend.",
                "action": "require_method",
                "args": {
                    "methods": ["get_st_status", "get_suspended", "get_delisting"],
                    "vendor_chain": get_vendor_chain("get_st_status"),
                },
                "observation": "required",
            })
        else:
            skipped_methods.extend(["get_st_status", "get_suspended", "get_delisting"])
            trace.append({
                "reason": "Risk pre-check data was not requested.",
                "action": "skip_method",
                "args": {"methods": ["get_st_status", "get_suspended", "get_delisting"]},
                "observation": "skipped",
            })

        adjusted_request = self._normalize_request(request)
        if adjusted_request.end_date != request.end_date:
            decision_summary.append(
                f"Normalize missing end_date to trade_date compact form: {adjusted_request.end_date}."
            )
        if getattr(request, "use_llm_news_filter", True):
            decision_summary.append("Use LLM news filtering with deterministic keyword fallback.")
        else:
            decision_summary.append("Use deterministic news filtering because LLM news filter is disabled.")
        if clarification_questions:
            trace.append({
                "reason": "Some user intent is ambiguous, so expose clarification questions before broad collection.",
                "action": "request_user_clarification",
                "observation": {"questions": clarification_questions},
            })
        trace.append({
            "reason": "Finalize a reproducible plan before touching vendors.",
            "action": "finalize_plan",
            "observation": {
                "required_methods": required_methods,
                "skipped_methods": skipped_methods,
                "adjusted_request": _model_dump(adjusted_request),
                "next_actions": next_actions,
                "clarification_questions": clarification_questions,
            },
        })

        return adjusted_request, DataAgentPlan(
            objective=objective,
            required_methods=required_methods,
            skipped_methods=skipped_methods,
            adjusted_request=_model_dump(adjusted_request),
            trace=trace,
            next_actions=next_actions,
            clarification_questions=clarification_questions,
            decision_summary=decision_summary,
        )

    @staticmethod
    def _objective(request: Any) -> str:
        parts = [
            f"collect auditable market data for {request.ticker}",
            "clean data",
            "analyze data",
            "return layered artifacts",
        ]
        if request.include_capital_flow:
            parts.insert(1, "collect capital flow")
        if getattr(request, "include_sector_context", True):
            parts.insert(1, "collect sector context")
        if getattr(request, "include_news", True):
            parts.insert(1, "collect news events")
            if getattr(request, "use_llm_news_filter", True):
                parts.insert(2, "filter news with LLM")
        if request.include_factors:
            parts.insert(-1, "compute factors")
        return ", ".join(parts)

    @staticmethod
    def _normalize_request(request: Any) -> Any:
        if request.end_date or not request.trade_date:
            return request
        return request.model_copy(update={"end_date": request.trade_date.replace("-", "")})

    @staticmethod
    def _next_action(
        name: str,
        method: str,
        rationale: str,
        *,
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "method": method,
            "depends_on": depends_on or [],
            "rationale": rationale,
        }

    @staticmethod
    def clarification_questions(request: Any) -> list[dict[str, Any]]:
        """Return user-facing questions for ambiguous but recoverable requests."""

        questions: list[dict[str, Any]] = []
        if not str(getattr(request, "ticker", "") or "").strip():
            questions.append({
                "id": "ticker",
                "question": "你想分析哪只股票？请提供 A 股代码，例如 000001.SZ。",
                "required": True,
                "default": None,
            })
        if not getattr(request, "trade_date", None) and not getattr(request, "end_date", None):
            today = date.today().isoformat()
            questions.append({
                "id": "trade_date",
                "question": f"没有指定交易日，是否按今天 {today} 的收盘数据分析？",
                "required": False,
                "default": today,
            })
        if getattr(request, "include_news", True) and not getattr(request, "news_keyword", None):
            questions.append({
                "id": "news_keyword",
                "question": "新闻检索没有关键词，是否用股票名称/代码自动推断？",
                "required": False,
                "default": "auto",
            })
        return questions
