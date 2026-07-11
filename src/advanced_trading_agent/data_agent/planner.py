"""ReAct-style planner for DataAgent.

The planner stays deterministic by default. It uses the ReAct shape
(thought -> action -> observation) to decide which data stages are needed,
then DataAgent performs the actual auditable ETL work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from .vendor_router import get_vendor_chain


@dataclass(frozen=True)
class DataAgentPlan:
    """Execution plan generated before DataAgent collection starts."""

    objective: str
    required_methods: list[str]
    skipped_methods: list[str]
    adjusted_request: dict[str, Any]
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataAgentPlanner:
    """Plan DataAgent runs with a small deterministic ReAct loop."""

    def plan(self, request: Any) -> tuple[Any, DataAgentPlan]:
        trace: list[dict[str, Any]] = []
        objective = self._objective(request)

        trace.append({
            "thought": "Clarify data objective from request flags.",
            "action": "inspect_request",
            "observation": {
                "ticker": request.ticker,
                "trade_date": request.normalized_trade_date(),
                "include_capital_flow": request.include_capital_flow,
                "include_news": getattr(request, "include_news", True),
                "include_factors": request.include_factors,
            },
        })

        required_methods = ["get_daily"]
        skipped_methods: list[str] = []

        trace.append({
            "thought": "Daily bars are the base input for cleaning and factor analysis.",
            "action": "require_method",
            "args": {"method": "get_daily", "vendor_chain": get_vendor_chain("get_daily")},
            "observation": "required",
        })

        if getattr(request, "include_market", True):
            required_methods.append("get_daily:index")
            trace.append({
                "thought": "Market index data is needed by Market Agent tier-1 context.",
                "action": "require_method",
                "args": {"method": "get_daily", "code": "000001.SH"},
                "observation": "required",
            })

        if request.include_capital_flow:
            required_methods.append("get_capital_flow")
            trace.append({
                "thought": "Capital flow was requested and should be collected when available.",
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
                "thought": "Capital flow was not requested.",
                "action": "skip_method",
                "args": {"method": "get_capital_flow"},
                "observation": "skipped",
            })

        if getattr(request, "include_news", True):
            required_methods.append("get_news")
            trace.append({
                "thought": "News events are needed by Event Agent tier-2 context.",
                "action": "require_method",
                "args": {"method": "get_news", "vendor_chain": get_vendor_chain("get_news")},
                "observation": "required",
            })
        else:
            skipped_methods.append("get_news")
            trace.append({
                "thought": "News events were not requested.",
                "action": "skip_method",
                "args": {"method": "get_news"},
                "observation": "skipped",
            })

        if request.include_factors:
            required_methods.append("compute_factors")
            trace.append({
                "thought": "Factors can be computed locally from cleaned daily bars.",
                "action": "require_method",
                "args": {"method": "compute_factors", "depends_on": "get_daily"},
                "observation": "required",
            })
        else:
            skipped_methods.append("compute_factors")
            trace.append({
                "thought": "Factor analysis was not requested.",
                "action": "skip_method",
                "args": {"method": "compute_factors"},
                "observation": "skipped",
            })

        if getattr(request, "include_risk", True):
            required_methods.extend(["get_st_status", "get_suspended", "get_delisting"])
            trace.append({
                "thought": "Risk gates need ST, suspended, and delisting lists before agents recommend.",
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
                "thought": "Risk pre-check data was not requested.",
                "action": "skip_method",
                "args": {"methods": ["get_st_status", "get_suspended", "get_delisting"]},
                "observation": "skipped",
            })

        adjusted_request = self._normalize_request(request)
        trace.append({
            "thought": "Finalize a reproducible plan before touching vendors.",
            "action": "finalize_plan",
            "observation": {
                "required_methods": required_methods,
                "skipped_methods": skipped_methods,
                "adjusted_request": asdict(adjusted_request),
            },
        })

        return adjusted_request, DataAgentPlan(
            objective=objective,
            required_methods=required_methods,
            skipped_methods=skipped_methods,
            adjusted_request=asdict(adjusted_request),
            trace=trace,
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
        if getattr(request, "include_news", True):
            parts.insert(1, "collect news events")
        if request.include_factors:
            parts.insert(-1, "compute factors")
        return ", ".join(parts)

    @staticmethod
    def _normalize_request(request: Any) -> Any:
        if request.end_date or not request.trade_date:
            return request
        return replace(request, end_date=request.trade_date.replace("-", ""))
