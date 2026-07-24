"""One-shot data collection plus workflow analysis for API/CLI callers."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import config
from .core.atomic_write import atomic_write_json, atomic_write_text
from .data_agent.data_agent import DataAgent, DataAgentRequest, DataAgentRun
from .data_agent.trading_calendar import resolve_market_trade_date


@dataclass(frozen=True)
class FullAnalysisRun:
    """Structured return payload for API/CLI callers."""

    ticker: str
    trade_date: str
    data_run: DataAgentRun
    final_state: dict[str, Any]
    final_report: str

    def to_dict(self) -> dict[str, Any]:
        return _json_roundtrip({
            "stage": "full_analysis",
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "analysis_mode": "workflow",
            "data_agent": {
                "run_id": self.data_run.run_id,
                "request": self.data_run.request,
                "response_path": self.data_run.response_path,
                "manifest_path": self.data_run.manifest_path,
                "artifacts": (self.data_run.to_dict() or {}).get("artifacts", {}),
                "collection_summary": self.data_run.collection_summary,
                "errors": self.data_run.final_data.get("errors", []),
            },
            "analysis": {
                "final_report": self.final_report,
                "final_report_path": _report_path(self.final_state),
                "audit_trace_path": self.final_state.get("audit_trace_path", ""),
                "execution_allowed": self.final_state.get("execution_allowed", False),
                "system_decision": _json_safe(self.final_state.get("system_decision_obj")),
                "round2_state": _json_safe(self.final_state.get("round2_state", {})),
                "risk_checks": {
                    "risk_check_1": _json_safe(self.final_state.get("risk_check_1", {})),
                    "risk_check_2": _json_safe(self.final_state.get("risk_check_2", {})),
                    "risk_check_3": _json_safe(self.final_state.get("risk_check_3", {})),
                },
            },
            "data_quality": self.data_run.final_data.get("analysis", {}).get("data_quality", {}),
            "agent_payload": self.data_run.final_data.get("analysis", {}).get("agent_payload", {}),
        })


def run_full_analysis(
    ticker: str,
    *,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str | None = None,
    use_react_planner: bool = True,
    news_keyword: str | None = None,
    sector_keyword: str | None = None,
    use_llm_news_filter: bool = True,
    use_llm_data_review: bool = False,
    fetch_news_full_text: bool = True,
    skip_backtest: bool = False,
    lookback_days: int = 90,
    data_agent: DataAgent | None = None,
    trading_system: Any | None = None,
) -> FullAnalysisRun:
    """Collect, clean, analyze data, then run the full LangGraph/LLM workflow."""
    effective_trade_date = resolve_market_trade_date(trade_date)
    effective_end_date = _normalize_compact_date(end_date or effective_trade_date)
    effective_start_date = _normalize_compact_date(
        start_date or _default_start_date(effective_trade_date, lookback_days=lookback_days)
    )
    request = DataAgentRequest(
        ticker=ticker,
        trade_date=effective_trade_date,
        start_date=effective_start_date,
        end_date=effective_end_date,
        output_dir=output_dir,
        use_react_planner=use_react_planner,
        news_keyword=news_keyword,
        sector_keyword=sector_keyword,
        use_llm_news_filter=use_llm_news_filter,
        use_llm_data_review=use_llm_data_review,
        fetch_news_full_text=fetch_news_full_text,
        include_market=True,
        include_capital_flow=True,
        include_news=True,
        include_factors=True,
        include_risk=True,
        include_sector_context=True,
    )
    agent = data_agent or DataAgent(results_dir=output_dir)
    data_run = agent.run(request)
    tier1, tier2 = _workflow_payload_from_data_run(data_run)

    if trading_system is None:
        from .graph.workflow import TradingSystem
        trading_system = TradingSystem(debug=False)
    final_state, final_report = trading_system.analyze(
        request.ticker,
        effective_trade_date,
        tier1_data=copy.deepcopy(tier1),
        tier2_data=copy.deepcopy(tier2),
        skip_backtest=skip_backtest,
    )

    return FullAnalysisRun(
        ticker=request.ticker,
        trade_date=effective_trade_date,
        data_run=data_run,
        final_state=final_state,
        final_report=final_report,
    )


def run_full_analysis_json(**kwargs: Any) -> str:
    """JSON convenience wrapper for CLI/API adapters."""
    return json.dumps(run_full_analysis(**kwargs).to_dict(), ensure_ascii=False, indent=2, default=str)


def _workflow_payload_from_data_run(data_run: DataAgentRun) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract workflow tier payload and attach audit metadata from DataAgent."""
    payload = data_run.final_data.get("analysis", {}).get("agent_payload", {})
    tier1 = copy.deepcopy(payload.get("tier1_data", {}) or {})
    tier2 = copy.deepcopy(payload.get("tier2_data", {}) or {})
    tier1["_data_manifest"] = copy.deepcopy(data_run.final_data.get("manifest"))
    tier1["_data_manifest_path"] = data_run.manifest_path
    tier1["_data_agent_run"] = data_run.to_dict()
    tier1["_collection_summary"] = copy.deepcopy(data_run.collection_summary)
    tier1["_audit_trail"] = copy.deepcopy(data_run.audit_trail)
    tier1["_errors"] = copy.deepcopy(data_run.final_data.get("errors", []))
    return tier1, tier2


def _normalize_iso_date(value: str) -> str:
    clean = value.replace("-", "")
    return datetime.strptime(clean, "%Y%m%d").date().isoformat()


def _normalize_compact_date(value: str) -> str:
    return _normalize_iso_date(value).replace("-", "")


def _default_start_date(trade_date: str, *, lookback_days: int) -> str:
    anchor = datetime.strptime(_normalize_iso_date(trade_date), "%Y-%m-%d").date()
    return (anchor - timedelta(days=max(0, lookback_days))).strftime("%Y%m%d")


def _report_path(final_state: dict[str, Any]) -> str:
    trace = final_state.get("audit_trace", {}) or {}
    return str(trace.get("report_path") or "")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_roundtrip(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
