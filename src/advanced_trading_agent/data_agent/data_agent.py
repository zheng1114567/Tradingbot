"""Standalone auditable data agent.

The agent records each stage of a data run:
1. input request
2. raw vendor data
3. cleaned data
4. analysis data
5. final layered response
"""
from __future__ import annotations

import logging
import json
import os
import re
from html import unescape
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from ..config import config
from ..core.atomic_write import atomic_write_json, atomic_write_text
from ..core.audit import audit_event, build_data_collection_summary
from ..core.vendor import timed_vendor_call
from .cleaner import DataCleaner
from .factors import FactorCalculator
from .manifest import DataManifest
from .planner import DataAgentPlan, DataAgentPlanner
from .request import DataAgentRequest
from .scanner import ScanBundle
from .stock_profile import StockProfile, StockProfileResolver
from .vendor_router import get_vendor_chain, route_to_vendor

logger = logging.getLogger(__name__)


RouteFn = Callable[..., Any]


_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.-]+")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NEWS_NOISE_PATTERNS = (
    re.compile(r"^原标题[:：]"),
    re.compile(r"^来源[:：]"),
    re.compile(r"^文章来源[:：]"),
    re.compile(r"^责任编辑[:：]"),
    re.compile(r"^编辑[:：]"),
    re.compile(r"^校对[:：]"),
    re.compile(r"^作者[:：]\s*$"),
    re.compile(r"^免责声明[:：]"),
    re.compile(r"^风险提示[:：]"),
    re.compile(r"^广告$"),
    re.compile(r"^更多精彩.*"),
    re.compile(r"^下载.*APP.*"),
    re.compile(r"^打开.*APP.*"),
    re.compile(r"^微信扫一扫.*"),
    re.compile(r"^海量资讯.*"),
    re.compile(r"^本文源自[:：]"),
    re.compile(r"^本文来自.*"),
    re.compile(r"^股市有风险.*"),
    re.compile(r"^投资需谨慎.*"),
    re.compile(r"^Copyright\b", re.IGNORECASE),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path_part(value: str, fallback: str) -> str:
    safe = _SAFE_PATH_PART.sub("_", value).strip("._")
    return safe or fallback


def _records_from_frame(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    frame = df if limit is None else df.tail(limit)
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


@dataclass
class DataAgentArtifact:
    """One persisted step in the data-agent trace."""

    stage: str
    path: str
    record_count: int | None = None
    columns: list[str] = field(default_factory=list)


@dataclass
class DataAgentRun:
    """Structured return payload for a standalone data-agent run."""

    run_id: str
    request: dict[str, Any]
    artifacts: dict[str, DataAgentArtifact]
    manifest_path: str
    response_path: str
    final_data: dict[str, Any]
    plan: dict[str, Any] | None = None
    collection_summary: dict[str, Any] = field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = {k: asdict(v) for k, v in self.artifacts.items()}
        return json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default))


class DataAgent:
    """Collect, clean, analyze, and persist market data in auditable layers."""

    def __init__(
        self,
        *,
        route_fn: RouteFn = route_to_vendor,
        results_dir: str | None = None,
        planner: DataAgentPlanner | None = None,
        llm_client: Any | None = None,
        profile_resolver: StockProfileResolver | None = None,
    ) -> None:
        self._route_fn = route_fn
        self._results_dir = Path(results_dir or config.get("results_dir"))
        self._planner = planner or DataAgentPlanner()
        self._llm_client = llm_client
        self._profile_resolver = profile_resolver or StockProfileResolver()

    @classmethod
    def from_bundle(
        cls,
        bundle: ScanBundle,
        ticker: str,
        trade_date: str | None = None,
        *,
        news_keyword: str | None = None,
        sector_keyword: str | None = None,
        results_dir: str | None = None,
        llm_client: Any | None = None,
        **request_kwargs: Any,
    ) -> DataAgentRun:
        """Run the full data pipeline for one ticker from a pre-collected *bundle*.

        Uses shared data (market index, sector context, risk lists) and
        per-ticker data (daily OHLCV, capital flow, news) already fetched
        by ``MarketScanner.scan_and_collect()``, eliminating redundant vendor calls.

        Example::

            bundle = MarketScanner().scan_and_collect()
            run = DataAgent.from_bundle(bundle, "000001.SZ")
        """
        ticker_raw = bundle.ticker_data.get(ticker, {})
        raw_data: dict[str, Any] = {
            **bundle.shared_raw,
            **ticker_raw,
            "route_trace": list(bundle.route_trace),
        }

        effective_trade_date = trade_date or bundle.trade_date
        request = DataAgentRequest(
            ticker=ticker,
            trade_date=effective_trade_date,
            use_react_planner=False,
            news_keyword=news_keyword,
            sector_keyword=sector_keyword,
            **request_kwargs,
        )

        agent = cls(results_dir=results_dir, llm_client=llm_client)
        return agent.run(request, raw_data=raw_data)

    def run(self, request: DataAgentRequest, raw_data: dict[str, Any] | None = None) -> DataAgentRun:
        # Ensure the free vendor adapters are available when DataAgent is used standalone.
        from .collector import register_all_vendors

        register_all_vendors()
        request, stock_profile = self._apply_stock_profile(request)
        run_dir = self._make_run_dir(request)
        manifest = DataManifest(
            ticker=request.ticker,
            trade_date=request.normalized_trade_date(),
        )

        artifacts: dict[str, DataAgentArtifact] = {}
        route_trace: list[dict[str, Any]] = []
        audit_trail: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        audit_trail.append(audit_event("data", f"开始数据收集: {request.ticker}", detail={"trade_date": request.normalized_trade_date()}))

        plan_payload: dict[str, Any] | None = None
        if request.use_react_planner:
            request, plan = self._planner.plan(request)
            plan_payload = plan.to_dict()
            artifacts["planner"] = self._write_json(run_dir / "00_planner" / "plan.json", {
                "stage": "planner",
                "created_at": _utc_now(),
                "plan": plan_payload,
            })

        input_payload = {
            "stage": "input",
            "created_at": _utc_now(),
            "request": request.model_dump(),
            "stock_profile": stock_profile,
            "planner": plan_payload,
            "vendor_chain": {
                "daily": get_vendor_chain("get_daily"),
                "market": get_vendor_chain("get_daily"),
                "capital_flow": get_vendor_chain("get_capital_flow"),
                "news": get_vendor_chain("get_news"),
                "sector_context": get_vendor_chain("get_sector"),
                "factors": get_vendor_chain("get_factors"),
                "risk": {
                    "st_status": get_vendor_chain("get_st_status"),
                    "suspended": get_vendor_chain("get_suspended"),
                    "delisting": get_vendor_chain("get_delisting"),
                },
            },
        }
        artifacts["input"] = self._write_json(run_dir / "01_input" / "request.json", input_payload)

        # --- Raw collection with error recovery ---
        try:
            if raw_data is not None:
                raw_payload = self._adopt_raw(raw_data, request, manifest, route_trace)
            else:
                raw_payload = self._collect_raw(request, manifest, route_trace)
            artifacts["raw"] = self._write_json(run_dir / "02_raw" / "raw_data.json", raw_payload)
        except Exception as exc:
            logger.error("Raw data collection failed: %s", exc)
            audit_trail.append(audit_event("data", f"数据收集失败: {exc}", level="error", detail={"stage": "raw_collection"}))
            errors.append({"stage": "raw_collection", "error": str(exc)})
            raw_payload = {"stage": "raw", "created_at": _utc_now(), "error": str(exc)}

        # --- Build collection summary ---
        vendor_health = self._summarize_vendor_health(route_trace)
        collection_summary = build_data_collection_summary(raw_payload, vendor_health, route_trace)
        audit_trail.append(audit_event("data", f"数据收集完成: {collection_summary.get('categories_with_data', 0)}/{collection_summary.get('total_categories', 0)} 类数据获取成功",
                                       detail={"failed": collection_summary.get('categories_failed', 0), "empty": collection_summary.get('categories_empty', 0)}))

        if collection_summary.get("categories_failed", 0) > 0:
            for key, cat in collection_summary.get("categories", {}).items():
                if cat["status"] == "error":
                    audit_trail.append(audit_event("data", f"{cat['label']} 获取失败: {cat.get('error', 'unknown')}", level="warning"))

        # --- Clean with error recovery ---
        try:
            cleaned_payload = self._clean(raw_payload)
            artifacts["cleaned"] = self._write_json(run_dir / "03_cleaned" / "cleaned_data.json", cleaned_payload)
        except Exception as exc:
            logger.error("Data cleaning failed: %s", exc)
            audit_trail.append(audit_event("data", f"数据清洗失败: {exc}", level="error"))
            errors.append({"stage": "cleaning", "error": str(exc)})
            cleaned_payload = {"stage": "cleaned", "created_at": _utc_now(), "error": str(exc)}

        # --- Analyze with error recovery ---
        try:
            analysis_payload = self._analyze(cleaned_payload, request, route_trace=route_trace)
            artifacts["analysis"] = self._write_json(
                run_dir / "04_analysis" / "analysis_data.json",
                analysis_payload,
            )
            audit_trail.append(audit_event("data", "因子计算与事件分析完成",
                                           detail={"factor_count": len(analysis_payload.get("factors", {}).get("records", [])),
                                                   "event_count": len(analysis_payload.get("events", {}).get("records", []))}))
        except Exception as exc:
            logger.error("Analysis failed: %s", exc)
            audit_trail.append(audit_event("data", f"分析阶段失败: {exc}", level="error"))
            errors.append({"stage": "analysis", "error": str(exc)})
            analysis_payload = {"stage": "analysis", "created_at": _utc_now(), "error": str(exc), "agent_payload": {}}

        artifacts["news_events"] = self._write_json(
            run_dir / "04_analysis" / "news_events.json",
            {
                "stage": "news_events",
                "created_at": _utc_now(),
                "events": analysis_payload.get("events", {}),
            },
        )
        agent_payload = {
            "stage": "agent_payload",
            "created_at": _utc_now(),
            **analysis_payload.get("agent_payload", {}),
        }
        artifacts["agent_payload"] = self._write_json(
            run_dir / "05_agent_payload" / "agent_payload.json",
            agent_payload,
        )

        final_payload = {
            "stage": "final",
            "created_at": _utc_now(),
            "input": input_payload,
            "raw": raw_payload,
            "cleaned": cleaned_payload,
            "analysis": analysis_payload,
            "agent_payload": agent_payload,
            "planner": plan_payload,
            "vendor_health": vendor_health,
            "manifest": manifest.to_dict(),
            "collection_summary": collection_summary,
            "audit_trail": audit_trail,
            "errors": errors,
        }
        artifacts["final"] = self._write_json(run_dir / "06_final" / "response.json", final_payload)

        manifest_path = manifest.save(results_dir=str(run_dir / "06_final"))

        if errors:
            audit_trail.append(audit_event("data", f"数据流水线完成，{len(errors)} 个阶段出错", level="warning" if len(errors) < 3 else "error"))
        else:
            audit_trail.append(audit_event("data", "数据流水线全部完成", level="info"))

        return DataAgentRun(
            run_id=run_dir.name,
            request=request.model_dump(),
            artifacts=artifacts,
            manifest_path=str(manifest_path),
            response_path=artifacts["final"].path,
            final_data=final_payload,
            plan=plan_payload,
            collection_summary=collection_summary,
            audit_trail=audit_trail,
        )

    def _make_run_dir(self, request: DataAgentRequest) -> Path:
        ticker = _safe_path_part(request.ticker.replace(".", "_"), "unknown_ticker")
        trade_date = _safe_path_part(request.normalized_trade_date(), "unknown_date")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._results_dir / "data_agent_runs" / f"{trade_date}_{ticker}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _apply_stock_profile(self, request: DataAgentRequest) -> tuple[DataAgentRequest, dict[str, Any]]:
        profile = self._profile_resolver.resolve(request.ticker)
        updates: dict[str, Any] = {}
        applied_fields: list[str] = []

        if not request.news_keyword and profile.company_name:
            updates["news_keyword"] = profile.company_name
            applied_fields.append("news_keyword")
        if not request.sector_keyword and profile.sector_keyword:
            updates["sector_keyword"] = profile.sector_keyword
            applied_fields.append("sector_keyword")

        effective_request = request.model_copy(update=updates) if updates else request
        profile_payload = {
            **asdict(profile),
            "applied_fields": applied_fields,
            "effective_news_keyword": effective_request.news_keyword,
            "effective_sector_keyword": effective_request.sector_keyword,
        }
        return effective_request, profile_payload

    def _adopt_raw(
        self,
        raw_data: dict[str, Any],
        request: DataAgentRequest,
        manifest: DataManifest,
        route_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Normalize pre-collected raw data into the standard raw payload format.

        When data is collected by MarketScanner.scan_and_collect(), this
        adapter merges the shared and per-ticker portions into a single
        payload matching _collect_raw's return shape.
        """
        if "route_trace" in raw_data:
            incoming = raw_data["route_trace"]
            if isinstance(incoming, list):
                route_trace.extend(incoming)

        daily = raw_data.get("daily", [])
        market = raw_data.get("market", [])
        sector_context = raw_data.get("sector_context", [])
        limit_up_summary = raw_data.get("limit_up_summary", {})
        dragon_tiger = raw_data.get("dragon_tiger", [])
        market_breadth = raw_data.get("market_breadth", {})
        capital_flow = raw_data.get("capital_flow", [])
        news = raw_data.get("news", [])
        risk = raw_data.get("risk", {})

        if request.fetch_news_full_text and isinstance(news, list) and news:
            news = self._enrich_news_full_text(news)

        field_checks = [
            ("stock.daily", daily),
            ("market.daily", market),
            ("sector.context", sector_context),
            ("market.limit_up_summary", [limit_up_summary] if isinstance(limit_up_summary, dict) and limit_up_summary else []),
            ("market.dragon_tiger", dragon_tiger),
            ("market.breadth", [market_breadth] if isinstance(market_breadth, dict) and market_breadth else []),
            ("stock.capital_flow", capital_flow),
            ("news.events", news),
        ]
        for field_name, value in field_checks:
            available = isinstance(value, list) and len(value) > 0
            manifest.add_field(
                field_name,
                available=available,
                source="scan_bundle",
                vendor_chain=[],
                record_count=len(value) if isinstance(value, list) else None,
            )

        return {
            "stage": "raw",
            "created_at": _utc_now(),
            "daily": daily,
            "market": market,
            "sector_context": sector_context,
            "limit_up_summary": limit_up_summary if isinstance(limit_up_summary, dict) else {},
            "dragon_tiger": dragon_tiger if isinstance(dragon_tiger, list) else [],
            "market_breadth": market_breadth if isinstance(market_breadth, dict) else {},
            "capital_flow": capital_flow,
            "news": news,
            "risk": risk if isinstance(risk, dict) else {},
            "route_trace": route_trace,
        }

    def _collect_raw(
        self,
        request: DataAgentRequest,
        manifest: DataManifest,
        route_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        daily = self._safe_route(
            "get_daily",
            manifest,
            field_name="stock.daily",
            route_trace=route_trace,
            code=request.ticker,
            start_date=request.start_date,
            end_date=request.normalized_end_date(),
        )
        market = []
        if request.include_market:
            market = self._safe_route(
                "get_daily",
                manifest,
                field_name="market.daily",
                route_trace=route_trace,
                code="000001.SH",
                start_date=request.start_date or request.normalized_end_date(),
                end_date=request.normalized_end_date(),
            )
        sector_context = []
        if request.include_sector_context:
            sector_context = self._safe_route(
                "get_sector",
                manifest,
                field_name="sector.context",
                route_trace=route_trace,
                top_n=request.sector_top_n,
                trade_date=request.normalized_trade_date(),
            )
        market_breadth = self._safe_route(
            "get_market_breadth",
            manifest,
            field_name="market.breadth",
            route_trace=route_trace,
            trade_date=request.normalized_trade_date(),
        )
        limit_up_summary = self._safe_route(
            "get_limit_up_tiers",
            manifest,
            field_name="market.limit_up_summary",
            route_trace=route_trace,
            trade_date=request.normalized_trade_date(),
        )
        dragon_tiger = self._safe_route(
            "get_dragon_tiger",
            manifest,
            field_name="market.dragon_tiger",
            route_trace=route_trace,
            trade_date=request.normalized_trade_date(),
        )
        capital_flow = []
        if request.include_capital_flow:
            capital_flow = self._safe_route(
                "get_capital_flow",
                manifest,
                field_name="stock.capital_flow",
                route_trace=route_trace,
                code=request.ticker,
                start_date=request.start_date,
                end_date=request.normalized_end_date(),
            )
        news = []
        if request.include_news:
            news = self._safe_route(
                "get_news",
                manifest,
                field_name="news.events",
                route_trace=route_trace,
                code=request.ticker,
                sector=request.sector_keyword,
                keyword=request.news_keyword,
                trade_date=request.normalized_trade_date(),
            )
            if request.fetch_news_full_text and isinstance(news, list):
                news = self._enrich_news_full_text(news)
        st_status: list[str] | dict[str, Any] = []
        suspended: list[str] | dict[str, Any] = []
        delisting: list[str] | dict[str, Any] = []
        if request.include_risk:
            st_status = self._safe_route(
                "get_st_status",
                manifest,
                field_name="risk.st_status",
                route_trace=route_trace,
                trade_date=request.normalized_trade_date(),
            )
            suspended = self._safe_route(
                "get_suspended",
                manifest,
                field_name="risk.suspended",
                route_trace=route_trace,
                trade_date=request.normalized_trade_date(),
            )
            delisting = self._safe_route(
                "get_delisting",
                manifest,
                field_name="risk.delisting",
                route_trace=route_trace,
                trade_date=request.normalized_trade_date(),
            )

        return {
            "stage": "raw",
            "created_at": _utc_now(),
            "daily": daily,
            "market": market,
            "sector_context": sector_context,
            "limit_up_summary": limit_up_summary if isinstance(limit_up_summary, dict) else {},
            "dragon_tiger": dragon_tiger if isinstance(dragon_tiger, list) else [],
            "market_breadth": market_breadth if isinstance(market_breadth, dict) else {},
            "capital_flow": capital_flow,
            "news": news,
            "risk": {
                "st_status": st_status,
                "suspended": suspended,
                "delisting": delisting,
            },
            "route_trace": route_trace,
        }

    def _safe_route(
        self,
        method: str,
        manifest: DataManifest,
        *,
        field_name: str,
        route_trace: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        vendor_chain = get_vendor_chain(method)
        try:
            result, _elapsed_ms = timed_vendor_call(
                method,
                route_trace=route_trace,
                route_fn=self._route_fn,
                **kwargs,
            )
            is_no_data = isinstance(result, str) and result.startswith("NO_DATA_AVAILABLE")
            count = len(result) if isinstance(result, list) else None
            manifest.add_field(
                field_name,
                available=not is_no_data,
                source=f"vendor_router:{method}",
                vendor_chain=vendor_chain,
                fallback_used=len(vendor_chain) > 1,
                error=result if is_no_data else None,
                record_count=count,
            )
            return result
        except Exception as exc:
            manifest.add_field(
                field_name,
                available=False,
                source=f"vendor_router:{method}",
                vendor_chain=vendor_chain,
                error=str(exc),
            )
            return {
                "error": str(exc),
                "method": method,
                "vendor_chain": vendor_chain,
            }

    def _clean(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        daily_raw = raw_payload.get("daily")
        if isinstance(daily_raw, list):
            daily_df = DataCleaner.clean_daily(daily_raw)
            daily_df = DataCleaner.detect_limit_up_down(daily_df)
        else:
            daily_df = pd.DataFrame()

        capital_flow_raw = raw_payload.get("capital_flow")
        capital_flow = capital_flow_raw if isinstance(capital_flow_raw, list) else []
        news_raw = raw_payload.get("news")
        news = news_raw if isinstance(news_raw, list) else []
        market_raw = raw_payload.get("market")
        if isinstance(market_raw, list):
            market_df = DataCleaner.clean_daily(market_raw)
        else:
            market_df = pd.DataFrame()
        limit_up_summary = raw_payload.get("limit_up_summary")
        dragon_tiger_raw = raw_payload.get("dragon_tiger")
        market_breadth = raw_payload.get("market_breadth")
        sector_raw = raw_payload.get("sector_context")
        sector_context = self._clean_sector_context(sector_raw if isinstance(sector_raw, list) else [])
        risk_raw = raw_payload.get("risk", {})

        return {
            "stage": "cleaned",
            "created_at": _utc_now(),
            "market": {
                "record_count": int(len(market_df)),
                "columns": list(market_df.columns),
                "records": _records_from_frame(market_df),
            },
            "daily": {
                "record_count": int(len(daily_df)),
                "columns": list(daily_df.columns),
                "records": _records_from_frame(daily_df),
            },
            "sector_context": {
                "record_count": len(sector_context),
                "records": sector_context,
            },
            "limit_up_summary": limit_up_summary if isinstance(limit_up_summary, dict) else {},
            "dragon_tiger": {
                "record_count": len(dragon_tiger_raw) if isinstance(dragon_tiger_raw, list) else 0,
                "records": dragon_tiger_raw if isinstance(dragon_tiger_raw, list) else [],
            },
            "market_breadth": market_breadth if isinstance(market_breadth, dict) else {},
            "capital_flow": {
                "record_count": len(capital_flow),
                "records": capital_flow,
            },
            "news": {
                "record_count": len(news),
                "records": self._clean_news(news),
            },
            "risk": risk_raw if isinstance(risk_raw, dict) else {},
        }

    def _analyze(
        self,
        cleaned_payload: dict[str, Any],
        request: DataAgentRequest,
        *,
        route_trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        daily_records = cleaned_payload.get("daily", {}).get("records", [])
        market_records = cleaned_payload.get("market", {}).get("records", [])
        sector_records = cleaned_payload.get("sector_context", {}).get("records", [])
        limit_up_summary = cleaned_payload.get("limit_up_summary", {})
        dragon_tiger_records = cleaned_payload.get("dragon_tiger", {}).get("records", [])
        market_breadth = cleaned_payload.get("market_breadth", {})
        news_records = cleaned_payload.get("news", {}).get("records", [])
        daily_df = pd.DataFrame(daily_records)
        market_df = pd.DataFrame(market_records)
        factor_records: list[dict[str, Any]] = []
        news_filter = self._filter_news_with_llm(news_records, request)
        event_records = self._build_event_records(news_filter["records"], request)
        capital_summary = self._summarize_capital(cleaned_payload.get("capital_flow", {}).get("records", []))
        summary: dict[str, Any] = {
            "ticker": request.ticker,
            "record_count": int(len(daily_df)),
            "latest": None,
            "price_change_pct": None,
            "volume_latest": None,
            "amount_latest": None,
        }

        if not daily_df.empty:
            if "trade_date" in daily_df.columns:
                daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])
                daily_df = daily_df.sort_values("trade_date")
            latest = daily_df.iloc[-1].to_dict()
            summary.update(
                {
                    "latest": self._json_safe_record(latest),
                    "price_change_pct": self._json_safe_value(latest.get("pct_chg")),
                    "volume_latest": self._json_safe_value(latest.get("volume")),
                    "amount_latest": self._json_safe_value(latest.get("amount")),
                }
            )

        if request.include_factors and not daily_df.empty:
            factor_df = FactorCalculator.run_all(daily_df.copy())
            factor_records = _records_from_frame(factor_df, limit=request.max_return_records)

        market_summary = self._summarize_market(
            market_df,
            limit_up_summary=limit_up_summary if isinstance(limit_up_summary, dict) else {},
            dragon_tiger_records=dragon_tiger_records if isinstance(dragon_tiger_records, list) else [],
            market_breadth=market_breadth if isinstance(market_breadth, dict) else {},
        )
        sector_summary = self._summarize_sector_context(sector_records, request)
        data_quality = {
            "daily_consistency": self._build_daily_consistency_report(
                daily_records,
                route_trace or [],
            )
        }
        risk_summary = self._summarize_risk(
            cleaned_payload.get("risk", {}),
            latest=summary.get("latest") if isinstance(summary.get("latest"), dict) else {},
        )
        tier1, tier2 = self._build_agent_payload(
            request=request,
            summary=summary,
            market_summary=market_summary,
            sector_summary=sector_summary,
            capital_summary=capital_summary,
            risk_summary=risk_summary,
            daily_records=daily_records,
            factor_records=factor_records,
            event_records=event_records,
            dragon_tiger_records=dragon_tiger_records if isinstance(dragon_tiger_records, list) else [],
            data_quality=data_quality,
        )

        return {
            "stage": "analysis",
            "created_at": _utc_now(),
            "summary": summary,
            "market": market_summary,
            "sector": sector_summary,
            "capital": capital_summary,
            "risk": risk_summary,
            "data_quality": data_quality,
            "dragon_tiger": {
                "record_count": len(dragon_tiger_records),
                "records": dragon_tiger_records,
            },
            "events": {
                "record_count": len(event_records),
                "records": event_records,
                "filter": news_filter["trace"],
            },
            "factors": {
                "record_count": len(factor_records),
                "records": factor_records,
            },
            "agent_payload": {
                "tier1_data": tier1,
                "tier2_data": tier2,
            },
        }

    @classmethod
    def _summarize_market(
        cls,
        market_df: pd.DataFrame,
        *,
        limit_up_summary: dict[str, Any] | None = None,
        dragon_tiger_records: list[dict[str, Any]] | None = None,
        market_breadth: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        limit_up_summary = limit_up_summary or {}
        dragon_tiger_records = dragon_tiger_records or []
        market_breadth = market_breadth or {}
        if market_df.empty:
            return {
                "index_close": 0,
                "index_change_pct": 0,
                "sentiment": "未知",
                "sentiment_score": 50,
                "advance_count": int(market_breadth.get("advance_count", 0) or 0),
                "decline_count": int(market_breadth.get("decline_count", 0) or 0),
                "limit_up_count": 0,
                "limit_down_count": 0,
                "dragon_tiger_count": len(dragon_tiger_records),
                "breadth_sample_size": int(market_breadth.get("sample_size", 0) or 0),
            }
        if "trade_date" in market_df.columns:
            market_df["trade_date"] = pd.to_datetime(market_df["trade_date"], errors="coerce")
            market_df = market_df.sort_values("trade_date")
        latest = market_df.iloc[-1].to_dict()
        pct = cls._json_safe_value(latest.get("pct_chg")) or 0
        try:
            pct_value = float(pct)
        except (TypeError, ValueError):
            pct_value = 0.0
        if pct_value <= -2:
            sentiment = "低迷"
            score = 35
        elif pct_value >= 2:
            sentiment = "温热"
            score = 65
        else:
            sentiment = "正常"
            score = 55
        return {
            "index_close": cls._json_safe_value(latest.get("close")) or 0,
            "index_change_pct": pct_value,
            "sentiment": sentiment,
            "sentiment_score": score,
            "advance_count": int(market_breadth.get("advance_count", 0) or 0),
            "decline_count": int(market_breadth.get("decline_count", 0) or 0),
            "limit_up_count": int(sum(limit_up_summary.get(key, 0) or 0 for key in ("first_board", "second_board", "third_plus"))),
            "limit_down_count": 0,
            "limit_up_breakdown": {
                "first_board": int(limit_up_summary.get("first_board", 0) or 0),
                "second_board": int(limit_up_summary.get("second_board", 0) or 0),
                "third_plus": int(limit_up_summary.get("third_plus", 0) or 0),
            },
            "dragon_tiger_count": len(dragon_tiger_records),
            "breadth_sample_size": int(market_breadth.get("sample_size", 0) or 0),
            "breadth_coverage_note": str(market_breadth.get("coverage_note", "")),
            "data_source": latest.get("data_source", ""),
        }

    @classmethod
    def _summarize_capital(cls, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {
                "sector_name": "",
                "sector_volume_cny": 0,
                "net_inflow_main": 0,
                "net_inflow_retail": 0,
                "consecutive_inflow_days": 0,
                "confirmation": "未知",
            }
        latest = records[-1]
        candidates = [
            "主力净流入-净额",
            "主力净流入",
            "net_inflow_main",
            "net_mf_amount",
        ]
        net = 0.0
        for key in candidates:
            if key in latest:
                parsed = cls._parse_number(latest.get(key))
                if parsed is not None:
                    net = parsed
                    break
        confirmation = "资金确认" if net > 0 else "资金背离" if net < 0 else "未知"
        return {
            "sector_name": latest.get("sector_name", latest.get("code", "")),
            "sector_volume_cny": cls._parse_number(latest.get("amount")) or 0,
            "net_inflow_main": net,
            "net_inflow_retail": cls._parse_number(latest.get("散户净流入-净额")) or 0,
            "consecutive_inflow_days": 1 if net > 0 else 0,
            "confirmation": confirmation,
            "data_source": latest.get("data_source", ""),
        }

    @classmethod
    def _summarize_risk(cls, risk_raw: dict[str, Any], *, latest: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        lists: dict[str, list[Any]] = {}
        for field in ["st_status", "suspended", "delisting"]:
            value = risk_raw.get(field)
            if isinstance(value, list):
                lists[field] = value
            else:
                lists[field] = []
                if isinstance(value, dict) and value.get("error"):
                    errors.append(f"{field}: {value['error']}")
        return {
            "st_list": lists["st_status"],
            "suspended_list": lists["suspended"],
            "delisting_list": lists["delisting"],
            "daily_volume": cls._json_safe_value(latest.get("amount")),
            "is_limit_up": bool(latest.get("is_limit_up", False)),
            "is_limit_down": bool(latest.get("is_limit_down", False)),
            "risk_data_available": not errors,
            "risk_data_errors": errors,
        }

    @staticmethod
    def _build_agent_payload(
        *,
        request: DataAgentRequest,
        summary: dict[str, Any],
        market_summary: dict[str, Any],
        sector_summary: dict[str, Any],
        capital_summary: dict[str, Any],
        risk_summary: dict[str, Any],
        daily_records: list[dict[str, Any]],
        factor_records: list[dict[str, Any]],
        event_records: list[dict[str, Any]],
        dragon_tiger_records: list[dict[str, Any]],
        data_quality: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        tier1 = {
            "market": {
                "index_close": market_summary.get("index_close", 0),
                "index_change_pct": market_summary.get("index_change_pct", 0),
                "advance_count": market_summary.get("advance_count", 0),
                "decline_count": market_summary.get("decline_count", 0),
                "limit_up_count": market_summary.get("limit_up_count", 0),
                "limit_down_count": market_summary.get("limit_down_count", 0),
                "limit_up_breakdown": market_summary.get("limit_up_breakdown", {}),
                "dragon_tiger_count": market_summary.get("dragon_tiger_count", 0),
                "breadth_sample_size": market_summary.get("breadth_sample_size", 0),
                "breadth_coverage_note": market_summary.get("breadth_coverage_note", ""),
            },
            "sentiment": {
                "sentiment": market_summary.get("sentiment", "未知"),
                "sentiment_score": market_summary.get("sentiment_score", 50),
            },
            "capital": capital_summary,
            "risk": risk_summary,
            "sector": {
                "status": sector_summary.get("status", "unavailable"),
                "matched_sector": sector_summary.get("matched_sector"),
                "match_confidence": sector_summary.get("match_confidence", 0),
                "top_sectors": sector_summary.get("top_sectors", []),
            },
        }
        tier2 = {
            "price_data": daily_records,
            "factors": factor_records,
            "events": event_records,
            "sector_context": sector_summary,
            "limit_up_summary": market_summary.get("limit_up_breakdown", {}),
            "dragon_tiger": dragon_tiger_records if isinstance(dragon_tiger_records, list) else [],
            "backtest_samples": [],
            "data_summary": summary,
            "data_quality": data_quality,
        }
        return tier1, tier2

    @classmethod
    def _clean_sector_context(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for idx, record in enumerate(records, start=1):
            sector_name = cls._first_present(record, ["sector_name", "板块名称", "行业", "name", "名称"]) or ""
            change_pct = cls._parse_number(cls._first_present(record, ["change_pct", "涨跌幅", "涨跌幅%", "change"])) or 0
            strength_score = cls._parse_number(record.get("strength_score"))
            if strength_score is None:
                strength_score = change_pct
            try:
                rank = int(record.get("rank", idx))
            except (TypeError, ValueError):
                rank = idx
            cleaned.append({
                "rank": rank,
                "sector_name": str(sector_name),
                "change_pct": change_pct,
                "strength_score": strength_score,
                "source": record.get("data_source", record.get("source", "")),
                "raw": record,
            })
        return cleaned

    @classmethod
    def _summarize_sector_context(
        cls,
        records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> dict[str, Any]:
        if not records:
            return {
                "status": "unavailable",
                "matched_sector": None,
                "match_confidence": 0.0,
                "match_strategy": "no_sector_records",
                "direct_stock_sector_supported": False,
                "top_sectors": [],
                "records": [],
                "reason": "No sector records were returned by the configured free data source.",
            }

        ranked = sorted(
            records,
            key=lambda item: (
                -float(item.get("strength_score") or 0),
                int(item.get("rank") or 999999),
            ),
        )
        top_sectors = [
            {
                "rank": item.get("rank"),
                "sector_name": item.get("sector_name", ""),
                "change_pct": item.get("change_pct", 0),
                "strength_score": item.get("strength_score", 0),
                "source": item.get("source", ""),
            }
            for item in ranked[:10]
        ]
        keywords = [
            ("sector_keyword", request.sector_keyword),
            ("news_keyword", request.news_keyword),
            ("ticker", request.ticker),
        ]
        matched: dict[str, Any] | None = None
        strategy = "top_rank_fallback"
        confidence = 0.3
        for label, value in keywords:
            keyword = str(value or "").strip().lower()
            if not keyword:
                continue
            for item in ranked:
                sector_name = str(item.get("sector_name") or "").strip()
                sector_lower = sector_name.lower()
                if not sector_lower:
                    continue
                if keyword in sector_lower or sector_lower in keyword:
                    matched = item
                    strategy = f"{label}_match"
                    confidence = 0.9 if label == "sector_keyword" else 0.75
                    break
            if matched is not None:
                break
        if matched is None:
            matched = ranked[0]

        return {
            "status": "matched" if strategy != "top_rank_fallback" else "fallback_top_sector",
            "matched_sector": matched.get("sector_name"),
            "match_confidence": confidence,
            "match_strategy": strategy,
            "direct_stock_sector_supported": False,
            "top_sectors": top_sectors,
            "records": ranked[: request.sector_top_n],
            "reason": (
                "Sector context is built from market-wide sector rankings. "
                "Without a paid/stock-membership endpoint, ticker-to-sector matching is heuristic."
            ),
        }

    @classmethod
    def _summarize_vendor_health(cls, route_trace: list[dict[str, Any]]) -> dict[str, Any]:
        vendors: dict[str, dict[str, Any]] = {}
        for attempt in route_trace:
            vendor = str(attempt.get("vendor") or "unknown")
            method = str(attempt.get("method") or "unknown")
            status = str(attempt.get("status") or "unknown")
            summary = vendors.setdefault(
                vendor,
                {
                    "attempt_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "methods": [],
                    "statuses": {},
                    "last_error": None,
                },
            )
            summary["attempt_count"] += 1
            if method not in summary["methods"]:
                summary["methods"].append(method)
            summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
            if status == "success":
                summary["success_count"] += 1
            elif status != "missing_impl":
                summary["error_count"] += 1
            if attempt.get("error"):
                summary["last_error"] = attempt["error"]
        return {
            "attempt_count": len(route_trace),
            "vendors": vendors,
            "attempts": route_trace,
        }

    @classmethod
    def _build_daily_consistency_report(
        cls,
        daily_records: list[dict[str, Any]],
        route_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not daily_records:
            return {
                "status": "unavailable",
                "confidence_score": 0.0,
                "sources": [],
                "recommended_action": "block_or_request_daily_data",
                "differences": [],
                "route_success_count": cls._route_success_count(route_trace, "get_daily"),
            }

        sources = sorted({
            str(record.get("data_source") or record.get("source") or "unknown")
            for record in daily_records
        })
        if len(sources) < 2:
            return {
                "status": "single_source",
                "confidence_score": 0.7,
                "sources": sources,
                "recommended_action": "use_single_source_with_manifest_trace",
                "differences": [],
                "route_success_count": cls._route_success_count(route_trace, "get_daily"),
            }

        frame = pd.DataFrame(daily_records)
        if "trade_date" not in frame.columns:
            return {
                "status": "insufficient_keys",
                "confidence_score": 0.5,
                "sources": sources,
                "recommended_action": "review_daily_schema_before_analysis",
                "differences": [],
                "route_success_count": cls._route_success_count(route_trace, "get_daily"),
            }

        checks = {
            "close": 0.001,
            "pct_chg": 0.01,
            "volume": 0.05,
            "amount": 0.05,
        }
        differences: list[dict[str, Any]] = []
        for trade_date, group in frame.groupby("trade_date", dropna=False):
            group_sources = {
                str(record.get("data_source") or record.get("source") or "unknown")
                for record in group.to_dict("records")
            }
            if len(group_sources) < 2:
                continue
            for column, threshold in checks.items():
                if column not in group.columns:
                    continue
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                if len(values) < 2:
                    continue
                min_value = float(values.min())
                max_value = float(values.max())
                base = max(abs(min_value), 1.0)
                relative_diff = abs(max_value - min_value) / base
                if relative_diff > threshold:
                    differences.append({
                        "trade_date": cls._json_safe_value(trade_date),
                        "field": column,
                        "min": min_value,
                        "max": max_value,
                        "relative_diff": relative_diff,
                        "threshold": threshold,
                        "sources": sorted(group_sources),
                    })

        if differences:
            return {
                "status": "conflict",
                "confidence_score": 0.4,
                "sources": sources,
                "recommended_action": "review_vendor_conflict_before_trading_decision",
                "differences": differences,
                "route_success_count": cls._route_success_count(route_trace, "get_daily"),
            }
        return {
            "status": "consistent",
            "confidence_score": 0.9,
            "sources": sources,
            "recommended_action": "use_primary_vendor_with_cross_source_support",
            "differences": [],
            "route_success_count": cls._route_success_count(route_trace, "get_daily"),
        }

    @staticmethod
    def _route_success_count(route_trace: list[dict[str, Any]], method: str) -> int:
        return sum(
            1
            for attempt in route_trace
            if attempt.get("method") == method and attempt.get("status") == "success"
        )

    @classmethod
    def _clean_news(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for idx, record in enumerate(records, start=1):
            title = cls._first_present(record, ["title", "标题", "新闻标题", "article_title", "name"]) or ""
            summary = cls._first_present(record, ["summary", "摘要", "内容", "content", "article_content"]) or title
            full_text = cls._first_present(record, ["full_text", "正文", "text", "article_text"])
            evidence_text = cls._select_evidence_text(full_text, summary, title)
            content_cleaning = record.get("content_cleaning")
            source = cls._first_present(record, ["source", "来源", "data_source"]) or record.get("data_source", "")
            event_time = cls._first_present(record, ["time", "时间", "datetime", "发布时间", "date", "日期"])
            url = cls._first_present(record, ["url", "链接", "link"])
            cleaned.append({
                "event_id": str(record.get("event_id") or f"news_{idx:04d}"),
                "event_type": str(record.get("event_type") or "新闻"),
                "summary": str(summary)[:500],
                "title": str(title or summary)[:200],
                "full_text": str(full_text or "")[:8000],
                "evidence_text": evidence_text,
                "content_status": str(record.get("content_status") or ("full_text" if full_text else "summary_only")),
                "content_cleaning": content_cleaning if isinstance(content_cleaning, dict) else {},
                "content_error": record.get("content_error"),
                "direction": str(record.get("direction") or "中性"),
                "confidence": float(record.get("confidence") or 0.5),
                "event_time": cls._json_safe_value(event_time),
                "source": source,
                "url": url,
                "raw": record,
            })
        return cleaned

    @classmethod
    def _select_evidence_text(cls, full_text: Any, summary: Any, title: Any) -> str:
        text = str(full_text or "").strip()
        if len(text) >= 80:
            return text[:3000]
        fallback = str(summary or title or "").strip()
        return fallback[:1500]

    @classmethod
    def _enrich_news_full_text(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for record in records:
            item = dict(record)
            title = cls._first_present(item, ["title", "标题", "新闻标题", "article_title", "name"]) or ""
            summary = cls._first_present(item, ["summary", "摘要", "内容", "content", "article_content"]) or title
            existing_text = cls._first_present(item, ["full_text", "正文", "text", "article_text"])
            url = cls._first_present(item, ["url", "链接", "link"])

            if existing_text:
                cleaned_text, cleaning_trace = cls._clean_article_text(str(existing_text))
                item["raw_full_text"] = str(existing_text)[:12000]
                item["full_text"] = cleaned_text[:8000]
                item["content_status"] = item.get("content_status") or ("full_text" if cleaned_text else "summary_only")
                item["content_cleaning"] = cleaning_trace
                item["evidence_text"] = cls._select_evidence_text(item["full_text"], summary, title)
                enriched.append(item)
                continue

            if not url:
                item["full_text"] = ""
                item["content_status"] = item.get("content_status") or "summary_only"
                item["content_error"] = item.get("content_error") or "missing_url"
                item["evidence_text"] = cls._select_evidence_text("", summary, title)
                enriched.append(item)
                continue

            try:
                full_text = cls._fetch_news_full_text(str(url))
            except Exception as exc:
                full_text = ""
                item["content_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

            if full_text:
                cleaned_text, cleaning_trace = cls._clean_article_text(full_text)
                item["raw_full_text"] = full_text[:12000]
                item["full_text"] = cleaned_text[:8000]
                item["content_status"] = "full_text" if cleaned_text else "summary_only"
                item["content_cleaning"] = cleaning_trace
                item["content_error"] = item.get("content_error")
            else:
                item["full_text"] = ""
                item["content_status"] = "summary_only"
                item["content_cleaning"] = {
                    "status": "skipped",
                    "raw_length": 0,
                    "cleaned_length": 0,
                    "removed_segments": 0,
                    "deduplicated_segments": 0,
                }
                item["content_error"] = item.get("content_error") or "extract_empty"
            item["evidence_text"] = cls._select_evidence_text(item.get("full_text"), summary, title)
            enriched.append(item)
        return enriched

    @classmethod
    def _fetch_news_full_text(cls, url: str) -> str:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                )
            },
            timeout=8,
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding or "utf-8"
        return cls._extract_article_text(response.text)

    @classmethod
    def _extract_article_text(cls, html: str) -> str:
        text = _SCRIPT_STYLE_RE.sub(" ", html or "")
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", text, flags=re.IGNORECASE | re.DOTALL)
        if paragraphs:
            candidates = [_WHITESPACE_RE.sub(" ", _HTML_TAG_RE.sub(" ", p)).strip() for p in paragraphs]
            body = "\n".join(p for p in candidates if len(p) >= 12)
        else:
            body = _HTML_TAG_RE.sub(" ", text)
        body = unescape(body).replace("\u3000", " ").replace("\xa0", " ")
        body = "\n".join(
            _WHITESPACE_RE.sub(" ", line).strip()
            for line in body.splitlines()
            if line.strip()
        )
        if len(body) < 80:
            return ""
        return body[:12000]

    @classmethod
    def _clean_article_text(cls, text: str) -> tuple[str, dict[str, Any]]:
        raw = unescape(text or "").replace("\u3000", " ").replace("\xa0", " ")
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        raw_length = len(raw)

        segments = [
            _WHITESPACE_RE.sub(" ", segment).strip()
            for segment in re.split(r"\n+|(?<=[。！？；])\s+", raw)
        ]
        cleaned_segments: list[str] = []
        seen: set[str] = set()
        removed = 0
        deduped = 0
        for segment in segments:
            if not segment:
                continue
            segment = cls._strip_news_noise(segment)
            if not segment:
                removed += 1
                continue
            if cls._is_noise_segment(segment):
                removed += 1
                continue
            normalized_key = re.sub(r"\W+", "", segment.lower())
            if normalized_key and normalized_key in seen:
                deduped += 1
                continue
            if normalized_key:
                seen.add(normalized_key)
            cleaned_segments.append(segment)

        cleaned = "\n".join(cleaned_segments).strip()
        if len(cleaned) < 80:
            cleaned = _WHITESPACE_RE.sub(" ", raw).strip()
            status = "raw_fallback" if cleaned else "empty"
        else:
            status = "cleaned"
        cleaned = cleaned[:12000]
        return cleaned, {
            "status": status,
            "raw_length": raw_length,
            "cleaned_length": len(cleaned),
            "removed_segments": removed,
            "deduplicated_segments": deduped,
        }

    @classmethod
    def _strip_news_noise(cls, text: str) -> str:
        text = re.sub(r"https?://\S+", "", text).strip()
        text = re.sub(r"（?文章来源[:：].*?）?$", "", text).strip()
        text = re.sub(r"（?责任编辑[:：].*?）?$", "", text).strip()
        text = re.sub(r"（?编辑[:：].*?）?$", "", text).strip()
        text = re.sub(r"（?原标题[:：].*?）?$", "", text).strip()
        return _WHITESPACE_RE.sub(" ", text).strip()

    @classmethod
    def _is_noise_segment(cls, text: str) -> bool:
        if len(text) < 6:
            return True
        if any(pattern.search(text) for pattern in _NEWS_NOISE_PATTERNS):
            return True
        if text.count(" ") > 0 and len(text.split()) <= 2 and len(text) < 16:
            return True
        return False

    @staticmethod
    def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            value = record.get(key)
            if value not in (None, ""):
                return value
        return None

    @classmethod
    def _build_event_records(
        cls,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> list[dict[str, Any]]:
        events = []
        for record in news_records[: request.max_news_records]:
            events.append({
                "event_id": record.get("event_id"),
                "event_type": record.get("event_type", "新闻"),
                "summary": record.get("summary", ""),
                "evidence_text": record.get("evidence_text") or record.get("summary", ""),
                "content_status": record.get("content_status", "summary_only"),
                "content_cleaning": record.get("content_cleaning", {}),
                "content_error": record.get("content_error"),
                "direction": record.get("direction", "中性"),
                "confidence": record.get("confidence", 0.5),
                "transmission_path": record.get("transmission_path", "新闻输入"),
                "direct_beneficiaries": record.get("direct_beneficiaries") or [request.sector_keyword or request.ticker],
                "evidence_level": "公开新闻",
                "pricing_status": "未定价",
                "chain_quality": record.get("chain_quality", "weak"),
                "event_time": record.get("event_time"),
                "source": record.get("source", ""),
                "url": record.get("url"),
                "news_scope": record.get("news_scope", "ticker"),
                "sector_name": record.get("sector_name"),
                "raw": record.get("raw", {}),
            })
        return events

    def _filter_news_with_llm(
        self,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> dict[str, Any]:
        if not news_records:
            return {
                "records": [],
                "trace": {
                    "mode": "empty",
                    "used_llm": False,
                    "reason": "no news records",
                    "input_count": 0,
                    "output_count": 0,
                },
            }

        if not request.use_llm_news_filter:
            records = self._filter_news_deterministically(news_records, request)
            return {
                "records": records,
                "trace": {
                    "mode": "deterministic",
                    "used_llm": False,
                    "reason": "LLM news filter disabled",
                    "input_count": len(news_records),
                    "output_count": len(records),
                },
            }

        try:
            llm = self._get_news_filter_llm()
            if not self._llm_news_filter_configured(llm):
                raise RuntimeError(f"{getattr(llm, 'provider', 'llm')} API key is not configured")
            decisions = self._ask_llm_to_filter_news(llm, news_records, request)
            selected: list[dict[str, Any]] = []
            decision_by_id = {str(item.get("event_id")): item for item in decisions}
            decision_trace: list[dict[str, Any]] = []
            for record in news_records:
                decision = decision_by_id.get(str(record.get("event_id")), {})
                try:
                    relevance = float(decision.get("relevance", 0))
                except (TypeError, ValueError):
                    relevance = 0.0
                keep = bool(decision.get("keep"))
                decision_trace.append({
                    "event_id": record.get("event_id"),
                    "title": record.get("title"),
                    "keep": keep,
                    "relevance": relevance,
                    "direction": decision.get("direction"),
                    "confidence": self._bounded_float(decision.get("confidence"), default=0.0),
                    "reason": str(decision.get("reason", ""))[:300],
                })
                if not keep or relevance < request.news_relevance_threshold:
                    continue
                selected.append({
                    **record,
                    "direction": decision.get("direction") or record.get("direction", "中性"),
                    "confidence": self._bounded_float(decision.get("confidence"), default=0.5),
                    "llm_relevance": relevance,
                    "llm_reason": str(decision.get("reason", ""))[:300],
                })
            guardrail_records = []
            if not selected:
                guardrail_records = self._filter_news_deterministically(news_records, request)
                for record in guardrail_records:
                    selected.append({
                        **record,
                        "llm_relevance": max(float(record.get("llm_relevance", 0.5)), request.news_relevance_threshold),
                        "llm_reason": "keyword guardrail after LLM filtered all candidates",
                    })
            return {
                "records": selected[: request.max_news_records],
                "trace": {
                    "mode": "llm" if not guardrail_records else "llm_with_keyword_guardrail",
                    "used_llm": True,
                    "model": getattr(llm, "model", ""),
                    "provider": getattr(llm, "provider", ""),
                    "input_count": len(news_records),
                    "output_count": len(selected[: request.max_news_records]),
                    "threshold": request.news_relevance_threshold,
                    "guardrail_added_count": len(guardrail_records),
                    "decisions": decision_trace,
                },
            }
        except Exception as exc:
            records = self._filter_news_deterministically(news_records, request)
            return {
                "records": records,
                "trace": {
                    "mode": "fallback",
                    "used_llm": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "input_count": len(news_records),
                    "output_count": len(records),
                    "threshold": request.news_relevance_threshold,
                },
            }

    def _get_news_filter_llm(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        from ..llm.client import create_llm

        return create_llm()

    def _llm_news_filter_configured(self, llm: Any) -> bool:
        if self._llm_client is not None:
            return True
        provider = str(getattr(llm, "provider", "deepseek")).upper()
        if provider == "DEEPSEEK":
            return bool(os.environ.get("DEEPSEEK_API_KEY"))
        if provider == "OPENAI":
            return bool(os.environ.get("OPENAI_API_KEY"))
        if provider == "ANTHROPIC":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        return bool(os.environ.get(f"{provider}_API_KEY"))

    @classmethod
    def _ask_llm_to_filter_news(
        cls,
        llm: Any,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> list[dict[str, Any]]:
        candidates = [
            {
                "event_id": record.get("event_id"),
                "title": record.get("title"),
                "summary": record.get("summary"),
                "evidence_text": str(record.get("evidence_text") or record.get("summary") or "")[:1200],
                "content_status": record.get("content_status"),
                "event_time": record.get("event_time"),
                "source": record.get("source"),
            }
            for record in news_records[: max(request.max_news_records * 3, request.max_news_records)]
        ]
        prompt = {
            "ticker": request.ticker,
            "trade_date": request.normalized_trade_date(),
            "news_keyword": request.news_keyword,
            "sector_keyword": request.sector_keyword,
            "task": (
                "Select news relevant to this ticker, its company, its business, or its sector for downstream trading agents. "
                "If sector_keyword is present, sector-level news is valid even when it does not mention the ticker directly. "
                "If title or summary directly contains the ticker keyword, company name, sector keyword, or clearly describes this company or its board, keep it unless it is obviously unrelated. "
                "Return one decision for every candidate in the same event_id space. "
                "Return only JSON with key decisions: list of objects containing event_id, keep, "
                "relevance from 0 to 1, direction in 正面/负面/中性, confidence from 0 to 1, and reason."
            ),
            "candidates": candidates,
        }
        response = llm.chat(
            [
                (
                    "system",
                    "你是量化交易数据管道里的新闻筛选器。只返回 JSON，不要输出解释文字。",
                ),
                ("human", json.dumps(prompt, ensure_ascii=False)),
            ],
            temperature=0,
            max_tokens=1200,
        )
        payload = json.loads(str(response))
        if isinstance(payload, list):
            decisions = payload
        elif isinstance(payload, dict):
            decisions = payload.get("decisions", [])
        else:
            decisions = []
        if not isinstance(decisions, list):
            raise ValueError("LLM news filter response must contain a decisions list")
        return [item for item in decisions if isinstance(item, dict)]

    @classmethod
    def _filter_news_deterministically(
        cls,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> list[dict[str, Any]]:
        keywords = [
            str(request.sector_keyword or "").lower(),
            str(request.news_keyword or "").lower(),
            str(request.ticker or "").lower(),
        ]
        keywords = [keyword for keyword in keywords if keyword]
        selected = []
        for record in news_records:
            haystack = json.dumps(record, ensure_ascii=False).lower()
            if not keywords or any(keyword in haystack for keyword in keywords):
                selected.append({
                    **record,
                    "llm_relevance": 0.5,
                    "llm_reason": "deterministic keyword fallback",
                })
        return selected[: request.max_news_records]

    @staticmethod
    def _bounded_float(value: Any, *, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, parsed))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> DataAgentArtifact:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload, default=self._json_default)
        return DataAgentArtifact(
            stage=str(payload.get("stage", path.parent.name)),
            path=str(path),
            record_count=self._record_count(payload),
            columns=self._columns(payload),
        )

    @staticmethod
    def _record_count(payload: dict[str, Any]) -> int | None:
        if isinstance(payload.get("daily"), list):
            return len(payload["daily"])
        daily = payload.get("daily")
        if isinstance(daily, dict) and "record_count" in daily:
            return int(daily["record_count"])
        return None

    @staticmethod
    def _columns(payload: dict[str, Any]) -> list[str]:
        daily = payload.get("daily")
        if isinstance(daily, dict):
            return list(daily.get("columns", []))
        return []

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    @staticmethod
    def _json_default(value: Any) -> Any:
        return _json_default(value)

    @staticmethod
    def _parse_number(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            if value in {"", "-", "--", "None", "nan"}:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _json_safe_record(cls, record: dict[str, Any]) -> dict[str, Any]:
        return {key: cls._json_safe_value(value) for key, value in record.items()}


def run_data_agent(
    ticker: str,
    *,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str | None = None,
    use_react_planner: bool = False,
    news_keyword: str | None = None,
    sector_keyword: str | None = None,
    use_llm_news_filter: bool = True,
    fetch_news_full_text: bool = True,
    include_sector_context: bool = True,
) -> DataAgentRun:
    """Convenience entry point for tests and CLI usage."""

    request = DataAgentRequest(
        ticker=ticker,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        use_react_planner=use_react_planner,
        news_keyword=news_keyword,
        sector_keyword=sector_keyword,
        use_llm_news_filter=use_llm_news_filter,
        fetch_news_full_text=fetch_news_full_text,
        include_sector_context=include_sector_context,
    )
    return DataAgent(results_dir=output_dir).run(request)
