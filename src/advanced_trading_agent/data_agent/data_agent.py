"""Standalone auditable data processor.

The agent records each stage of a data run:
1. input request
2. scan-collected raw data
3. scan-cleaned data
4. structured data processing
5. final layered response
"""
from __future__ import annotations

import logging
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import config
from ..core.atomic_write import atomic_write_json
from ..core.audit import audit_event, build_data_collection_summary
from .a_share_signals import AShareSignalBuilder
from .agent_payload import build_agent_payload
from .cache_manifest import CacheManifest
from .data_health import build_daily_health_report
from .factors import FactorCalculator
from .manifest import DataManifest
from .news_filter import (
    NewsFilter,
    ask_llm_to_filter_news,
    bounded_float,
    build_event_records,
    filter_news_deterministically,
)
from .news_text import select_evidence_text
from .planner import DataAgentPlanner
from .raw_collection import RawDataAdopter
from .request import DataAgentRequest
from .scan_cleaning import clean_news, first_present
from .scanner import MarketScanner, ScanBundle, ScanDataPackage
from .sector_context import clean_sector_context, summarize_sector_context
from .stock_profile import StockProfileResolver
from .vendor_router import get_vendor_chain, route_to_vendor

logger = logging.getLogger(__name__)


RouteFn = Callable[..., Any]


_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_path_part(value: str, fallback: str) -> str:
    safe = _SAFE_PATH_PART.sub("_", value).strip("._")
    return safe or fallback


def _parse_financial_value(value: Any) -> float | None:
    """Parse baostock financial values which may be strings with commas."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value != 0 else None
    text = str(value).strip().replace(",", "").replace(" ", "")
    if not text or text in ("0", "0.0", "None", "nan"):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


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
    """Process scan-owned market data into auditable downstream payloads."""

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
        return agent.run(request, scan_package=bundle.package_for_ticker(ticker))

    def run(
        self,
        request: DataAgentRequest,
        raw_data: dict[str, Any] | None = None,
        *,
        scan_package: ScanDataPackage | None = None,
    ) -> DataAgentRun:
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

        # --- Scan-owned collection and cleaning with error recovery ---
        try:
            if scan_package is not None:
                scan_package = self._adopt_scan_package(scan_package, request, manifest, route_trace)
            elif raw_data is not None:
                scan_package = self._package_from_scan_raw(raw_data, request, manifest, route_trace)
            else:
                scan_package = self._collect_scan_package(request, manifest, route_trace)
            raw_payload = scan_package.raw_payload
            cleaned_payload = scan_package.cleaned_payload
            artifacts["raw"] = self._write_json(run_dir / "02_raw" / "raw_data.json", raw_payload)
            artifacts["cleaned"] = self._write_json(run_dir / "03_cleaned" / "cleaned_data.json", cleaned_payload)
        except Exception as exc:
            logger.error("Scan input preparation failed: %s", exc)
            audit_trail.append(audit_event("data", f"Scan 数据准备失败: {exc}", level="error", detail={"stage": "scan_input"}))
            errors.append({"stage": "scan_input", "error": str(exc)})
            raw_payload = {"stage": "raw", "created_at": _utc_now(), "error": str(exc)}
            cleaned_payload = {"stage": "cleaned", "created_at": _utc_now(), "error": str(exc)}

        # --- Build collection summary ---
        vendor_health = self._summarize_vendor_health(route_trace)
        collection_summary = build_data_collection_summary(raw_payload, vendor_health, route_trace)
        audit_trail.append(audit_event("data", f"数据收集完成: {collection_summary.get('categories_with_data', 0)}/{collection_summary.get('total_categories', 0)} 类数据获取成功",
                                       detail={"failed": collection_summary.get('categories_failed', 0), "empty": collection_summary.get('categories_empty', 0)}))

        if collection_summary.get("categories_failed", 0) > 0:
            for key, cat in collection_summary.get("categories", {}).items():
                if cat["status"] == "error":
                    audit_trail.append(audit_event("data", f"{cat['label']} 获取失败: {cat.get('error', 'unknown')}", level="warning"))

        # --- Process with error recovery ---
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

    def _raw_adopter(self) -> RawDataAdopter:
        return RawDataAdopter(now_fn=_utc_now)

    def _adopt_raw(
        self,
        raw_data: dict[str, Any],
        request: DataAgentRequest,
        manifest: DataManifest,
        route_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._raw_adopter().adopt(raw_data, request, manifest, route_trace)

    def _package_from_scan_raw(
        self,
        raw_data: dict[str, Any],
        request: DataAgentRequest,
        manifest: DataManifest,
        route_trace: list[dict[str, Any]],
    ) -> ScanDataPackage:
        raw_payload = self._adopt_raw(raw_data, request, manifest, route_trace)
        return ScanDataPackage(
            raw_payload=raw_payload,
            cleaned_payload=MarketScanner.clean_data_agent_raw(raw_payload),
            route_trace=route_trace,
        )

    def _adopt_scan_package(
        self,
        scan_package: ScanDataPackage,
        request: DataAgentRequest,
        manifest: DataManifest,
        route_trace: list[dict[str, Any]],
    ) -> ScanDataPackage:
        if scan_package.route_trace and scan_package.route_trace is not route_trace:
            route_trace.extend(scan_package.route_trace)
        raw_payload = self._adopt_raw(scan_package.raw_payload, request, manifest, route_trace)
        return ScanDataPackage(
            raw_payload=raw_payload,
            cleaned_payload=scan_package.cleaned_payload,
            route_trace=route_trace,
        )

    def _collect_scan_package(
        self,
        request: DataAgentRequest,
        manifest: DataManifest,
        route_trace: list[dict[str, Any]],
    ) -> ScanDataPackage:
        scanner = self._scanner_for_collection()
        scan_package = scanner.collect_data_agent_package(request, route_trace)
        raw_payload = self._adopt_raw(scan_package.raw_payload, request, manifest, route_trace)
        return ScanDataPackage(
            raw_payload=raw_payload,
            cleaned_payload=MarketScanner.clean_data_agent_raw(raw_payload),
            route_trace=route_trace,
        )

    def _scanner_for_collection(self) -> MarketScanner:
        """Build the Scan boundary used by standalone DataAgent runs."""
        if self._route_fn is route_to_vendor:
            return MarketScanner(auto_refresh_cache=True)
        return MarketScanner(route_fn=self._route_fn, cache_only=False)

    def _collect_raw(
        self,
        request: DataAgentRequest,
        manifest: DataManifest,
        route_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compatibility wrapper: raw fetching is delegated to scan."""
        return self._collect_scan_package(request, manifest, route_trace).raw_payload

    def _clean(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Compatibility wrapper: cleaning is owned by scan."""
        return MarketScanner.clean_data_agent_raw(raw_payload)

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
            factor_df = FactorCalculator.run_all(self._enrich_daily_with_financials(daily_df.copy(), request.ticker))
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
            ),
            "daily_health": build_daily_health_report(
                daily_records,
                start_date=request.start_date,
                end_date=request.normalized_end_date(),
                cache_entry=self._daily_cache_manifest_entry(request.ticker),
            ),
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
        # Phase 0.5: inject A-share specialist signals into tier2
        a_share_signals = AShareSignalBuilder.build(tier2)
        tier2["a_share_signals"] = a_share_signals

        # LLM data quality review
        review = self._llm_review_data(tier1, tier2, request)

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
            "llm_review": review,
        }

    def _llm_review_data(
        self,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        request: DataAgentRequest,
    ) -> dict[str, Any]:
        """LLM sanity check on collected data quality."""
        try:
            from ..llm.client import create_llm

            # Build a compact summary for the LLM
            factors = tier2.get("factors", [])
            latest_factor = factors[-1] if factors else {}
            events = tier2.get("events", [])
            sector = tier1.get("sector", {})
            market = tier1.get("market", {})

            summary = {
                "ticker": request.ticker,
                "trade_date": request.normalized_trade_date(),
                "sector": sector.get("matched_sector"),
                "sector_confidence": sector.get("match_confidence"),
                "market_index": market.get("index_close"),
                "market_change": market.get("index_change_pct"),
                "events_count": len(events),
                "factors": {
                    k: latest_factor.get(k)
                    for k in ["roe", "profit_growth", "revenue_growth", "pe",
                              "momentum_20d", "composite_score"]
                    if latest_factor.get(k) is not None
                },
                "event_directions": [e.get("direction") for e in events[:5]],
                "signals": {
                    k: tier2.get("a_share_signals", {}).get(k, {}).get("signal")
                    for k in ["hot_money", "policy", "multifactor"]
                },
            }

            llm = create_llm()
            prompt = json.dumps(summary, ensure_ascii=False, indent=2)
            response = llm.chat(
                [
                    ("system",
                     "你是量化数据管道的质量审查员。检查以下数据摘要，找出异常或矛盾之处。"
                     "关注：因子值是否在合理范围、板块匹配是否可信、事件方向是否一致。"
                     "只返回 JSON：{\"ok\": true/false, \"issues\": [\"问题描述\"], \"confidence\": 0-1}。"
                     "如果没有问题，ok=true, issues=[]。50字以内。"),
                    ("human", prompt),
                ],
                temperature=0,
                max_tokens=200,
            )
            result = json.loads(str(response))
            return {
                "ok": bool(result.get("ok", True)),
                "issues": result.get("issues", []),
                "confidence": float(result.get("confidence", 0.8)),
            }
        except Exception as e:
            return {"ok": True, "issues": [], "confidence": 0.5, "error": str(e)[:100]}

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
        for risk_field in ["st_status", "suspended", "delisting"]:
            value = risk_raw.get(risk_field)
            if isinstance(value, list):
                lists[risk_field] = value
            else:
                lists[risk_field] = []
                if isinstance(value, dict) and value.get("error"):
                    errors.append(f"{risk_field}: {value['error']}")
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
        return build_agent_payload(
            summary=summary,
            market_summary=market_summary,
            sector_summary=sector_summary,
            capital_summary=capital_summary,
            risk_summary=risk_summary,
            daily_records=daily_records,
            factor_records=factor_records,
            event_records=event_records,
            dragon_tiger_records=dragon_tiger_records,
            data_quality=data_quality,
        )

    @classmethod
    def _clean_sector_context(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return clean_sector_context(records)

    @staticmethod
    def _enrich_daily_with_financials(daily_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Merge latest quarterly financial data into OHLCV for FactorCalculator."""
        try:
            import baostock as bs
            from datetime import date as dt_date

            parts = ticker.split(".")
            code = f"{parts[1].lower()}.{parts[0]}" if len(parts) == 2 else ticker
            today = dt_date.today()
            y, q = today.year, (today.month - 1) // 3 + 1

            bs.login()
            try:
                # Try quarters from current back to find latest available
                profit_rows, growth_rows = [], []
                for _ in range(3):
                    rs = bs.query_profit_data(code=code, year=y, quarter=q)
                    while rs.error_code == "0" and rs.next():
                        profit_rows.append(dict(zip(rs.fields, rs.get_row_data())))
                    if profit_rows:
                        break
                    q -= 1
                    if q == 0:
                        y -= 1
                        q = 4
                # Fetch growth data for the same quarter we found profit data
                if profit_rows:
                    latest_q = profit_rows[0]
                    stat_date = latest_q.get("statDate", "")
                    if stat_date:
                        sy, sq = int(stat_date[:4]), (int(stat_date[5:7]) - 1) // 3 + 1
                    else:
                        sy, sq = y, q
                    rs2 = bs.query_growth_data(code=code, year=sy, quarter=sq)
                    while rs2.error_code == "0" and rs2.next():
                        growth_rows.append(dict(zip(rs2.fields, rs2.get_row_data())))
            finally:
                bs.logout()

            if not profit_rows:
                return daily_df

            latest = profit_rows[0]
            growth = growth_rows[0] if growth_rows else {}

            def _g(key: str) -> float | None:
                return _parse_financial_value(growth.get(key))

            net_profit = _parse_financial_value(latest.get("netProfit"))
            roe = _parse_financial_value(latest.get("roeAvg"))
            eps = _parse_financial_value(latest.get("epsTTM"))
            gp_margin = _parse_financial_value(latest.get("gpMargin"))
            np_margin = _parse_financial_value(latest.get("npMargin"))
            total_share = _parse_financial_value(latest.get("totalShare"))
            revenue_raw = _parse_financial_value(latest.get("MBRevenue"))

            if np_margin and np_margin > 0 and net_profit and not revenue_raw:
                revenue_raw = net_profit / np_margin

            if net_profit is not None: daily_df["net_profit"] = net_profit
            if roe is not None:        daily_df["roe"] = roe
            if revenue_raw is not None: daily_df["revenue"] = revenue_raw
            if eps is not None:        daily_df["eps"] = eps
            if total_share is not None: daily_df["total_share"] = total_share
            if gp_margin is not None:  daily_df["gross_margin"] = gp_margin
            if np_margin is not None:  daily_df["net_margin"] = np_margin
            if eps and eps > 0 and "close" in daily_df.columns:
                daily_df["pe"] = pd.to_numeric(daily_df["close"], errors="coerce") / eps

            # Official YoY growth from baostock
            yoy_ni = _parse_financial_value(growth.get("YOYNI")) if growth else None
            if yoy_ni is not None:
                daily_df["profit_growth"] = yoy_ni
                daily_df["revenue_growth"] = yoy_ni  # proxy: no direct revenue growth in growth_data

        except Exception:
            pass
        return daily_df

    @classmethod
    def _summarize_sector_context(
        cls,
        records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> dict[str, Any]:
        stock_boards: list[str] | None = None
        try:
            import efinance as ef
            digits = request.ticker.split(".")[0]
            df = ef.stock.get_belong_board(digits)
            if df is not None and not df.empty:
                stock_boards = [str(b) for b in df.get("板块名称", []) if b]
        except Exception:
            pass

        return summarize_sector_context(
            records,
            sector_keyword=request.sector_keyword,
            news_keyword=request.news_keyword,
            ticker=request.ticker,
            sector_top_n=request.sector_top_n,
            stock_boards=stock_boards,
        )

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

    @staticmethod
    def _daily_cache_manifest_entry(ticker: str) -> dict[str, Any]:
        try:
            from .local_cache import LocalCache

            manifest = CacheManifest(LocalCache().cache_dir)
            entry = manifest.get_daily(ticker)
            return asdict(entry) if entry is not None else {}
        except Exception:
            return {}

    @classmethod
    def _clean_news(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return clean_news(records)

    @classmethod
    def _select_evidence_text(cls, full_text: Any, summary: Any, title: Any) -> str:
        return select_evidence_text(full_text, summary, title)

    @staticmethod
    def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
        return first_present(record, keys)

    @classmethod
    def _build_event_records(
        cls,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> list[dict[str, Any]]:
        return build_event_records(news_records, request)

    def _filter_news_with_llm(
        self,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> dict[str, Any]:
        return NewsFilter(llm_client=self._llm_client).filter(news_records, request)

    def _get_news_filter_llm(self) -> Any:
        return NewsFilter(llm_client=self._llm_client).get_llm()

    def _llm_news_filter_configured(self, llm: Any) -> bool:
        return NewsFilter(llm_client=self._llm_client).llm_configured(llm)

    @classmethod
    def _ask_llm_to_filter_news(
        cls,
        llm: Any,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> list[dict[str, Any]]:
        return ask_llm_to_filter_news(llm, news_records, request)

    @classmethod
    def _filter_news_deterministically(
        cls,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> list[dict[str, Any]]:
        return filter_news_deterministically(news_records, request)

    @staticmethod
    def _bounded_float(value: Any, *, default: float) -> float:
        return bounded_float(value, default=default)

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
