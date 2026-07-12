"""
Run scan/data pipeline with a cache-first workflow.

Workflow:
1. Build free local cache snapshots.
2. Scan from local cache only.
3. Cache news/risk locally for shortlisted tickers.
4. Feed cached raw data into DataAgent and write reports.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

MODEL = "deepseek-chat"
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "data" / "results"

os.environ.setdefault("ATA_LLM_PROVIDER", "deepseek")
os.environ["ATA_DEEP_THINK_LLM"] = MODEL
os.environ["ATA_QUICK_THINK_LLM"] = MODEL
os.environ.setdefault("ATA_RESULTS_DIR", str(RESULTS_ROOT))

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(RESULTS_ROOT / "runtime.log"), encoding="utf-8"),
    ],
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from advanced_trading_agent.config import config
from advanced_trading_agent.data_agent.build_cache import (
    build,
    cache_news_for_sector,
    cache_risk_snapshot,
    ensure_candidate_daily_cache,
)
from advanced_trading_agent.data_agent.data_agent import DataAgent, DataAgentRequest
from advanced_trading_agent.core.atomic_write import atomic_write_text, atomic_write_json
from advanced_trading_agent.data_agent.local_cache import (
    get_cached_daily,
    get_cached_dragon_tiger,
    get_cached_limit_up,
    get_cached_market_breadth,
    get_cached_news,
    get_cached_risk_snapshot,
    get_cached_sector_data,
    get_cached_sector_constituents,
    get_cached_sector_news,
)
from advanced_trading_agent.data_agent.reports import write_dataagent_report, write_scan_report
from advanced_trading_agent.data_agent.scanner import MarketScanner, ScanBundle
from advanced_trading_agent.data_agent.vendor_router import ensure_default_vendor_registration, get_vendor_impl

TRADE_DATE = "2026-07-10"
TOP_N = 3
TOP_SECTORS = 5


def _configure_vendors() -> None:
    config.update(
        {
            "data_vendors": {
                "market_data": "local_cache,mootdx,baostock",
                "fundamental_data": "baostock",
                "news_data": "sina,cls",
                "capital_flow": "local_cache",
                "a_share_specific": "local_cache,efinance,eastmoney",
                "analysis": "baostock",
                "risk_data": "local_cache,baostock",
            }
        }
    )


def _call_vendor(method: str, vendor: str, **kwargs: Any) -> Any:
    ensure_default_vendor_registration()
    impl = get_vendor_impl(method, vendor)
    if impl is None:
        raise RuntimeError(f"missing vendor impl: {vendor}/{method}")
    return impl(**kwargs)


def _local_scan_route(method: str, **kwargs: Any) -> Any:
    if method in {"get_sector", "get_sector_constituents", "get_limit_up_tiers", "get_dragon_tiger"}:
        try:
            return _call_vendor(method, "local_cache", **kwargs)
        except Exception:
            pass
    if method == "get_limit_up_tiers":
        return {"first_board": 0, "second_board": 0, "third_plus": 0, "stocks": []}
    return []


def _build_cached_raw_data(ticker: str, trade_date: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    route_trace: list[dict[str, Any]] = []

    def _record(method: str, vendor: str, payload: Any) -> Any:
        status = "success" if payload else "no_data"
        route_trace.append({"method": method, "vendor": vendor, "status": status})
        return payload

    market = _record(
        "get_daily",
        "local_cache",
        get_cached_daily("000001.SH", start_date=trade_date, end_date=trade_date),
    )
    sector_context = _record(
        "get_sector",
        "local_cache",
        get_cached_sector_data(trade_date=trade_date, top_n=20),
    )
    daily = _record(
        "get_daily",
        "local_cache",
        get_cached_daily(ticker, end_date=trade_date),
    )

    try:
        capital_flow = _record(
            "get_capital_flow",
            "local_cache",
            _call_vendor("get_capital_flow", "local_cache", code=ticker, end_date=trade_date, trade_date=trade_date),
        )
    except Exception as exc:
        route_trace.append({"method": "get_capital_flow", "vendor": "local_cache", "status": "error", "error": str(exc)})
        capital_flow = []

    news = _record(
        "get_news",
        "local_cache",
        get_cached_news(ticker, trade_date=trade_date),
    )
    risk_snapshot = _record(
        "get_risk_snapshot",
        "local_cache",
        get_cached_risk_snapshot(trade_date=trade_date),
    )
    limit_up_summary = _record(
        "get_limit_up_tiers",
        "local_cache",
        get_cached_limit_up(trade_date=trade_date),
    )
    dragon_tiger = _record(
        "get_dragon_tiger",
        "local_cache",
        get_cached_dragon_tiger(trade_date=trade_date),
    )
    market_breadth = _record(
        "get_market_breadth",
        "local_cache",
        get_cached_market_breadth(trade_date=trade_date),
    )

    raw_data = {
        "daily": daily if isinstance(daily, list) else [],
        "market": market if isinstance(market, list) else [],
        "sector_context": sector_context if isinstance(sector_context, list) else [],
        "limit_up_summary": limit_up_summary if isinstance(limit_up_summary, dict) else {},
        "dragon_tiger": dragon_tiger if isinstance(dragon_tiger, list) else [],
        "market_breadth": market_breadth if isinstance(market_breadth, dict) else {},
        "capital_flow": capital_flow if isinstance(capital_flow, list) else [],
        "news": news if isinstance(news, list) else [],
        "risk": risk_snapshot if isinstance(risk_snapshot, dict) else {},
        "route_trace": route_trace,
    }
    return raw_data, route_trace


def _build_sector_cached_raw_data(ticker: str, sector_name: str, trade_date: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_data, route_trace = _build_cached_raw_data(ticker, trade_date)
    sector_news = get_cached_sector_news(sector_name, trade_date=trade_date)
    raw_data["news"] = sector_news if isinstance(sector_news, list) else []
    route_trace.append(
        {
            "method": "get_sector_news",
            "vendor": "local_cache",
            "status": "success" if raw_data["news"] else "no_data",
            "sector": sector_name,
        }
    )
    return raw_data, route_trace


def main() -> None:
    _configure_vendors()

    print(f"\n{'=' * 60}")
    print(f"Cache-First Scan/Data Run | {TRADE_DATE} | Model: {MODEL}")
    print(f"{'=' * 60}\n")

    results_dir = RESULTS_ROOT / f"data_run_{TRADE_DATE}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building free local cache...")
    cache_dir = build(trade_date=TRADE_DATE, output_dir=str(RESULTS_ROOT), compute_signals=False)
    print(f"  Cache dir: {cache_dir}")

    print("\n[2/4] Scanning from local cache...")
    scanner = MarketScanner(top_sectors=TOP_SECTORS, top_n=TOP_N, route_fn=_local_scan_route)
    results = scanner.scan(TRADE_DATE)
    if not results:
        print("No results from local scan.")
        return

    route_trace: list[dict[str, Any]] = []
    shared_raw = scanner.collect_shared_data(TRADE_DATE, route_trace=route_trace)
    bundle = ScanBundle(
        trade_date=TRADE_DATE,
        results=results[:TOP_N],
        shared_raw=shared_raw,
        ticker_data={},
        route_trace=route_trace,
    )
    for result in bundle.results:
        print(f"  {result.ticker} {result.name}: score={result.score:.1f} ({result.reason})")

    print("  Ensuring candidate daily cache...")
    daily_statuses = ensure_candidate_daily_cache(
        [result.ticker for result in bundle.results],
        TRADE_DATE,
        output_dir=str(RESULTS_ROOT),
    )
    for item in daily_statuses:
        print(
            f"    {item['ticker']}: {item['status']} "
            f"({item.get('source') or '-'}, rows={item.get('record_count', 0)})"
        )

    llm_summary = scanner.summarize_with_llm(bundle.results)
    llm_review = scanner.review_with_llm(bundle.results)
    scan_report_path = results_dir / f"scan_report_{TRADE_DATE}.md"
    write_scan_report(bundle, scan_report_path, llm_summary=llm_summary, llm_review=llm_review, model=MODEL)
    print(f"  Scan report: {scan_report_path}")

    print("\n[3/4] Caching ticker news/risk and running DataAgent...")
    data_runs = []
    summaries = []
    for idx, result in enumerate(bundle.results, start=1):
        print(f"  [{idx}/{len(bundle.results)}] {result.ticker} {result.name}")

        sector_name = result.sector or "未识别板块"
        sector_constituents = get_cached_sector_constituents(sector_name, trade_date=TRADE_DATE)
        constituent_tickers = [str(item.get("code", "")) for item in sector_constituents[:5] if item.get("code")]
        cache_news_for_sector(
            sector_name,
            trade_date=TRADE_DATE,
            output_dir=str(RESULTS_ROOT),
            constituent_tickers=constituent_tickers,
        )
        cache_risk_snapshot(trade_date=TRADE_DATE, output_dir=str(RESULTS_ROOT))

        raw_data, trace = _build_sector_cached_raw_data(result.ticker, sector_name, TRADE_DATE)
        bundle.route_trace.extend(trace)
        bundle.ticker_data[result.ticker] = {
            "daily": raw_data["daily"],
            "capital_flow": raw_data["capital_flow"],
            "news": raw_data["news"],
        }

        run = DataAgent(results_dir=str(results_dir)).run(
            DataAgentRequest(
                ticker=result.ticker,
                trade_date=TRADE_DATE,
                start_date=TRADE_DATE,
                end_date=TRADE_DATE,
                include_market=True,
                include_capital_flow=True,
                include_news=True,
                include_factors=True,
                include_risk=True,
                use_react_planner=False,
                use_llm_news_filter=True,
                fetch_news_full_text=False,
                news_keyword=sector_name,
                sector_keyword=sector_name,
            ),
            raw_data=raw_data,
        )
        data_runs.append(run)

        final = run.final_data
        cleaned = final.get("cleaned", {})
        analysis = final.get("analysis", {})
        summaries.append(
            {
                "ticker": result.ticker,
                "name": result.name,
                "score": result.score,
                "response_path": run.response_path,
                "daily_records": cleaned.get("daily", {}).get("record_count", 0),
                "news_records": cleaned.get("news", {}).get("record_count", 0),
                "factor_records": analysis.get("factors", {}).get("record_count", 0),
                "event_records": analysis.get("events", {}).get("record_count", 0),
                "sector_match": analysis.get("sector", {}).get("matched_sector", "N/A"),
                "capital_confirmation": analysis.get("capital", {}).get("confirmation", "N/A"),
            }
        )

    print("\n[4/4] Writing layered reports...")
    summary_path = results_dir / "run_summary.json"
    atomic_write_json(summary_path, summaries)
    data_report_path = results_dir / f"dataagent_layered_report_{TRADE_DATE}.md"
    write_dataagent_report(data_runs, data_report_path, bundle=bundle, model=MODEL)

    print(f"  Summary: {summary_path}")
    print(f"  DataAgent report: {data_report_path}")
    print(f"\n{'=' * 60}")
    print(f"Results dir: {results_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
