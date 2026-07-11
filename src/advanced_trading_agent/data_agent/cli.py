"""Standalone CLI for running DataAgent only."""
from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from .data_agent import DataAgent, DataAgentRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataagent",
        description="Run DataAgent only: collect, clean, analyze, and persist layered data artifacts.",
    )
    parser.add_argument("--ticker", "-t", required=True, help="Ticker, e.g. 000001.SZ")
    parser.add_argument("--date", "-d", dest="trade_date", help="Trade date, e.g. 2026-07-10")
    parser.add_argument("--start-date", help="Start date, e.g. 20260701")
    parser.add_argument("--end-date", help="End date, e.g. 20260710")
    parser.add_argument("--output-dir", help="Output directory, default uses configured results_dir")
    parser.add_argument("--news-keyword", help="Keyword used by news collection and fallback filtering")
    parser.add_argument("--react-planner", action="store_true", help="Enable ReAct-style data planning trace")
    parser.add_argument(
        "--no-llm-news-filter",
        action="store_true",
        help="Disable LLM news screening and use deterministic keyword filtering only",
    )
    parser.add_argument("--no-market", action="store_true", help="Skip market index data")
    parser.add_argument("--no-capital-flow", action="store_true", help="Skip capital-flow data")
    parser.add_argument("--no-news", action="store_true", help="Skip news data")
    parser.add_argument("--no-factors", action="store_true", help="Skip factor calculation")
    parser.add_argument("--no-risk", action="store_true", help="Skip ST/suspended/delisting risk data")
    parser.add_argument("--max-news-records", type=int, default=20, help="Max news records to keep")
    parser.add_argument("--max-return-records", type=int, default=20, help="Max daily/factor records in outputs")
    parser.add_argument("--json", action="store_true", help="Print full DataAgentRun JSON")
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    request = DataAgentRequest(
        ticker=args.ticker,
        trade_date=args.trade_date,
        start_date=args.start_date,
        end_date=args.end_date,
        include_market=not args.no_market,
        include_capital_flow=not args.no_capital_flow,
        include_news=not args.no_news,
        include_factors=not args.no_factors,
        include_risk=not args.no_risk,
        news_keyword=args.news_keyword,
        use_llm_news_filter=not args.no_llm_news_filter,
        use_react_planner=args.react_planner,
        output_dir=args.output_dir,
        max_news_records=args.max_news_records,
        max_return_records=args.max_return_records,
    )
    result = DataAgent(results_dir=args.output_dir).run(request)
    return result.to_dict()


def summarize_run(payload: dict[str, Any]) -> dict[str, Any]:
    final_data = payload.get("final_data", {})
    cleaned = final_data.get("cleaned", {})
    analysis = final_data.get("analysis", {})
    events = analysis.get("events", {})
    data_quality = analysis.get("data_quality", {})
    vendor_health = final_data.get("vendor_health", {})
    return {
        "run_id": payload.get("run_id"),
        "response_path": payload.get("response_path"),
        "manifest_path": payload.get("manifest_path"),
        "daily_records": cleaned.get("daily", {}).get("record_count"),
        "news_records": cleaned.get("news", {}).get("record_count"),
        "event_records": events.get("record_count"),
        "news_filter": events.get("filter"),
        "daily_consistency": data_quality.get("daily_consistency"),
        "vendor_health": {
            "attempt_count": vendor_health.get("attempt_count", 0),
            "vendors": vendor_health.get("vendors", {}),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_from_args(args)
    output = payload if args.json else summarize_run(payload)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
