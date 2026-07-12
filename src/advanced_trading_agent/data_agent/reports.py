"""Markdown reports for the scan -> data pipeline.

These reports are intentionally deterministic: LLM output is accepted as an
optional summary section, but the tables and tiered data inventory come from
persisted pipeline artifacts.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..core.atomic_write import atomic_write_text
from .data_agent import DataAgentRun
from .scanner import ScanBundle, ScanResult


def write_scan_report(
    bundle: ScanBundle,
    output_path: str | Path,
    *,
    llm_summary: str = "",
    llm_review: str = "",
    model: str = "deepseek-chat",
) -> str:
    """Write a standalone scan report and return its Markdown."""

    report = format_scan_report(bundle, llm_summary=llm_summary, llm_review=llm_review, model=model)
    atomic_write_text(Path(output_path), report)
    return report


def write_dataagent_report(
    runs: list[DataAgentRun],
    output_path: str | Path,
    *,
    bundle: ScanBundle | None = None,
    model: str = "deepseek-chat",
) -> str:
    """Write a standalone DataAgent layered report and return its Markdown."""

    report = format_dataagent_report(runs, bundle=bundle, model=model)
    atomic_write_text(Path(output_path), report)
    return report


def format_scan_report(
    bundle: ScanBundle,
    *,
    llm_summary: str = "",
    llm_review: str = "",
    model: str = "deepseek-chat",
) -> str:
    """Render scan results, collection readiness, and routing health."""

    results = bundle.results
    source_counts = Counter()
    sector_counts = Counter()
    for result in results:
        sector_counts[result.sector or "未识别"] += 1
        for source in result.source.split("+"):
            if source:
                source_counts[source] += 1

    lines = [
        "# Scan Report",
        "",
        f"- **Trade date**: {bundle.trade_date}",
        f"- **Candidates**: {len(results)}",
        f"- **LLM model**: {model}",
        f"- **Shared categories**: {_shared_readiness(bundle.shared_raw)}",
        f"- **Ticker datasets**: {len(bundle.ticker_data)} tickers collected",
        "",
    ]
    if llm_summary:
        lines.extend(["## LLM Market Summary", "", llm_summary.strip(), ""])
    if llm_review:
        lines.extend(["## LLM Candidate Review", "", llm_review.strip(), ""])

    lines.extend([
        "## Candidate Ranking",
        "",
        "| Rank | Ticker | Name | Score | Sector | Sources | Reason | Data Ready |",
        "|---:|---|---|---:|---|---|---|---|",
    ])
    if results:
        for idx, result in enumerate(results, start=1):
            data_ready = _ticker_readiness(bundle.ticker_data.get(result.ticker, {}))
            lines.append(
                "| "
                f"{idx} | {result.ticker} | {result.name or '-'} | {result.score:.1f} | "
                f"{result.sector or '-'} | {result.source or '-'} | {_trim(result.reason, 80)} | {data_ready} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | No candidates found | - |")

    lines.extend([
        "",
        "## Signal Mix",
        "",
        f"- **Top sectors**: {_counter_summary(sector_counts)}",
        f"- **Signal sources**: {_counter_summary(source_counts)}",
        "",
        "## Collection Health",
        "",
        "| Method | Attempts | Success | Error | Vendors |",
        "|---|---:|---:|---:|---|",
    ])
    for method, item in _route_summary(bundle.route_trace).items():
        lines.append(
            f"| {method} | {item['attempts']} | {item['success']} | "
            f"{item['error']} | {', '.join(item['vendors']) or '-'} |"
        )
    if not bundle.route_trace:
        lines.append("| - | 0 | 0 | 0 | - |")

    return "\n".join(lines).rstrip() + "\n"


def format_dataagent_report(
    runs: list[DataAgentRun],
    *,
    bundle: ScanBundle | None = None,
    model: str = "deepseek-chat",
) -> str:
    """Render DataAgent outputs with explicit Tier 1 / Tier 2 layering."""

    trade_date = bundle.trade_date if bundle else _first_request_value(runs, "trade_date")
    lines = [
        "# DataAgent Layered Report",
        "",
        f"- **Trade date**: {trade_date or '-'}",
        f"- **Runs**: {len(runs)}",
        f"- **LLM model**: {model}",
        "- **Scope**: scan/data only; downstream trading agents are not included.",
        "",
        "## Run Summary",
        "",
        "| Rank | Ticker | Name | Score | Daily | Factors | Events | Tier Status | Response |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]

    scan_by_ticker = {r.ticker: (idx, r) for idx, r in enumerate(bundle.results, start=1)} if bundle else {}
    for fallback_rank, run in enumerate(runs, start=1):
        final = run.final_data
        request = run.request
        ticker = request.get("ticker", "-")
        rank, scan_result = scan_by_ticker.get(ticker, (fallback_rank, None))
        analysis = final.get("analysis", {})
        cleaned = final.get("cleaned", {})
        tier_status = _tier_status(final.get("agent_payload", {}))
        lines.append(
            "| "
            f"{rank} | {ticker} | {(scan_result.name if scan_result else '-') or '-'} | "
            f"{(scan_result.score if scan_result else 0):.1f} | "
            f"{cleaned.get('daily', {}).get('record_count', 0)} | "
            f"{analysis.get('factors', {}).get('record_count', 0)} | "
            f"{analysis.get('events', {}).get('record_count', 0)} | "
            f"{tier_status} | {run.response_path} |"
        )

    lines.extend(["", "## Tier 1 Default Context", ""])
    for run in runs:
        ticker = run.request.get("ticker", "-")
        tier1 = run.final_data.get("agent_payload", {}).get("tier1_data", {})
        lines.extend([
            f"### {ticker}",
            "",
            "| Layer | Key Fields |",
            "|---|---|",
            f"| Market | {_market_summary(tier1.get('market', {}), tier1.get('sentiment', {}))} |",
            f"| Capital | {_capital_summary(tier1.get('capital', {}))} |",
            f"| Sector | {_sector_summary(tier1.get('sector', {}))} |",
            f"| Risk | {_risk_summary(tier1.get('risk', {}))} |",
            "",
        ])

    lines.extend(["## Tier 2 On-Demand Data", ""])
    for run in runs:
        ticker = run.request.get("ticker", "-")
        tier2 = run.final_data.get("agent_payload", {}).get("tier2_data", {})
        data_quality = tier2.get("data_quality", {}).get("daily_consistency", {})
        lines.extend([
            f"### {ticker}",
            "",
            "| Data Layer | Records / Status |",
            "|---|---|",
            f"| Price data | {len(tier2.get('price_data', []) or [])} records |",
            f"| Factors | {len(tier2.get('factors', []) or [])} records |",
            f"| Events | {len(tier2.get('events', []) or [])} records |",
            f"| Event scope | {_event_scope_summary(tier2.get('events', []) or [])} |",
            f"| Sector context | {tier2.get('sector_context', {}).get('status', 'unavailable')} |",
            f"| Data quality | {data_quality.get('status', 'unknown')} / confidence {data_quality.get('confidence_score', '-')} |",
            "",
        ])

    lines.extend(["## DataAgent Collection Health", ""])
    for run in runs:
        summary = run.collection_summary or {}
        lines.extend([
            f"### {run.request.get('ticker', '-')}",
            "",
            f"- **Categories with data**: {summary.get('categories_with_data', 0)}/{summary.get('total_categories', 0)}",
            f"- **Empty categories**: {summary.get('categories_empty', 0)}",
            f"- **Failed categories**: {summary.get('categories_failed', 0)}",
            f"- **Artifacts**: {', '.join(sorted(run.artifacts.keys()))}",
            "",
        ])

    return "\n".join(lines).rstrip() + "\n"


def _shared_readiness(shared_raw: dict[str, Any]) -> str:
    parts = []
    for key in ("market", "sector_context"):
        value = shared_raw.get(key, [])
        parts.append(f"{key}={len(value) if isinstance(value, list) else 0}")
    risk = shared_raw.get("risk", {})
    if isinstance(risk, dict):
        parts.append("risk=" + ",".join(f"{k}:{len(v) if isinstance(v, list) else 0}" for k, v in risk.items()))
    return "; ".join(parts) if parts else "none"


def _ticker_readiness(raw: dict[str, Any]) -> str:
    if not raw:
        return "missing"
    return ", ".join(f"{key}:{len(raw.get(key, [])) if isinstance(raw.get(key), list) else 0}" for key in ("daily", "capital_flow", "news"))


def _route_summary(route_trace: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for entry in route_trace:
        method = str(entry.get("method") or "unknown")
        item = summary.setdefault(method, {"attempts": 0, "success": 0, "error": 0, "vendors": set()})
        item["attempts"] += 1
        if entry.get("vendor"):
            item["vendors"].add(str(entry["vendor"]))
        status = str(entry.get("status", "")).lower()
        if status in {"ok", "success"}:
            item["success"] += 1
        elif status == "error":
            item["error"] += 1
    return {
        method: {**item, "vendors": sorted(item["vendors"])}
        for method, item in sorted(summary.items())
    }


def _counter_summary(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}({count})" for key, count in counter.most_common(5))


def _trim(value: str, limit: int) -> str:
    value = str(value or "").replace("|", "/").replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "..."


def _tier_status(agent_payload: dict[str, Any]) -> str:
    tier1_ok = bool(agent_payload.get("tier1_data"))
    tier2_ok = bool(agent_payload.get("tier2_data"))
    if tier1_ok and tier2_ok:
        return "tier1+tier2"
    if tier1_ok:
        return "tier1 only"
    if tier2_ok:
        return "tier2 only"
    return "missing"


def _first_request_value(runs: list[DataAgentRun], key: str) -> Any:
    for run in runs:
        value = run.request.get(key)
        if value:
            return value
    return None


def _market_summary(market: dict[str, Any], sentiment: dict[str, Any]) -> str:
    return (
        f"close={market.get('index_close', 0)}, "
        f"chg={market.get('index_change_pct', 0)}%, "
        f"sentiment={sentiment.get('sentiment', '未知')}({sentiment.get('sentiment_score', '-')})"
    )


def _capital_summary(capital: dict[str, Any]) -> str:
    return (
        f"confirmation={capital.get('confirmation', '未知')}, "
        f"net_main={capital.get('net_inflow_main', 0)}"
    )


def _sector_summary(sector: dict[str, Any]) -> str:
    return (
        f"status={sector.get('status', 'unavailable')}, "
        f"matched={sector.get('matched_sector') or '-'}, "
        f"confidence={sector.get('match_confidence', 0)}"
    )


def _risk_summary(risk: dict[str, Any]) -> str:
    return (
        f"available={risk.get('risk_data_available', False)}, "
        f"limit_up={risk.get('is_limit_up', False)}, "
        f"limit_down={risk.get('is_limit_down', False)}"
    )


def _event_scope_summary(events: list[dict[str, Any]]) -> str:
    if not events:
        return "none"
    scopes = Counter(str(item.get("news_scope", "ticker")) for item in events)
    return ", ".join(f"{scope}({count})" for scope, count in scopes.most_common())
