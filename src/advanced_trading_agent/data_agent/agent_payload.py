"""Build the compact Tier 1 / Tier 2 payload consumed by downstream agents."""

from __future__ import annotations

from typing import Any


def build_agent_payload(
    *,
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
    """Return the stable DataAgent contract for workflow agents."""
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
