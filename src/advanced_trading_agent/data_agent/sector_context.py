"""Sector-context normalization and heuristic matching.

This module deliberately treats market-wide sector rankings separately from a
stock's official industry. Free data sources used here do not provide a
reliable point-in-time stock membership endpoint, so matches are heuristic.
"""

from __future__ import annotations

import re
from typing import Any


def clean_sector_context(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        sector_name = _first_present(record, ["sector_name", "板块名称", "行业", "name", "名称"]) or ""
        change_pct = _parse_number(_first_present(record, ["change_pct", "涨跌幅", "涨跌幅%", "change"])) or 0
        strength_score = _parse_number(record.get("strength_score"))
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


def summarize_sector_context(
    records: list[dict[str, Any]],
    *,
    sector_keyword: str | None,
    news_keyword: str | None,
    ticker: str,
    sector_top_n: int,
    stock_boards: list[str] | None = None,
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

    matched: dict[str, Any] | None = None
    strategy = "top_rank_fallback"
    confidence = 0.3
    direct_supported = False

    # 1. Direct board membership match via efinance (highest confidence)
    if stock_boards:
        direct_supported = True
        board_set = {b.strip().lower() for b in stock_boards if b.strip()}
        board_set.update(b.replace("板块", "").replace("概念", "").strip().lower() for b in stock_boards)
        for item in ranked:
            sector_name = str(item.get("sector_name") or "").strip().lower()
            if not sector_name:
                continue
            if sector_name in board_set or any(
                b in sector_name or sector_name in b for b in board_set
            ):
                matched = item
                strategy = "direct_board_match"
                confidence = 0.85
                break

    # 2. Keyword heuristic match (medium confidence)
    if matched is None:
        keywords = [
            ("sector_keyword", sector_keyword),
            ("news_keyword", news_keyword),
            ("ticker", ticker),
        ]
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

    # 3. Fallback: use stock's first actual board from efinance
    if matched is None and stock_boards:
        first_board = stock_boards[0]
        # Try to find it in the ranked hot sectors
        for item in ranked:
            if str(item.get("sector_name", "")).strip().lower() == first_board.lower():
                matched = item
                strategy = "direct_board_match"
                confidence = 0.70
                break
        if matched is None:
            matched = {"sector_name": first_board, "change_pct": 0, "strength_score": 0, "rank": 99, "source": "efinance_direct"}
            strategy = "direct_board_unranked"
            confidence = 0.60

    # 4. Last resort: top-ranked hot sector
    if matched is None:
        matched = ranked[0]

    return {
        "status": "matched" if strategy != "top_rank_fallback" else "fallback_top_sector",
        "matched_sector": matched.get("sector_name"),
        "match_confidence": confidence,
        "match_strategy": strategy,
        "direct_stock_sector_supported": direct_supported,
        "top_sectors": top_sectors,
        "records": ranked[:sector_top_n],
        "reason": (
            "Direct board lookup via efinance get_belong_board." if direct_supported
            else "Sector context is built from market-wide sector rankings. "
            "Without a paid/stock-membership endpoint, ticker-to-sector matching is heuristic."
        ),
    }


def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    if text.endswith("万"):
        multiplier = 10_000.0
        text = text[:-1]
    elif text.endswith("亿"):
        multiplier = 100_000_000.0
        text = text[:-1]
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        return None
    try:
        return float(text) * multiplier
    except ValueError:
        return None
