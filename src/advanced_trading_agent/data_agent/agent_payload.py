"""Build the compact Tier 1 / Tier 2 payload consumed by downstream agents."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of past signals to simulate for backtest samples
_BACKTEST_SAMPLE_COUNT = 60
# Gap between simulated signals (trading days)
_BACKTEST_SIGNAL_GAP = 5
# Holding horizon for backtest (trading days)
_BACKTEST_HOLDING_DAYS = 5


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
        "backtest_samples": _compute_backtest_samples(daily_records),
        "data_summary": summary,
        "data_quality": data_quality,
    }
    return tier1, tier2


def _compute_backtest_samples(
    daily_records: list[dict[str, Any]],
    *,
    holding_days: int = _BACKTEST_HOLDING_DAYS,
    sample_count: int = _BACKTEST_SAMPLE_COUNT,
    signal_gap: int = _BACKTEST_SIGNAL_GAP,
) -> list[dict[str, Any]]:
    """Compute point-in-time backtest samples from historical daily data.

    Simulates hypothetical past buy signals on sliding dates and computes
    the holding-period return for each, treating them as if the system
    had recommended the stock on that date.

    Returns a list ready to inject into ``tier2_data.backtest_samples``.
    """
    if not daily_records or len(daily_records) < holding_days + signal_gap:
        return []

    df = _prepare_price_df(daily_records)
    if df is None or len(df) < holding_days + signal_gap:
        return []

    # Generate signal dates: start from earliest date, step by signal_gap
    # Skip dates too close to the end (no exit price available)
    dates = df["trade_date"].tolist()
    signal_indices = list(range(0, len(dates) - holding_days, signal_gap))[-sample_count:]

    if not signal_indices:
        return []

    returns: list[float] = []
    bench_excess: list[float] = []
    tradable_count = 0

    for signal_idx in signal_indices:
        entry_idx = signal_idx + 1  # T+1
        exit_idx = entry_idx + holding_days

        if exit_idx >= len(df):
            continue

        entry_row = df.iloc[entry_idx]
        exit_row = df.iloc[exit_idx]

        # Skip untradable entries
        if entry_row.get("is_limit_up", False) or entry_row.get("is_limit_down", False):
            continue
        if entry_row.get("volume", 0) == 0:
            continue

        entry_price = float(entry_row.get("open", entry_row.get("close", 0)))
        exit_price = float(exit_row.get("close", exit_row.get("open", 0)))
        if entry_price <= 0:
            continue

        tradable_count += 1
        ret = (exit_price - entry_price) / entry_price
        returns.append(ret)

        # Excess vs benchmark (if available)
        bench_entry = entry_row.get("bench_close") if "bench_close" in df.columns else None
        bench_exit = exit_row.get("bench_close") if "bench_close" in df.columns else None
        if bench_entry is not None and bench_exit is not None and bench_entry > 0:
            bench_ret = (float(bench_exit) - float(bench_entry)) / float(bench_entry)
            bench_excess.append(ret - bench_ret)

    if not returns:
        return [{
            "sample_size": 0,
            "win_rate": 0.0,
            "avg_excess_return": 0.0,
            "holding_days": holding_days,
            "note": "no_tradable_signals",
        }]

    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    avg_return = float(np.mean(returns))
    avg_excess = float(np.mean(bench_excess)) if bench_excess else 0.0

    return [{
        "sample_size": len(returns),
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_return, 4),
        "avg_excess_return": round(avg_excess, 4),
        "holding_days": holding_days,
        "tradable_count": tradable_count,
        "computed_at": datetime.now().isoformat(),
    }]


def _prepare_price_df(records: list[dict[str, Any]]) -> pd.DataFrame | None:
    """Convert daily_records list to a DataFrame with required columns."""
    df = pd.DataFrame(records)
    if df.empty:
        return None

    required_cols = {"trade_date", "close"}
    if not required_cols.issubset(df.columns):
        return None

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    # Fill optional columns
    for col in ("open", "is_limit_up", "is_limit_down", "volume", "bench_close"):
        if col not in df.columns:
            if col in ("is_limit_up", "is_limit_down"):
                df[col] = False
            elif col == "volume":
                df[col] = 1
            elif col == "open":
                df[col] = df["close"]

    return df
