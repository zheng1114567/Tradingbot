"""Data quality checks for local and vendor-sourced market records."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def build_daily_health_report(
    records: list[dict[str, Any]],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact health report for daily OHLCV-like records."""
    issues: list[dict[str, Any]] = []
    if not records:
        return {
            "status": "unavailable",
            "score": 0.0,
            "row_count": 0,
            "issues": [{"code": "no_records", "severity": "error", "message": "No daily records available"}],
            "cache": cache_entry or {},
        }

    frame = pd.DataFrame(records)
    date_col = _date_column(frame)
    if date_col is None:
        return {
            "status": "invalid_schema",
            "score": 0.2,
            "row_count": int(len(frame)),
            "issues": [{"code": "missing_date_column", "severity": "error", "message": "No date column found"}],
            "cache": cache_entry or {},
        }

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    invalid_dates = int(frame[date_col].isna().sum())
    if invalid_dates:
        issues.append({
            "code": "invalid_dates",
            "severity": "error",
            "count": invalid_dates,
            "message": f"{invalid_dates} records have invalid dates",
        })
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)

    duplicate_count = int(frame.duplicated(subset=[date_col]).sum())
    if duplicate_count:
        issues.append({
            "code": "duplicate_dates",
            "severity": "warning",
            "count": duplicate_count,
            "message": f"{duplicate_count} duplicate trade dates found",
        })

    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column not in frame.columns:
            continue
        null_count = int(pd.to_numeric(frame[column], errors="coerce").isna().sum())
        if null_count:
            issues.append({
                "code": f"null_{column}",
                "severity": "warning",
                "count": null_count,
                "message": f"{null_count} null/invalid {column} values",
            })

    pct_col = "pct_chg" if "pct_chg" in frame.columns else "pctChg" if "pctChg" in frame.columns else None
    if pct_col:
        pct = pd.to_numeric(frame[pct_col], errors="coerce").abs()
        abnormal_count = int((pct > 25).sum())
        if abnormal_count:
            issues.append({
                "code": "abnormal_pct_change",
                "severity": "warning",
                "count": abnormal_count,
                "message": f"{abnormal_count} daily changes exceed 25%",
            })

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    observed_start = frame[date_col].dt.date.min()
    observed_end = frame[date_col].dt.date.max()
    if start and observed_start and observed_start > start:
        issues.append({
            "code": "start_gap",
            "severity": "warning",
            "message": f"First cached date {observed_start} is after requested start {start}",
        })
    if end and observed_end and observed_end < end:
        issues.append({
            "code": "end_gap",
            "severity": "warning",
            "message": f"Last cached date {observed_end} is before requested end {end}",
        })

    severities = {issue["severity"] for issue in issues}
    if "error" in severities:
        status = "invalid"
        score = 0.3
    elif issues:
        status = "warning"
        score = 0.75
    else:
        status = "ok"
        score = 0.95

    return {
        "status": status,
        "score": score,
        "row_count": int(len(frame)),
        "date_column": date_col,
        "start_date": str(observed_start) if observed_start else None,
        "end_date": str(observed_end) if observed_end else None,
        "issues": issues,
        "cache": cache_entry or {},
    }


def _date_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("trade_date", "datetime", "date"):
        if candidate in frame.columns:
            return candidate
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    raw = str(value)
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    try:
        return pd.to_datetime(raw).date()
    except (TypeError, ValueError):
        return None
