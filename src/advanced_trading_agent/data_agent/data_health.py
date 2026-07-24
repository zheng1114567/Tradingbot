"""Data quality checks for local and vendor-sourced market records."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from .trading_calendar import resolve_market_trade_date


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


def run_data_source_health(
    trade_date: str | None = None,
    *,
    route_fn: Any | None = None,
) -> dict[str, Any]:
    """Probe the minimum data sources needed before running the strategy.

    This is intentionally shallow: it verifies that the routing layer can get
    representative A-share daily, sector, limit-up, and ETF spot data, and it
    returns route traces so failures are attributable to a vendor.
    """
    from .vendor_router import route_to_vendor

    td = resolve_market_trade_date(trade_date)
    route = route_fn or route_to_vendor
    probes = [
        {
            "name": "a_share_daily",
            "method": "get_daily",
            "kwargs": {"code": "000001.SZ", "start_date": td, "end_date": td},
            "required": True,
        },
        {
            "name": "sector_ranking",
            "method": "get_sector",
            "kwargs": {"top_n": 3, "trade_date": td},
            "required": True,
        },
        {
            "name": "limit_up_pool",
            "method": "get_limit_up_tiers",
            "kwargs": {"trade_date": td},
            "required": False,
        },
        {
            "name": "ticker_news",
            "method": "get_news",
            "kwargs": {"code": "000001.SZ", "keyword": "平安银行", "trade_date": td, "limit": 5},
            "required": False,
        },
        {
            "name": "sector_news",
            "method": "get_news",
            "kwargs": {"sector": "半导体", "keyword": "半导体", "trade_date": td, "limit": 5},
            "required": False,
        },
        {
            "name": "etf_spot",
            "method": "get_etf_spot",
            "kwargs": {"trade_date": td},
            "required": True,
        },
    ]

    results: dict[str, Any] = {}
    for probe in probes:
        trace: list[dict[str, Any]] = []
        result: Any = None
        error: str | None = None
        try:
            result = _call_route(route, probe["method"], trace, **probe["kwargs"])
        except Exception as exc:
            error = str(exc)
        results[probe["name"]] = _summarize_probe_result(
            result,
            trace,
            error=error,
            required=bool(probe["required"]),
        )

    required = [item for item in results.values() if item["required"]]
    if all(item["status"] == "ok" for item in required):
        overall = "ok"
    elif any(item["status"] == "ok" for item in required):
        overall = "degraded"
    else:
        overall = "unavailable"
    news_probe_names = ("ticker_news", "sector_news")
    news_ok = sum(1 for name in news_probe_names if results.get(name, {}).get("status") == "ok")
    daily_rows = int(results.get("a_share_daily", {}).get("row_count") or 0)
    return {
        "requested_date": trade_date or date.today().isoformat(),
        "trade_date": td,
        "effective_trade_date": td,
        "overall_status": overall,
        "coverage": {
            "daily_probe_rows": daily_rows,
            "daily_coverage_status": "ok" if daily_rows else "missing",
            "news_probe_count": len(news_probe_names),
            "news_ok_count": news_ok,
            "news_coverage_status": "ok" if news_ok == len(news_probe_names) else "partial" if news_ok else "missing",
            "ticker_news_status": results.get("ticker_news", {}).get("status", "unavailable"),
            "sector_news_status": results.get("sector_news", {}).get("status", "unavailable"),
        },
        "probes": results,
    }


def refresh_etf_cache(
    trade_date: str | None = None,
    *,
    etf_codes: list[str] | None = None,
    daily_limit: int = 20,
    route_fn: Any | None = None,
) -> dict[str, Any]:
    """Refresh ETF spot data and a small ETF daily cache slice.

    If *etf_codes* is omitted, the function picks the most liquid ETF rows from
    the fetched spot universe. The underlying akshare adapter writes through to
    local cache on success.
    """
    from .vendor_router import route_to_vendor

    td = resolve_market_trade_date(trade_date)
    route = route_fn or route_to_vendor
    trace: list[dict[str, Any]] = []
    spot_error: str | None = None
    spot_rows: list[dict[str, Any]] = []
    try:
        spot_result = _call_route(route, "get_etf_spot", trace, trade_date=td)
        if isinstance(spot_result, list):
            spot_rows = [row for row in spot_result if isinstance(row, dict)]
        elif isinstance(spot_result, str):
            spot_error = spot_result
    except Exception as exc:
        spot_error = str(exc)

    selected_codes = list(dict.fromkeys(etf_codes or _select_liquid_etf_codes(spot_rows, daily_limit)))
    daily_results: list[dict[str, Any]] = []
    for code in selected_codes[:daily_limit]:
        item_trace: list[dict[str, Any]] = []
        try:
            rows = _call_route(
                route,
                "get_etf_daily",
                item_trace,
                code=code,
                start_date=td,
                end_date=td,
            )
            row_count = len(rows) if isinstance(rows, list) else 0
            daily_results.append({
                "code": code,
                "status": "ok" if row_count else "unavailable",
                "row_count": row_count,
                "route_trace": item_trace,
            })
        except Exception as exc:
            daily_results.append({
                "code": code,
                "status": "error",
                "error": str(exc),
                "route_trace": item_trace,
            })

    return {
        "trade_date": td,
        "spot": {
            "status": "ok" if spot_rows else "unavailable",
            "row_count": len(spot_rows),
            "error": spot_error,
        },
        "daily": {
            "requested": len(selected_codes[:daily_limit]),
            "success_count": sum(1 for item in daily_results if item["status"] == "ok"),
            "results": daily_results,
        },
        "route_trace": trace,
    }


def _call_route(route_fn: Any, method: str, route_trace: list[dict[str, Any]], **kwargs: Any) -> Any:
    try:
        return route_fn(method, _route_trace=route_trace, **kwargs)
    except TypeError:
        return route_fn(method, **kwargs)


def _summarize_probe_result(
    result: Any,
    route_trace: list[dict[str, Any]],
    *,
    error: str | None,
    required: bool,
) -> dict[str, Any]:
    row_count = len(result) if isinstance(result, list) else len(result) if isinstance(result, dict) else 0
    is_sentinel = isinstance(result, str) and result.startswith("NO_DATA_AVAILABLE")
    status = "ok" if row_count > 0 and not is_sentinel and not error else "unavailable"
    successful = [attempt for attempt in route_trace if attempt.get("status") == "success"]
    failed_before_success = bool(successful and route_trace and route_trace[0] is not successful[0])
    return {
        "status": status,
        "required": required,
        "row_count": row_count,
        "fallback_used": failed_before_success,
        "success_vendor": successful[-1].get("vendor") if successful else None,
        "error": error or result if is_sentinel else error,
        "route_trace": route_trace,
    }


def _select_liquid_etf_codes(rows: list[dict[str, Any]], limit: int) -> list[str]:
    def amount(row: dict[str, Any]) -> float:
        value = row.get("amount")
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    sorted_rows = sorted(rows, key=amount, reverse=True)
    codes = [str(row.get("code") or row.get("raw_code") or "") for row in sorted_rows]
    return [code for code in codes if code][:limit]


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
