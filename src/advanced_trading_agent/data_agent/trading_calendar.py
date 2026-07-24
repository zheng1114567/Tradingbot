"""Small A-share trading-date helpers.

This intentionally handles the common no-session case first: weekends.  Public
holiday calendars can be layered in later through the same functions.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TypeAlias

DateLike: TypeAlias = str | date | datetime | None


def normalize_iso_date(value: DateLike = None) -> str:
    """Return *value* as YYYY-MM-DD, defaulting to today."""
    return _parse_date(value).isoformat()


def compact_date(value: DateLike = None) -> str:
    """Return *value* as YYYYMMDD."""
    return normalize_iso_date(value).replace("-", "")


def resolve_market_trade_date(value: DateLike = None) -> str:
    """Resolve an analysis date to the latest weekday market session.

    A-share daily/spot data is not produced on Saturday or Sunday.  If callers
    pass a weekend date, use the preceding Friday so data-source probes and
    roundtable inputs do not fail just because the default date landed on a
    closed weekend.
    """
    current = _parse_date(value)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current.isoformat()


def compact_market_trade_date(value: DateLike = None) -> str:
    """Return the resolved market trade date as YYYYMMDD."""
    return resolve_market_trade_date(value).replace("-", "")


def _parse_date(value: DateLike) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    raw = str(value).strip()
    clean = raw.replace("-", "")
    if len(clean) != 8 or not clean.isdigit():
        raise ValueError(f"Invalid date format: {value!r}. Expected YYYY-MM-DD or YYYYMMDD")
    return datetime.strptime(clean, "%Y%m%d").date()
