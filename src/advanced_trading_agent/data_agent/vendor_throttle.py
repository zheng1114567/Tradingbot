"""Conservative throttling for free market-data vendors.

The goal is not to bypass vendor controls. It is to keep this project from
bursting public endpoints during cache builds and to fail over cleanly when a
source starts rejecting requests.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


class VendorCoolingDownError(RuntimeError):
    """Raised when a vendor is in local cooldown after repeated failures."""


@dataclass
class _VendorState:
    last_call_at: float = 0.0
    consecutive_failures: int = 0
    blocked_until: float = 0.0


_LOCK = threading.Lock()
_STATE: dict[str, _VendorState] = {}

_DEFAULT_MIN_INTERVALS = {
    "akshare": 3.0,
    "eastmoney": 3.0,
    "eastmoney_global": 3.0,
    "efinance": 1.5,
    "sina": 1.5,
    "cls": 1.5,
    "tencent": 1.0,
    "baostock": 0.35,
    "mootdx": 0.25,
    "free_http": 1.0,
    "local_cache": 0.0,
}
_DEFAULT_JITTERS = {
    "akshare": (0.8, 2.5),
    "eastmoney": (0.8, 2.5),
    "eastmoney_global": (0.8, 2.5),
    "efinance": (0.4, 1.2),
    "sina": (0.4, 1.2),
    "cls": (0.4, 1.2),
    "tencent": (0.2, 0.8),
    "baostock": (0.1, 0.35),
    "mootdx": (0.05, 0.2),
    "free_http": (0.25, 0.75),
    "local_cache": (0.0, 0.0),
}
_FAILURE_THRESHOLD = 3
_COOLDOWN_SECONDS = 180.0


def _state_for(vendor: str) -> _VendorState:
    return _STATE.setdefault(vendor, _VendorState())


def _rate_limit_like(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "429",
        "403",
        "rate limit",
        "too many",
        "blocked",
        "proxyerror",
        "max retries",
        "remote end closed",
        "connection reset",
        "timed out",
        "read timed out",
    )
    return any(marker in text for marker in markers)


def throttle_vendor(
    vendor: str,
    *,
    min_interval: float | None = None,
    jitter: tuple[float, float] | None = None,
) -> None:
    """Sleep before a public vendor call and enforce local cooldowns."""
    vendor_key = (vendor or "free_http").lower()
    min_interval = _DEFAULT_MIN_INTERVALS.get(vendor_key, _DEFAULT_MIN_INTERVALS["free_http"]) if min_interval is None else min_interval
    jitter = _DEFAULT_JITTERS.get(vendor_key, _DEFAULT_JITTERS["free_http"]) if jitter is None else jitter

    with _LOCK:
        state = _state_for(vendor_key)
        now = time.monotonic()
        if state.blocked_until > now:
            remaining = state.blocked_until - now
            raise VendorCoolingDownError(f"{vendor_key} cooling down for {remaining:.0f}s")
        wait = max(0.0, state.last_call_at + min_interval - now)
        if jitter[1] > 0:
            wait += random.uniform(jitter[0], jitter[1])
        state.last_call_at = now + wait

    if wait > 0:
        time.sleep(wait)


def record_vendor_success(vendor: str) -> None:
    vendor_key = (vendor or "free_http").lower()
    with _LOCK:
        state = _state_for(vendor_key)
        state.consecutive_failures = 0
        state.blocked_until = 0.0


def record_vendor_failure(vendor: str, exc: Exception) -> None:
    vendor_key = (vendor or "free_http").lower()
    if not _rate_limit_like(exc):
        return
    with _LOCK:
        state = _state_for(vendor_key)
        state.consecutive_failures += 1
        if state.consecutive_failures >= _FAILURE_THRESHOLD:
            state.blocked_until = time.monotonic() + _COOLDOWN_SECONDS


def call_with_vendor_guard(vendor: str, fn: Callable[[], T]) -> T:
    """Run one vendor call under throttle and circuit-breaker accounting."""
    throttle_vendor(vendor)
    try:
        result = fn()
    except Exception as exc:
        record_vendor_failure(vendor, exc)
        raise
    record_vendor_success(vendor)
    return result


def reset_vendor_throttle() -> None:
    """Test helper: clear in-memory throttling state."""
    with _LOCK:
        _STATE.clear()
