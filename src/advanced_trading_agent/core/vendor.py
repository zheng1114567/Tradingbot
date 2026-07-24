"""Vendor call wrapper — timing and route_trace for DataAgent and MarketScanner."""

from __future__ import annotations

import time
from typing import Any, Callable


def timed_vendor_call(
    method: str,
    *,
    route_trace: list[dict[str, Any]] | None = None,
    route_fn: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> tuple[Any, float]:
    """Call a vendor method with timing and route_trace tracking.

    Returns:
        ``(result, elapsed_ms)``

    Raises:
        Any exception from the underlying vendor call.
    """
    from ..data_agent.vendor_router import route_to_vendor  # noqa: F811

    effective_route_fn = route_fn or route_to_vendor
    call_kwargs = dict(kwargs)
    if effective_route_fn is route_to_vendor and route_trace is not None:
        call_kwargs.setdefault("_route_trace", route_trace)

    start = time.perf_counter()
    try:
        result = effective_route_fn(method, **call_kwargs)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if route_trace is not None and effective_route_fn is not route_to_vendor:
            route_trace.append({
                "method": method,
                "vendor": "custom_route_fn",
                "status": "error",
                "error": str(exc),
            })
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    if route_trace is not None and effective_route_fn is not route_to_vendor:
        route_trace.append({
            "method": method,
            "vendor": "custom_route_fn",
            "status": "success",
            "elapsed_ms": round(elapsed_ms, 3),
            "record_count": len(result) if isinstance(result, list) else None,
        })
    return result, elapsed_ms
