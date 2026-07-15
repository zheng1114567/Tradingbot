"""Raw scan-payload adoption for DataAgent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .manifest import DataManifest
from .request import DataAgentRequest


NowFn = Callable[[], str]


class RawDataAdopter:
    """Adopt scan-owned raw data while updating DataAgent lineage.

    MarketScanner owns vendor fetching and article-text enrichment. DataAgent
    receives that payload, records provenance, then moves on to data processing.
    """

    def __init__(self, *, now_fn: NowFn) -> None:
        self._now_fn = now_fn

    def adopt(
        self,
        raw_data: dict[str, Any],
        request: DataAgentRequest,
        manifest: DataManifest,
        route_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Normalize scan-collected raw data into the persisted raw layer."""
        if "route_trace" in raw_data:
            incoming = raw_data["route_trace"]
            if isinstance(incoming, list) and incoming is not route_trace:
                for attempt in incoming:
                    if attempt not in route_trace:
                        route_trace.append(attempt)

        daily = raw_data.get("daily", [])
        market = raw_data.get("market", [])
        sector_context = raw_data.get("sector_context", [])
        limit_up_summary = raw_data.get("limit_up_summary", {})
        dragon_tiger = raw_data.get("dragon_tiger", [])
        market_breadth = raw_data.get("market_breadth", {})
        capital_flow = raw_data.get("capital_flow", [])
        news = raw_data.get("news", [])
        risk = raw_data.get("risk", {})

        field_checks = [
            ("stock.daily", daily),
            ("market.daily", market),
            ("sector.context", sector_context),
            ("market.limit_up_summary", [limit_up_summary] if isinstance(limit_up_summary, dict) and limit_up_summary else []),
            ("market.dragon_tiger", dragon_tiger),
            ("market.breadth", [market_breadth] if isinstance(market_breadth, dict) and market_breadth else []),
            ("stock.capital_flow", capital_flow),
            ("news.events", news),
        ]
        for field_name, value in field_checks:
            available = isinstance(value, list) and len(value) > 0
            manifest.add_field(
                field_name,
                available=available,
                source="scan",
                vendor_chain=[],
                record_count=len(value) if isinstance(value, list) else None,
            )

        return {
            "stage": "raw",
            "created_at": self._now_fn(),
            "source_stage": "scan",
            "requested_full_text": request.fetch_news_full_text,
            "daily": daily,
            "market": market,
            "sector_context": sector_context,
            "limit_up_summary": limit_up_summary if isinstance(limit_up_summary, dict) else {},
            "dragon_tiger": dragon_tiger if isinstance(dragon_tiger, list) else [],
            "market_breadth": market_breadth if isinstance(market_breadth, dict) else {},
            "capital_flow": capital_flow,
            "news": news,
            "risk": risk if isinstance(risk, dict) else {},
            "route_trace": route_trace,
        }


RawDataCollector = RawDataAdopter
