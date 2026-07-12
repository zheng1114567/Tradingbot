"""LangGraph tool whitelist for agent ReAct execution."""

from __future__ import annotations

from typing import Any

from .analysis_tools import check_crowding, get_factors, rank_stocks
from .backtest_tools import find_similar, run_backtest
from .event_tools import get_announcements, get_calendar, search_news
from .market_tools import (
    get_capital_flow,
    get_limit_up_tiers,
    get_market_sentiment,
    get_northbound_flow,
    get_sector_rotation,
)


_TOOLS_BY_NAME: dict[str, Any] = {
    "get_market_sentiment": get_market_sentiment,
    "get_northbound_flow": get_northbound_flow,
    "get_capital_flow": get_capital_flow,
    "get_sector_rotation": get_sector_rotation,
    "get_limit_up_tiers": get_limit_up_tiers,
    "search_news": search_news,
    "get_announcements": get_announcements,
    "get_calendar": get_calendar,
    "get_factors": get_factors,
    "rank_stocks": rank_stocks,
    "check_crowding": check_crowding,
    "run_backtest": run_backtest,
    "find_similar": find_similar,
}


AGENT_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "market": (
        "get_market_sentiment",
        "get_northbound_flow",
        "get_capital_flow",
        "get_sector_rotation",
        "get_limit_up_tiers",
    ),
    "event": ("search_news", "get_announcements", "get_calendar"),
    "analysis": ("get_factors", "rank_stocks", "check_crowding"),
    "backtest": ("run_backtest", "find_similar"),
}


def get_agent_tools(agent_key: str) -> list[Any]:
    """Return whitelisted tools for an agent."""
    normalized = agent_key.lower()
    try:
        tool_names = AGENT_TOOL_NAMES[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown agent tool whitelist: {agent_key}") from exc
    return [_TOOLS_BY_NAME[name] for name in tool_names]


def get_allowed_tool_names(agent_key: str) -> tuple[str, ...]:
    """Return whitelisted tool names for an agent (used in prompt building)."""
    normalized = agent_key.lower()
    return AGENT_TOOL_NAMES.get(normalized, ())
