"""
Agent 模块初始化 — 重写版

每个 Agent 的工厂函数返回一个 LangGraph 可调用的节点函数。

Note: agent factories are lazy-imported to break a circular chain:
  tool_nodes.registry → backtest_tools → backtest.review → agents.memory_agent
  → agents.__init__ → market_agent → tool_nodes.registry
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy factory resolution to break circular imports."""
    _LAZY: dict[str, str] = {
        "create_market_agent": ".market_agent",
        "create_event_agent": ".event_agent",
        "create_analysis_agent": ".analysis_agent",
        "create_backtest_agent": ".backtest_agent",
        "create_system_agent": ".system_agent",
        "create_memory_agent": ".memory_agent",
        "create_approval_agent": ".approval_agent",
        "create_report_agent": ".report_agent",
    }
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __package__)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "create_market_agent", "create_event_agent",
    "create_analysis_agent", "create_backtest_agent",
    "create_system_agent", "create_memory_agent",
    "create_approval_agent", "create_report_agent",
]
