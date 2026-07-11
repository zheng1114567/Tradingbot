"""
Agent 模块初始化 — 重写版

每个 Agent 的工厂函数返回一个 LangGraph 可调用的节点函数。
"""
from ..llm.client import LLMClient
from .schemas import (
    DecisionType, RiskVerdict, Confidence,
    MarketReport, EventReport, AnalysisReport, StockRanking,
    BacktestReport, SystemDecision, MemoryRecall, FinalReport,
)
from .market_agent import create_market_agent
from .event_agent import create_event_agent
from .analysis_agent import create_analysis_agent
from .backtest_agent import create_backtest_agent
from .system_agent import create_system_agent
from .memory_agent import create_memory_agent
from .approval_agent import create_approval_agent
from .report_agent import create_report_agent

__all__ = [
    "create_market_agent", "create_event_agent",
    "create_analysis_agent", "create_backtest_agent",
    "create_system_agent", "create_memory_agent",
    "create_approval_agent", "create_report_agent",
]
