"""
LangGraph 工作流定义 — 借鉴 TradingAgents' setup.py + trading_graph.py

工作流拓扑:
  Memory Agent (历史召回)
    → Market Agent (市场温度)
    → Event Agent (事件分析)
    → Analysis Agent (因子评分)
    → Backtest Agent (历史证据)
    → System Agent (综合裁定 + 硬风控)
    → Report Agent (输出报告)

对比 TradingAgents:
  TradingAgents: 4 Analysts → Bull/Bear Researchers → Trader → 3 Risk Debators → PM
  我们的方案: Memory → Market → Event → Analysis → Backtest → System → Report

区别:
  1. 以数据工作流为中心, 而非交易辩论
  2. 硬风控在 System Agent 中由代码执行, 不在 LLM 风控辩论中
  3. 输出为"明日观察池"而非 Buy/Sell 信号
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..agents import (
    create_analysis_agent,
    create_backtest_agent,
    create_event_agent,
    create_market_agent,
    create_memory_agent,
    create_report_agent,
    create_system_agent,
)
from ..config import config
from ..llm.client import LLMClient
from .state import AgentState

logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:
    """创建并编译工作流图

    工作流:
    START → Memory → Market → Event → Analysis → Backtest → System → Report → END
    """
    # 初始化 LLM
    llm = LLMClient()

    # 创建 Agent 节点
    memory_node = create_memory_agent(llm)
    market_node = create_market_agent(llm)
    event_node = create_event_agent(llm)
    analysis_node = create_analysis_agent(llm)
    backtest_node = create_backtest_agent(llm)
    system_node = create_system_agent(llm)
    report_node = create_report_agent()

    # 构建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("Memory Agent", memory_node)
    workflow.add_node("Market Agent", market_node)
    workflow.add_node("Event Agent", event_node)
    workflow.add_node("Analysis Agent", analysis_node)
    workflow.add_node("Backtest Agent", backtest_node)
    workflow.add_node("System Agent", system_node)
    workflow.add_node("Report Agent", report_node)

    # 连线: 顺序执行
    workflow.add_edge(START, "Memory Agent")
    workflow.add_edge("Memory Agent", "Market Agent")
    workflow.add_edge("Market Agent", "Event Agent")
    workflow.add_edge("Event Agent", "Analysis Agent")
    workflow.add_edge("Analysis Agent", "Backtest Agent")
    workflow.add_edge("Backtest Agent", "System Agent")
    workflow.add_edge("System Agent", "Report Agent")
    workflow.add_edge("Report Agent", END)

    return workflow.compile()


class TradingSystem:
    """交易系统主入口 — 借鉴 TradingAgents' TradingAgentsGraph"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.workflow = create_workflow()

    def analyze(self, ticker: str, trade_date: str | None = None,
                tier1_data: dict[str, Any] | None = None,
                tier2_data: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
        """运行分析工作流

        Args:
            ticker: 股票代码 (如 "000001.SZ")
            trade_date: 交易日 (默认今天)
            tier1_data: Tier 1 数据 (可选)
            tier2_data: Tier 2 数据 (可选)

        Returns:
            (final_state, final_report_markdown)
        """
        from datetime import date
        trade_date = trade_date or str(date.today())

        # 初始化状态
        init_state = {
            "messages": [("human", f"分析 {ticker} {trade_date}")],
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "sender": "system",
            "tier1_data": tier1_data or {},
            "tier2_data": tier2_data or {},
            "memory_context": "",
            "memory_recall_obj": None,
            "market_report": "",
            "event_report": "",
            "analysis_report": "",
            "backtest_report": "",
            "market_report_obj": None,
            "event_report_obj": None,
            "analysis_report_obj": None,
            "backtest_report_obj": None,
            "system_decision_obj": None,
            "system_decision_state": "",
            "final_report": "",
            "final_report_obj": None,
        }

        # 执行
        if self.debug:
            final_state = None
            for chunk in self.workflow.stream(init_state):
                if self.debug:
                    node_name = list(chunk.keys())[0]
                    logger.info("Completed: %s", node_name)
                if chunk:
                    for k, v in chunk.items():
                        final_state = v if isinstance(v, dict) else final_state
        else:
            final_state = self.workflow.invoke(init_state)

        report = final_state.get("final_report", "") if final_state else ""
        return final_state or init_state, report
