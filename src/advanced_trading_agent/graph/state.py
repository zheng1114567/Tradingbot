"""
LangGraph Agent State — 借鉴 TradingAgents' agent_states.py

定义工作流中所有 Agent 共享的状态结构。
"""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """扩展的 Agent State — 包含所有 Agent 的输入和输出"""

    # --- 基本信息 ---
    company_of_interest: Annotated[str, "分析标的"]
    trade_date: Annotated[str, "交易日"]
    sender: Annotated[str, "当前发言 Agent"]

    # --- Tier 数据 ---
    tier1_data: Annotated[dict[str, Any], "Tier 1 数据 (大盘摘要)"]
    tier2_data: Annotated[dict[str, Any], "Tier 2 数据 (个股明细)"]

    # --- Memory ---
    memory_context: Annotated[str, "历史上下文"]
    memory_recall_obj: Annotated[Any, "Memory 召回结果"]

    # --- Agent 报告 (字符串格式, 用于 LLM 读取) ---
    market_report: Annotated[str, "市场分析报告"]
    event_report: Annotated[str, "事件分析报告"]
    analysis_report: Annotated[str, "因子分析报告"]
    backtest_report: Annotated[str, "回测验证报告"]

    # --- Agent 报告 (对象格式, 用于下游读取) ---
    market_report_obj: Annotated[Any, "市场报告对象"]
    event_report_obj: Annotated[Any, "事件报告对象"]
    analysis_report_obj: Annotated[Any, "因子报告对象"]
    backtest_report_obj: Annotated[Any, "回测报告对象"]

    # --- 系统裁定 ---
    system_decision_obj: Annotated[Any, "系统裁定对象"]
    system_decision_state: Annotated[str, "裁定状态"]

    # --- 最终输出 ---
    final_report: Annotated[str, "最终报告 Markdown"]
    final_report_obj: Annotated[Any, "最终报告对象"]
