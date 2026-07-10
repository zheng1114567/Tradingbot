"""
LangGraph State — 重写版

借鉴 TradingAgents' agent_states.py 的结构,
但增加了:
1. Round 2 辩论状态 (cross_questioning_state)
2. 三层硬风控结果 (risk_check_1/2/3)
3. 数据质量报告 (data_quality_report)
4. Memory 直接注入 System Agent (memory_context)
5. 运行模式 (live / backtest)
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import MessagesState


class QuestionItem(TypedDict):
    """Round 2 交叉质询中的单个问题"""
    source_agent: str   # 提问 Agent
    target_agent: str   # 被质询 Agent
    question: str
    answer: str | None
    data_source: str  # 质询基于的数据矛盾


class Round2State(TypedDict):
    """Round 2 辩论状态"""
    active: bool                 # 是否启动 Round 2
    round_count: int             # 当前轮数
    max_rounds: int              # 最大轮数 (8)
    questions: list[QuestionItem]
    contradictions: list[str]    # 发现的矛盾摘要
    current_speaker: str         # 当前发言 Agent
    completed: bool              # 是否完成


class AgentState(MessagesState):
    """扩展的 Agent State — 重写版"""

    # === 基本信息 ===
    company_of_interest: str
    trade_date: str
    sender: str

    # === 运行模式 ===
    run_mode: str  # live / backtest

    # === 数据层 ===
    tier1_data: dict
    tier2_data: dict
    data_quality_report: Any | None
    pit_manifest: Any | None

    # === Memory ===
    memory_context: str
    memory_recall: dict

    # === 风控 ===
    risk_check_1: dict
    risk_check_2: dict
    risk_check_3: dict

    # === Agent 报告 (字符串格式) ===
    market_report: str
    event_report: str
    analysis_report: str
    backtest_report: str

    # === Agent 报告 (对象格式) ===
    market_report_obj: Any | None
    event_report_obj: Any | None
    analysis_report_obj: Any | None
    backtest_report_obj: Any | None

    # === Round 2 辩论 ===
    round2_state: Round2State

    # === System 裁定 ===
    system_decision_obj: Any | None
    system_state: str

    # === 最终输出 ===
    final_report: str
    final_report_obj: Any | None
