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

from typing import Annotated, Any, TypedDict

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
    company_of_interest: Annotated[str, "分析标的代码"]
    trade_date: Annotated[str, "交易日"]
    sender: Annotated[str, "当前发言 Agent"]

    # === 运行模式 ===
    run_mode: Annotated[str, "live / backtest"]  # live: 真实 LLM, backtest: 规则模拟

    # === 数据层 ===
    tier1_data: Annotated[dict[str, Any], "Tier 1 数据"]
    tier2_data: Annotated[dict[str, Any], "Tier 2 数据"]
    data_quality_report: Annotated[Any, "数据质量报告"]
    pit_manifest: Annotated[Any, "Point-in-time manifest"]

    # === Memory (直接从 System Agent 注入) ===
    memory_context: Annotated[str, "历史上下文 (格式化为 prompt)"]
    memory_recall: Annotated[dict[str, Any], "Memory 召回结果 {success, failure, warnings}"]

    # === 风控 (三层结果) ===
    risk_check_1: Annotated[dict[str, Any], "硬风控 1: ST/停牌/退市 (分析前)"]
    risk_check_2: Annotated[dict[str, Any], "硬风控 2: 流动性/涨跌停 (Round1后)"]
    risk_check_3: Annotated[dict[str, Any], "硬风控 3: 冲击成本/仓位 (裁定前)"]

    # === Agent 报告 (字符串格式, 用于下游 LLM 读取) ===
    market_report: Annotated[str, "市场分析报告 (markdown)"]
    event_report: Annotated[str, "事件分析报告 (markdown)"]
    analysis_report: Annotated[str, "因子分析报告 (markdown)"]
    backtest_report: Annotated[str, "回测验证报告 (markdown)"]

    # === Agent 报告 (对象格式, 用于下游代码读取) ===
    market_report_obj: Annotated[Any, "MarketReport Pydantic"]
    event_report_obj: Annotated[Any, "EventReport Pydantic"]
    analysis_report_obj: Annotated[Any, "AnalysisReport Pydantic"]
    backtest_report_obj: Annotated[Any, "BacktestReport Pydantic"]

    # === Round 2 辩论 ===
    round2_state: Annotated[Round2State, "Round 2 交叉质询状态"]

    # === System 裁定 ===
    system_decision_obj: Annotated[Any, "SystemDecision Pydantic"]
    system_state: Annotated[str, "pending / completed / vetoed"]

    # === 最终输出 ===
    final_report: Annotated[str, "最终报告 Markdown"]
    final_report_obj: Annotated[Any, "FinalReport Pydantic"]
