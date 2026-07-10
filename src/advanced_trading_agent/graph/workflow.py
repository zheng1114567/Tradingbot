"""
LangGraph 工作流 — 重写版 (含 Round 2 交叉质询)

完整拓扑:

START
  → [硬风控1] ST/停牌/退市检查 (代码节点, 无 LLM)
    → HARD_VETO → END
    → PASS → [Memory Agent] 历史召回 → [Market Agent]
      → Market 冰点 → 跳过后续分析 → [Risk检2] → [Report]
      → Market 正常 → [Event Agent] → [Analysis Agent] → [Backtest Agent]
        → [硬风控2] 流动性/涨跌停检查
          → [System Agent] 判断是否进 Round 2
            → 有矛盾 → [Round 2 交叉质询] (≤8轮)
            → 无矛盾 → [硬风控3] 冲击成本/仓位检查
              → HARD_VETO → END
              → PASS → [System Agent 裁定] → [Report Agent] → [Memory存储] → END

对比 v0.1.0:
- 硬风控从 System Agent 内部移到独立代码节点
- 增加 Round 2 辩论
- Memory 从第一个节点移到 System Agent 前
- 冰点模式跳过深度分析省 token
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..agents import (
    create_analysis_agent,
    create_backtest_agent,
    create_event_agent,
    create_market_agent,
    create_memory_agent,
    create_report_agent,
)
from ..agents.system_agent import create_system_agent
from ..config import config
from ..llm.client import LLMClient
from .conditional import (
    after_market,
    after_risk_check_1,
    after_risk_check_3,
    after_round1,
    after_round2_turn,
)
from .risk_nodes import create_risk_check_1, create_risk_check_2, create_risk_check_3
from .state import AgentState

logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:
    """创建完整工作流图"""
    llm = LLMClient()

    # === 创建所有节点 ===

    # 硬风控节点 (代码, 无 LLM)
    risk1_node = create_risk_check_1()
    risk2_node = create_risk_check_2()
    risk3_node = create_risk_check_3()

    # Memory 节点 (直接注入 System Agent)
    memory_node = create_memory_agent(llm)

    # Agent 节点 (LLM + ToolNode)
    market_node = create_market_agent(llm)
    event_node = create_event_agent(llm)
    analysis_node = create_analysis_agent(llm)
    backtest_node = create_backtest_agent(llm)

    # System Agent (三个内部节点: init + round2_judge + final)
    sa = create_system_agent(llm)
    sa_init = sa["init"]          # 数据质量 + Memory + 硬风控1检查
    sa_round2_judge = sa["round2_judge"]  # 判断是否进 Round 2
    sa_final = sa["final"]        # 最终裁定

    # Report Agent (无 LLM)
    report_node = create_report_agent()

    # === 构建图 ===
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("Risk Check 1", risk1_node)
    workflow.add_node("Memory Agent", memory_node)
    workflow.add_node("Market Agent", market_node)
    workflow.add_node("Event Agent", event_node)
    workflow.add_node("Analysis Agent", analysis_node)
    workflow.add_node("Backtest Agent", backtest_node)
    workflow.add_node("Risk Check 2", risk2_node)
    workflow.add_node("System Init", sa_init)
    workflow.add_node("Round 2 Judge", sa_round2_judge)
    workflow.add_node("Round 2 Debate", _create_round2_node(llm))
    workflow.add_node("Risk Check 3", risk3_node)
    workflow.add_node("System Final Decision", sa_final)
    workflow.add_node("Report Agent", report_node)

    # === 连边 ===

    # START → 硬风控1
    workflow.add_edge(START, "Risk Check 1")

    # 硬风控1: HARD_VETO → END, PASS → 继续
    workflow.add_conditional_edges(
        "Risk Check 1",
        after_risk_check_1,
        {"round1": "Memory Agent", "end": END},
    )

    # Memory → Market
    workflow.add_edge("Memory Agent", "Market Agent")

    # Market Agent: 冰点 → 跳过后续深度分析
    workflow.add_conditional_edges(
        "Market Agent",
        after_market,
        {"continue_round1": "Event Agent", "skip_round1": "Risk Check 2"},
    )

    # Round 1 顺序
    workflow.add_edge("Event Agent", "Analysis Agent")
    workflow.add_edge("Analysis Agent", "Backtest Agent")

    # Round 1 完成 → 硬风控2
    workflow.add_edge("Backtest Agent", "Risk Check 2")

    # 硬风控2 → System Init (System Agent 加载+检查 Memory)
    workflow.add_edge("Risk Check 2", "System Init")

    # System Init → Round 2 Judge
    workflow.add_edge("System Init", "Round 2 Judge")

    # Round 2 Judge: 判断是否进辩论
    workflow.add_conditional_edges(
        "Round 2 Judge",
        after_round1,
        {"round2": "Round 2 Debate", "finalize": "Risk Check 3"},
    )

    # Round 2 辩论: 每轮结束后判断是否继续
    workflow.add_conditional_edges(
        "Round 2 Debate",
        after_round2_turn,
        {"continue_round2": "Round 2 Judge", "finalize": "Risk Check 3"},
    )

    # 硬风控3: HARD_VETO → END, PASS → 裁定
    workflow.add_conditional_edges(
        "Risk Check 3",
        after_risk_check_3,
        {"finalize": "System Final Decision", "end": END},
    )

    # 裁定 → 报告 → END
    workflow.add_edge("System Final Decision", "Report Agent")
    workflow.add_edge("Report Agent", END)

    return workflow.compile()


def _create_round2_node(llm: LLMClient):
    """创建 Round 2 交叉质询节点

    一轮交叉质询:
    1. System Agent 提出矛盾点
    2. 相关 Agent 被质询
    3. 被质询 Agent 回答
    4. 更新 round_count
    """
    def round2_node(state: AgentState) -> dict[str, Any]:
        round2 = state.get("round2_state", {})
        count = round2.get("round_count", 0)
        contradictions = round2.get("contradictions", [])
        max_rounds = round2.get("max_rounds", 8)

        # 当前轮质询
        if count >= max_rounds or not contradictions:
            return {
                "round2_state": {
                    **round2,
                    "completed": True,
                    "round_count": count,
                },
            }

        # 提出质询 (LLM 生成问题)
        contradiction = contradictions[count % len(contradictions)] if contradictions else ""
        market_rpt = state.get("market_report", "")
        event_rpt = state.get("event_report", "")
        analysis_rpt = state.get("analysis_report", "")
        backtest_rpt = state.get("backtest_report", "")

        prompt = f"""Round 2 交叉质询 - 第 {count + 1} 轮

发现的矛盾:
{contradiction}

各 Agent 报告:
Market: {market_rpt[:200]}
Event: {event_rpt[:200]}
Analysis: {analysis_rpt[:200]}
Backtest: {backtest_rpt[:200]}

请针对上述矛盾提出一个精准的质询问题。
质询必须基于数据矛盾, 不能是泛泛的问题。"""

        try:
            response = llm.chat([
                ("system", "你是在 Round 2 交叉质询中提出问题的 System Agent。"),
                ("human", prompt),
            ])
            question = response if isinstance(response, str) else str(response)
        except Exception:
            question = f"关于矛盾 '{contradiction[:50]}', 请相关 Agent 提供更多数据支撑。"

        return {
            "round2_state": {
                **round2,
                "round_count": count + 1,
                "current_speaker": "System",
                "completed": count + 1 >= max_rounds,
            },
        }

    return round2_node


class TradingSystem:
    """交易系统主入口"""

    def __init__(self, debug: bool = False, mode: str = "live"):
        self.debug = debug
        self.mode = mode  # "live" or "backtest"
        self.workflow = create_workflow()

    def analyze(self, ticker: str, trade_date: str | None = None,
                tier1_data: dict[str, Any] | None = None,
                tier2_data: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
        """运行分析工作流

        Args:
            ticker: 股票代码
            trade_date: 交易日 (默认今天)
            tier1_data: Tier 1 数据 (可选)
            tier2_data: Tier 2 数据 (可选)

        Returns:
            (final_state, final_report_markdown)
        """
        trade_date = trade_date or str(date.today())

        init_state = {
            "messages": [("human", f"分析 {ticker} {trade_date}")],
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "sender": "system",
            "run_mode": self.mode,
            "tier1_data": tier1_data or {},
            "tier2_data": tier2_data or {},
            "data_quality_report": None,
            "pit_manifest": None,
            "memory_context": "",
            "memory_recall": {},
            "risk_check_1": {},
            "risk_check_2": {},
            "risk_check_3": {},
            "market_report": "",
            "event_report": "",
            "analysis_report": "",
            "backtest_report": "",
            "market_report_obj": None,
            "event_report_obj": None,
            "analysis_report_obj": None,
            "backtest_report_obj": None,
            "round2_state": {
                "active": False,
                "round_count": 0,
                "max_rounds": 8,
                "questions": [],
                "contradictions": [],
                "current_speaker": "",
                "completed": False,
            },
            "system_decision_obj": None,
            "system_state": "",
            "final_report": "",
            "final_report_obj": None,
        }

        if self.debug:
            final_state = None
            for chunk in self.workflow.stream(init_state):
                node_name = list(chunk.keys())[0]
                logger.info("Completed node: %s", node_name)
                if chunk:
                    for k, v in chunk.items():
                        final_state = v if isinstance(v, dict) else final_state
        else:
            final_state = self.workflow.invoke(init_state)

        report = final_state.get("final_report", "") if final_state else ""
        return final_state or init_state, report
