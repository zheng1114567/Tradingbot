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
    create_approval_agent,
    create_backtest_agent,
    create_event_agent,
    create_market_agent,
    create_memory_agent,
    create_report_agent,
)
from ..agents.contract import basic_self_check, build_agent_update, build_node_audit_update
from ..agents.schemas import BacktestReport, Confidence
from ..agents.system_agent import create_system_agent
from ..config import config
from ..data_agent.data_agent import DataAgent, DataAgentRequest
from ..llm.client import LLMClient
from ..roundtable import AutoGenRoundtable, RoundtableHarness
from .conditional import (
    after_market,
    after_risk_check_1,
    after_risk_check_3,
    after_round1,
)
from .risk_nodes import create_risk_check_1, create_risk_check_2, create_risk_check_3
from .state import AgentState

logger = logging.getLogger(__name__)


def _create_skip_backtest_node():
    """Create an auditable placeholder when backtest is explicitly skipped."""

    def skip_backtest_node(state: dict[str, Any]) -> dict[str, Any]:
        report = BacktestReport(
            sample_size=0,
            win_rate=0,
            avg_excess_return=0,
            failure_pattern="用户参数 skip_backtest=True，跳过回测证据审查",
            confidence=Confidence.LOW,
            reasoning="本次运行明确跳过 Backtest Agent；最终裁定不得将回测作为支持项。",
        )
        evidence = [
            "skip_backtest=True",
            "sample_size=0",
            "confidence=low",
        ]
        return build_agent_update(
            state,
            sender="Backtest Agent",
            report_key="backtest_report",
            report=report.to_markdown(),
            report_obj_key="backtest_report_obj",
            report_obj=report,
            evidence=evidence,
            tool_calls=[],
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=["backtest_explicitly_skipped"],
                warnings=["回测已跳过，不能作为推荐依据"],
                confidence="low",
            ),
        )

    return skip_backtest_node


def _after_analysis(state: AgentState) -> str:
    if state.get("skip_backtest"):
        return "skip_backtest"
    return "backtest"


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
    skip_backtest_node = _create_skip_backtest_node()

    # System Agent (三个内部节点: init + round2_judge + final)
    sa = create_system_agent(llm)
    sa_init = sa["init"]          # 数据质量 + Memory + 硬风控1检查
    sa_round2_judge = sa["round2_judge"]  # 判断是否进 Round 2
    sa_final = sa["final"]        # 最终裁定

    # Report Agent (无 LLM)
    approval_node = create_approval_agent()
    report_node = create_report_agent()

    # === 构建图 ===
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("Risk Check 1", risk1_node)
    workflow.add_node("System Init", sa_init)
    workflow.add_node("Memory Agent", memory_node)
    workflow.add_node("Market Agent", market_node)
    workflow.add_node("Event Agent", event_node)
    workflow.add_node("Analysis Agent", analysis_node)
    workflow.add_node("Backtest Agent", backtest_node)
    workflow.add_node("Skip Backtest", skip_backtest_node)
    workflow.add_node("Risk Check 2", risk2_node)
    workflow.add_node("Round 2 Judge", sa_round2_judge)
    workflow.add_node("Round 2 Debate", _create_round2_subgraph(llm))
    workflow.add_node("Risk Check 3", risk3_node)
    workflow.add_node("System Final Decision", sa_final)
    workflow.add_node("Approval Agent", approval_node)
    workflow.add_node("Report Agent", report_node)

    # === 连边 ===

    # START → 硬风控1
    workflow.add_edge(START, "Risk Check 1")

    # 硬风控1: HARD_VETO → END, PASS → System Init
    workflow.add_conditional_edges(
        "Risk Check 1",
        after_risk_check_1,
        {"round1": "System Init", "end": END},
    )

    # System Init → Memory → Market
    workflow.add_edge("System Init", "Memory Agent")
    workflow.add_edge("Memory Agent", "Market Agent")

    # Market Agent: 冰点 → 跳过后续深度分析
    workflow.add_conditional_edges(
        "Market Agent",
        after_market,
        {"continue_round1": "Event Agent", "skip_round1": "Risk Check 2"},
    )

    # Round 1 顺序
    workflow.add_edge("Event Agent", "Analysis Agent")
    workflow.add_conditional_edges(
        "Analysis Agent",
        _after_analysis,
        {"backtest": "Backtest Agent", "skip_backtest": "Skip Backtest"},
    )

    # Round 1 完成 → 硬风控2
    workflow.add_edge("Backtest Agent", "Risk Check 2")
    workflow.add_edge("Skip Backtest", "Risk Check 2")

    # 硬风控2 → Round 2 Judge
    workflow.add_edge("Risk Check 2", "Round 2 Judge")

    # Round 2 Judge -> Round 2 subgraph -> Risk Check 3
    workflow.add_conditional_edges(
        "Round 2 Judge",
        after_round1,
        {"round2": "Round 2 Debate", "finalize": "Risk Check 3"},
    )
    workflow.add_edge("Round 2 Debate", "Risk Check 3")

    # 硬风控3: HARD_VETO → 直接生成报告, PASS → 进入裁定
    workflow.add_conditional_edges(
        "Risk Check 3",
        after_risk_check_3,
        {"finalize": "System Final Decision", "end": "Report Agent"},
    )

    # 裁定 → 人工审批记录 → 报告 → END
    workflow.add_edge("System Final Decision", "Approval Agent")
    workflow.add_edge("Approval Agent", "Report Agent")
    workflow.add_edge("Report Agent", END)

    return workflow.compile()


def _create_round2_subgraph(llm: LLMClient) -> StateGraph:
    """Round 2 辩论子图 — 内部管理多轮辩论循环

    strict_autogen 模式:
    - Round 2 只允许 AutoGen 执行
    - AutoGen 失败时不再回退到 DebateEngine
    - 失败信息写入 round2_state, 由上游显式暴露
    """

    def debate_turn(state: AgentState) -> dict[str, Any]:
        """执行一轮辩论"""
        round2 = state.get("round2_state", {})
        count = round2.get("round_count", 0)
        max_rounds = round2.get("max_rounds", 8)

        contradictions_raw = round2.get("contradiction_records", [])
        if not contradictions_raw:
            return {
                "round2_state": {
                    **round2,
                    "completed": True,
                    "round_count": count,
                },
            }

        autogen_roundtable = AutoGenRoundtable(harness=RoundtableHarness())
        try:
            autogen_result = autogen_roundtable.run(state)
            result = {
                "round_count": autogen_result.round_count or max(count, 1),
                "completed": True,
                "current_speaker": "System_Moderator",
                "summary": autogen_result.summary,
                "questions": autogen_result.questions,
                "final_pressure": autogen_result.final_pressure,
                "unresolved_conflicts": autogen_result.unresolved_conflicts,
                "provider": autogen_result.provider,
                "fallback_reason": autogen_result.fallback_reason,
                "contradiction_records": contradictions_raw,
                "evidence_board": autogen_result.evidence_board,
                "round_history": autogen_result.round_history,
                "moderator_output": autogen_result.moderator_output,
            }
            tool_calls = [
                {"agent": agent_name, **call}
                for agent_name, calls in autogen_result.tool_calls_by_agent.items()
                for call in calls
            ]
            evidence = [
                "provider=autogen",
                f"contradictions={len(contradictions_raw)}",
                f"round_count={result['round_count']}",
                f"tool_events={len(tool_calls)}",
            ]
            return build_node_audit_update(
                sender="Round 2 Debate",
                round2_state={**round2, **result},
                evidence=evidence,
                tool_calls=tool_calls,
                self_check=basic_self_check(
                    evidence=evidence,
                    passed_rules=["autogen_roundtable_executed", "roundtable_tools_whitelisted"],
                    warnings=[],
                    confidence=autogen_result.final_pressure,
                ),
            )
        except Exception as e:
            logger.error("AutoGen roundtable failed in strict_autogen mode, round %d: %s", count, e)
            result = {
                "round_count": count + 1,
                "completed": True,
                "current_speaker": "",
                "summary": f"Round {count + 1} AutoGen 圆桌失败: {e}",
                "final_pressure": "neutral",
                "unresolved_conflicts": [
                    c.get("description", str(c))
                    for c in contradictions_raw
                ],
                "provider": "autogen",
                "fallback_reason": f"strict_autogen_failed: {type(e).__name__}: {e}",
                "contradiction_records": contradictions_raw,
                "evidence_board": round2.get("evidence_board", []),
                "round_history": round2.get("round_history", []),
                "moderator_output": None,
            }
            evidence = [
                "provider=autogen",
                "mode=strict_autogen",
                f"autogen_failed={type(e).__name__}",
                f"contradictions={len(contradictions_raw)}",
            ]
            return build_node_audit_update(
                sender="Round 2 Debate",
                round2_state={**round2, **result},
                evidence=evidence,
                tool_calls=[],
                self_check=basic_self_check(
                    evidence=evidence,
                    passed_rules=["strict_autogen_enforced"],
                    warnings=[f"AutoGen roundtable failed: {e}"],
                    confidence="neutral",
                ),
            )

    def after_turn(state: AgentState) -> str:
        round2 = state.get("round2_state", {})
        if round2.get("completed") or round2.get("round_count", 0) >= round2.get("max_rounds", 8):
            return "finalize"
        return "continue"

    builder = StateGraph(AgentState)
    builder.add_node("debate_turn", debate_turn)
    builder.add_edge(START, "debate_turn")
    builder.add_conditional_edges(
        "debate_turn",
        after_turn,
        {"continue": "debate_turn", "finalize": END},
    )
    return builder.compile()


class TradingSystem:
    """交易系统主入口"""

    def __init__(self, debug: bool = False, mode: str = "live"):
        self.debug = debug
        self.mode = mode  # "live" or "backtest"
        self.workflow = create_workflow()

    @staticmethod
    def _load_data(ticker: str, trade_date: str) -> tuple[dict, dict]:
        """自动加载 Tier 1 / Tier 2 数据, 返回 (tier1_data, tier2_data)

        统一通过 DataAgent 采集、清洗、分析并生成后续 Agent 消费结构。
        """
        result = DataAgent().run(
            DataAgentRequest(
                ticker=ticker,
                trade_date=trade_date,
                start_date=trade_date,
                end_date=trade_date,
                include_market=True,
                include_capital_flow=True,
                include_news=True,
                include_factors=True,
                include_risk=True,
                use_react_planner=True,
            )
        )
        payload = result.final_data.get("analysis", {}).get("agent_payload", {})
        tier1 = payload.get("tier1_data", {})
        tier2 = payload.get("tier2_data", {})
        tier1["_data_manifest"] = result.final_data.get("manifest")
        tier1["_data_manifest_path"] = result.manifest_path
        tier1["_data_agent_run"] = result.to_dict()
        tier1["_collection_summary"] = result.collection_summary
        tier1["_audit_trail"] = result.audit_trail
        tier1["_errors"] = result.final_data.get("errors", [])
        return tier1, tier2

    def analyze(self, ticker: str, trade_date: str | None = None,
                tier1_data: dict[str, Any] | None = None,
                tier2_data: dict[str, Any] | None = None,
                skip_backtest: bool = False) -> tuple[dict[str, Any], str]:
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

        # 自动加载数据: 优先使用传入数据, 缺失时从供应商加载
        if not tier1_data and not tier2_data:
            tier1_data, tier2_data = self._load_data(ticker, trade_date)
        tier1_data = tier1_data or {}
        tier2_data = tier2_data or {}
        pit_manifest = tier1_data.pop("_data_manifest", None)
        manifest_path = tier1_data.pop("_data_manifest_path", None)
        manifest_save_error = tier1_data.pop("_data_manifest_save_error", None)
        if isinstance(pit_manifest, dict):
            if manifest_path:
                pit_manifest["path"] = manifest_path
            if manifest_save_error:
                pit_manifest["save_error"] = manifest_save_error

        init_state = {
            "messages": [("human", f"分析 {ticker} {trade_date}")],
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "sender": "system",
            "run_mode": self.mode,
            "skip_backtest": skip_backtest,
            "tier1_data": tier1_data or {},
            "tier2_data": tier2_data or {},
            "tier2_decision": {},
            "data_quality_report": None,
            "pit_manifest": pit_manifest,
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
            "agent_evidence": {},
            "agent_tool_calls": {},
            "agent_self_checks": {},
            "round2_state": {
                "active": False,
                "round_count": 0,
                "max_rounds": 8,
                
                
                "current_speaker": "",
                "completed": False,
                "summary": "",
                "provider": "none",
                "fallback_reason": "",
                "final_pressure": "neutral",
                "unresolved_conflicts": [],
                "contradiction_records": [],
                "evidence_board": [],
                "round_history": [],
                "moderator_output": None,
            },
            "round2_summary": "",
            "system_decision_obj": None,
            "system_rubric": {},
            "system_state": "",
            "approval_input": {},
            "approval_record": {},
            "execution_allowed": False,
            "final_report": "",
            "final_report_obj": None,
            "audit_trace": {},
            "audit_trace_path": "",
        }

        if self.debug:
            final_state: dict[str, Any] = {}
            for chunk in self.workflow.stream(init_state):
                node_name = list(chunk.keys())[0]
                logger.info("Completed node: %s", node_name)
                if chunk:
                    for _k, v in chunk.items():
                        if isinstance(v, dict):
                            final_state.update(v)
        else:
            final_state = self.workflow.invoke(init_state)

        report = final_state.get("final_report", "") if final_state else ""
        return final_state or init_state, report
