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
from ..agents.contract import basic_self_check, build_agent_update
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
    after_round2_turn,
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
    workflow.add_node("Round 2 Debate", _create_round2_node(llm))
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
        {"continue_round2": "Round 2 Debate", "finalize": "Risk Check 3"},
    )

    # 硬风控3: 无论 PASS/HARD_VETO 都进入裁定节点生成报告
    workflow.add_conditional_edges(
        "Risk Check 3",
        after_risk_check_3,
        {"finalize": "System Final Decision", "end": "System Final Decision"},
    )

    # 裁定 → 人工审批记录 → 报告 → END
    workflow.add_edge("System Final Decision", "Approval Agent")
    workflow.add_edge("Approval Agent", "Report Agent")
    workflow.add_edge("Report Agent", END)

    return workflow.compile()


def _create_round2_node(llm: LLMClient):
    """创建 Round 2 圆桌质询节点

    一轮圆桌:
    1. System Agent 针对矛盾点提出问题
    2. 选择相关 Agent 发言
    3. 用各 Agent 的既有报告形成回答
    4. 写回问题、回答和圆桌总结
    """
    def _targets_for(contradiction: str) -> list[str]:
        targets = []
        lowered = contradiction.lower()
        if "market" in lowered:
            targets.append("Market")
        if "event" in lowered:
            targets.append("Event")
        if "analysis" in lowered:
            targets.append("Analysis")
        if "backtest" in lowered:
            targets.append("Backtest")
        return targets or ["Market", "Event", "Analysis", "Backtest"]

    harness = RoundtableHarness()

    def _fallback_answer(target: str, contradiction: str, report: str, evidence: str) -> str:
        report_excerpt = report.replace("\n", " ")[:180]
        evidence_excerpt = evidence.replace("\n", " ")[:360]
        return (
            f"{target} 回答: 基于 AgentContext 和当前报告，针对矛盾“{contradiction}”，"
            f"可引用数据证据为：{evidence_excerpt}；报告依据为：{report_excerpt}。"
            "若上述证据不足以消除矛盾，应进入最终裁定的反对意见。"
        )

    def _build_summary(questions: list[dict[str, Any]]) -> str:
        if not questions:
            return ""
        lines = ["Round 2 圆桌会议总结:"]
        for idx, item in enumerate(questions, start=1):
            lines.append(f"{idx}. 矛盾: {item.get('data_source', '')}")
            lines.append(f"   质询: {item.get('question', '')}")
            for answer in item.get("answers", []):
                lines.append(
                    f"   - {answer.get('target_agent', 'Agent')}: "
                    f"{answer.get('answer', '')}"
                )
        lines.append("结论: 未消除的分歧必须进入最终裁定和风控理由。")
        return "\n".join(lines)

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

        try:
            result = AutoGenRoundtable().run(state)
            if result.summary:
                return {
                    "round2_state": {
                        **round2,
                        "round_count": max_rounds,
                        "questions": result.questions,
                        "current_speaker": "AutoGenRoundtable",
                        "completed": True,
                        "summary": result.summary,
                        "unresolved_conflicts": result.unresolved_conflicts,
                        "final_pressure": result.final_pressure,
                        "provider": result.provider,
                        "fallback_reason": result.fallback_reason,
                    },
                    "round2_summary": result.summary,
                }
        except Exception as e:
            fallback_reason = f"{type(e).__name__}: {e}"
            logger.warning(
                "AutoGen roundtable failed, falling back to deterministic roundtable: %s",
                e,
            )
        else:
            fallback_reason = "AutoGen returned empty summary"

        # 提出质询 (LLM 生成问题, 失败则确定性降级)
        contradiction = contradictions[count % len(contradictions)] if contradictions else ""
        context = harness.build_context(state, contradictions)

        prompt = f"""Round 2 交叉质询 - 第 {count + 1} 轮

发现的矛盾:
{contradiction}

DATA_AGENT_BRIEF:
{context.shared_evidence_text[:1200]}

请针对上述矛盾提出一个精准的质询问题。
质询必须基于 DataAgent 数据矛盾和各 Agent 可见证据, 不能是泛泛的问题。"""

        try:
            response = llm.chat([
                ("system", "你是在 Round 2 交叉质询中提出问题的 System Agent。"),
                ("human", prompt),
            ])
            question = response if isinstance(response, str) else str(response)
        except Exception:
            question = f"关于矛盾 '{contradiction[:50]}', 请相关 Agent 提供更多数据支撑。"

        questions = list(round2.get("questions", []))
        answers = []
        for target in _targets_for(contradiction):
            agent_context = context.agent_contexts[target]
            answer_prompt = f"""Round 2 圆桌会议 - {target} Agent 发言

矛盾:
{contradiction}

System 质询:
{question}

AgentContext:
{agent_context.evidence_text[:1200]}

{target} Agent 当前报告:
{agent_context.report[:1200]}

请只基于该 AgentContext、DATA_AGENT_BRIEF 和该 Agent 报告回答:
1. 是否坚持原判断
2. 支撑证据，必须点名引用 DataAgent 字段或报告片段
3. 对最终裁定的影响
回答要简洁。"""
            try:
                response = llm.chat([
                    ("system", agent_context.system_message),
                    ("human", answer_prompt),
                ])
                answer = response if isinstance(response, str) else str(response)
            except Exception:
                answer = _fallback_answer(
                    target,
                    contradiction,
                    agent_context.report,
                    agent_context.evidence_text,
                )
            answers.append({
                "target_agent": target,
                "answer": answer,
                "evidence": agent_context.evidence_text[:800],
            })

        questions.append({
            "source_agent": "System",
            "target_agent": ",".join(a["target_agent"] for a in answers),
            "question": question,
            "answer": "\n".join(f"{a['target_agent']}: {a['answer']}" for a in answers),
            "answers": answers,
            "data_source": contradiction,
        })
        summary = _build_summary(questions)

        return {
            "round2_state": {
                **round2,
                "round_count": count + 1,
                "questions": questions,
                "current_speaker": "System",
                "completed": count + 1 >= max_rounds,
                "summary": summary,
                "provider": "deterministic",
                "fallback_reason": fallback_reason,
                "final_pressure": "neutral",
                "unresolved_conflicts": contradictions,
            },
            "round2_summary": summary,
        }

    return round2_node


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
                "questions": [],
                "contradictions": [],
                "current_speaker": "",
                "completed": False,
                "summary": "",
                "provider": "none",
                "fallback_reason": "",
                "final_pressure": "neutral",
                "unresolved_conflicts": [],
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
