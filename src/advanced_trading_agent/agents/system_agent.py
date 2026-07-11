"""
System Agent — 组长与裁判

职责:
1. 初始化: 加载 Tier 1 → 数据质量检查 → Memory 召回
2. Round 1 调度 (数据流控制)
3. 判断是否进 Round 2
4. 执行硬风控 (三层)
5. 最终裁定

硬风控三时间点:
- ① 分析前: ST/停牌/退市 → HARD_VETO 直接终止
- ② Round1 后: 流动性/涨跌停 → SOFT_VETO 带入讨论
- ③ 裁定前: 冲击成本/仓位 → 最终 HARD_VETO

借鉴 TradingAgents' portfolio_manager.py 的裁定模式,
但硬风控由代码执行, LLM 不可覆盖。
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from ..core.cache_manager import Tier1Data, WINTER_SENTIMENTS, decide_tier2_loading
from ..core.data_quality import DataQualityChecker
from ..llm.client import LLMClient
from ..risk.soft_risk import SignalType, SoftRiskController
from .contract import basic_self_check, build_node_audit_update
from .schemas import DecisionType, RiskVerdict, SystemDecision, SystemRubric

logger = logging.getLogger(__name__)


_DECISION_RANK = {
    DecisionType.REJECT: 0,
    DecisionType.WATCH: 1,
    DecisionType.RECOMMEND: 2,
}


def _build_system_rubric(
    *,
    market_rpt: Any,
    event_rpt: Any,
    analysis_rpt: Any,
    backtest_rpt: Any,
    memory_ctx: str,
    risk3_verdict: str,
    risk3_reasons: list[str],
    risk2: dict[str, Any],
    soft_assessment: Any,
    round2_summary: str,
    round2_final_pressure: str,
) -> SystemRubric:
    """Build a deterministic rubric before asking the LLM to express the decision."""
    support: list[str] = []
    objections: list[str] = []
    forced_downgrades: list[str] = []

    market_score = 0
    if market_rpt:
        if market_rpt.market_state in ("冰点", "高潮") or market_rpt.capital_confirmation in ("资金背离", "资金不足"):
            market_score = 1 if market_rpt.position_cap > 0 else 0
            objections.append(f"市场/资金约束: {market_rpt.market_state}, {market_rpt.capital_confirmation}")
        else:
            market_score = 2
            support.append(f"市场环境可交易: {market_rpt.market_state}, {market_rpt.capital_confirmation}")
    else:
        objections.append("缺少市场报告")

    event_score = 0
    if event_rpt:
        if event_rpt.direction == "利好" and event_rpt.chain_quality == "direct" and event_rpt.confidence >= 0.6:
            event_score = 2
            support.append(f"事件链条直接且偏利好: {event_rpt.event_type}")
        elif event_rpt.direction == "利好" and event_rpt.chain_quality != "weak":
            event_score = 1
            objections.append("事件存在但置信度/链条强度不足")
        else:
            objections.append(f"事件不支持推荐: {event_rpt.direction}/{event_rpt.chain_quality}")
    else:
        objections.append("缺少事件报告")

    analysis_score = 0
    if analysis_rpt and analysis_rpt.stock_rankings:
        top = analysis_rpt.stock_rankings[0]
        warnings = list(top.warnings or []) + list(analysis_rpt.factor_warnings or [])
        if top.composite_score >= 7 and not warnings:
            analysis_score = 2
            support.append(f"因子排序靠前: {top.code} {top.composite_score:.1f}")
        else:
            analysis_score = 1
            objections.append("因子分数或拥挤警告不足以直接推荐")
    else:
        objections.append("缺少可用因子排序")

    backtest_score = 0
    if backtest_rpt:
        if (
            backtest_rpt.sample_size >= 30
            and backtest_rpt.win_rate >= 0.5
            and backtest_rpt.avg_excess_return > 0
        ):
            backtest_score = 2
            support.append(
                f"回测样本有效: n={backtest_rpt.sample_size}, "
                f"win={backtest_rpt.win_rate:.0%}, excess={backtest_rpt.avg_excess_return:+.2%}"
            )
        elif backtest_rpt.sample_size > 0:
            backtest_score = 1
            objections.append("回测样本/胜率/超额不足以支撑强推荐")
        else:
            objections.append("回测无有效样本")
    else:
        objections.append("缺少回测报告")

    memory_score = 1 if memory_ctx else 0
    if memory_ctx:
        support.append("存在同标的历史复盘记忆")

    risk_score = 2
    if risk3_verdict == "HARD_VETO":
        risk_score = 0
        forced_downgrades.extend(risk3_reasons or ["硬风控否决"])
    elif risk3_verdict == "SOFT_VETO" or risk2.get("verdict") == "SOFT_VETO":
        risk_score = 1
        forced_downgrades.extend(risk3_reasons + risk2.get("reasons", []))

    if soft_assessment.signal in (SignalType.AVOID, SignalType.SELL):
        risk_score = 0
        forced_downgrades.extend(soft_assessment.reasons or ["软风控触发退出/回避"])
    elif soft_assessment.signal == SignalType.REDUCE:
        risk_score = min(risk_score, 1)
        forced_downgrades.extend(soft_assessment.reasons or ["软风控建议减仓"])

    if round2_summary:
        forced_downgrades.append("Round 2 存在争议，需写入反对意见或降级")
    if round2_final_pressure == "downgrade":
        forced_downgrades.append("Round 2 Moderator 要求降级")

    total_score = market_score + event_score + analysis_score + backtest_score + memory_score + risk_score
    if risk_score == 0:
        recommendation_floor = DecisionType.REJECT
    elif total_score >= 9 and not forced_downgrades:
        recommendation_floor = DecisionType.RECOMMEND
    else:
        recommendation_floor = DecisionType.WATCH

    return SystemRubric(
        market_score=market_score,
        event_score=event_score,
        analysis_score=analysis_score,
        backtest_score=backtest_score,
        memory_score=memory_score,
        risk_score=risk_score,
        total_score=total_score,
        recommendation_floor=recommendation_floor,
        support=support,
        objections=objections,
        forced_downgrades=forced_downgrades,
    )


def _enforce_rubric_floor(decision: SystemDecision, rubric: SystemRubric) -> bool:
    """Downgrade LLM output when deterministic rubric allows only a weaker conclusion."""
    if _DECISION_RANK[decision.decision] <= _DECISION_RANK[rubric.recommendation_floor]:
        return False

    original = decision.decision
    decision.decision = rubric.recommendation_floor
    if rubric.recommendation_floor == DecisionType.REJECT:
        decision.position = 0
        decision.risk_verdict = RiskVerdict.SOFT_VETO if rubric.risk_score > 0 else RiskVerdict.HARD_VETO
    elif rubric.recommendation_floor == DecisionType.WATCH:
        decision.position = 0

    reason = f"结构化rubric限制: {original.value} 降级为 {rubric.recommendation_floor.value}"
    decision.reasons = [reason, *decision.reasons]
    decision.objections = [*decision.objections, *rubric.objections, *rubric.forced_downgrades]
    decision.risk_details = [*decision.risk_details, *rubric.forced_downgrades]
    return True


def create_system_agent(llm: LLMClient):
    """创建 System Agent 节点函数"""

    def init_node(state: dict[str, Any]) -> dict[str, Any]:
        """初始化节点: 数据质量检查 + 判断 Winter 模式 + Tier2 加载决策

        这个节点在 Market Agent 之前执行。
        硬风控1 (ST/停牌/退市) 已在 Risk Check 1 节点完成。
        """
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", str(date.today()))
        tier1 = state.get("tier1_data", {})

        # 数据质量检查
        quality = DataQualityChecker.check_tier1(tier1)

        # 判断是否 Winter 模式
        sentiment = tier1.get("sentiment", {}).get("sentiment", "正常")
        winter = sentiment in WINTER_SENTIMENTS

        tier2_decision = decide_tier2_loading(
            Tier1Data(
                market=tier1.get("market", {}),
                sentiment=tier1.get("sentiment", {}),
                capital=tier1.get("capital", {}),
                winter_mode=winter,
            )
        )

        evidence = [
            f"ticker={ticker}",
            f"trade_date={trade_date}",
            f"sentiment={sentiment}",
            f"winter_mode={winter}",
        ]

        return build_node_audit_update(
            sender="System Init",
            data_quality_report=quality,
            system_state="running",
            tier1_data={**tier1, "winter_mode": winter},
            tier2_decision=tier2_decision,
            evidence=evidence,
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=["tier1_quality_checked", "tier2_loading_decided"],
                warnings=[],
                confidence="winter" if winter else "normal",
            ),
        )

    def round2_judge_node(state: dict[str, Any]) -> dict[str, Any]:
        """判断是否需要 Round 2 交叉质询

        进入 Round 2 的条件:
        1. 存在明确的矛盾 (如 Market 说资金背离但 Event 说利好)
        2. Backtest 样本不足 (< 30) 但 Analysis 排序靠前
        3. 至少 2 个 Agent 的观点存在分歧
        """
        market_rpt = state.get("market_report_obj")
        event_rpt = state.get("event_report_obj")
        analysis_rpt = state.get("analysis_report_obj")
        backtest_rpt = state.get("backtest_report_obj")

        contradictions = []

        # 检查矛盾
        if market_rpt and event_rpt:
            if (market_rpt.capital_confirmation in ("资金背离", "资金不足")
                    and event_rpt.direction == "利好"):
                contradictions.append(
                    f"Market:资金{market_rpt.capital_confirmation} ↔ Event:{event_rpt.direction}"
                )

        if backtest_rpt and analysis_rpt:
            if (backtest_rpt.sample_size < 30
                    and analysis_rpt.stock_rankings
                    and analysis_rpt.stock_rankings[0].composite_score > 7):
                contradictions.append(
                    f"Backtest:样本不足({backtest_rpt.sample_size}) ↔ Analysis:高分"
                )

        needs_round2 = len(contradictions) > 0

        evidence = [
            f"contradictions={len(contradictions)}",
            f"needs_round2={needs_round2}",
        ]

        return build_node_audit_update(
            sender="Round 2 Judge",
            round2_state={
                "active": needs_round2,
                "round_count": 0,
                "max_rounds": 8,
                "questions": [],
                "contradictions": contradictions,
                "current_speaker": "",
                "completed": not needs_round2,
            },
            system_state="round2" if needs_round2 else "finalizing",
            evidence=evidence + contradictions,
            self_check=basic_self_check(
                evidence=evidence + contradictions,
                passed_rules=["round2_gate_evaluated"],
                warnings=contradictions,
                confidence=len(contradictions),
            ),
        )

    def final_decision_node(state: dict[str, Any]) -> dict[str, Any]:
        """最终裁定: LLM 综合 + 硬风控 3

        读取所有 Agent 报告和 Risk Check 3 结果, LLM 输出最终裁定。
        硬风控 3 结果不可被 LLM 覆盖。
        """
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", str(date.today()))

        market_rpt = state.get("market_report_obj")
        event_rpt = state.get("event_report_obj")
        analysis_rpt = state.get("analysis_report_obj")
        backtest_rpt = state.get("backtest_report_obj")
        memory_ctx = state.get("memory_context", "")
        round2 = state.get("round2_state", {})
        round2_table_summary = state.get("round2_summary", "") or round2.get("summary", "")

        # 读取硬风控 3 结果 (由 Risk Check 3 节点计算)
        risk3 = state.get("risk_check_3", {})
        risk3_verdict = risk3.get("verdict", "PASS")
        risk3_reasons = risk3.get("reasons", [])
        risk2 = state.get("risk_check_2", {})
        tier1_risk = state.get("tier1_data", {}).get("risk", {})
        tier2 = state.get("tier2_data", {})

        # 软风控
        soft_risk = SoftRiskController()
        soft_state = tier2.get("current_state", {})
        soft_assessment = soft_risk.assess_all(
            holding_days=tier1_risk.get("holding_days", soft_state.get("holding_days", 0)),
            current_return=tier1_risk.get("current_return", soft_state.get("current_return", 0)),
            portfolio_drawdown=tier1_risk.get(
                "portfolio_drawdown",
                soft_state.get("portfolio_drawdown", 0),
            ),
            half_life_days=tier1_risk.get("half_life_days", soft_state.get("half_life_days")),
            invalid_conditions=tier1_risk.get(
                "invalid_conditions",
                soft_state.get("invalid_conditions"),
            ),
            current_state=soft_state or tier1_risk.get("current_state"),
        )

        # 汇总各 Agent 结论
        summary_lines = []

        if market_rpt:
            summary_lines.append(
                f"Market Agent: 市场{market_rpt.market_state}, "
                f"仓位上限{market_rpt.position_cap:.0%}, "
                f"资金{market_rpt.capital_confirmation}"
            )
        if event_rpt:
            summary_lines.append(
                f"Event Agent: [{event_rpt.event_type}] "
                f"{event_rpt.direction}({event_rpt.confidence:.0%}), "
                f"链条{event_rpt.chain_quality}"
            )
        if analysis_rpt:
            summary_lines.append(
                f"Analysis Agent: 板块评分{analysis_rpt.sector_score or 'N/A'}, "
                f"Top股票{len(analysis_rpt.stock_rankings)}只"
            )
        if backtest_rpt:
            summary_lines.append(
                f"Backtest Agent: {backtest_rpt.sample_size}样本, "
                f"胜率{backtest_rpt.win_rate:.0%}, "
                f"超额{backtest_rpt.avg_excess_return:+.2%}"
            )

        agent_summary = "\n".join(summary_lines)

        round2_summary = ""
        if round2 and round2.get("contradictions"):
            round2_summary = "\n".join(
                f"- {c}" for c in round2["contradictions"]
            )
        round2_final_pressure = str(round2.get("final_pressure", "neutral") or "neutral")
        round2_provider = str(round2.get("provider", "") or "")
        round2_fallback_reason = str(round2.get("fallback_reason", "") or "")
        if round2_table_summary:
            round2_summary = f"{round2_summary}\n\n{round2_table_summary}".strip()

        rubric = _build_system_rubric(
            market_rpt=market_rpt,
            event_rpt=event_rpt,
            analysis_rpt=analysis_rpt,
            backtest_rpt=backtest_rpt,
            memory_ctx=memory_ctx,
            risk3_verdict=risk3_verdict,
            risk3_reasons=risk3_reasons,
            risk2=risk2,
            soft_assessment=soft_assessment,
            round2_summary=round2_summary,
            round2_final_pressure=round2_final_pressure,
        )

        prompt = f"""你是 System Agent, 负责最终裁定。

## 标的
{ticker} ({trade_date})

## 各 Agent 分析汇总
{agent_summary}

## Round 2 争议
{round2_summary or '无明显矛盾'}

## Round 2 元数据
provider={round2_provider or 'none'}
final_pressure={round2_final_pressure}
fallback_reason={round2_fallback_reason or 'none'}

## Round 2 圆桌要求
如果圆桌仍存在未消除分歧, 最终裁定必须降级或写入反对意见。
如果 final_pressure=downgrade, 最终裁定不得为推荐。

## 硬风控
{risk3}

## 软风控
{soft_assessment}

## 历史记忆
{memory_ctx[:400] if memory_ctx else "暂无"}

## 结构化 Rubric
{rubric.model_dump_json(indent=2)}

## 裁定规则
推荐: Alpha 清晰 + 资金确认 + 因子支持 + 回测有效 + 风控通过
观察: Alpha 存在但样本不足/资金背离/估值过高/链条偏弱
拒绝: 风控不通过/Alpha 不清晰/反例强/成本过高

每个推荐必须绑定 Alpha 来源。
硬风控 HARD_VETO 不可覆盖, 必须输出拒绝。
结构化 Rubric 给出的是最高允许裁定等级，不能被 LLM 上调。"""

        try:
            decision = llm.chat(
                messages=[
                    ("system",
                     "你是 A 股交易系统组长。"
                     "严格遵守硬风控结果。"
                     "不输出无 Alpha 来源的推荐。"),
                    ("human", prompt),
                ],
                response_format=SystemDecision,
            )
        except Exception as e:
            logger.warning("LLM decision failed, defaulting to watch: %s", e)
            decision = SystemDecision(
                decision=DecisionType.WATCH,
                alpha_source=[],
                horizon_days=5,
                reasons=["LLM 不可用, 默认观察"],
                objections=["无法获取 LLM 裁定"],
                risk_verdict=RiskVerdict.PASS if risk3_verdict == "PASS" else RiskVerdict.SOFT_VETO,
                risk_details=risk3_reasons,
                reasoning="LLM 降级: 默认观察",
            )

        # 硬风控 3 覆盖 (HARD_VETO 不可被 LLM 覆盖)
        is_veto = risk3_verdict == "HARD_VETO"
        if is_veto:
            decision.decision = DecisionType.REJECT
            decision.risk_verdict = RiskVerdict.HARD_VETO
            decision.position = 0
            decision.reasons = [f"硬风控3否决: {'; '.join(risk3_reasons)}"]
            decision.risk_details = risk3_reasons
            decision.reasoning = "硬风控不通过"
        elif risk3_verdict == "SOFT_VETO" or risk2.get("verdict") == "SOFT_VETO":
            decision.risk_verdict = RiskVerdict.SOFT_VETO
            decision.risk_details = risk3_reasons + risk2.get("reasons", [])
            if decision.decision == DecisionType.RECOMMEND:
                decision.decision = DecisionType.WATCH
                decision.position = 0
                decision.reasons = ["软风控未通过，推荐降级为观察", *decision.reasons]
        elif soft_assessment.signal in (SignalType.AVOID, SignalType.SELL):
            decision.decision = DecisionType.REJECT
            decision.risk_verdict = RiskVerdict.SOFT_VETO
            decision.position = 0
            decision.reasons = ["软风控触发退出/回避", *soft_assessment.reasons]
            decision.risk_details = soft_assessment.reasons
        elif soft_assessment.signal == SignalType.REDUCE:
            decision.risk_verdict = RiskVerdict.SOFT_VETO
            decision.risk_details = soft_assessment.reasons
            if decision.decision == DecisionType.RECOMMEND:
                decision.decision = DecisionType.WATCH
                decision.position = min(decision.position, 0.05)
                decision.reasons = ["软风控建议减仓，推荐降级为观察", *decision.reasons]

        rubric_downgraded = False
        if not is_veto:
            rubric_downgraded = _enforce_rubric_floor(decision, rubric)

        evidence = [
            f"decision={decision.decision.value}",
            f"risk_verdict={decision.risk_verdict.value}",
            f"rubric_total={rubric.total_score}",
            f"rubric_floor={rubric.recommendation_floor.value}",
            f"rubric_downgraded={rubric_downgraded}",
            f"round2_provider={round2_provider or 'none'}",
            f"round2_final_pressure={round2_final_pressure}",
            f"alpha_sources={len(decision.alpha_source)}",
            f"reasons={len(decision.reasons)}",
            f"objections={len(decision.objections)}",
        ]
        warnings = []
        if not decision.alpha_source and decision.decision == DecisionType.RECOMMEND:
            warnings.append("推荐缺少 Alpha 来源")
        if is_veto:
            warnings.extend(risk3_reasons)
        warnings.extend(soft_assessment.reasons)

        return build_node_audit_update(
            sender="System Final Decision",
            system_decision_obj=decision,
            system_rubric=rubric.model_dump(mode="json"),
            risk_check_3={
                "verdict": decision.risk_verdict.value,
                "reasons": risk3_reasons + soft_assessment.reasons,
            },
            system_state="completed",
            evidence=evidence,
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=[
                    "hard_veto_override_enforced",
                    "soft_risk_state_checked",
                    "structured_rubric_floor_enforced",
                ],
                warnings=warnings,
                confidence=decision.decision.value,
            ),
        )

    return {"init": init_node, "round2_judge": round2_judge_node, "final": final_decision_node}
