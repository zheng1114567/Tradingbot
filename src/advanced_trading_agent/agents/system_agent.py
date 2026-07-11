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
from ..risk.soft_risk import SoftRiskController
from .schemas import DecisionType, RiskVerdict, SystemDecision

logger = logging.getLogger(__name__)


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

        return {
            "data_quality_report": quality,
            "system_state": "running",
            "tier1_data": {**tier1, "winter_mode": winter},
            "tier2_decision": tier2_decision,
        }

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

        return {
            "round2_state": {
                "active": needs_round2,
                "round_count": 0,
                "max_rounds": 8,
                "questions": [],
                "contradictions": contradictions,
                "current_speaker": "",
                "completed": not needs_round2,
            },
            "system_state": "round2" if needs_round2 else "finalizing",
        }

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

        # 软风控
        soft_risk = SoftRiskController()
        soft_assessment = soft_risk.assess_all()

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
        if round2_table_summary:
            round2_summary = f"{round2_summary}\n\n{round2_table_summary}".strip()

        prompt = f"""你是 System Agent, 负责最终裁定。

## 标的
{ticker} ({trade_date})

## 各 Agent 分析汇总
{agent_summary}

## Round 2 争议
{round2_summary or '无明显矛盾'}

## Round 2 圆桌要求
如果圆桌仍存在未消除分歧, 最终裁定必须降级或写入反对意见。

## 硬风控
{risk3}

## 软风控
{soft_assessment}

## 历史记忆
{memory_ctx[:400] if memory_ctx else "暂无"}

## 裁定规则
推荐: Alpha 清晰 + 资金确认 + 因子支持 + 回测有效 + 风控通过
观察: Alpha 存在但样本不足/资金背离/估值过高/链条偏弱
拒绝: 风控不通过/Alpha 不清晰/反例强/成本过高

每个推荐必须绑定 Alpha 来源。
硬风控 HARD_VETO 不可覆盖, 必须输出拒绝。"""

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

        return {
            "system_decision_obj": decision,
            "risk_check_3": {
                "verdict": decision.risk_verdict.value,
                "reasons": risk3_reasons,
            },
            "system_state": "completed",
        }

    return {"init": init_node, "round2_judge": round2_judge_node, "final": final_decision_node}
