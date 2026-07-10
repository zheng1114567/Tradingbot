"""
System Agent — 组长与裁判

职责:
- 加载 Tier 1 / Tier 2 数据
- 调度各 Agent 发言顺序
- 拦截越权发言
- 汇总争议, 推动 Round 2 质询
- 执行软风控, 输出裁定

借鉴 TradingAgents' research_manager.py + portfolio_manager.py 的控场模式,
但这里 System Agent 是工作流的调度者 + 最终裁定者。
"""
from __future__ import annotations

import logging
from typing import Any

from ..data_service.schema import DecisionType
from ..llm.client import LLMClient
from ..risk.hard_risk import HardRiskController, RiskVerdictType
from .schemas import SystemDecision

logger = logging.getLogger(__name__)


def create_system_agent(llm: LLMClient):
    """创建 System Agent 节点函数"""

    def system_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        # 读取各 Agent 报告
        market_rpt = state.get("market_report_obj", None)
        event_rpt = state.get("event_report_obj", None)
        analysis_rpt = state.get("analysis_report_obj", None)
        backtest_rpt = state.get("backtest_report_obj", None)
        memory_ctx = state.get("memory_context", "")

        # 读取风控数据
        tier1 = state.get("tier1_data", {})
        risk_data = tier1.get("risk", {})

        # 硬风控检查
        hard_risk = HardRiskController()
        risk_verdict = hard_risk.check_all(
            code=ticker,
            direction="buy",
            st_list=risk_data.get("st_list"),
            suspended_list=risk_data.get("suspended_list"),
            delisting_list=risk_data.get("delisting_list"),
            daily_volume_cny=risk_data.get("daily_volume", 0),
        )

        if risk_verdict.verdict == RiskVerdictType.HARD_VETO:
            return {
                "system_decision_obj": SystemDecision(
                    decision=DecisionType.REJECT,
                    position=0,
                    alpha_source=[],
                    horizon_days=5,
                    reasons=[f"硬风控否决: {'; '.join(risk_verdict.reasons)}"],
                    objections=[],
                    invalid_conditions=[],
                    risk_verdict="HARD_VETO",
                    risk_details=risk_verdict.reasons,
                    reasoning="硬风控不通过, 终止",
                ),
                "system_decision_state": "completed",
            }

        # LLM 综合裁定
        prompt = f"""你是 System Agent, 负责最终裁定。

## 标的
{ticker} ({trade_date})

## 硬风控结果
{risk_verdict}

## 市场分析
{market_rpt}

## 事件分析
{event_rpt}

## 因子分析
{analysis_rpt}

## 回测验证
{backtest_rpt}

## 历史记忆
{memory_ctx[:500] if memory_ctx else "暂无"}

请进行最终裁定:
1. 推荐: Alpha 清晰 + 资金确认 + 因子支持 + 回测有效 + 风控通过
2. 观察: Alpha 存在但样本不足/资金背离/估值过高/链条偏弱
3. 拒绝: 风控不通过/Alpha 不清晰/反例强/成本过高

每条推荐必须绑定 Alpha 来源。输出结构化裁定结果。"""

        try:
            decision = llm.chat(
                messages=[
                    ("system", "你是 A 股交易系统组长。严格遵守硬风控, 不输出无 Alpha 来源的推荐。"),
                    ("human", prompt),
                ],
                response_format=SystemDecision,
            )
        except Exception as e:
            logger.warning("LLM system decision failed, defaulting to watch: %s", e)
            decision = SystemDecision(
                decision=DecisionType.WATCH,
                alpha_source=[],
                horizon_days=5,
                reasons=["LLM 不可用, 默认观察"],
                objections=["无法获取 LLM 裁定"],
                risk_verdict="PASS" if risk_verdict.verdict == RiskVerdictType.PASS else "SOFT_VETO",
                risk_details=risk_verdict.reasons,
                reasoning="LLM 降级: 默认观察",
            )

        return {
            "system_decision_obj": decision,
            "system_decision_state": "completed",
        }

    return system_node
