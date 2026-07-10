"""
硬风控代码节点 — 在 LangGraph 中作为独立节点执行

三个硬风控检查时间点:
1. risk_check_1: 分析前 (ST/停牌/退市)
2. risk_check_2: Round 1 完成后 (流动性/涨跌停)
3. risk_check_3: 裁定前 (冲击成本/仓位)

所有检查由代码执行, 没有 LLM 参与。
"""
from __future__ import annotations

import logging
from typing import Any

from ..risk.hard_risk import HardRiskController, RiskVerdictType

logger = logging.getLogger(__name__)


def create_risk_check_1():
    """硬风控 1: ST/停牌/退市检查 (分析前)

    在 Round 1 之前执行。
    HARD_VETO → 直接终止, 不进 Round 1。
    """
    def risk_check_1_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        tier1 = state.get("tier1_data", {})
        risk_data = tier1.get("risk", {})

        controller = HardRiskController()
        verdict = controller.check_all(
            code=ticker,
            st_list=risk_data.get("st_list"),
            suspended_list=risk_data.get("suspended_list"),
            delisting_list=risk_data.get("delisting_list"),
        )

        return {
            "risk_check_1": {
                "verdict": verdict.verdict.value,
                "reasons": verdict.reasons,
                "suggested": verdict.suggested_actions,
            },
            "system_state": "vetoed" if verdict.verdict == RiskVerdictType.HARD_VETO else "running",
        }

    return risk_check_1_node


def create_risk_check_2():
    """硬风控 2: 流动性/涨跌停检查 (Round 1 完成后)

    检查标的是否可成交。
    结果作为 System Agent 的输入, 影响 Round 2 讨论。
    """
    def risk_check_2_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        tier2 = state.get("tier2_data", {})
        risk_data = state.get("tier1_data", {}).get("risk", {})

        controller = HardRiskController()

        # 涨跌停检查
        limit_up = tier2.get("limit_up", {})
        is_limit_up = limit_up.get("is_limit_up", False)
        is_limit_down = limit_up.get("is_limit_down", False)

        verdict = controller.check_limit_up_down(
            is_limit_up=is_limit_up,
            is_limit_down=is_limit_down,
            direction="buy",
        )

        if verdict.verdict == RiskVerdictType.HARD_VETO:
            return {
                "risk_check_2": {
                    "verdict": "HARD_VETO",
                    "reasons": verdict.reasons,
                },
            }

        # 流动性检查
        daily_volume = risk_data.get("daily_volume", 0)
        liquidity_verdict = controller.check_liquidity(daily_volume)

        return {
            "risk_check_2": {
                "verdict": liquidity_verdict.verdict.value,
                "reasons": liquidity_verdict.reasons,
            },
        }

    return risk_check_2_node


def create_risk_check_3():
    """硬风控 3: 冲击成本/仓位检查 (裁定前)

    最终硬风控, 结果不可被 LLM 覆盖。
    HARD_VETO → 必须输出拒绝。
    """
    def risk_check_3_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        risk_data = state.get("tier1_data", {}).get("risk", {})

        controller = HardRiskController()
        verdict = controller.check_all(
            code=ticker,
            daily_volume_cny=risk_data.get("daily_volume", 0),
            current_position_pct=risk_data.get("current_position", 0),
            proposed_pct=0.10,
        )

        return {
            "risk_check_3": {
                "verdict": verdict.verdict.value,
                "reasons": verdict.reasons,
                "suggested": verdict.suggested_actions,
            },
        }

    return risk_check_3_node
