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

from ..agents.contract import basic_self_check, build_node_audit_update
from ..risk.hard_risk import HardRiskController, RiskVerdictType

logger = logging.getLogger(__name__)


def _latest_price_record(tier2: dict[str, Any]) -> dict[str, Any]:
    price_data = tier2.get("price_data", [])
    if isinstance(price_data, list) and price_data:
        return price_data[-1] if isinstance(price_data[-1], dict) else {}
    if isinstance(price_data, dict):
        return price_data
    return {}


def create_risk_check_1():
    """硬风控 1: ST/停牌/退市检查 (分析前)

    在 Round 1 之前执行。
    HARD_VETO → 直接终止, 不进 Round 1。
    """
    def risk_check_1_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        tier1 = state.get("tier1_data", {})
        risk_data = tier1.get("risk", {})

        if not risk_data or not risk_data.get("risk_data_available", True):
            reasons = [
                "风险基础数据缺失，无法确认 ST/停牌/退市状态",
                *risk_data.get("risk_data_errors", []),
            ]
            return build_node_audit_update(
                sender="Risk Check 1",
                risk_check_1={
                    "verdict": "SOFT_VETO",
                    "reasons": reasons,
                    "suggested": ["补齐风险数据后再做推荐"],
                },
                system_state="running",
                evidence=["risk_data_available=False"],
                self_check=basic_self_check(
                    evidence=["risk_data_available=False"],
                    passed_rules=["missing_risk_data_soft_vetoed"],
                    warnings=reasons,
                    confidence="SOFT_VETO",
                ),
            )

        controller = HardRiskController()
        verdict = controller.check_all(
            code=ticker,
            st_list=risk_data.get("st_list"),
            suspended_list=risk_data.get("suspended_list"),
            delisting_list=risk_data.get("delisting_list"),
        )

        risk_update = {
            "verdict": verdict.verdict.value,
            "reasons": verdict.reasons,
            "suggested": verdict.suggested_actions,
        }
        return build_node_audit_update(
            sender="Risk Check 1",
            risk_check_1=risk_update,
            system_state="vetoed" if verdict.verdict == RiskVerdictType.HARD_VETO else "running",
            evidence=[
                f"ticker={ticker}",
                f"verdict={verdict.verdict.value}",
                f"reasons={len(verdict.reasons)}",
            ],
            self_check=basic_self_check(
                evidence=[f"ticker={ticker}", f"verdict={verdict.verdict.value}"],
                passed_rules=["st_suspended_delisting_checked"],
                warnings=verdict.reasons,
                confidence=verdict.verdict.value,
            ),
        )

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
        price_record = _latest_price_record(tier2)
        is_limit_up = limit_up.get("is_limit_up", price_record.get("is_limit_up", False))
        is_limit_down = limit_up.get("is_limit_down", price_record.get("is_limit_down", False))

        verdict = controller.check_limit_up_down(
            is_limit_up=is_limit_up,
            is_limit_down=is_limit_down,
            direction="buy",
        )

        if verdict.verdict == RiskVerdictType.HARD_VETO:
            return build_node_audit_update(
                sender="Risk Check 2",
                risk_check_2={
                    "verdict": "HARD_VETO",
                    "reasons": verdict.reasons,
                },
                evidence=[
                    f"ticker={ticker}",
                    "limit_up_down_verdict=HARD_VETO",
                ],
                self_check=basic_self_check(
                    evidence=[f"ticker={ticker}", "limit_up_down_verdict=HARD_VETO"],
                    passed_rules=["limit_up_down_checked"],
                    warnings=verdict.reasons,
                    confidence="HARD_VETO",
                ),
            )

        # 流动性检查
        daily_volume = risk_data.get("daily_volume")
        if daily_volume is None:
            reasons = ["流动性数据缺失，无法确认可成交性"]
            return build_node_audit_update(
                sender="Risk Check 2",
                risk_check_2={
                    "verdict": "SOFT_VETO",
                    "reasons": reasons,
                },
                evidence=[f"ticker={ticker}", "daily_volume=None"],
                self_check=basic_self_check(
                    evidence=[f"ticker={ticker}", "daily_volume=None"],
                    passed_rules=["missing_liquidity_soft_vetoed"],
                    warnings=reasons,
                    confidence="SOFT_VETO",
                ),
            )
        liquidity_verdict = controller.check_liquidity(daily_volume)

        return build_node_audit_update(
            sender="Risk Check 2",
            risk_check_2={
                "verdict": liquidity_verdict.verdict.value,
                "reasons": liquidity_verdict.reasons,
            },
            evidence=[
                f"ticker={ticker}",
                f"daily_volume={daily_volume}",
                f"verdict={liquidity_verdict.verdict.value}",
            ],
            self_check=basic_self_check(
                evidence=[f"ticker={ticker}", f"daily_volume={daily_volume}"],
                passed_rules=["limit_up_down_checked", "liquidity_checked"],
                warnings=liquidity_verdict.reasons,
                confidence=liquidity_verdict.verdict.value,
            ),
        )

    return risk_check_2_node


def create_risk_check_3():
    """硬风控 3: 冲击成本/仓位检查 (裁定前)

    最终硬风控, 结果不可被 LLM 覆盖。
    HARD_VETO → 必须输出拒绝。
    """
    def risk_check_3_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        risk_data = state.get("tier1_data", {}).get("risk", {})
        decision = state.get("system_decision_obj")
        proposed_pct = getattr(decision, "position", None) or 0.10

        controller = HardRiskController()
        verdict = controller.check_all(
            code=ticker,
            daily_volume_cny=risk_data.get("daily_volume"),
            is_limit_up=risk_data.get("is_limit_up", False),
            is_limit_down=risk_data.get("is_limit_down", False),
            estimated_impact_bps=risk_data.get("estimated_impact_bps", 0),
            expected_return_bps=risk_data.get("expected_return_bps", 0),
            current_position_pct=risk_data.get("current_position", 0),
            proposed_pct=proposed_pct,
        )

        return build_node_audit_update(
            sender="Risk Check 3",
            risk_check_3={
                "verdict": verdict.verdict.value,
                "reasons": verdict.reasons,
                "suggested": verdict.suggested_actions,
            },
            evidence=[
                f"ticker={ticker}",
                f"proposed_pct={proposed_pct:.2%}",
                f"verdict={verdict.verdict.value}",
            ],
            self_check=basic_self_check(
                evidence=[f"ticker={ticker}", f"proposed_pct={proposed_pct:.2%}"],
                passed_rules=["impact_cost_position_checked"],
                warnings=verdict.reasons,
                confidence=verdict.verdict.value,
            ),
        )

    return risk_check_3_node
