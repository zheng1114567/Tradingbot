"""
条件路由逻辑 — 控制 LangGraph 流程

1. after_risk_check_1: 硬风控1通过 → Round 1, 否决 → END
2. after_market: 冰点模式跳过深度分析
3. after_round1: 判断矛盾检测后是否需要进入风控3
4. after_risk_check_3: 硬风控3通过 → 裁定, 否决 → END
"""
from __future__ import annotations

from typing import Literal

from .state import AgentState


def after_risk_check_1(state: AgentState) -> Literal["round1", "end"]:
    """硬风控1: 跳过或继续"""
    risk1 = state.get("risk_check_1", {})
    if risk1.get("verdict") == "HARD_VETO":
        return "end"
    return "round1"


def after_market(state: AgentState) -> Literal["continue_round1", "skip_round1"]:
    """Market Agent 之后: 冰点模式跳过后续深度分析"""
    market = state.get("market_report_obj")
    if market and market.market_state in ("冰点",):
        return "skip_round1"
    return "continue_round1"


def after_round1(state: AgentState) -> Literal["round2", "finalize"]:
    """Round 1 完成后: 判断是否进 Round 2"""
    round2 = state.get("round2_state", {})
    if round2.get("active") and not round2.get("completed"):
        return "round2"
    return "finalize"


def after_risk_check_3(state: AgentState) -> Literal["finalize", "end"]:
    """硬风控3: 通过则裁定, 否决则结束"""
    risk3 = state.get("risk_check_3", {})
    if risk3.get("verdict") == "HARD_VETO":
        return "end"
    return "finalize"
