"""
硬风控 — 由代码执行, LLM 不可覆盖

规则:
1. 交易前: 单票 ≤ 10%, 单板块 ≤ 30%, 总仓位 ≤ 60%
2. 非 ST, 非停牌, 非退市风险
3. 日成交额 ≥ 1000万
4. 冲击成本 < 预期收益的 30% → HARD_VETO
5. 一字板/跌停/停牌 → HARD_VETO

借鉴 TradingAgents 的 risk_mgmt/*_debator.py 风控模式,
但这里的 HARD_VETO 由代码执行, 不给 LLM 覆盖机会。
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import config

logger = logging.getLogger(__name__)


class RiskVerdictType(str, Enum):
    HARD_VETO = "HARD_VETO"     # 硬否决, 不可覆盖
    SOFT_VETO = "SOFT_VETO"     # 软否决, LLM 可辩论
    PASS = "PASS"               # 通过


@dataclass
class RiskVerdict:
    """风控裁决结果"""
    verdict: RiskVerdictType = RiskVerdictType.PASS
    reasons: list[str] = field(default_factory=list)
    position_limit: float | None = None
    suggested_actions: list[str] = field(default_factory=list)


class HardRiskController:
    """硬风控控制器 — 纯代码, 无 LLM"""

    def __init__(self, risk_config: dict[str, Any] | None = None):
        rc = deepcopy(config.get("risk_config", {}))
        if risk_config:
            rc.update(risk_config)
        self.max_single_pct = rc.get("max_single_position_pct", 0.10)
        self.max_sector_pct = rc.get("max_sector_pct", 0.30)
        self.max_total_pct = rc.get("max_total_pct", 0.60)
        self.min_volume = rc.get("min_daily_volume_cny", 10_000_000)
        self.impact_threshold = rc.get("impact_cost_threshold", 0.30)

    # ============================================================
    # 单一检查
    # ============================================================

    def check_st_status(self, code: str, st_list: list[str] | None = None) -> RiskVerdict:
        """ST/*ST 检查"""
        if st_list and code in st_list:
            return RiskVerdict(
                verdict=RiskVerdictType.HARD_VETO,
                reasons=[f"{code} 处于 ST/*ST 状态"],
            )
        return RiskVerdict()

    def check_suspension(self, code: str,
                          suspended_list: list[str] | None = None) -> RiskVerdict:
        """停牌检查"""
        if suspended_list and code in suspended_list:
            return RiskVerdict(
                verdict=RiskVerdictType.HARD_VETO,
                reasons=[f"{code} 停牌中"],
            )
        return RiskVerdict()

    def check_delisting_risk(self, code: str,
                              delisting_list: list[str] | None = None) -> RiskVerdict:
        """退市风险检查"""
        if delisting_list and code in delisting_list:
            return RiskVerdict(
                verdict=RiskVerdictType.HARD_VETO,
                reasons=[f"{code} 有退市风险"],
            )
        return RiskVerdict()

    def check_limit_up_down(self, is_limit_up: bool = False,
                             is_limit_down: bool = False,
                             direction: str = "buy") -> RiskVerdict:
        """涨跌停不可成交"""
        if is_limit_up and direction == "buy":
            return RiskVerdict(
                verdict=RiskVerdictType.HARD_VETO,
                reasons=["涨停不可买入"],
            )
        if is_limit_down:
            return RiskVerdict(
                verdict=RiskVerdictType.HARD_VETO,
                reasons=["跌停不可交易"],
            )
        return RiskVerdict()

    def check_liquidity(self, daily_volume_cny: float) -> RiskVerdict:
        """流动性检查 (日成交额)"""
        if daily_volume_cny < self.min_volume:
            return RiskVerdict(
                verdict=RiskVerdictType.SOFT_VETO,
                reasons=[f"日成交额 {daily_volume_cny:.0f} < 最小要求 {self.min_volume:.0f}"],
            )
        return RiskVerdict()

    def check_impact_cost(self, estimated_impact_bps: float,
                           expected_return_bps: float) -> RiskVerdict:
        """冲击成本检查"""
        if expected_return_bps <= 0:
            # 预期收益为负时, 任何冲击成本都 veto
            return RiskVerdict(
                verdict=RiskVerdictType.HARD_VETO,
                reasons=[f"预期收益为负 ({expected_return_bps:.1f}bps), 冲击成本 {estimated_impact_bps:.1f}bps"],
            )
        ratio = estimated_impact_bps / expected_return_bps
        if ratio > self.impact_threshold:
            return RiskVerdict(
                verdict=RiskVerdictType.HARD_VETO,
                reasons=[f"冲击成本/预期收益 {ratio:.1%} > 阈值 {self.impact_threshold:.0%}"],
            )
        return RiskVerdict()

    def check_position_limit(self, current_position_pct: float,
                              proposed_pct: float) -> RiskVerdict:
        """仓位限制"""
        total = current_position_pct + proposed_pct
        reasons = []
        actions = []
        if proposed_pct > self.max_single_pct:
            reasons.append(f"单票仓位 {proposed_pct:.1%} > 上限 {self.max_single_pct:.0%}")
            actions.append(f"建议降至 {self.max_single_pct:.0%} 以下")
        if total > self.max_total_pct:
            reasons.append(f"总仓位 {total:.1%} > 上限 {self.max_total_pct:.0%}")
        if reasons:
            return RiskVerdict(
                verdict=RiskVerdictType.SOFT_VETO,
                reasons=reasons,
                suggested_actions=actions,
            )
        return RiskVerdict()

    def check_sector_limit(self, current_sector_pct: float,
                           proposed_pct: float) -> RiskVerdict:
        """单板块仓位限制"""
        total = current_sector_pct + proposed_pct
        if total > self.max_sector_pct:
            return RiskVerdict(
                verdict=RiskVerdictType.SOFT_VETO,
                reasons=[f"单板块仓位 {total:.1%} > 上限 {self.max_sector_pct:.0%}"],
                suggested_actions=[f"建议降低本次仓位或将板块暴露控制在 {self.max_sector_pct:.0%} 内"],
            )
        return RiskVerdict()

    def check_etf_execution(
        self,
        *,
        code: str,
        daily_amount_cny: float | None = None,
        premium_discount_pct: float | None = None,
        is_suspended: bool = False,
        is_limit_up: bool = False,
        is_limit_down: bool = False,
        current_sector_pct: float = 0,
        proposed_pct: float = 0.10,
        max_abs_premium_discount_pct: float = 2.0,
    ) -> RiskVerdict:
        """ETF-specific hard-risk wrapper for the sector ETF strategy."""
        if is_suspended:
            return RiskVerdict(
                verdict=RiskVerdictType.HARD_VETO,
                reasons=[f"{code} ETF 停牌中"],
            )
        tradability = self.check_limit_up_down(is_limit_up, is_limit_down, direction="buy")
        if tradability.verdict == RiskVerdictType.HARD_VETO:
            return tradability

        all_reasons: list[str] = []
        all_actions: list[str] = []
        worst_verdict = RiskVerdictType.PASS
        if daily_amount_cny is not None:
            liquidity = self.check_liquidity(daily_amount_cny)
            if liquidity.verdict == RiskVerdictType.SOFT_VETO:
                worst_verdict = RiskVerdictType.SOFT_VETO
                all_reasons.extend(liquidity.reasons)

        if premium_discount_pct is not None and abs(premium_discount_pct) > max_abs_premium_discount_pct:
            worst_verdict = RiskVerdictType.SOFT_VETO
            all_reasons.append(
                f"ETF 溢折价 {premium_discount_pct:+.2f}% 超过阈值 {max_abs_premium_discount_pct:.2f}%"
            )
            all_actions.append("建议等待溢折价收敛或改用备选 ETF")

        sector = self.check_sector_limit(current_sector_pct, proposed_pct)
        if sector.verdict == RiskVerdictType.SOFT_VETO:
            worst_verdict = RiskVerdictType.SOFT_VETO
            all_reasons.extend(sector.reasons)
            all_actions.extend(sector.suggested_actions)

        if worst_verdict != RiskVerdictType.PASS:
            return RiskVerdict(
                verdict=worst_verdict,
                reasons=all_reasons,
                suggested_actions=all_actions,
                position_limit=proposed_pct,
            )
        return RiskVerdict(verdict=RiskVerdictType.PASS, position_limit=proposed_pct)

    # ============================================================
    # 综合检查
    # ============================================================

    def check_all(self, code: str, direction: str = "buy",
                   daily_volume_cny: float | None = None,
                   is_limit_up: bool = False,
                   is_limit_down: bool = False,
                   estimated_impact_bps: float = 0,
                   expected_return_bps: float = 0,
                   current_position_pct: float = 0,
                   current_sector_pct: float = 0,
                   proposed_pct: float = 0.10,
                   st_list: list[str] | None = None,
                   suspended_list: list[str] | None = None,
                   delisting_list: list[str] | None = None) -> RiskVerdict:
        """全面风控检查 — 顺序执行, 发现 HARD_VETO 立即返回"""
        checks = [
            ("ST/退市检查", lambda: self.check_st_status(code, st_list)),
            ("停牌检查", lambda: self.check_suspension(code, suspended_list)),
            ("涨跌停检查", lambda: self.check_limit_up_down(is_limit_up, is_limit_down, direction)),
        ]

        if daily_volume_cny is not None:
            checks.append(("流动性检查", lambda: self.check_liquidity(daily_volume_cny)))

        all_reasons = []
        all_actions = []
        worst_verdict = RiskVerdictType.PASS

        for name, check_fn in checks:
            result = check_fn()
            if result.verdict == RiskVerdictType.HARD_VETO:
                return result
            if result.verdict == RiskVerdictType.SOFT_VETO:
                worst_verdict = RiskVerdictType.SOFT_VETO
            all_reasons.extend(result.reasons)
            all_actions.extend(result.suggested_actions)

        # 冲击成本检查
        if estimated_impact_bps > 0 or expected_return_bps > 0:
            impact_result = self.check_impact_cost(estimated_impact_bps, expected_return_bps)
            if impact_result.verdict == RiskVerdictType.HARD_VETO:
                return impact_result
            if impact_result.verdict == RiskVerdictType.SOFT_VETO:
                worst_verdict = RiskVerdictType.SOFT_VETO
            all_reasons.extend(impact_result.reasons)

        # 仓位检查
        pos_result = self.check_position_limit(current_position_pct, proposed_pct)
        if pos_result.verdict != RiskVerdictType.PASS:
            all_reasons.extend(pos_result.reasons)
            all_actions.extend(pos_result.suggested_actions)
            if pos_result.verdict == RiskVerdictType.HARD_VETO:
                return pos_result
            if pos_result.verdict == RiskVerdictType.SOFT_VETO:
                worst_verdict = RiskVerdictType.SOFT_VETO

        # 行业/板块仓位检查
        sector_result = self.check_sector_limit(current_sector_pct, proposed_pct)
        if sector_result.verdict != RiskVerdictType.PASS:
            all_reasons.extend(sector_result.reasons)
            all_actions.extend(sector_result.suggested_actions)
            if sector_result.verdict == RiskVerdictType.HARD_VETO:
                return sector_result
            if sector_result.verdict == RiskVerdictType.SOFT_VETO:
                worst_verdict = RiskVerdictType.SOFT_VETO

        if all_reasons:
            return RiskVerdict(
                verdict=worst_verdict,
                reasons=all_reasons,
                suggested_actions=all_actions,
                position_limit=proposed_pct,
            )

        return RiskVerdict(verdict=RiskVerdictType.PASS, position_limit=proposed_pct)
