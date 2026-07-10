"""风控模块边缘测试 — 补充 test_risk.py 未覆盖的场景"""
import pytest
from advanced_trading_agent.risk.hard_risk import HardRiskController, RiskVerdictType


class TestHardRiskEdgeCases:
    """硬风控边缘情况"""

    def setup_method(self):
        self.controller = HardRiskController()

    def test_impact_cost_zero_expected_return(self):
        """预期收益为 0 时应返回 HARD_VETO"""
        verdict = self.controller.check_impact_cost(
            estimated_impact_bps=10,
            expected_return_bps=0,
        )
        assert verdict.verdict == RiskVerdictType.HARD_VETO

    def test_impact_cost_negative_expected_return(self):
        """预期收益为负时应返回 HARD_VETO"""
        verdict = self.controller.check_impact_cost(
            estimated_impact_bps=5,
            expected_return_bps=-10,
        )
        assert verdict.verdict == RiskVerdictType.HARD_VETO

    def test_impact_cost_below_threshold(self):
        """冲击成本低于阈值时应返回 PASS"""
        verdict = self.controller.check_impact_cost(
            estimated_impact_bps=10,
            expected_return_bps=100,  # 10/100 = 10% < 30%
        )
        assert verdict.verdict == RiskVerdictType.PASS

    def test_liquidity_zero_volume(self):
        """日成交额为 0 时应返回 SOFT_VETO"""
        verdict = self.controller.check_liquidity(daily_volume_cny=0)
        assert verdict.verdict == RiskVerdictType.SOFT_VETO

    def test_liquidity_exactly_at_min(self):
        """日成交额恰好等于阈值时应返回 PASS"""
        verdict = self.controller.check_liquidity(daily_volume_cny=10_000_000)
        assert verdict.verdict == RiskVerdictType.PASS

    def test_limit_up_sell_should_pass(self):
        """涨停时卖出不应被 veto"""
        verdict = self.controller.check_limit_up_down(
            is_limit_up=True, direction="sell"
        )
        assert verdict.verdict == RiskVerdictType.PASS

    def test_limit_down_buy_should_veto(self):
        """跌停时买入也应被 veto (跌停无人卖出, 无法成交)"""
        verdict = self.controller.check_limit_up_down(
            is_limit_down=True, direction="buy"
        )
        assert verdict.verdict == RiskVerdictType.HARD_VETO

    def test_position_limit_exact_boundary(self):
        """仓位恰好等于上限时应返回 PASS"""
        verdict = self.controller.check_position_limit(
            current_position_pct=0, proposed_pct=0.10
        )
        assert verdict.verdict == RiskVerdictType.PASS

    def test_position_limit_exceeded(self):
        """仓位超出上限时应返回 SOFT_VETO"""
        verdict = self.controller.check_position_limit(
            current_position_pct=0, proposed_pct=0.15
        )
        assert verdict.verdict == RiskVerdictType.SOFT_VETO

    def test_check_all_empty_lists(self):
        """空列表应正常通过"""
        verdict = self.controller.check_all(
            code="000001.SZ",
            st_list=[],
            suspended_list=[],
            delisting_list=[],
            daily_volume_cny=100_000_000,
        )
        assert verdict.verdict == RiskVerdictType.PASS

    def test_check_all_none_lists(self):
        """None 列表应正常通过"""
        verdict = self.controller.check_all(
            code="000001.SZ",
            st_list=None,
            suspended_list=None,
            delisting_list=None,
        )
        assert verdict.verdict == RiskVerdictType.PASS

    def test_check_all_code_not_in_list(self):
        """代码不在 ST 列表中应正常通过"""
        verdict = self.controller.check_all(
            code="000001.SZ",
            st_list=["600001.SH", "600002.SH"],
            suspended_list=[],
        )
        assert verdict.verdict == RiskVerdictType.PASS
