"""风控模块测试"""
import pytest
from advanced_trading_agent.risk.hard_risk import (
    HardRiskController, RiskVerdict, RiskVerdictType,
)
from advanced_trading_agent.risk.soft_risk import (
    SoftRiskController, SoftRiskAssessment, SignalType,
)


class TestHardRisk:
    """测试硬风控"""

    def setup_method(self):
        self.controller = HardRiskController()

    def test_st_status_veto(self):
        verdict = self.controller.check_st_status("600001.SH", st_list=["600001.SH"])
        assert verdict.verdict == RiskVerdictType.HARD_VETO
        assert "ST" in str(verdict.reasons)

    def test_suspension_veto(self):
        verdict = self.controller.check_suspension("600001.SH", suspended_list=["600001.SH"])
        assert verdict.verdict == RiskVerdictType.HARD_VETO

    def test_limit_up_veto(self):
        verdict = self.controller.check_limit_up_down(is_limit_up=True, direction="buy")
        assert verdict.verdict == RiskVerdictType.HARD_VETO

    def test_liquidity_warning(self):
        verdict = self.controller.check_liquidity(daily_volume_cny=1_000_000)
        assert verdict.verdict == RiskVerdictType.SOFT_VETO

    def test_impact_cost_veto(self):
        verdict = self.controller.check_impact_cost(
            estimated_impact_bps=50,
            expected_return_bps=100,
        )
        assert verdict.verdict == RiskVerdictType.HARD_VETO  # 50/100 = 50% > 30%

    def test_pass(self):
        verdict = self.controller.check_liquidity(daily_volume_cny=100_000_000)
        assert verdict.verdict == RiskVerdictType.PASS

    def test_check_all_veto(self):
        verdict = self.controller.check_all(
            code="ST0001",
            st_list=["ST0001"],
            suspended_list=[],
        )
        assert verdict.verdict == RiskVerdictType.HARD_VETO

    def test_check_all_pass(self):
        verdict = self.controller.check_all(
            code="000001.SZ",
            daily_volume_cny=1e8,
        )
        assert verdict.verdict == RiskVerdictType.PASS

    def test_sector_limit_warning(self):
        verdict = self.controller.check_sector_limit(
            current_sector_pct=0.28,
            proposed_pct=0.05,
        )
        assert verdict.verdict == RiskVerdictType.SOFT_VETO
        assert "单板块仓位" in verdict.reasons[0]


class TestSoftRisk:
    """测试软风控"""

    def setup_method(self):
        self.controller = SoftRiskController()

    def test_stop_loss(self):
        result = self.controller.check_stop_loss(-0.10)
        assert result.signal == SignalType.SELL

    def test_take_profit(self):
        result = self.controller.check_take_profit(0.20)
        assert result.signal == SignalType.REDUCE

    def test_holding_period(self):
        result = self.controller.check_holding_period(25)
        assert result.signal == SignalType.REDUCE

    def test_drawdown_fuse(self):
        result = self.controller.check_drawdown(-0.20)
        assert result.signal == SignalType.AVOID

    def test_normal_pass(self):
        result = self.controller.assess_all(
            holding_days=3, current_return=0.02,
            portfolio_drawdown=-0.03,
        )
        assert result.signal in (SignalType.BUY, SignalType.WATCH)

    def test_half_life_expired(self):
        result = self.controller.check_half_life(
            half_life_days=5, holding_days=10,
        )
        assert result.signal == SignalType.REDUCE
