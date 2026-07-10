"""
软风控 — LLM 辅助风控, 但 LLM 不能覆盖硬风控

软风控检查:
1. 事件半衰期是否已过
2. 资金流是否恶化
3. 组合回撤是否触发熔断
4. 失效条件是否触发
5. 新闻/舆情异常
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from ..config import config
from .hard_risk import RiskVerdict, RiskVerdictType

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    STRONG_BUY = "强烈推荐"
    BUY = "推荐"
    WATCH = "观察"
    REDUCE = "减仓"
    SELL = "卖出"
    AVOID = "回避"


@dataclass
class SoftRiskAssessment:
    """软风控评估结果"""
    signal: SignalType = SignalType.WATCH
    confidence: float = 0.5
    reasons: list[str] = field(default_factory=list)
    triggered_invalid_conditions: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)


class SoftRiskController:
    """软风控控制器 — LLM 辅助 + 规则"""

    def __init__(self):
        rc = config.get("risk_config", {})
        self.stop_loss = rc.get("stop_loss_pct", -0.07)
        self.take_profit = rc.get("take_profit_pct", 0.15)
        self.max_holding_days = rc.get("max_holding_days", 20)
        self.drawdown_fuse = rc.get("drawdown_fuse_pct", -0.15)  # 组合回撤熔断

    def check_half_life(self, half_life_days: int | None,
                         holding_days: int,
                         events: list[dict[str, Any]] | None = None) -> SoftRiskAssessment:
        """检查事件半衰期是否已过"""
        if half_life_days is None:
            return SoftRiskAssessment(signal=SignalType.WATCH)

        if holding_days > half_life_days:
            has_new_catalyst = False
            if events:
                for e in events:
                    if e.get("is_new_catalyst", False):
                        has_new_catalyst = True
                        break
            if not has_new_catalyst:
                return SoftRiskAssessment(
                    signal=SignalType.REDUCE,
                    confidence=0.6,
                    reasons=[f"持有期 {holding_days}d > 半衰期 {half_life_days}d, 且无新增催化"],
                    suggested_actions=["考虑减仓或退出"],
                )
        return SoftRiskAssessment(signal=SignalType.BUY)

    def check_stop_loss(self, current_return: float) -> SoftRiskAssessment:
        """止损检查"""
        if current_return <= self.stop_loss:
            return SoftRiskAssessment(
                signal=SignalType.SELL,
                confidence=0.9,
                reasons=[f"触发止损线: {current_return:.1%} <= {self.stop_loss:.0%}"],
                suggested_actions=["立即止损"],
            )
        return SoftRiskAssessment(signal=SignalType.BUY)

    def check_take_profit(self, current_return: float) -> SoftRiskAssessment:
        """止盈检查"""
        if current_return >= self.take_profit:
            return SoftRiskAssessment(
                signal=SignalType.REDUCE,
                confidence=0.7,
                reasons=[f"触发止盈线: {current_return:.1%} >= {self.take_profit:.0%}"],
                suggested_actions=["考虑部分止盈"],
            )
        return SoftRiskAssessment(signal=SignalType.BUY)

    def check_holding_period(self, holding_days: int) -> SoftRiskAssessment:
        """持有天数检查"""
        if holding_days >= self.max_holding_days:
            return SoftRiskAssessment(
                signal=SignalType.REDUCE,
                confidence=0.8,
                reasons=[f"持有 {holding_days}d 超过上限 {self.max_holding_days}d"],
                suggested_actions=["考虑退出"],
            )
        return SoftRiskAssessment(signal=SignalType.BUY)

    def check_drawdown(self, portfolio_drawdown: float) -> SoftRiskAssessment:
        """组合回撤熔断"""
        if portfolio_drawdown <= self.drawdown_fuse:
            return SoftRiskAssessment(
                signal=SignalType.AVOID,
                confidence=0.95,
                reasons=[f"组合回撤 {portfolio_drawdown:.1%} 触发熔断 {self.drawdown_fuse:.0%}"],
                suggested_actions=["停止所有新开仓"],
            )
        return SoftRiskAssessment(signal=SignalType.BUY)

    def check_invalid_conditions(self, conditions: list[str],
                                   current_state: dict[str, Any]) -> SoftRiskAssessment:
        """检查失效条件是否触发"""
        triggered = []
        # 规则的失效条件检查 (每个条件需自定义判断逻辑)
        for condition in conditions:
            if "放量下跌" in condition:
                if (current_state.get("volume_change", 0) > 1.5 and
                        current_state.get("price_change", 0) < -0.03):
                    triggered.append(condition)
            elif "大盘" in condition and "跌破" in condition:
                if current_state.get("index_change", 0) < -0.05:
                    triggered.append(condition)
            elif "政策" in condition and "低于预期" in condition:
                triggered.append(condition)  # 需要 LLM 判断

        if triggered:
            return SoftRiskAssessment(
                signal=SignalType.REDUCE,
                confidence=0.7,
                reasons=triggered,
                triggered_invalid_conditions=triggered,
                suggested_actions=["失效条件触发, 考虑减仓或退出"],
            )
        return SoftRiskAssessment(signal=SignalType.BUY)

    def assess_all(self, holding_days: int = 0,
                    current_return: float = 0,
                    portfolio_drawdown: float = 0,
                    half_life_days: int | None = None,
                    invalid_conditions: list[str] | None = None,
                    current_state: dict[str, Any] | None = None) -> SoftRiskAssessment:
        """综合软风控评估"""
        checks = [
            self.check_stop_loss(current_return),
            self.check_take_profit(current_return),
            self.check_holding_period(holding_days),
            self.check_drawdown(portfolio_drawdown),
        ]
        if half_life_days is not None:
            checks.append(self.check_half_life(half_life_days, holding_days))
        if invalid_conditions and current_state:
            checks.append(self.check_invalid_conditions(invalid_conditions, current_state))

        # 取最差信号
        signal_priority = {
            SignalType.AVOID: 0,
            SignalType.SELL: 1,
            SignalType.REDUCE: 2,
            SignalType.WATCH: 3,
            SignalType.BUY: 4,
            SignalType.STRONG_BUY: 5,
        }

        worst = min(checks, key=lambda x: signal_priority.get(x.signal, 3))
        all_reasons = []
        all_actions = []
        all_triggered = []
        for c in checks:
            all_reasons.extend(c.reasons)
            all_actions.extend(c.suggested_actions)
            all_triggered.extend(c.triggered_invalid_conditions)

        return SoftRiskAssessment(
            signal=worst.signal,
            confidence=worst.confidence,
            reasons=all_reasons,
            triggered_invalid_conditions=list(set(all_triggered)),
            suggested_actions=list(set(all_actions)),
        )
