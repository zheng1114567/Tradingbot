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
                                   current_state: dict[str, Any],
                                   llm: Any = None) -> SoftRiskAssessment:
        """检查失效条件是否触发

        数值条件由确定性规则检查；自然语言条件（政策、舆情等）
        通过 LLM 评估 current_state 中的事件和新闻数据。

        Args:
            conditions: 失效条件文案列表
            current_state: 包含 events、price_change、volume_change、index_change 等字段
            llm: 可选的 LLMClient，用于自然语言条件判断
        """
        triggered = []
        llm_candidates: list[str] = []

        for condition in conditions:
            if "放量下跌" in condition:
                if (current_state.get("volume_change", 0) > 1.5 and
                        current_state.get("price_change", 0) < -0.03):
                    triggered.append(condition)
            elif "大盘" in condition and "跌破" in condition:
                if current_state.get("index_change", 0) < -0.05:
                    triggered.append(condition)
            elif "政策" in condition or "低于预期" in condition or "舆情" in condition or "负面" in condition:
                llm_candidates.append(condition)
            elif "涨幅" in condition and "过高" in condition:
                if current_state.get("price_change", 0) > 0.15:
                    triggered.append(condition)

        # LLM 评估需要自然语言理解的失效条件
        if llm_candidates and llm is not None:
            llm_verdicts = self._evaluate_conditions_with_llm(
                llm_candidates, current_state, llm
            )
            triggered.extend(llm_verdicts)
        elif llm_candidates:
            # 无 LLM 可用时，对政策/舆情类条件保守处理：存在即提示，但标记低置信度
            triggered.extend(llm_candidates)

        if triggered:
            return SoftRiskAssessment(
                signal=SignalType.REDUCE,
                confidence=0.7 if (llm is not None and llm_candidates) else 0.55,
                reasons=triggered,
                triggered_invalid_conditions=triggered,
                suggested_actions=["失效条件触发, 考虑减仓或退出"],
            )
        return SoftRiskAssessment(signal=SignalType.BUY)

    @staticmethod
    def _evaluate_conditions_with_llm(
        conditions: list[str],
        state: dict[str, Any],
        llm: Any,
    ) -> list[str]:
        """Use LLM to judge whether policy/news conditions are triggered.

        Returns the subset of conditions that the LLM considers active.
        """
        events = state.get("events", []) or []
        news_text = state.get("news_summary", "")
        if not events and not news_text:
            return conditions  # conservative: trigger if we have conditions but no data

        event_snippets = []
        for ev in events[:10]:
            title = str(ev.get("title") or ev.get("headline") or "")
            direction = str(ev.get("direction") or ev.get("sentiment") or "")
            if title:
                event_snippets.append(f"- [{direction}] {title[:120]}")

        prompt = (
            "你是一个 A 股量化交易系统的失效条件评估器。\n"
            "根据当前事件和新闻数据，判断以下失效条件是否已经触发。\n"
            "只返回一个 JSON 数组，包含已触发的条件原文，不要输出其他内容。\n"
            "如果条件没有触发，返回空数组 []。\n"
            "触发标准：事件或新闻中明确出现了对应条件的证据。\n"
            f"\n失效条件清单:\n"
            f"{chr(10).join(f'- {c}' for c in conditions)}\n"
            f"\n当前事件摘要:\n"
            f"{chr(10).join(event_snippets) if event_snippets else '无事件数据'}\n"
            f"\n新闻摘要: {news_text[:800] if news_text else '无'}"
        )

        try:
            raw = llm.chat(
                [("system", "只返回 JSON 数组，不要解释"), ("human", prompt)],
                temperature=0,
                max_tokens=256,
            )
            import json
            parsed = json.loads(str(raw).strip())
            if isinstance(parsed, list):
                return [c for c in conditions if any(c in str(item) for item in parsed)]
        except Exception:
            pass

        return conditions  # conservative fallback on LLM error

    def assess_all(self, holding_days: int = 0,
                    current_return: float = 0,
                    portfolio_drawdown: float = 0,
                    half_life_days: int | None = None,
                    invalid_conditions: list[str] | None = None,
                    current_state: dict[str, Any] | None = None,
                    llm: Any = None) -> SoftRiskAssessment:
        """综合软风控评估 — 支持 LLM 辅助评估自然语言失效条件"""
        checks = [
            self.check_stop_loss(current_return),
            self.check_take_profit(current_return),
            self.check_holding_period(holding_days),
            self.check_drawdown(portfolio_drawdown),
        ]
        if half_life_days is not None:
            checks.append(self.check_half_life(half_life_days, holding_days))
        if invalid_conditions and current_state:
            checks.append(self.check_invalid_conditions(invalid_conditions, current_state, llm=llm))

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
