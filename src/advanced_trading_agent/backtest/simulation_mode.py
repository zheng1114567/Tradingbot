"""
回测模拟模式 — 用规则代替真实 LLM Agent, 大幅节省回测成本

Live 模式: 每个 Agent 用真实 LLM (一次分析 ¥1-5)
Backtest 模式: 每个 Agent 用规则模拟 (一次分析 ¥0.001, 快 1000 倍)

模拟策略:
- Market Agent: 基于情绪 → 仓位映射规则
- Event Agent: 基于事件类型 → 方向规则
- Analysis Agent: 基于因子排序规则
- Backtest Agent: 基于统计规则
- System Agent: 基于规则裁定 + 硬风控

用于:
1. 大规模回测 (1000 天 × 100 只股票)
2. 对比实验 (多 Agent vs 规则基线)
3. 因子有效性测试

借鉴 TradingAgents: 没有模拟模式 (回测就是跑真实 LLM),
这是我们的创新点。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..agents.schemas import (
    AnalysisReport,
    BacktestReport,
    Confidence,
    DecisionType,
    EventReport,
    MarketReport,
    RiskVerdict,
    StockRanking,
    SystemDecision,
)
# 情绪 → 仓位上限映射
SENTIMENT_POSITION_CAP = {
    "冰点": 0.20, "低迷": 0.40, "正常": 0.60,
    "温热": 0.50, "高潮": 0.30,
}
from ..risk.hard_risk import HardRiskController, RiskVerdictType

logger = logging.getLogger(__name__)


class SimulationAgent:
    """规则模拟 Agent — 代替真实 LLM"""

    @staticmethod
    def simulate_market(tier1: dict[str, Any]) -> MarketReport:
        """模拟 Market Agent"""
        sentiment = tier1.get("sentiment", {}).get("sentiment", "正常")
        position_cap = SENTIMENT_POSITION_CAP.get(sentiment, 0.50)
        capital_conf = tier1.get("capital", {}).get("confirmation", "未知")

        return MarketReport(
            market_state=sentiment,
            position_cap=position_cap,
            capital_confirmation=capital_conf,
            sector_preference=[],
            reasoning=f"模拟: 情绪{sentiment} → 仓位{position_cap:.0%}",
        )

    @staticmethod
    def simulate_event(events: list[dict[str, Any]]) -> EventReport | None:
        """模拟 Event Agent"""
        if not events:
            return None

        # 取第一个重要事件
        event = events[0]
        return EventReport(
            event_id=event.get("event_id", "sim_event"),
            event_type=event.get("event_type", "情绪"),
            direction=event.get("direction", "中性"),
            confidence=event.get("confidence", 0.5),
            transmission_path=event.get("transmission_path", "模拟路径"),
            direct_beneficiaries=event.get("direct_beneficiaries", []),
            evidence_level=event.get("evidence_level", "权威媒体"),
            pricing_status=event.get("pricing_status", "未定价"),
            chain_quality=event.get("chain_quality", "indirect"),
            reasoning="模拟: 基于结构化事件数据的规则分析",
            invalid_conditions=event.get("invalid_conditions", []),
        )

    @staticmethod
    def simulate_analysis(factors: list[dict[str, Any]]) -> AnalysisReport:
        """模拟 Analysis Agent (纯规则排序)"""
        sorted_factors = sorted(
            factors,
            key=lambda x: x.get("composite_score", 0) or 0,
            reverse=True,
        )
        rankings = [
            StockRanking(
                code=f.get("code", ""),
                name=f.get("name", ""),
                composite_score=f.get("composite_score", 5) or 5,
                main_driver=f"综合分{f.get('composite_score', 'N/A')}",
                warnings=(
                    ["流动性不足"] if (f.get("liquidity_score") or 5) < 3
                    else ["估值偏高"] if (f.get("valuation_score") or 5) > 8
                    else []
                ),
            )
            for f in sorted_factors[:10]
        ]

        avg_score = sum(r.composite_score for r in rankings) / len(rankings) if rankings else 0
        return AnalysisReport(
            sector_score=avg_score,
            stock_rankings=rankings,
            factor_explanation="模拟: 基于综合因子评分的确定性排序",
            reasoning="模拟: 规则排序",
        )

    @staticmethod
    def simulate_backtest(samples: list[dict[str, Any]]) -> BacktestReport:
        """模拟 Backtest Agent"""
        if not samples:
            return BacktestReport(
                sample_size=0,
                win_rate=0,
                avg_excess_return=0,
                confidence=Confidence.LOW,
                reasoning="模拟: 无回测样本",
            )

        s = samples[0]
        sample_size = s.get("sample_size", 0)
        return BacktestReport(
            sample_size=sample_size,
            win_rate=s.get("win_rate", 0),
            avg_excess_return=s.get("avg_excess_return", 0),
            best_holding_period=s.get("best_holding_period"),
            failure_pattern=s.get("failure_common_pattern"),
            confidence=Confidence.HIGH if sample_size >= 30 else Confidence.MEDIUM if sample_size >= 10 else Confidence.LOW,
            reasoning=f"模拟: {sample_size}样本, 规则判定置信度",
        )

    @staticmethod
    def simulate_system(
        market: MarketReport,
        event: EventReport | None,
        analysis: AnalysisReport,
        backtest: BacktestReport,
        risk_status: dict[str, Any],
    ) -> SystemDecision:
        """模拟 System Agent (规则裁定)

        规则:
        1. 硬风控 HARD_VETO → 拒绝
        2. 回测胜率 < 50% → 观察
        3. 样本 < 30 + 链条质量不是 direct → 观察
        4. 链条 direct + 胜率 > 55% + 资金确认 → 推荐
        5. 其他 → 观察
        """
        # 硬风控优先
        if risk_status.get("hard_veto"):
            return SystemDecision(
                decision=DecisionType.REJECT,
                position=0,
                alpha_source=[],
                horizon_days=5,
                reasons=[f"硬风控否决: {risk_status.get('reason', '')}"],
                objections=[],
                risk_verdict=RiskVerdict.HARD_VETO,
                risk_details=[risk_status.get("reason", "")],
                reasoning="模拟: 硬风控不通过",
            )

        # 规则裁定
        chain_quality = event.chain_quality if event else "weak"
        capital_conf = market.capital_confirmation
        sample_size = backtest.sample_size
        win_rate = backtest.win_rate

        if (
            chain_quality == "direct"
            and capital_conf == "资金确认"
            and sample_size >= 30
            and win_rate >= 0.55
            and analysis.stock_rankings
            and analysis.stock_rankings[0].composite_score >= 6
        ):
            return SystemDecision(
                decision=DecisionType.RECOMMEND,
                position=market.position_cap * 0.3,
                alpha_source=["模拟规则: factors+event+backtest"],
                horizon_days=5,
                reasons=["Alpha清晰", "资金确认", "因子支持", "回测有效"],
                objections=[],
                invalid_conditions=event.invalid_conditions if event else [],
                risk_verdict=RiskVerdict.PASS,
                reasoning="模拟: 规则裁定-推荐 (满足全部条件)",
            )

        if win_rate < 0.5:
            return SystemDecision(
                decision=DecisionType.REJECT,
                position=0,
                alpha_source=[],
                horizon_days=5,
                reasons=[f"回测胜率 {win_rate:.0%} < 50%"],
                objections=["历史不支持"],
                risk_verdict=RiskVerdict.PASS,
                reasoning="模拟: 规则裁定-拒绝 (胜率不足)",
            )

        return SystemDecision(
            decision=DecisionType.WATCH,
            position=market.position_cap * 0.1,
            alpha_source=[],
            horizon_days=5,
            reasons=["Alpha不清晰或证据不足", "建议继续观察"],
            objections=[],
            risk_verdict=RiskVerdict.PASS,
            reasoning="模拟: 规则裁定-观察 (不满足推荐条件)",
        )
