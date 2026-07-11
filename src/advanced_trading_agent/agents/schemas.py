"""
Agent 输出 Schema — 重写版

与 data_service/schema.py 的分工:
- data_service/schema.py: 数据格式 (行情/资金/因子等原始数据)
- agents/schemas.py: Agent 分析结果 (报告/决策等)

借鉴 TradingAgents' schemas.py 的 Pydantic 结构化输出模式
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# 通用类型
# ============================================================

class DecisionType(str, Enum):
    RECOMMEND = "推荐"
    WATCH = "观察"
    REJECT = "拒绝"


class RiskVerdict(str, Enum):
    HARD_VETO = "HARD_VETO"
    SOFT_VETO = "SOFT_VETO"
    PASS = "PASS"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================
# Market Agent
# ============================================================

class MarketReport(BaseModel):
    """市场温度分析报告"""
    market_state: str = Field(description="市场情绪: 冰点/低迷/正常/温热/高潮")
    position_cap: float = Field(ge=0, le=1, description="建议组合仓位上限")
    capital_confirmation: str = Field(description="资金确认: 资金确认/资金背离/资金不足")
    sector_preference: list[str] = Field(default_factory=list, description="当前适合进攻的板块")
    risk_warning: str | None = Field(default=None, description="市场风险警告")
    reasoning: str = Field(description="核心判断逻辑")

    def to_markdown(self) -> str:
        return (
            f"**市场状态**: {self.market_state}\n"
            f"**仓位上限**: {self.position_cap:.0%}\n"
            f"**资金状态**: {self.capital_confirmation}\n"
            f"**偏好板块**: {', '.join(self.sector_preference) if self.sector_preference else '无'}\n"
            f"**风控警告**: {self.risk_warning or '无'}\n"
            f"**逻辑**: {self.reasoning}"
        )


# ============================================================
# Event Agent
# ============================================================

class EventReport(BaseModel):
    """事件分析报告"""
    event_id: str = Field(description="事件ID")
    event_type: Literal["政策", "订单", "业绩", "合规", "情绪", "供给冲击"] = Field(description="事件类型")
    direction: Literal["利好", "利空", "中性"] = Field(description="方向")
    confidence: float = Field(ge=0, le=1, description="置信度")
    transmission_path: str = Field(description="传导路径")
    direct_beneficiaries: list[str] = Field(default_factory=list, description="直接受益标的")
    half_life_days: int | None = Field(default=None, description="半衰期(交易日)")
    invalid_conditions: list[str] = Field(default_factory=list, description="证伪条件")
    evidence_level: str = Field(description="证据等级: 公告/权威媒体/行业媒体/社交传闻")
    pricing_status: str = Field(description="定价状态: 未定价/部分定价/过度定价")
    chain_quality: Literal["direct", "indirect", "weak"] = Field(description="链条质量")
    reasoning: str = Field(description="分析逻辑")

    def to_markdown(self) -> str:
        return (
            f"**事件**: [{self.event_type}] {self.event_id}\n"
            f"**方向**: {self.direction} (置信度: {self.confidence:.0%})\n"
            f"**传导**: {self.transmission_path}\n"
            f"**受益**: {', '.join(self.direct_beneficiaries) if self.direct_beneficiaries else '无明确受益'}\n"
            f"**半衰期**: {self.half_life_days or 'N/A'}天\n"
            f"**证据**: {self.evidence_level} | **定价**: {self.pricing_status}\n"
            f"**链条质量**: {self.chain_quality}\n"
            f"**失效条件**: {'; '.join(self.invalid_conditions) if self.invalid_conditions else '无'}\n"
            f"**逻辑**: {self.reasoning}"
        )


# ============================================================
# Analysis Agent
# ============================================================

class StockRanking(BaseModel):
    """个股排序"""
    code: str
    name: str
    composite_score: float = Field(ge=0, le=10)
    main_driver: str = Field(description="主要驱动因子")
    warnings: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """因子分析报告"""
    sector_score: float | None = Field(default=None, description="板块综合评分")
    stock_rankings: list[StockRanking] = Field(default_factory=list, description="个股排序")
    factor_explanation: str = Field(description="主要驱动因子说明")
    timing_filter: str | None = Field(default=None, description="择时过滤条件")
    factor_warnings: list[str] = Field(default_factory=list, description="因子失效/拥挤提醒")
    reasoning: str = Field(description="分析逻辑")

    def to_markdown(self) -> str:
        lines = [
            f"**板块评分**: {self.sector_score or 'N/A'}",
            f"**因子说明**: {self.factor_explanation}",
        ]
        if self.stock_rankings:
            lines.append(f"**个股Top {len(self.stock_rankings)}**:")
            for s in self.stock_rankings:
                warnings = f" ⚠{','.join(s.warnings)}" if s.warnings else ""
                lines.append(f"  {s.name}({s.code}): {s.composite_score:.1f}分 - {s.main_driver}{warnings}")
        if self.timing_filter:
            lines.append(f"**择时条件**: {self.timing_filter}")
        if self.factor_warnings:
            lines.append(f"**因子警告**: {'; '.join(self.factor_warnings)}")
        lines.append(f"**逻辑**: {self.reasoning}")
        return "\n".join(lines)


# ============================================================
# Backtest Agent
# ============================================================

class BacktestReport(BaseModel):
    """回测验证报告"""
    sample_size: int = Field(description="样本数")
    win_rate: float = Field(ge=0, le=1, description="胜率")
    avg_excess_return: float = Field(description="平均超额收益率")
    best_holding_period: int | None = Field(default=None, description="最佳持仓周期")
    failure_pattern: str | None = Field(default=None, description="失败案例共性")
    confidence: Confidence = Field(description="置信度")
    reasoning: str = Field(description="验证逻辑")

    def to_markdown(self) -> str:
        return (
            f"**样本数**: {self.sample_size} "
            f"({'✅ 充足' if self.sample_size >= 30 else '⚠ 不足'})\n"
            f"**胜率**: {self.win_rate:.1%}\n"
            f"**平均超额**: {self.avg_excess_return:+.2%}\n"
            f"**最佳持仓**: {self.best_holding_period or 'N/A'}天\n"
            f"**失败共性**: {self.failure_pattern or '无'}\n"
            f"**置信度**: {self.confidence.value}\n"
            f"**逻辑**: {self.reasoning}"
        )


# ============================================================
# System Agent
# ============================================================

class SystemDecision(BaseModel):
    """系统最终裁定 — 所有 Agent 的综合结果"""
    decision: DecisionType = Field(description="推荐/观察/拒绝")
    position: float | None = Field(default=None, ge=0, le=1, description="建议仓位")
    alpha_source: list[str] = Field(description="Alpha 来源")
    horizon_days: int = Field(default=5, description="评估周期")
    reasons: list[str] = Field(description="支持理由")
    objections: list[str] = Field(description="反对意见")
    invalid_conditions: list[str] = Field(default_factory=list, description="失效条件")
    risk_verdict: RiskVerdict = Field(default=RiskVerdict.PASS, description="风控结果")
    risk_details: list[str] = Field(default_factory=list, description="风控详情")
    reasoning: str = Field(description="裁定逻辑")

    def to_markdown(self) -> str:
        return (
            f"**裁定**: {self.decision.value}\n"
            f"**仓位**: {f'{self.position:.0%}' if self.position else 'N/A'}\n"
            f"**Alpha来源**: {'/'.join(self.alpha_source) if self.alpha_source else '无'}\n"
            f"**评估周期**: {self.horizon_days}天\n"
            f"**风控**: {self.risk_verdict.value}\n"
        )


class SystemRubric(BaseModel):
    """系统裁定前的结构化评分表"""
    market_score: int = Field(ge=0, le=2, description="市场环境评分")
    event_score: int = Field(ge=0, le=2, description="事件链条评分")
    analysis_score: int = Field(ge=0, le=2, description="因子/排序评分")
    backtest_score: int = Field(ge=0, le=2, description="历史证据评分")
    memory_score: int = Field(ge=0, le=1, description="历史记忆支持评分")
    risk_score: int = Field(ge=0, le=2, description="风控评分")
    total_score: int = Field(ge=0, le=11, description="总分")
    recommendation_floor: DecisionType = Field(description="规则给出的最高裁定等级")
    support: list[str] = Field(default_factory=list, description="支持项")
    objections: list[str] = Field(default_factory=list, description="反对项")
    forced_downgrades: list[str] = Field(default_factory=list, description="强制降级原因")


# ============================================================
# Memory Agent
# ============================================================

class MemoryRecall(BaseModel):
    """Memory 召回结果"""
    success_cases: list[dict] = Field(default_factory=list, description="相似成功案例")
    failure_cases: list[dict] = Field(default_factory=list, description="相似失败案例")
    agent_accuracy: dict[str, float] = Field(default_factory=dict, description="Agent历史准确率")
    historical_warnings: list[str] = Field(default_factory=list, description="历史误判提醒")
    reasoning: str = Field(description="分析逻辑")


# ============================================================
# Report Agent
# ============================================================

class FinalReport(BaseModel):
    """最终报告"""
    code: str = Field(description="标的代码")
    name: str = Field(description="标的名称")
    trade_date: str = Field(description="交易日")
    decision: DecisionType = Field(description="结论")
    position: float | None = Field(default=None, description="建议仓位")
    alpha_source: list[str] = Field(description="Alpha 来源")
    horizon_days: int = Field(default=5, description="评估周期")
    reasons: list[str] = Field(description="支持理由")
    objections: list[str] = Field(description="反对意见")
    invalid_conditions: list[str] = Field(default_factory=list, description="失效条件")
    risk_verdict: str = Field(default="PASS", description="风控结果")
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        """输出为明日观察池格式"""
        lines = [
            "# 明日观察池报告",
            "",
            f"**标的**: {self.name}({self.code})",
            f"**交易日**: {self.trade_date}",
            f"**生成时间**: {self.generated_at}",
            "",
            "---",
            "",
            f"## 结论: {self.decision.value}",
            "",
        ]
        if self.position:
            lines.append(f"**建议仓位**: {self.position:.0%}")
            lines.append("")
        if self.alpha_source:
            lines.append(f"**Alpha来源**: {'/'.join(self.alpha_source)}")
            lines.append("")
        if self.reasons:
            lines.append("**支持理由**:")
            lines.extend(f"- {r}" for r in self.reasons)
            lines.append("")
        if self.objections:
            lines.append("**反对意见**:")
            lines.extend(f"- {o}" for o in self.objections)
            lines.append("")
        if self.invalid_conditions:
            lines.append("**失效条件**:")
            lines.extend(f"- {c}" for c in self.invalid_conditions)
            lines.append("")
        lines.extend([
            f"**风控**: {self.risk_verdict}",
            "",
            "---",
            "## 复盘字段",
            f"- 评估周期: {self.horizon_days}天",
            f"- 主要Alpha: {'/'.join(self.alpha_source) if self.alpha_source else '无'}",
            "- 需追踪: 1/3/5/10日超额收益",
        ])
        return "\n".join(lines)
