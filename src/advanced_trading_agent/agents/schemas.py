"""
Agent 结构化输出 Schema — 借鉴 TradingAgents' schemas.py

使用 Pydantic 确保:
1. Agent 输出结构一致
2. Field description = LLM 输出指令
3. field_validator 处理边界值
4. render helper 保持下游兼容
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..data_service.schema import (
    DecisionType,
    MarketSentiment as SentimentLevel,
    RiskLevel,
)


# ============================================================
# Market Agent 输出
# ============================================================

class MarketReport(BaseModel):
    """市场温度分析报告"""
    market_state: SentimentLevel = Field(description="市场情绪: 冰点/低迷/正常/温热/高潮")
    position_cap: float = Field(ge=0, le=1, description="建议组合仓位上限 0-1")
    capital_confirmation: str = Field(description="资金确认: 资金确认/资金背离/资金不足")
    sector_preference: list[str] = Field(default_factory=list, description="当前适合进攻的板块")
    reasoning: str = Field(description="核心判断逻辑")
    risk_warning: str | None = Field(default=None, description="市场风险警告")


# ============================================================
# Event Agent 输出
# ============================================================

class EventReport(BaseModel):
    """事件分析报告"""
    event_id: str = Field(description="事件ID")
    event_type: Literal["政策", "订单", "业绩", "合规", "情绪", "供给冲击"] = Field(description="事件类型")
    direction: Literal["利好", "利空", "中性"] = Field(description="方向")
    confidence: float = Field(ge=0, le=1, description="置信度 0-1")
    transmission_path: str = Field(description="传导路径: 事件→业绩/估值/资金")
    direct_beneficiaries: list[str] = Field(description="直接受益标的")
    half_life_days: int | None = Field(default=None, description="半衰期(交易日)")
    invalid_conditions: list[str] = Field(default_factory=list, description="证伪条件")
    evidence_level: str = Field(description="证据等级")
    pricing_status: str = Field(description="定价状态: 未定价/部分定价/过度定价")
    chain_quality: Literal["direct", "indirect", "weak"] = Field(description="链条质量")
    reasoning: str = Field(description="分析逻辑")


# ============================================================
# Analysis Agent 输出
# ============================================================

class StockRanking(BaseModel):
    """个股排序"""
    code: str
    name: str
    composite_score: float = Field(ge=0, le=10)
    main_driver: str = Field(description="主要驱动因子")
    warnings: list[str] = Field(default_factory=list)
    rank: int | None = Field(default=None)


class AnalysisReport(BaseModel):
    """因子分析报告"""
    sector_score: float | None = Field(default=None, description="板块综合评分")
    stock_rankings: list[StockRanking] = Field(default_factory=list, description="个股排序")
    factor_explanation: str = Field(description="主要驱动因子说明")
    timing_filter: str | None = Field(default=None, description="择时过滤条件")
    factor_warnings: list[str] = Field(default_factory=list, description="因子失效或拥挤提醒")
    reasoning: str = Field(description="分析逻辑")


# ============================================================
# Backtest Agent 输出
# ============================================================

class BacktestReport(BaseModel):
    """回测验证报告"""
    sample_size: int = Field(description="样本数")
    win_rate: float = Field(ge=0, le=1, description="胜率")
    avg_excess_return: float = Field(description="平均超额收益率")
    best_holding_period: int | None = Field(default=None, description="最佳持仓周期")
    failure_pattern: str | None = Field(default=None, description="失败案例共性")
    confidence: Literal["high", "medium", "low"] = Field(description="置信度")
    similar_success_cases: int = Field(default=0, description="相似成功案例数")
    similar_failure_cases: int = Field(default=0, description="相似失败案例数")
    reasoning: str = Field(description="验证逻辑")


# ============================================================
# System Agent 输出
# ============================================================

class SystemDecision(BaseModel):
    """系统最终裁定"""
    decision: DecisionType = Field(description="推荐/观察/拒绝")
    position: float | None = Field(default=None, ge=0, le=1, description="建议仓位")
    alpha_source: list[str] = Field(description="Alpha 来源")
    horizon_days: int = Field(default=5, description="评估周期")
    reasons: list[str] = Field(description="支持理由")
    objections: list[str] = Field(description="反对意见")
    invalid_conditions: list[str] = Field(default_factory=list, description="失效条件")
    risk_verdict: str = Field(default="PASS", description="风控结果: HARD_VETO/SOFT_VETO/PASS")
    risk_details: list[str] = Field(default_factory=list, description="风控详情")
    reasoning: str = Field(description="裁定逻辑")


# ============================================================
# Memory Agent 输出
# ============================================================

class MemoryRecall(BaseModel):
    """Memory 召回结果"""
    success_cases: list[dict] = Field(default_factory=list, description="相似成功案例")
    failure_cases: list[dict] = Field(default_factory=list, description="相似失败案例")
    agent_accuracy: dict[str, float] = Field(default_factory=dict, description="各 Agent 历史准确率")
    historical_warnings: list[str] = Field(default_factory=list, description="历史误判提醒")
    reasoning: str = Field(description="召回逻辑")


# ============================================================
# Report Agent 输出
# ============================================================

class FinalReport(BaseModel):
    """最终报告"""
    decision: DecisionType = Field(description="结论")
    position: float | None = Field(default=None, description="建议仓位")
    alpha_source: list[str] = Field(description="Alpha 来源")
    horizon_days: int = Field(default=5, description="评估周期")
    reasons: list[str] = Field(description="支持理由")
    objections: list[str] = Field(description="反对意见")
    invalid_conditions: list[str] = Field(default_factory=list, description="失效条件")
    risk_result: str = Field(default="PASS", description="风控结果")
    code: str | None = Field(default=None, description="标的代码")
    name: str | None = Field(default=None, description="标的名称")
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
