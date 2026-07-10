"""
数据 Schema 定义 — Tier 1 / Tier 2

Tier 1: 每次讨论默认加载的轻量摘要 (大盘、情绪、板块 Top 10、资金概览、风险)
Tier 2: 需要时再加载的详细数据 (个股因子、具体事件、回测样本)

与 TradingAgents schemas.py 的区别:
- 这里定义的是数据层 Schema, Agent 输出 Schema 在 agents/schemas.py
- 每个字段记录 point-in-time 信息 (as_of_date, available_at)

借鉴 TradingAgents 的 Pydantic 结构化模式:
- Field description = 字段说明
- field_validator 处理边界值
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ============================================================
# 点时间信息
# ============================================================

class PointInTime(BaseModel):
    """数据点位信息 — 每个数据字段都需要记录"""
    as_of_date: date = Field(description="数据所属交易日")
    event_time: datetime | None = Field(default=None, description="事件实际发生时间")
    available_at: datetime | None = Field(default=None, description="系统可使用该数据的时间")
    ingested_at: datetime | None = Field(default=None, description="数据进入本系统的时间")
    source: str = Field(default="", description="数据来源")
    vendor_version: str | None = Field(default=None, description="供应商数据版本")


# ============================================================
# Tier 1 Schema — 每次讨论默认加载
# ============================================================

class MarketSentiment(str, Enum):
    ICE = "冰点"
    LOW = "低迷"
    NORMAL = "正常"
    WARM = "温热"
    HIGH = "高潮"


class CapitalConfirmation(str, Enum):
    CONFIRMED = "资金确认"
    DIVERGENCE = "资金背离"
    INSUFFICIENT = "资金不足"
    UNKNOWN = "未知"


class RiskLevel(str, Enum):
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    CRITICAL = "极危"  # 触发 HARD_VETO


class DecisionType(str, Enum):
    RECOMMEND = "推荐"
    WATCH = "观察"
    REJECT = "拒绝"


class MarketSchema(BaseModel):
    """大盘状态摘要 — Tier 1"""
    pit: PointInTime
    index_close: float = Field(description="大盘收盘价")
    index_change_pct: float = Field(description="大盘涨跌幅%")
    advance_count: int = Field(description="上涨家数")
    decline_count: int = Field(description="下跌家数")
    limit_up_count: int = Field(description="涨停家数")
    limit_down_count: int = Field(description="跌停家数")
    total_volume_cny: float = Field(description="两市总成交额")


class SentimentSchema(BaseModel):
    """市场情绪 — Tier 1"""
    pit: PointInTime
    sentiment: MarketSentiment = Field(description="情绪档位")
    sentiment_score: float = Field(ge=0, le=100, description="情绪综合分 0-100")
    limit_up_break_rate: float = Field(default=0.0, description="炸板率%")
    media_ratio: float | None = Field(default=None, description="媒体情绪比 (正面/负面)")


class CapitalSchema(BaseModel):
    """资金流向 — Tier 1"""
    pit: PointInTime
    sector_name: str = Field(description="板块名称")
    sector_volume_cny: float = Field(description="板块成交额")
    net_inflow_main: float = Field(description="主力净流入")
    net_inflow_retail: float = Field(description="散户净流入")
    consecutive_inflow_days: int = Field(default=0, description="连续净流入天数")
    confirmation: CapitalConfirmation = Field(description="资金确认状态")


class SectorSchema(BaseModel):
    """板块概况 — Tier 1"""
    pit: PointInTime
    sector_rank: list[SectorRankItem] = Field(default_factory=list, description="板块 Top N 排名")


class SectorRankItem(BaseModel):
    """板块排名单项"""
    rank: int
    sector_name: str
    change_pct: float
    strength_score: float  # 板块强度评分


class RiskSchema(BaseModel):
    """风险状态 — Tier 1"""
    pit: PointInTime
    overall_risk: RiskLevel = Field(description="整体风险评估")
    st_stocks_count: int = Field(default=0, description="ST 股票数")
    suspension_count: int = Field(default=0, description="停牌股票数")
    delisting_risk_count: int = Field(default=0, description="退市风险股票数")


# ============================================================
# Tier 2 Schema — 按需加载
# ============================================================

class FactorSchema(BaseModel):
    """个股因子明细 — Tier 2"""
    pit: PointInTime
    code: str = Field(description="股票代码")
    name: str = Field(description="股票名称")
    sector: str = Field(default="", description="所属板块")
    # 六因子
    quality_score: float | None = Field(default=None, description="质量因子 (ROE, 毛利率等)")
    growth_score: float | None = Field(default=None, description="成长因子 (营收/利润增速)")
    valuation_score: float | None = Field(default=None, description="估值因子 (PE/PB分位数)")
    momentum_score: float | None = Field(default=None, description="动量因子 (过去N日收益)")
    volatility_score: float | None = Field(default=None, description="波动率因子")
    liquidity_score: float | None = Field(default=None, description="流动性因子")
    # 综合
    composite_score: float | None = Field(default=None, description="因子综合分")
    factor_warning: str | None = Field(default=None, description="因子拥挤或失效警告")


class EventSchema(BaseModel):
    """结构化事件 — Tier 2"""
    pit: PointInTime
    event_id: str = Field(description="事件唯一ID")
    event_type: Literal["政策", "订单", "业绩", "合规", "情绪", "供给冲击"] = Field(description="事件类型")
    summary: str = Field(description="事件摘要")
    direction: Literal["利好", "利空", "中性"] = Field(description="方向")
    confidence: float = Field(ge=0, le=1, description="置信度")
    # 传导分析
    transmission_path: str = Field(description="传导路径: 事件→业绩/估值/资金")
    direct_beneficiaries: list[str] = Field(default_factory=list, description="直接受益板块或标的")
    half_life_days: int | None = Field(default=None, description="半衰期(交易日)")
    invalid_conditions: list[str] = Field(default_factory=list, description="证伪条件")
    # 证据
    evidence_level: Literal["公告/披露", "权威媒体", "行业媒体", "社交传闻"] = Field(description="证据等级")
    pricing_status: Literal["未定价", "部分定价", "疑似过度定价"] = Field(description="定价状态")
    chain_quality: Literal["direct", "indirect", "weak"] = Field(
        description="链条质量: direct=可推荐, indirect=仅观察, weak=不进入"
    )

    @field_validator("chain_quality")
    @classmethod
    def check_chain_quality(cls, v):
        if v == "weak":
            return v  # weak 需要由 System Agent 决定是否降级
        return v


class BacktestSampleSchema(BaseModel):
    """回测样本 — Tier 2"""
    pit: PointInTime
    sample_size: int = Field(description="样本数")
    win_rate: float = Field(description="胜率", ge=0, le=1)
    avg_excess_return: float = Field(description="平均超额收益率")
    avg_return_1d: float = Field(default=0, description="1日平均收益")
    avg_return_3d: float = Field(default=0, description="3日平均收益")
    avg_return_5d: float = Field(default=0, description="5日平均收益")
    avg_return_10d: float = Field(default=0, description="10日平均收益")
    sharpe_ratio: float | None = Field(default=None, description="夏普比")
    max_drawdown: float | None = Field(default=None, description="最大回撤")
    best_holding_period: int | None = Field(default=None, description="最佳持仓周期")
    failure_cases_count: int = Field(default=0, description="失败案例数")
    failure_common_pattern: str | None = Field(default=None, description="失败共性")
    confidence: Literal["high", "medium", "low"] = Field(description="置信度")


# ============================================================
# Tier Manifest — 数据时间边界
# ============================================================

class PointInTimeManifest(BaseModel):
    """本次运行的数据时间边界"""
    run_time: datetime = Field(description="运行时间")
    trade_date: date = Field(description="数据所属交易日")
    available_fields: dict[str, bool] = Field(default_factory=dict, description="字段可用性")
    field_versions: dict[str, str] = Field(default_factory=dict, description="字段版本")
    data_quality_report: str | None = Field(default=None, description="数据质量报告")
