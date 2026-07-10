"""
Point-in-time 数据点位强制校验

每个数据字段记录:
- as_of_date: 数据所属交易日
- event_time: 事件实际发生时间
- available_at: 系统可使用该数据的时间 (盘后/盘中等)
- source: 数据来源
- vendor_version: 供应商版本

回测强制规则:
- available_at > run_time → 该字段标记为 unavailable
- 回测时每个样本只能读取当时的 point_in_time_manifest
- 若关键字段缺失 → 系统输出数据报告, 不给交易建议

借鉴 TradingAgents' yfinance_news.py 的 news_window 回溯安全模式,
但这里做了更严格的 point-in-time 框架。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any


@dataclass
class PointInTime:
    """单个数据字段的点位信息"""
    as_of_date: date                       # 数据所属交易日
    event_time: datetime | None = None     # 事件实际发生时间
    available_at: datetime | None = None   # 系统可用的最早时间
    ingested_at: datetime | None = None    # 进入本系统的时间
    source: str = ""                       # 数据来源
    vendor_version: str | None = None      # 供应商版本
    is_critical: bool = False              # 是否关键字段 (缺失则不能推荐)


@dataclass
class PointInTimeManifest:
    """整次运行的数据时间边界清单"""
    run_time: datetime                     # 运行时间
    trade_date: date                       # 数据所属交易日
    available_fields: dict[str, bool] = field(default_factory=dict)   # 字段可用性
    critical_missing: list[str] = field(default_factory=list)         # 缺失的关键字段
    data_quality: str = "ok"               # ok / degraded / failed
    quality_reason: str = ""               # 降级原因

    def can_recommend(self) -> bool:
        """是否可以输出推荐 (关键字段全部可用)"""
        return len(self.critical_missing) == 0 and self.data_quality != "failed"

    def can_output_report(self) -> bool:
        """是否可以输出数据报告 (非关键字段可用)"""
        return self.data_quality != "failed"


def check_field_availability(
    field_pit: PointInTime,
    run_time: datetime,
) -> bool:
    """检查某个字段在 run_time 是否可用

    规则:
    - available_at 未设置 → 假设可用
    - available_at > run_time → 不可用 (未来数据)
    """
    if field_pit.available_at is None:
        return True
    return field_pit.available_at <= run_time


# A 股关键数据的时间约束
A_SHARE_DATA_AVAILABILITY = {
    "daily_close": {
        "available_at": "15:05",     # 收盘后约 5 分钟
        "is_critical": True,
        "description": "日线收盘价",
    },
    "capital_flow": {
        "available_at": "15:30",     # 资金流收盘后约 30 分钟
        "is_critical": False,
        "description": "资金流向",
    },
    "northbound_flow": {
        "available_at": "17:30",     # 北向资金盘后才出
        "is_critical": False,
        "description": "北向资金",
    },
    "financial_data": {
        "available_at": "next_day",  # 财务数据次日才完整
        "is_critical": True,
        "description": "财务指标",
    },
    "news": {
        "available_at": "realtime",  # 新闻实时可用
        "is_critical": False,
        "description": "新闻",
    },
    "limit_up_down": {
        "available_at": "15:05",     # 收盘后可知涨跌停
        "is_critical": True,
        "description": "涨跌停状态",
    },
}
