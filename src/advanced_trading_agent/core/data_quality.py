"""
数据质量检查 — 在每次分析之前执行

检查项:
1. 关键字段是否缺失
2. 数据是否在 point-in-time 窗口内
3. 数据源是否返回了有效数据
4. 数据是否过时 (stale)

借鉴 TradingAgents' market_data_validation_tools.py 的数据验证模式,
但做了更严格的检查并跟 point-in-time 框架整合。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """数据质量报告"""
    passed: bool = True
    critical_missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stale_fields: list[str] = field(default_factory=list)
    grade: str = "A"  # A/B/C/D/F

    def to_prompt(self) -> str:
        """格式化为 Agent prompt"""
        if self.grade == "F":
            return (
                f"[数据质量 - 失败]\n"
                f"关键字段缺失: {', '.join(self.critical_missing)}\n"
                f"无法输出交易建议, 仅输出数据报告\n"
            )
        lines = [f"[数据质量 - {self.grade}]"]
        if self.warnings:
            lines.append(f"警告: {'; '.join(self.warnings)}")
        if self.stale_fields:
            lines.append(f"过时字段: {', '.join(self.stale_fields)}")
        return "\n".join(lines)


class DataQualityChecker:
    """数据质量检查器"""

    # 必须存在的 Tier 1 关键字段
    CRITICAL_TIER1_FIELDS = [
        "index_close", "index_change_pct",
        "advance_count", "decline_count",
        "limit_up_count", "limit_down_count",
    ]

    @classmethod
    def check_tier1(cls, data: dict[str, Any],
                     trade_date: date | None = None) -> DataQualityReport:
        """检查 Tier 1 数据质量"""
        report = DataQualityReport()

        market = data.get("market", {})
        sentiment = data.get("sentiment", {})

        # 检查关键字段
        for field in cls.CRITICAL_TIER1_FIELDS:
            if market.get(field) is None:
                report.critical_missing.append(f"market.{field}")
                report.passed = False

        index_close = market.get("index_close")
        try:
            if index_close is not None and float(index_close) <= 0:
                if "market.index_close" not in report.critical_missing:
                    report.critical_missing.append("market.index_close")
                report.passed = False
        except (TypeError, ValueError):
            if "market.index_close" not in report.critical_missing:
                report.critical_missing.append("market.index_close")
            report.passed = False

        if (market.get("advance_count") or 0) == 0 and (market.get("decline_count") or 0) == 0:
            report.warnings.append("市场涨跌家数全为 0，可能是空数据默认值")

        # 检查情绪数据
        if not sentiment.get("sentiment") or sentiment.get("sentiment") == "未知":
            report.warnings.append("情绪数据缺失")

        # 检查数据过时 (如果传入了交易日)
        if trade_date and market.get("as_of_date"):
            try:
                d = datetime.strptime(str(market["as_of_date"]), "%Y-%m-%d").date()
                if d < trade_date:
                    report.stale_fields.append(f"行情感 {d} 早于 {trade_date}")
                    report.warnings.append("行情数据可能滞后")
            except (ValueError, TypeError):
                pass

        # 评分
        critical_count = len(report.critical_missing)
        warning_count = len(report.warnings)
        if critical_count > 0:
            report.grade = "F"
        elif warning_count > 2:
            report.grade = "C"
        elif warning_count > 0:
            report.grade = "B"
        else:
            report.grade = "A"

        return report

    @classmethod
    def check_event_data(cls, events: list[dict[str, Any]]) -> DataQualityReport:
        """检查事件数据质量"""
        report = DataQualityReport()
        if not events:
            report.warnings.append("无可用事件数据")
            return report

        # 检查事件是否有必要的字段
        for i, e in enumerate(events):
            if not e.get("summary"):
                report.warnings.append(f"事件 #{i} 缺少摘要")

        return report

    @classmethod
    def check_factor_data(cls, factors: list[dict[str, Any]]) -> DataQualityReport:
        """检查因子数据质量"""
        report = DataQualityReport()
        if not factors:
            report.critical_missing.append("factor_data")
            report.passed = False
            report.grade = "F"
            return report

        # 检查因子值是否合理
        valid_count = sum(
            1 for f in factors if f.get("composite_score") is not None
        )
        if valid_count < len(factors) * 0.5:
            report.warnings.append(f"仅 {valid_count}/{len(factors)} 个标的因子完整")

        return report
