"""Data quality gates for backtest and paper-trading evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataQAIssue:
    """One data quality issue."""

    severity: str
    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "field": self.field,
            "message": self.message,
        }


@dataclass
class DataQAReport:
    """Data quality gate report."""

    passed: bool
    issues: list[DataQAIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": self.summary,
        }


class DataQualityGate:
    """Validate the minimum dataset needed for profit evaluation."""

    required_price_columns = {"code", "trade_date", "open", "close"}
    required_signal_columns = {"code", "signal_date", "decision"}

    def validate(
        self,
        *,
        signals: pd.DataFrame | list[dict[str, Any]],
        prices: pd.DataFrame | list[dict[str, Any]],
        run_time: str | None = None,
    ) -> DataQAReport:
        signals_df = pd.DataFrame(signals)
        prices_df = pd.DataFrame(prices)
        issues: list[DataQAIssue] = []

        issues.extend(self._missing_columns(signals_df, self.required_signal_columns, "signals"))
        issues.extend(self._missing_columns(prices_df, self.required_price_columns, "prices"))

        if issues:
            return DataQAReport(False, issues, {
                "signals_count": len(signals_df),
                "prices_count": len(prices_df),
            })

        signals_df = signals_df.copy()
        prices_df = prices_df.copy()
        signals_df["signal_date"] = pd.to_datetime(signals_df["signal_date"], errors="coerce")
        prices_df["trade_date"] = pd.to_datetime(prices_df["trade_date"], errors="coerce")

        if signals_df["signal_date"].isna().any():
            issues.append(DataQAIssue("error", "signals.signal_date", "存在无法解析的信号日期"))
        if prices_df["trade_date"].isna().any():
            issues.append(DataQAIssue("error", "prices.trade_date", "存在无法解析的行情日期"))
        if (prices_df["open"].astype(float) <= 0).any() or (prices_df["close"].astype(float) <= 0).any():
            issues.append(DataQAIssue("error", "prices.open_close", "存在非正价格"))

        if "available_at" in signals_df.columns and run_time:
            available_at = pd.to_datetime(signals_df["available_at"], errors="coerce")
            if (available_at > pd.Timestamp(run_time)).any():
                issues.append(DataQAIssue("error", "signals.available_at", "信号包含运行时点之后才可得的数据"))

        duplicate_prices = prices_df.duplicated(["code", "trade_date"]).sum()
        if duplicate_prices:
            issues.append(DataQAIssue("warning", "prices", f"存在 {duplicate_prices} 条重复 code/trade_date 行情"))

        return DataQAReport(
            passed=not any(issue.severity == "error" for issue in issues),
            issues=issues,
            summary={
                "signals_count": len(signals_df),
                "prices_count": len(prices_df),
                "unique_codes": int(prices_df["code"].nunique()) if "code" in prices_df else 0,
                "duplicate_prices": int(duplicate_prices),
            },
        )

    @staticmethod
    def _missing_columns(df: pd.DataFrame, required: set[str], label: str) -> list[DataQAIssue]:
        missing = sorted(required - set(df.columns))
        if not missing:
            return []
        return [
            DataQAIssue("error", label, f"缺少必要字段: {', '.join(missing)}")
        ]

