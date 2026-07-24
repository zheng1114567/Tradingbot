"""Data quality gates for backtest and paper-trading evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
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


@dataclass
class BacktestDataset:
    """Local-cache dataset prepared before a backtest run."""

    prices_by_code: dict[str, pd.DataFrame]
    coverage: dict[str, Any]


class BacktestDatasetBuilder:
    """Build point-in-time price datasets from local cache before simulation.

    The backtest engine should receive a prepared in-memory dataset.  This
    keeps network/cache repair outside the simulation loop and makes missing
    30/60-day history explicit.
    """

    def __init__(self, *, lookback_days: int = 60, min_coverage_days: int = 30):
        self.lookback_days = lookback_days
        self.min_coverage_days = min_coverage_days

    def build(
        self,
        signals: list[dict[str, Any]],
        *,
        end_date: str,
        etf: bool = False,
    ) -> BacktestDataset:
        from ..data_agent.local_cache import get_cached_daily, get_cached_etf_daily

        end = _parse_iso(end_date)
        start = end - timedelta(days=self.lookback_days)
        loader = get_cached_etf_daily if etf else get_cached_daily
        prices_by_code: dict[str, pd.DataFrame] = {}
        per_code: dict[str, dict[str, Any]] = {}

        codes = sorted({str(item.get("code") or item.get("ticker") or "") for item in signals if item.get("code") or item.get("ticker")})
        for code in codes:
            rows = loader(code, start.isoformat(), end.isoformat())
            frame = pd.DataFrame(rows)
            trade_days = _unique_trade_day_count(frame)
            status = "ok" if trade_days >= self.min_coverage_days else "insufficient_cache"
            if not frame.empty:
                prices_by_code[code] = frame
            per_code[code] = {
                "status": status,
                "row_count": int(len(frame)),
                "trade_day_count": trade_days,
                "required_min_trade_days": self.min_coverage_days,
            }

        return BacktestDataset(
            prices_by_code=prices_by_code,
            coverage={
                "status": "ok" if per_code and all(item["status"] == "ok" for item in per_code.values()) else "insufficient_cache",
                "lookback_days": self.lookback_days,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "codes": per_code,
            },
        )


def _unique_trade_day_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    for column in ("trade_date", "datetime", "date"):
        if column in frame.columns:
            return int(pd.to_datetime(frame[column], errors="coerce").dt.date.nunique())
    return 0


def _parse_iso(value: str) -> date:
    raw = str(value)
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return pd.to_datetime(raw).date()
