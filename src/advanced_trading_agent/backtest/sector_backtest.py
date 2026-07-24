"""Sector ETF backtest — validate roundtable decisions against history.

Wire this into the batch ETF watchlist workflow so every roundtable
decision gets a point-in-time backtest for auditability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from ..config import config
from .engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)

# Default holding horizons in trading days
_DEFAULT_HORIZONS = [1, 3, 5, 10, 20]


@dataclass
class SectorBacktestEntry:
    """Backtest result for one sector-ETF decision."""

    sector: str
    etf_code: str
    etf_name: str
    entry_date: str
    status: str  # "active" | "monitor"
    confidence: str  # "high" | "medium" | "low"
    primary_return: float | None = None
    primary_horizon: int = 5
    excess_return: float | None = None
    returns_by_horizon: dict[int, float | None] = field(default_factory=dict)
    max_drawdown: float | None = None
    tradable: bool = False
    invalid_reason: str = ""
    cost_bps: float = 0
    raw: BacktestResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector": self.sector,
            "etf_code": self.etf_code,
            "etf_name": self.etf_name,
            "entry_date": self.entry_date,
            "status": self.status,
            "confidence": self.confidence,
            "primary_return": self.primary_return,
            "primary_horizon": self.primary_horizon,
            "excess_return": self.excess_return,
            "returns_by_horizon": self.returns_by_horizon,
            "max_drawdown": self.max_drawdown,
            "tradable": self.tradable,
            "invalid_reason": self.invalid_reason,
            "cost_bps": self.cost_bps,
        }


@dataclass
class SectorBacktestSummary:
    """Aggregated backtest for a batch of sector ETF decisions."""

    trade_date: str
    entries: list[SectorBacktestEntry] = field(default_factory=list)
    win_count: int = 0
    total_tradable: int = 0
    avg_return: float | None = None
    avg_excess: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date,
            "entries": [e.to_dict() for e in self.entries],
            "win_count": self.win_count,
            "total_tradable": self.total_tradable,
            "avg_return": self.avg_return,
            "avg_excess": self.avg_excess,
        }


class SectorETFBacktester:
    """Point-in-time backtest for ETF roundtable decisions.

    For each sector decision with a primary ETF, fetch historical daily
    data, simulate a T+1 entry, and compute returns at multiple horizons.

    Intended call site (after AutoGen/rules roundtable):
        backtester = SectorETFBacktester()
        summary = backtester.backtest_decisions(
            decisions=watchlist_report.decisions,
            trade_date=report.trade_date,
        )
    """

    def __init__(
        self,
        *,
        holding_horizons: list[int] | None = None,
        primary_horizon: int = 5,
        price_loader: Any = None,
    ) -> None:
        self.holding_horizons = holding_horizons or _DEFAULT_HORIZONS
        self.primary_horizon = primary_horizon
        self._price_loader = price_loader
        self._engine_cache: dict[str, BacktestEngine] = {}

    def backtest_decisions(
        self,
        *,
        decisions: list[dict[str, Any]],
        trade_date: str,
        price_data: dict[str, pd.DataFrame] | None = None,
    ) -> SectorBacktestSummary:
        """Backtest a list of roundtable final_decisions.

        Args:
            decisions: List of decision dicts from roundtable output.
                       Each must have: sector, primary_etf_code, status, confidence.
            trade_date: Signal date (YYYY-MM-DD). Entry is at T+1.
            price_data: Optional pre-loaded {etf_code: DataFrame} map.

        Returns:
            SectorBacktestSummary with per-decision entries and aggregates.
        """
        entries: list[SectorBacktestEntry] = []
        engine = self._engine(trade_date)

        for dec in decisions:
            etf_code = str(dec.get("primary_etf_code") or "").strip()
            sector = str(dec.get("sector") or "").strip()
            status = str(dec.get("status") or "monitor")
            confidence = str(dec.get("confidence") or "medium")

            if not etf_code:
                entries.append(SectorBacktestEntry(
                    sector=sector, etf_code="", etf_name="",
                    entry_date=trade_date, status=status,
                    confidence=confidence, invalid_reason="no_etf_code",
                ))
                continue

            etf_name = str(dec.get("sector") or sector)
            df = price_data.get(etf_code) if price_data else None
            if df is None:
                df = self._load_etf_prices(etf_code)

            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                entries.append(SectorBacktestEntry(
                    sector=sector, etf_code=etf_code, etf_name=etf_name,
                    entry_date=trade_date, status=status,
                    confidence=confidence, invalid_reason="no_price_data",
                ))
                continue

            result = engine.run_single(
                price_df=df,
                entry_date=date.fromisoformat(trade_date),
                code=etf_code,
                decision="推荐" if status == "active" else "观察",
            )

            entry = SectorBacktestEntry(
                sector=sector,
                etf_code=etf_code,
                etf_name=etf_name,
                entry_date=trade_date,
                status=status,
                confidence=confidence,
                primary_return=result.returns.get(self.primary_horizon),
                primary_horizon=self.primary_horizon,
                excess_return=result.excess_returns.get(self.primary_horizon),
                returns_by_horizon=result.returns,
                max_drawdown=result.max_drawdown,
                tradable=result.tradable,
                invalid_reason=result.invalid_reason,
                cost_bps=result.cost_bps,
                raw=result,
            )
            entries.append(entry)

        tradable = [e for e in entries if e.tradable and e.primary_return is not None]
        win_count = sum(1 for e in tradable if (e.primary_return or 0) > 0)
        avg_return = float(sum(e.primary_return or 0 for e in tradable) / len(tradable)) if tradable else None
        avg_excess = float(sum(e.excess_return or 0 for e in tradable) / len(tradable)) if tradable else None

        return SectorBacktestSummary(
            trade_date=trade_date,
            entries=entries,
            win_count=win_count,
            total_tradable=len(tradable),
            avg_return=avg_return,
            avg_excess=avg_excess,
        )

    def _engine(self, trade_date: str) -> BacktestEngine:
        key = f"{trade_date}_{self.primary_horizon}"
        if key not in self._engine_cache:
            self._engine_cache[key] = BacktestEngine(config_override={
                "default_holding_days": self.holding_horizons,
                "primary_holding_days": self.primary_horizon,
            })
        return self._engine_cache[key]

    def _load_etf_prices(self, code: str) -> pd.DataFrame | None:
        """Load ETF daily prices from vendor or local cache."""
        if self._price_loader is not None:
            try:
                result = self._price_loader(code)
                if isinstance(result, pd.DataFrame):
                    return result
            except Exception:
                pass

        try:
            from ..data_agent.local_cache import get_cached_etf_daily, DEFAULT_CACHE_DIR

            cache_dir = config.get("cache_dir") or str(DEFAULT_CACHE_DIR)
            records = get_cached_etf_daily(code, cache_dir=cache_dir)
            if records:
                df = pd.DataFrame(records)
                if "trade_date" in df.columns:
                    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                    df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
                return df
        except Exception as exc:
            logger.debug("Failed to load ETF prices for %s: %s", code, exc)

        return None


def format_sector_backtest_summary(summary: SectorBacktestSummary) -> str:
    """Render a sector backtest summary as Markdown for audit reports."""
    lines = [
        "## ETF 板块回测验证",
        "",
        f"- 信号日期: {summary.trade_date}",
        f"- 可成交: {summary.total_tradable}/{len(summary.entries)}",
    ]
    if summary.avg_return is not None:
        lines.append(f"- 平均收益 (T+{summary.entries[0].primary_horizon if summary.entries else 5}): {summary.avg_return:+.2%}")
    if summary.avg_excess is not None:
        lines.append(f"- 平均超额: {summary.avg_excess:+.2%}")
    lines.append(f"- 胜率: {summary.win_count}/{summary.total_tradable}" if summary.total_tradable else "- 胜率: N/A")
    lines.append("")

    if not summary.entries:
        lines.append("无回测条目。")
        return "\n".join(lines)

    lines.extend([
        "| 板块 | ETF | 状态 | 置信度 | T+5 收益 | 超额 | 最大回撤 | 可成交 |",
        "|------|-----|------|--------|---------|------|---------|--------|",
    ])
    for e in sorted(summary.entries, key=lambda x: (x.tradable, x.primary_return or -999), reverse=True):
        ret = f"{e.primary_return:+.2%}" if e.primary_return is not None else "N/A"
        ex = f"{e.excess_return:+.2%}" if e.excess_return is not None else "N/A"
        dd = f"{e.max_drawdown:.1%}" if e.max_drawdown is not None else "N/A"
        tradable = "Y" if e.tradable else f"N({e.invalid_reason[:20]})"
        lines.append(
            f"| {e.sector} | {e.etf_code} {e.etf_name} | {e.status} | "
            f"{e.confidence} | {ret} | {ex} | {dd} | {tradable} |"
        )
    return "\n".join(lines)
