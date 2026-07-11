"""Post-trade review loop for signal outcomes and strategy calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..agents.memory_agent import MemoryStore
from ..config import config
from .engine import BacktestEngine, BacktestResult


PriceLoader = Callable[[str, str], pd.DataFrame | list[dict[str, Any]] | None]


@dataclass
class AlphaPerformance:
    """Performance bucket for one alpha source."""

    alpha_source: str
    rule_version: str = "UNKNOWN"
    sample_size: int = 0
    tradable_count: int = 0
    hit_rate: float = 0.0
    avg_return: float = 0.0
    avg_excess_return: float = 0.0
    max_drawdown: float = 0.0
    recommendation: str = "KEEP"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_source": self.alpha_source,
            "rule_version": self.rule_version,
            "sample_size": self.sample_size,
            "tradable_count": self.tradable_count,
            "hit_rate": self.hit_rate,
            "avg_return": self.avg_return,
            "avg_excess_return": self.avg_excess_return,
            "max_drawdown": self.max_drawdown,
            "recommendation": self.recommendation,
            "reasons": self.reasons,
        }


class ReviewEngine:
    """Resolve pending decisions and summarize alpha-source performance."""

    def __init__(self, review_config: dict[str, Any] | None = None):
        cfg = {**config.get("review_config", {}), **(review_config or {})}
        self.holding_days = cfg.get("review_horizons", [1, 3, 5, 10, 20])
        self.primary_days = cfg.get("primary_horizon_days", 5)
        self.min_samples_to_adjust = cfg.get("min_samples_to_adjust", 30)
        self.min_hit_rate = cfg.get("min_hit_rate", 0.45)
        self.min_avg_excess = cfg.get("min_avg_excess_return", 0.0)
        self.pause_avg_excess = cfg.get("pause_avg_excess_return", -0.02)
        self.pause_hit_rate = cfg.get("pause_hit_rate", 0.35)

    def outcome_from_prices(
        self,
        entry: dict[str, Any],
        price_df: pd.DataFrame | list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Build one outcome payload from historical prices."""
        ticker = str(entry.get("ticker", ""))
        signal_date = str(entry.get("date", ""))
        if not ticker or not signal_date:
            return None

        df = pd.DataFrame(price_df)
        if df.empty:
            return None

        decision_data = entry.get("decision_data", {}) or {}
        horizon = int(decision_data.get("horizon_days") or self.primary_days)
        if horizon not in self.holding_days:
            self.holding_days = sorted(set([*self.holding_days, horizon]))
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            if len(df[df["trade_date"] > pd.Timestamp(signal_date)]) < horizon:
                return None

        engine = BacktestEngine(config_override={
            "default_holding_days": self.holding_days,
            "primary_holding_days": horizon,
        })
        result = engine.run_single(
            price_df=df,
            entry_date=date.fromisoformat(signal_date),
            code=ticker,
            decision=str(entry.get("decision", "")),
            alpha_source=decision_data.get("alpha_source", []),
        )
        return self.outcome_from_result(result, horizon=horizon)

    @staticmethod
    def outcome_from_result(result: BacktestResult, horizon: int) -> dict[str, Any]:
        """Convert a BacktestResult into MemoryStore.resolve_pending outcome shape."""
        absolute_return = result.returns.get(horizon)
        excess_return = result.excess_returns.get(horizon)
        if excess_return is None:
            excess_return = absolute_return
        return {
            "horizon_days": horizon,
            "absolute_return": absolute_return,
            "excess_return": excess_return,
            "returns_by_horizon": result.returns,
            "excess_returns_by_horizon": result.excess_returns,
            "max_drawdown": result.max_drawdown,
            "tradable": result.tradable,
            "cost_bps": result.cost_bps,
            "benchmark": result.benchmark,
        }

    def resolve_due(
        self,
        store: MemoryStore,
        *,
        price_loader: PriceLoader,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve pending Memory entries when enough price data is available."""
        as_of_date = as_of or datetime.now().date().isoformat()

        def loader(entry: dict[str, Any]) -> dict[str, Any] | None:
            prices = price_loader(str(entry.get("ticker", "")), str(entry.get("date", "")))
            if prices is None:
                return None
            return self.outcome_from_prices(entry, prices)

        return store.resolve_pending(outcome_loader=loader, as_of=as_of_date)

    def summarize_entries(self, entries: list[dict[str, Any]]) -> list[AlphaPerformance]:
        """Group resolved entries by alpha source and rule version."""
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for entry in entries:
            if entry.get("pending"):
                continue
            decision_data = entry.get("decision_data", {}) or {}
            rulebook = entry.get("rulebook", {}) or {}
            rule_version = str(rulebook.get("version") or "UNKNOWN")
            sources = decision_data.get("alpha_source") or ["UNKNOWN"]
            for source in sources:
                buckets.setdefault((str(source), rule_version), []).append(entry)

        return [
            self._summarize_bucket(source, rule_version, rows)
            for (source, rule_version), rows in sorted(buckets.items())
        ]

    def _summarize_bucket(
        self,
        source: str,
        rule_version: str,
        rows: list[dict[str, Any]],
    ) -> AlphaPerformance:
        returns: list[float] = []
        excess_returns: list[float] = []
        drawdowns: list[float] = []
        hits = 0
        tradable = 0

        for row in rows:
            outcome = row.get("outcome", {}) or {}
            if outcome.get("tradable", True):
                tradable += 1
            abs_ret = outcome.get("absolute_return")
            ex_ret = outcome.get("excess_return")
            if abs_ret is not None:
                returns.append(float(abs_ret))
            if ex_ret is not None:
                ex = float(ex_ret)
                excess_returns.append(ex)
                if ex > 0:
                    hits += 1
            dd = outcome.get("max_drawdown")
            if dd is not None:
                drawdowns.append(float(dd))

        sample_size = len(excess_returns)
        hit_rate = hits / sample_size if sample_size else 0.0
        avg_return = float(np.mean(returns)) if returns else 0.0
        avg_excess = float(np.mean(excess_returns)) if excess_returns else 0.0
        max_drawdown = float(np.min(drawdowns)) if drawdowns else 0.0
        recommendation, reasons = self._recommend(sample_size, hit_rate, avg_excess)

        return AlphaPerformance(
            alpha_source=source,
            rule_version=rule_version,
            sample_size=sample_size,
            tradable_count=tradable,
            hit_rate=hit_rate,
            avg_return=avg_return,
            avg_excess_return=avg_excess,
            max_drawdown=max_drawdown,
            recommendation=recommendation,
            reasons=reasons,
        )

    def _recommend(
        self,
        sample_size: int,
        hit_rate: float,
        avg_excess: float,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if sample_size < self.min_samples_to_adjust:
            return "KEEP_OBSERVING", [f"样本不足 {sample_size} < {self.min_samples_to_adjust}，不自动调权"]

        if avg_excess <= self.pause_avg_excess or hit_rate <= self.pause_hit_rate:
            if avg_excess <= self.pause_avg_excess:
                reasons.append(f"平均超额 {avg_excess:+.2%} <= 暂停阈值 {self.pause_avg_excess:+.2%}")
            if hit_rate <= self.pause_hit_rate:
                reasons.append(f"命中率 {hit_rate:.1%} <= 暂停阈值 {self.pause_hit_rate:.1%}")
            return "PAUSE", reasons

        if avg_excess <= self.min_avg_excess or hit_rate < self.min_hit_rate:
            if avg_excess <= self.min_avg_excess:
                reasons.append(f"平均超额 {avg_excess:+.2%} 未跑赢阈值")
            if hit_rate < self.min_hit_rate:
                reasons.append(f"命中率 {hit_rate:.1%} < 最低要求 {self.min_hit_rate:.1%}")
            return "DOWNWEIGHT", reasons

        return "KEEP", ["表现达标，维持当前权重"]

    @staticmethod
    def format_summary(summary: list[AlphaPerformance]) -> str:
        lines = ["# 复盘绩效与策略建议", ""]
        if not summary:
            return "# 复盘绩效与策略建议\n\n暂无已结算样本。"
        for item in summary:
            lines.extend([
                f"## {item.alpha_source} ({item.rule_version})",
                f"- 样本数: {item.sample_size}",
                f"- 可成交数: {item.tradable_count}",
                f"- 命中率: {item.hit_rate:.1%}",
                f"- 平均收益: {item.avg_return:+.2%}",
                f"- 平均超额: {item.avg_excess_return:+.2%}",
                f"- 最大回撤: {item.max_drawdown:.1%}",
                f"- 策略建议: {item.recommendation}",
                f"- 原因: {'; '.join(item.reasons)}",
                "",
            ])
        return "\n".join(lines).rstrip()
