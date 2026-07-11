"""
绩效指标 — 回测结果分析

借鉴 TradingAgents 的 rating.py 集中式设计,
所有绩效指标统一管理和计算。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .engine import BacktestResult

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """回测绩效指标计算"""

    @staticmethod
    def _valid_returns(results: list[BacktestResult], days: int) -> list[float]:
        return [
            r.returns[days]
            for r in results
            if r.tradable and r.returns.get(days) is not None
        ]

    @staticmethod
    def win_rate(results: list[BacktestResult], days: int = 5) -> float:
        """胜率 (指定持有期的正收益比例)"""
        valid = PerformanceMetrics._valid_returns(results, days)
        if not valid:
            return 0.0
        wins = sum(1 for r in valid if r > 0)
        return wins / len(valid)

    @staticmethod
    def avg_return(results: list[BacktestResult], days: int = 5) -> float:
        """平均收益率"""
        valid = PerformanceMetrics._valid_returns(results, days)
        if not valid:
            return 0.0
        return float(np.mean(valid))

    @staticmethod
    def avg_excess_return(results: list[BacktestResult], days: int = 5) -> float:
        """平均超额收益率"""
        valid = [
            r.excess_returns[days]
            for r in results
            if r.tradable and r.excess_returns.get(days) is not None
        ]
        if not valid:
            return 0.0
        return float(np.mean(valid))

    @staticmethod
    def profit_loss_ratio(results: list[BacktestResult], days: int = 5) -> float:
        """盈亏比"""
        valid = PerformanceMetrics._valid_returns(results, days)
        if not valid:
            return 0.0
        wins = [r for r in valid if r > 0]
        losses = [r for r in valid if r < 0]
        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = abs(float(np.mean(losses))) if losses else 1
        return avg_win / avg_loss if avg_loss > 0 else 0

    @staticmethod
    def max_drawdown(results: list[BacktestResult]) -> float:
        """最大回撤 (所有结果中的最大回撤)"""
        dd = [r.max_drawdown for r in results if r.max_drawdown is not None]
        return float(np.min(dd)) if dd else 0.0

    @staticmethod
    def sharpe_ratio(results: list[BacktestResult], days: int = 5,
                     risk_free_rate: float = 0.02) -> float:
        """年化夏普比"""
        valid = PerformanceMetrics._valid_returns(results, days)
        if len(valid) < 5:
            return 0.0
        arr = np.array(valid)
        period_rf = risk_free_rate * days / 245
        excess = arr - period_rf
        if np.std(arr) == 0:
            return 0.0
        return float(np.mean(excess) / np.std(arr) * np.sqrt(245 / days))

    @staticmethod
    def tradable_ratio(results: list[BacktestResult]) -> float:
        """可成交率"""
        if not results:
            return 0.0
        return sum(1 for r in results if r.tradable) / len(results)

    @staticmethod
    def decision_differentiation(results: list[BacktestResult], days: int = 5) -> dict[str, float]:
        """推荐/观察/拒绝 三类决策的后验表现差异"""
        groups = {}
        for r in results:
            dec = r.decision
            if dec not in groups:
                groups[dec] = []
            groups[dec].append(r)

        diff = {}
        for dec, group in groups.items():
            diff[dec] = {
                "count": len(group),
                "avg_return": PerformanceMetrics.avg_return(group, days),
                "win_rate": PerformanceMetrics.win_rate(group, days),
            }
        return diff

    @staticmethod
    def summary(results: list[BacktestResult]) -> dict[str, Any]:
        """综合绩效摘要"""
        return {
            "total_signals": len(results),
            "tradable_ratio": PerformanceMetrics.tradable_ratio(results),
            "win_rate_5d": PerformanceMetrics.win_rate(results, 5),
            "avg_return_1d": PerformanceMetrics.avg_return(results, 1),
            "avg_return_3d": PerformanceMetrics.avg_return(results, 3),
            "avg_return_5d": PerformanceMetrics.avg_return(results, 5),
            "avg_return_10d": PerformanceMetrics.avg_return(results, 10),
            "profit_loss_ratio": PerformanceMetrics.profit_loss_ratio(results, 5),
            "max_drawdown": PerformanceMetrics.max_drawdown(results),
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(results, 5),
            "decision_diff": PerformanceMetrics.decision_differentiation(results),
        }

    @staticmethod
    def format_summary(results: list[BacktestResult]) -> str:
        """格式化的绩效报告"""
        s = PerformanceMetrics.summary(results)
        lines = [
            "## 回测绩效报告",
            f"总信号数: {s['total_signals']}",
            f"可成交率: {s['tradable_ratio']:.1%}",
            "",
            "### 不同持有期收益",
        ]
        for d in [1, 3, 5, 10]:
            lines.append(f"  {d}日: {s[f'avg_return_{d}d']:+.2%}")
        lines.extend([
            "",
            f"5日胜率: {s['win_rate_5d']:.1%}",
            f"盈亏比: {s['profit_loss_ratio']:.2f}",
            f"最大回撤: {s['max_drawdown']:.1%}",
            f"夏普比: {s['sharpe_ratio']:.2f}",
            "",
            "### 决策分化",
        ])
        for dec, info in s.get("decision_diff", {}).items():
            lines.append(
                f"  {dec}: {info['count']}次, "
                f"收益 {info['avg_return']:+.2%}, "
                f"胜率 {info['win_rate']:.1%}"
            )
        return "\n".join(lines)
