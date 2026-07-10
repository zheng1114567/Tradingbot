"""
对比实验 — 衡量多 Agent 系统效果

对比维度 (按设计方案):
1. 纯规则版本 vs 多 Agent 版本
2. 无 Memory 版本
3. 参考开源项目复刻口径

借鉴 TradingAgents 的 test_*.py 测试模式
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..backtest.engine import BacktestEngine, BacktestResult
from ..backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """对比实验结果"""
    name: str
    results: list[BacktestResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def compute(self) -> "ComparisonResult":
        self.summary = PerformanceMetrics.summary(self.results)
        return self


def run_comparison(price_data: dict[str, Any],
                    signals: list[dict[str, Any]]) -> list[ComparisonResult]:
    """运行对比实验

    Args:
        price_data: {code: DataFrame} 行情数据
        signals: 信号列表

    Returns:
        各实验对比结果
    """
    engine = BacktestEngine()
    comparisons = []

    # 1. 完整版本
    full_results = engine.run_batch(price_data, signals)
    comparisons.append(
        ComparisonResult(
            name="完整版 (多 Agent + Memory + 风控)",
            results=full_results,
        ).compute()
    )

    # 2. 无风控版本
    no_risk_signals = [
        {**s, "skip_risk": True} for s in signals
    ]
    no_risk_results = engine.run_batch(price_data, no_risk_signals)
    comparisons.append(
        ComparisonResult(
            name="无风控版本",
            results=no_risk_results,
        ).compute()
    )

    # 3. 纯规则版本 (总是推荐)
    rule_signals = [
        {**s, "decision": "推荐", "alpha_source": ["规则基线"]}
        for s in signals
    ]
    rule_results = engine.run_batch(price_data, rule_signals)
    comparisons.append(
        ComparisonResult(
            name="纯规则版本 (全推荐)",
            results=rule_results,
        ).compute()
    )

    return comparisons


def format_comparison(comparisons: list[ComparisonResult]) -> str:
    """格式化为对比报告"""
    lines = ["# 对比实验报告", ""]

    for comp in comparisons:
        s = comp.summary
        lines.extend([
            f"## {comp.name}",
            f"- 总信号: {s.get('total_signals', 0)}",
            f"- 可成交率: {s.get('tradable_ratio', 0):.1%}",
            f"- 5日胜率: {s.get('win_rate_5d', 0):.1%}",
            f"- 5日收益: {s.get('avg_return_5d', 0):+.2%}",
            f"- 夏普比: {s.get('sharpe_ratio', 0):.2f}",
            f"- 最大回撤: {s.get('max_drawdown', 0):.1%}",
            "",
        ])

    return "\n".join(lines)
