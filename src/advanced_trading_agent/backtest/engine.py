"""
回测引擎 — A 股专用

核心设计:
1. Point-in-time: 每个历史交易日只能使用当时可得的数据
2. A 股约束: T+1, 涨跌停不可成交, 停牌, ST, 一字板
3. 成本扣除: 滑点, 印花税, 佣金, 冲击成本
4. 对比实验: 支持多 Agent vs 无 Agent 对比

借鉴 TradingAgents 的 backtrader 集成模式,
但针对 A 股做了重写 (涨跌停约束+复权处理+T+1)。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..config import config

logger = logging.getLogger(__name__)


# ============================================================
# 输出结构
# ============================================================

@dataclass
class BacktestResult:
    """单次回测结果"""
    run_date: date                     # 生成日期
    target_date: date                  # 目标交易日
    code: str                          # 股票代码
    decision: str                      # 推荐/观察/拒绝
    alpha_source: list[str] = field(default_factory=list)
    entry_price: float | None = None
    exit_price: float | None = None
    holding_days: int = 0
    returns: dict[int, float | None] = field(default_factory=dict)  # {holding_days: return}
    excess_returns: dict[int, float | None] = field(default_factory=dict)
    max_drawdown: float | None = None
    tradable: bool = True
    invalid_triggered: bool = False
    invalid_reason: str = ""
    cost_bps: float = 0
    benchmark: str = "000300.SH"  # 沪深300

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_date": str(self.run_date),
            "target_date": str(self.target_date),
            "code": self.code,
            "decision": self.decision,
            "alpha_source": self.alpha_source,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "holding_days": self.holding_days,
            "returns": self.returns,
            "excess_returns": self.excess_returns,
            "max_drawdown": self.max_drawdown,
            "tradable": self.tradable,
            "invalid_triggered": self.invalid_triggered,
            "invalid_reason": self.invalid_reason,
            "cost_bps": self.cost_bps,
            "benchmark": self.benchmark,
        }


# ============================================================
# 回测引擎
# ============================================================

class BacktestEngine:
    """事件驱动回测引擎 — A 股专用"""

    def __init__(self, config_override: dict[str, Any] | None = None):
        bc = deepcopy(config.get("backtest_config", {}))
        if config_override:
            bc.update(config_override)

        self.holding_days = bc.get("default_holding_days", [1, 3, 5, 10])
        self.primary_days = bc.get("primary_holding_days", 5)
        self.benchmark = bc.get("benchmark", "000300.SH")
        self.slippage_bps = bc.get("slippage_bps", 3)
        self.stamp_tax_bps = bc.get("stamp_tax_bps", 10)
        self.commission_bps = bc.get("commission_bps", 3)
        self.min_sample_size = bc.get("min_sample_size", 30)

    # ============================================================
    # 成本计算
    # ============================================================

    def calc_trade_cost(self, is_buy: bool = True) -> float:
        """计算交易成本 (bps)"""
        cost = self.commission_bps
        if not is_buy:
            cost += self.stamp_tax_bps  # 卖出才交印花税
        return cost + self.slippage_bps

    # ============================================================
    # 可成交性检查
    # ============================================================

    def is_tradable(self, row: pd.Series, direction: str = "buy") -> tuple[bool, str]:
        """检查某日是否可成交"""
        if row.get("is_limit_up", False) and direction == "buy":
            return False, "涨停不可买"
        if row.get("is_limit_down", False) and direction == "sell":
            return False, "跌停不可卖"
        if row.get("is_limit_down", False) and direction == "buy":
            return False, "跌停不可买"
        if row.get("volume", 0) == 0:
            return False, "无成交量"
        if row.get("amount", 0) < 10_000_000:  # 日成交额 >= 1000万
            return False, "成交额过低"
        return True, ""

    def _empty_returns(self) -> tuple[dict[int, None], dict[int, None]]:
        """Return a holding-period map with no executable returns."""
        return ({d: None for d in self.holding_days},
                {d: None for d in self.holding_days})

    # ============================================================
    # 单笔回测
    # ============================================================

    def run_single(self, price_df: pd.DataFrame, entry_date: date,
                   code: str, decision: str = "推荐",
                   alpha_source: list[str] | None = None) -> BacktestResult:
        """对单条推荐运行回测

        Args:
            price_df: 日K DataFrame, 需包含 'close', 'open', 'pct_chg' 等列
            entry_date: 买入日期 (T日)
            code: 股票代码
            decision: 决策类型
            alpha_source: Alpha 来源
        """
        if alpha_source is None:
            alpha_source = []

        if price_df.empty:
            return BacktestResult(
                run_date=date.today(), target_date=entry_date,
                code=code, decision=decision, tradable=False,
                invalid_reason="empty_price_data", benchmark=self.benchmark,
            )

        if "trade_date" not in price_df.columns:
            return BacktestResult(
                run_date=date.today(), target_date=entry_date,
                code=code, decision=decision, alpha_source=alpha_source,
                tradable=False, invalid_reason="missing_trade_date", benchmark=self.benchmark,
            )

        # Point-in-time: locate the signal date before entering on the next trading day.
        # The trade_date is the after-close signal date; execution starts at T+1.
        price_df = price_df.copy()
        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"], errors="coerce")
        price_df = price_df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        try:
            signal_idx = price_df[
                price_df["trade_date"] == pd.Timestamp(entry_date)
            ].index[0]
        except IndexError:
            return BacktestResult(
                run_date=date.today(), target_date=entry_date,
                code=code, decision=decision, tradable=False,
                invalid_reason="entry_date_not_found", benchmark=self.benchmark,
            )

        entry_idx = signal_idx + 1
        if entry_idx >= len(price_df):
            returns, excess_returns = self._empty_returns()
            return BacktestResult(
                run_date=date.today(), target_date=entry_date,
                code=code, decision=decision, alpha_source=alpha_source,
                returns=returns, excess_returns=excess_returns,
                tradable=False, invalid_reason="no_next_trading_day", benchmark=self.benchmark,
            )

        # 检查可成交性
        entry_row = price_df.loc[entry_idx]
        tradable, reason = self.is_tradable(entry_row, "buy")
        if not tradable:
            returns, excess_returns = self._empty_returns()
            return BacktestResult(
                run_date=date.today(), target_date=entry_date,
                code=code, decision=decision, alpha_source=alpha_source,
                entry_price=entry_row.get("open", entry_row.get("close")),
                returns=returns, excess_returns=excess_returns,
                tradable=False, invalid_reason=reason, benchmark=self.benchmark,
            )

        # T+1: 最早次日卖出
        entry_price = entry_row.get("open", entry_row["close"])
        exit_price = entry_price
        cost_buy = self.calc_trade_cost(is_buy=True)

        returns: dict[int, float | None] = {}
        excess_returns: dict[int, float | None] = {}

        max_drawdown = 0.0
        peak = entry_price
        invalid_triggered = False

        for days in self.holding_days:
            # 找到 days 个交易日后的位置 (T+1 起算)
            exit_idx = entry_idx + days
            # 跳过第一个交易日 (T+1)
            if exit_idx >= len(price_df):
                returns[days] = None
                excess_returns[days] = None
                continue

            exit_row = price_df.loc[exit_idx]
            exit_price = exit_row.get("open", exit_row["close"])

            # 检查卖出是否可成交
            sell_tradable, _ = self.is_tradable(exit_row, "sell")
            if not sell_tradable:
                # 顺延到下一个可成交日
                found_exit = False
                for lookahead in range(1, 10):
                    if exit_idx + lookahead >= len(price_df):
                        break
                    next_row = price_df.loc[exit_idx + lookahead]
                    if self.is_tradable(next_row, "sell")[0]:
                        exit_price = next_row.get("open", next_row["close"])
                        exit_idx = exit_idx + lookahead
                        found_exit = True
                        break
                if not found_exit:
                    returns[days] = None
                    excess_returns[days] = None
                    continue

            # 计算收益 (扣除成本)
            cost_sell = self.calc_trade_cost(is_buy=False)
            raw_return = (exit_price - entry_price) / entry_price
            net_return = raw_return - (cost_buy + cost_sell) / 10000
            returns[days] = net_return

            # 相对基准超额收益 (调用方可传入 bench_close 列)
            if "bench_close" in price_df.columns:
                bench_entry = price_df.loc[entry_idx, "bench_close"]
                bench_exit = price_df.loc[exit_idx, "bench_close"]
                if bench_entry and not pd.isna(bench_entry) and not pd.isna(bench_exit):
                    bench_return = (bench_exit - bench_entry) / bench_entry
                    excess_returns[days] = net_return - bench_return
                else:
                    excess_returns[days] = None
            else:
                excess_returns[days] = None

            # 最大回撤
            for look_idx in range(entry_idx + 1, exit_idx + 1):
                if look_idx < len(price_df):
                    cur = price_df.loc[look_idx, "close"]
                    peak = max(peak, cur)
                    dd = (cur - peak) / peak
                    max_drawdown = min(max_drawdown, dd)

        actual_days = max(
            (d for d in self.holding_days if returns.get(d) is not None),
            default=0
        )

        return BacktestResult(
            run_date=date.today(),
            target_date=entry_date,
            code=code,
            decision=decision,
            alpha_source=alpha_source,
            entry_price=entry_price,
            exit_price=exit_price,
            holding_days=actual_days,
            returns=returns,
            excess_returns=excess_returns,
            max_drawdown=max_drawdown,
            tradable=tradable,
            invalid_triggered=invalid_triggered,
            cost_bps=cost_buy + self.calc_trade_cost(is_buy=False),
            benchmark=self.benchmark,
        )

    # ============================================================
    # 批量回测
    # ============================================================

    def run_batch(self, price_data: dict[str, pd.DataFrame],
                  signals: list[dict[str, Any]]) -> list[BacktestResult]:
        """批量回测

        Args:
            price_data: {code: DataFrame} 的行情数据
            signals: [{"code": str, "date": str, "decision": str, ...}]
        """
        results = []
        for signal in signals:
            code = signal["code"]
            entry_date = signal["date"]
            if isinstance(entry_date, str):
                entry_date = datetime.strptime(entry_date, "%Y-%m-%d").date()
            df = price_data.get(code)
            if df is None:
                logger.warning("No price data for %s", code)
                continue
            result = self.run_single(
                price_df=df,
                entry_date=entry_date,
                code=code,
                decision=signal.get("decision", "推荐"),
                alpha_source=signal.get("alpha_source", []),
            )
            results.append(result)
        return results
