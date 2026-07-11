"""Portfolio-level backtest for the daily observation pool."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortfolioBacktestResult:
    """Daily portfolio backtest result."""

    nav: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nav": self.nav.to_dict("records"),
            "trades": self.trades.to_dict("records"),
            "summary": self.summary,
        }


class ObservationPortfolioBacktester:
    """Backtest a daily observation pool as a simple equal-weight portfolio.

    Signals are generated after market close on signal_date and enter on the next
    trading day. Positions are held for holding_days, then exited at the open.
    """

    def __init__(
        self,
        *,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 10,
        max_single_position_pct: float = 0.10,
        holding_days: int = 5,
        cost_bps: float = 22,
    ):
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.max_single_position_pct = max_single_position_pct
        self.holding_days = holding_days
        self.cost_bps = cost_bps

    def run(self, signals: pd.DataFrame | list[dict[str, Any]],
            prices: pd.DataFrame | list[dict[str, Any]]) -> PortfolioBacktestResult:
        signals_df = pd.DataFrame(signals)
        price_df = pd.DataFrame(prices)
        if signals_df.empty or price_df.empty:
            return PortfolioBacktestResult(
                nav=pd.DataFrame(),
                trades=pd.DataFrame(),
                summary=self._empty_summary(),
            )

        signals_df = self._normalize_signals(signals_df)
        price_df = self._normalize_prices(price_df)
        trading_days = sorted(price_df["trade_date"].dropna().unique())
        price_by_code = {code: group.sort_values("trade_date").reset_index(drop=True)
                         for code, group in price_df.groupby("code")}

        cash = self.initial_cash
        open_positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        nav_rows: list[dict[str, Any]] = []

        for current_day in trading_days:
            current_ts = pd.Timestamp(current_day)

            # Exit due positions first at today's open.
            remaining_positions = []
            for position in open_positions:
                if position["exit_date"] <= current_ts:
                    exit_price = self._price_on(price_by_code.get(position["code"]), current_ts, "open")
                    if exit_price is None:
                        remaining_positions.append(position)
                        continue
                    proceeds = position["shares"] * exit_price
                    cost = proceeds * self.cost_bps / 10000
                    cash += proceeds - cost
                    ret = (exit_price - position["entry_price"]) / position["entry_price"] - self.cost_bps / 10000
                    trades.append({
                        **position,
                        "actual_exit_date": current_ts.date().isoformat(),
                        "exit_price": exit_price,
                        "return": ret,
                        "pnl": proceeds - cost - position["notional"],
                    })
                else:
                    remaining_positions.append(position)
            open_positions = remaining_positions

            # Enter signals generated on the previous available trading day.
            signal_day = self._previous_trading_day(trading_days, current_ts)
            if signal_day is not None:
                todays_signals = self._select_signals(signals_df, signal_day, open_positions)
                slots = max(0, self.max_positions - len(open_positions))
                for _, signal in todays_signals.head(slots).iterrows():
                    code = signal["code"]
                    if any(pos["code"] == code for pos in open_positions):
                        continue
                    entry_price = self._price_on(price_by_code.get(code), current_ts, "open")
                    if entry_price is None or entry_price <= 0:
                        continue
                    notional = min(self.initial_cash * self.max_single_position_pct, cash)
                    if notional <= 0:
                        break
                    shares = notional / entry_price
                    cost = notional * self.cost_bps / 10000
                    cash -= notional + cost
                    exit_date = self._nth_trading_day_after(trading_days, current_ts, self.holding_days)
                    if exit_date is None:
                        cash += notional
                        continue
                    open_positions.append({
                        "code": code,
                        "signal_date": pd.Timestamp(signal["signal_date"]).date().isoformat(),
                        "entry_date": current_ts.date().isoformat(),
                        "exit_date": pd.Timestamp(exit_date),
                        "entry_price": entry_price,
                        "shares": shares,
                        "notional": notional,
                        "alpha_source": signal.get("alpha_source", ""),
                        "score": signal.get("score", 0.0),
                    })

            market_value = 0.0
            for position in open_positions:
                close_price = self._price_on(price_by_code.get(position["code"]), current_ts, "close")
                if close_price is None:
                    close_price = position["entry_price"]
                market_value += position["shares"] * close_price
            equity = cash + market_value
            nav_rows.append({
                "trade_date": current_ts.date().isoformat(),
                "cash": cash,
                "market_value": market_value,
                "equity": equity,
                "open_positions": len(open_positions),
            })

        nav = pd.DataFrame(nav_rows)
        trades_df = pd.DataFrame(trades)
        return PortfolioBacktestResult(
            nav=nav,
            trades=trades_df,
            summary=self._summarize(nav, trades_df),
        )

    @staticmethod
    def _normalize_signals(signals: pd.DataFrame) -> pd.DataFrame:
        signals = signals.copy()
        if "signal_date" not in signals.columns and "trade_date" in signals.columns:
            signals["signal_date"] = signals["trade_date"]
        if "score" not in signals.columns:
            signals["score"] = 0.0
        if "decision" not in signals.columns:
            signals["decision"] = "推荐"
        if "alpha_source" not in signals.columns:
            signals["alpha_source"] = ""
        signals["signal_date"] = pd.to_datetime(signals["signal_date"])
        return signals.sort_values(["signal_date", "score"], ascending=[True, False])

    @staticmethod
    def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.copy()
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        return prices.sort_values(["trade_date", "code"]).reset_index(drop=True)

    def _select_signals(self, signals: pd.DataFrame, signal_day: pd.Timestamp,
                        open_positions: list[dict[str, Any]]) -> pd.DataFrame:
        held_codes = {pos["code"] for pos in open_positions}
        selected = signals[
            (signals["signal_date"] == signal_day)
            & (signals["decision"].isin(["推荐", "RECOMMEND"]))
            & (~signals["code"].isin(held_codes))
        ].copy()
        return selected.sort_values("score", ascending=False)

    @staticmethod
    def _price_on(price_df: pd.DataFrame | None, trade_day: pd.Timestamp, field: str) -> float | None:
        if price_df is None or price_df.empty:
            return None
        row = price_df[price_df["trade_date"] == trade_day]
        if row.empty or field not in row.columns:
            return None
        value = row.iloc[0][field]
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _previous_trading_day(trading_days: list[Any], current_day: pd.Timestamp) -> pd.Timestamp | None:
        prior = [pd.Timestamp(day) for day in trading_days if pd.Timestamp(day) < current_day]
        return prior[-1] if prior else None

    @staticmethod
    def _nth_trading_day_after(trading_days: list[Any], current_day: pd.Timestamp, n: int) -> pd.Timestamp | None:
        future = [pd.Timestamp(day) for day in trading_days if pd.Timestamp(day) > current_day]
        if len(future) < n:
            return None
        return future[n - 1]

    @staticmethod
    def _empty_summary() -> dict[str, Any]:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
        }

    def _summarize(self, nav: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
        if nav.empty:
            return self._empty_summary()
        equity = nav["equity"].astype(float)
        total_return = equity.iloc[-1] / self.initial_cash - 1
        running_peak = equity.cummax()
        drawdown = equity / running_peak - 1
        trade_returns = trades["return"].astype(float) if not trades.empty and "return" in trades else pd.Series(dtype=float)
        return {
            "total_return": float(total_return),
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "trade_count": int(len(trades)),
            "win_rate": float((trade_returns > 0).mean()) if not trade_returns.empty else 0.0,
            "avg_trade_return": float(trade_returns.mean()) if not trade_returns.empty else 0.0,
        }

