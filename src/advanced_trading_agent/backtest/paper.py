"""Paper-trading ledger for observation-pool signals."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import config
from ..core.atomic_write import atomic_write_text
from .portfolio import ObservationPortfolioBacktester


@dataclass
class PaperTradingRun:
    """Result of one paper-trading update."""

    recorded_count: int
    resolved_count: int
    ledger_path: str
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_count": self.recorded_count,
            "resolved_count": self.resolved_count,
            "ledger_path": self.ledger_path,
            "summary": self.summary,
        }


class PaperTradingLedger:
    """Append-only paper ledger with deterministic settlement from prices."""

    def __init__(self, path: str | None = None):
        default_path = Path(config.get("results_dir")) / "paper_trading_ledger.jsonl"
        self.path = Path(path or default_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def write(self, rows: list[dict[str, Any]]) -> None:
        lines = [
            json.dumps(self._json_safe(row), ensure_ascii=False, sort_keys=True)
            for row in rows
        ]
        atomic_write_text(self.path, "\n".join(lines) + ("\n" if lines else ""))

    def record_signals(self, signals: pd.DataFrame | list[dict[str, Any]]) -> int:
        existing = self.load()
        seen = {self._key(row) for row in existing}
        count = 0
        for signal in pd.DataFrame(signals).to_dict("records"):
            row = {
                "signal_date": str(pd.Timestamp(signal.get("signal_date", signal.get("trade_date"))).date()),
                "code": str(signal.get("code", "")),
                "decision": str(signal.get("decision", "推荐")),
                "score": float(signal.get("score", 0.0) or 0.0),
                "alpha_source": signal.get("alpha_source", ""),
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "trade": {},
            }
            if not row["code"] or self._key(row) in seen:
                continue
            existing.append(row)
            seen.add(self._key(row))
            count += 1
        self.write(existing)
        return count

    def resolve(
        self,
        prices: pd.DataFrame | list[dict[str, Any]],
        *,
        holding_days: int = 5,
    ) -> PaperTradingRun:
        rows = self.load()
        pending = [row for row in rows if row.get("status") == "pending"]
        if not pending:
            return PaperTradingRun(0, 0, str(self.path), {})

        result = ObservationPortfolioBacktester(holding_days=holding_days).run(pending, prices)
        resolved_by_key = {
            (str(trade.get("signal_date")), str(trade.get("code"))): trade
            for trade in result.trades.to_dict("records")
        }
        resolved_count = 0
        for row in rows:
            key = (str(row.get("signal_date")), str(row.get("code")))
            trade = resolved_by_key.get(key)
            if row.get("status") == "pending" and trade:
                row["status"] = "resolved"
                row["trade"] = trade
                row["resolved_at"] = datetime.now().isoformat()
                resolved_count += 1
        self.write(rows)
        return PaperTradingRun(
            recorded_count=0,
            resolved_count=resolved_count,
            ledger_path=str(self.path),
            summary=result.summary,
        )

    def record_and_resolve(
        self,
        *,
        signals: pd.DataFrame | list[dict[str, Any]],
        prices: pd.DataFrame | list[dict[str, Any]],
        holding_days: int = 5,
    ) -> PaperTradingRun:
        recorded = self.record_signals(signals)
        resolved = self.resolve(prices, holding_days=holding_days)
        resolved.recorded_count = recorded
        return resolved

    @staticmethod
    def _key(row: dict[str, Any]) -> tuple[str, str]:
        return str(row.get("signal_date", "")), str(row.get("code", ""))

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): PaperTradingLedger._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [PaperTradingLedger._json_safe(item) for item in value]
        return value
