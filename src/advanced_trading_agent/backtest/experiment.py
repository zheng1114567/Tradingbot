"""Experiment registry for profit-oriented evaluation runs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import config
from ..strategy_rules import current_rulebook


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _records(value: pd.DataFrame | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return list(value)


@dataclass
class ExperimentRecord:
    """One reproducible evaluation run."""

    run_id: str
    name: str
    created_at: str
    rulebook: dict[str, Any]
    inputs: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at,
            "rulebook": self.rulebook,
            "inputs": self.inputs,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }


class ExperimentRegistry:
    """Append-only registry for backtests, paper runs, and ablations."""

    def __init__(self, path: str | None = None):
        default_path = Path(config.get("results_dir")) / "experiment_registry.jsonl"
        self.path = Path(path or default_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        *,
        name: str,
        metrics: dict[str, Any],
        signals: pd.DataFrame | list[dict[str, Any]] | None = None,
        prices: pd.DataFrame | list[dict[str, Any]] | None = None,
        artifacts: dict[str, str] | None = None,
        notes: list[str] | None = None,
    ) -> ExperimentRecord:
        """Persist one experiment with input fingerprints, not full datasets."""
        signal_rows = _records(signals)
        price_rows = _records(prices)
        inputs = {
            "signals_count": len(signal_rows),
            "prices_count": len(price_rows),
            "signals_hash": _stable_hash(signal_rows),
            "prices_hash": _stable_hash(price_rows),
        }
        payload = {
            "name": name,
            "inputs": inputs,
            "metrics": metrics,
            "rulebook": current_rulebook().to_dict(),
        }
        record = ExperimentRecord(
            run_id=f"exp_{_stable_hash(payload)}",
            name=name,
            created_at=datetime.now().isoformat(),
            rulebook=payload["rulebook"],
            inputs=inputs,
            metrics=metrics,
            artifacts=artifacts or {},
            notes=notes or [],
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def load(self) -> list[ExperimentRecord]:
        """Load all registry entries."""
        if not self.path.exists():
            return []
        records: list[ExperimentRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(ExperimentRecord(**data))
        return records

