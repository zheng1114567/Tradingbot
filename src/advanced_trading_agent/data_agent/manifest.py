"""Data lineage manifest for each analysis run."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import config


_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_path_part(value: str, fallback: str) -> str:
    safe = _SAFE_PATH_PART.sub("_", value).strip("._")
    return safe or fallback


@dataclass
class DataFieldStatus:
    """Availability and source detail for one data field."""

    available: bool
    source: str
    vendor_chain: list[str] = field(default_factory=list)
    fallback_used: bool = False
    error: str | None = None
    record_count: int | None = None


@dataclass
class DataManifest:
    """Auditable data boundary for a single trading analysis run."""

    ticker: str
    trade_date: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    fields: dict[str, DataFieldStatus] = field(default_factory=dict)
    vendor_errors: list[str] = field(default_factory=list)
    soft_veto_reasons: list[str] = field(default_factory=list)

    def add_field(
        self,
        name: str,
        *,
        available: bool,
        source: str,
        vendor_chain: list[str] | None = None,
        fallback_used: bool = False,
        error: str | None = None,
        record_count: int | None = None,
    ) -> None:
        self.fields[name] = DataFieldStatus(
            available=available,
            source=source,
            vendor_chain=vendor_chain or [],
            fallback_used=fallback_used,
            error=error,
            record_count=record_count,
        )
        if error:
            self.vendor_errors.append(f"{name}: {error}")
        if not available:
            self.soft_veto_reasons.append(f"{name} unavailable")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, results_dir: str | None = None) -> Path:
        base_dir = Path(results_dir or config.get("results_dir", "data/results"))
        manifest_dir = base_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        ticker = _safe_path_part(self.ticker.replace(".", "_"), "unknown_ticker")
        trade_date = _safe_path_part(self.trade_date, "unknown_date")
        path = manifest_dir / f"manifest_{ticker}_{trade_date}.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
