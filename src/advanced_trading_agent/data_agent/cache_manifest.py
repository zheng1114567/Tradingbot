"""Cache metadata for local market data files."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.atomic_write import atomic_write_json


SCHEMA_VERSION = "daily-cache-v1"


@dataclass
class CacheManifestEntry:
    """Metadata for one cached dataset."""

    key: str
    kind: str
    ticker: str
    freq: str
    path: str
    start_date: str | None = None
    end_date: str | None = None
    source: str = ""
    row_count: int = 0
    schema_version: str = SCHEMA_VERSION
    last_updated_at: str = ""
    checksum: str = ""
    status: str = "unknown"
    notes: list[str] | None = None


class CacheManifest:
    """Small JSON index describing cached local data coverage."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / "cache_manifest.json"
        self.entries: dict[str, CacheManifestEntry] = {}
        self._load()

    def get(self, key: str) -> CacheManifestEntry | None:
        return self.entries.get(key)

    def get_daily(self, ticker: str, freq: str = "1d") -> CacheManifestEntry | None:
        return self.get(self.daily_key(ticker, freq))

    def update_daily(
        self,
        *,
        ticker: str,
        path: str | Path,
        start_date: str | None,
        end_date: str | None,
        source: str,
        row_count: int,
        freq: str = "1d",
        status: str = "ready",
        notes: list[str] | None = None,
    ) -> CacheManifestEntry:
        entry = CacheManifestEntry(
            key=self.daily_key(ticker, freq),
            kind="daily",
            ticker=ticker,
            freq=freq,
            path=str(Path(path)),
            start_date=start_date,
            end_date=end_date,
            source=source,
            row_count=int(row_count),
            last_updated_at=datetime.now(timezone.utc).isoformat(),
            checksum=self._checksum(path),
            status=status,
            notes=notes or [],
        )
        self.entries[entry.key] = entry
        self.save()
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entries": {key: asdict(entry) for key, entry in sorted(self.entries.items())},
        }

    def save(self) -> None:
        atomic_write_json(self.path, self.to_dict())

    @staticmethod
    def daily_key(ticker: str, freq: str = "1d") -> str:
        return f"daily:{ticker}:{freq}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text("utf-8"))
        except json.JSONDecodeError:
            return
        raw_entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        if not isinstance(raw_entries, dict):
            return
        for key, raw in raw_entries.items():
            if not isinstance(raw, dict):
                continue
            try:
                self.entries[key] = CacheManifestEntry(**raw)
            except TypeError:
                continue

    @staticmethod
    def _checksum(path: str | Path) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return ""
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
