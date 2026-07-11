"""Atomic file writes — write to temp then rename, minimising corruption windows.

On POSIX os.replace is atomic when the target is on the same filesystem.
On Windows it is a best-effort replace (MoveFileEx with MOVEFILE_REPLACE_EXISTING),
which is still far safer than a direct write_text that can leave a truncated file.

Usage:
    from advanced_trading_agent.core.atomic_write import atomic_write_text, atomic_write_json

    atomic_write_text(path, "content")
    atomic_write_json(path, {"key": "value"})
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, content: str) -> None:
    """Atomically write a text string to *path*."""
    _atomic_write(Path(path), content)


def atomic_write_json(path: str | Path, data: Any, **kwargs: Any) -> None:
    """Atomically write JSON-serializable *data* to *path*."""
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("indent", 2)
    content = json.dumps(data, **kwargs)
    _atomic_write(Path(path), content)


def atomic_write_jsonl(path: str | Path, entries: list[dict[str, Any]]) -> None:
    """Atomically write a list of dicts as JSONL to *path*."""
    lines = [
        json.dumps(entry, ensure_ascii=False, sort_keys=True)
        for entry in entries
    ]
    _atomic_write(Path(path), "\n".join(lines) + ("\n" if lines else ""))
