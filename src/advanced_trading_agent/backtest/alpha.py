"""Alpha-source attribution utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AlphaAttribution:
    """Performance contribution for one alpha source."""

    alpha_source: str
    sample_size: int
    win_rate: float
    avg_return: float
    total_pnl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_source": self.alpha_source,
            "sample_size": self.sample_size,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "total_pnl": self.total_pnl,
        }


def _split_sources(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ["UNKNOWN"]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)] or ["UNKNOWN"]
    text = str(value).strip()
    if not text:
        return ["UNKNOWN"]
    for sep in ["|", "/", ",", "，"]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()] or ["UNKNOWN"]
    return [text]


class AlphaAttributionAnalyzer:
    """Attribute trade outcomes to declared alpha sources."""

    def summarize(self, trades: pd.DataFrame | list[dict[str, Any]]) -> list[AlphaAttribution]:
        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return []
        if "return" not in trades_df.columns:
            trades_df["return"] = 0.0
        if "pnl" not in trades_df.columns:
            trades_df["pnl"] = 0.0
        if "alpha_source" not in trades_df.columns:
            trades_df["alpha_source"] = "UNKNOWN"

        rows: list[dict[str, Any]] = []
        for _, trade in trades_df.iterrows():
            sources = _split_sources(trade.get("alpha_source"))
            weight = 1 / len(sources)
            for source in sources:
                rows.append({
                    "alpha_source": source,
                    "return": float(trade.get("return", 0.0) or 0.0),
                    "pnl": float(trade.get("pnl", 0.0) or 0.0) * weight,
                })

        expanded = pd.DataFrame(rows)
        result: list[AlphaAttribution] = []
        for source, group in expanded.groupby("alpha_source"):
            returns = group["return"].astype(float)
            result.append(AlphaAttribution(
                alpha_source=str(source),
                sample_size=int(len(group)),
                win_rate=float((returns > 0).mean()) if len(group) else 0.0,
                avg_return=float(np.mean(returns)) if len(group) else 0.0,
                total_pnl=float(group["pnl"].sum()),
            ))
        return sorted(result, key=lambda item: item.total_pnl, reverse=True)

