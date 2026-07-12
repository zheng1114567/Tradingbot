"""Stock identity resolution — ticker → company name, sector hints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_STATIC_PROFILES: dict[str, StockProfile] = {}


@dataclass(frozen=True)
class StockProfile:
    """Resolved stock identity used to drive purposeful data collection."""

    ticker: str
    company_name: str | None = None
    sector_keyword: str | None = None
    sector_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    source: str = "unknown"
    confidence: float = 0.0


class StockProfileResolver:
    """Best-effort A-share profile resolver with deterministic local fallbacks."""

    def resolve(self, ticker: str) -> StockProfile:
        normalized = self._normalize_ticker(ticker)
        if normalized in _STATIC_PROFILES:
            return _STATIC_PROFILES[normalized]
        return StockProfile(
            ticker=normalized or ticker,
            aliases=[item for item in {ticker, normalized, self._ticker_digits(normalized)} if item],
            source="ticker_only",
            confidence=0.2,
        )

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        value = str(ticker or "").strip().upper()
        if not value:
            return value
        match = re.match(r"^(SZ|SH)(\d{6})$", value)
        if match:
            return f"{match.group(2)}.{match.group(1)}"
        return value

    @staticmethod
    def _ticker_digits(ticker: str) -> str:
        match = re.search(r"(\d{6})", str(ticker or ""))
        return match.group(1) if match else ""


# Pre-populate well-known static profiles
_STATIC_PROFILES.update({
    "000001.SZ": StockProfile(
        ticker="000001.SZ",
        company_name="平安银行",
        sector_keyword="银行",
        sector_name="银行",
        aliases=["平安银行", "平安银行股份有限公司", "000001", "000001.SZ"],
        source="built_in_a_share_profile",
        confidence=0.95,
    ),
    "000001.SH": StockProfile(
        ticker="000001.SH",
        company_name="上证指数",
        sector_keyword=None,
        sector_name=None,
        aliases=["上证指数", "上证综指", "000001", "000001.SH"],
        source="built_in_a_share_profile",
        confidence=0.95,
    ),
})


__all__ = ["StockProfile", "StockProfileResolver"]
