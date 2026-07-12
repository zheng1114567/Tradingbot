"""Stock identity resolution — ticker → company name, sector hints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SECTOR_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("银行",), "银行"),
    (("证券", "券商"), "证券"),
    (("保险",), "保险"),
    (("煤炭",), "煤炭"),
    (("钢铁",), "钢铁"),
    (("医药", "生物"), "医药"),
    (("电力",), "电力"),
    (("汽车",), "汽车"),
    (("地产", "房地产"), "房地产"),
    (("白酒", "酒"), "酿酒"),
)

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
        akshare_profile = self._resolve_from_akshare(normalized)
        if akshare_profile.company_name:
            return akshare_profile
        return StockProfile(
            ticker=normalized or ticker,
            aliases=[item for item in {ticker, normalized, self._ticker_digits(normalized)} if item],
            source="ticker_only",
            confidence=0.2,
        )

    @classmethod
    def _resolve_from_akshare(cls, ticker: str) -> StockProfile:
        code = cls._ticker_digits(ticker)
        if not code:
            return StockProfile(ticker=ticker, source="akshare_code_name_unavailable", confidence=0.0)
        try:
            import akshare as ak

            frame = ak.stock_info_a_code_name()
        except Exception:
            return StockProfile(ticker=ticker, source="akshare_code_name_unavailable", confidence=0.0)
        if not hasattr(frame, "empty") or frame.empty:
            return StockProfile(ticker=ticker, source="akshare_code_name_empty", confidence=0.0)

        code_column = cls._first_existing_column(frame, ("code", "代码", "证券代码"))
        name_column = cls._first_existing_column(frame, ("name", "名称", "证券简称"))
        if not code_column or not name_column:
            return StockProfile(ticker=ticker, source="akshare_code_name_schema_mismatch", confidence=0.0)

        matched = frame[frame[code_column].astype(str).str.zfill(6) == code]
        if matched.empty:
            return StockProfile(ticker=ticker, source="akshare_code_name_not_found", confidence=0.0)

        company_name = str(matched.iloc[0][name_column]).strip() or None
        sector_keyword = cls._infer_sector_keyword(company_name)
        aliases = [item for item in {company_name, ticker, code} if item]
        return StockProfile(
            ticker=ticker,
            company_name=company_name,
            sector_keyword=sector_keyword,
            sector_name=sector_keyword,
            aliases=aliases,
            source="akshare_code_name",
            confidence=0.75 if company_name else 0.0,
        )

    @classmethod
    def _infer_sector_keyword(cls, company_name: str | None) -> str | None:
        text = str(company_name or "")
        for needles, sector in _SECTOR_HINTS:
            if any(needle in text for needle in needles):
                return sector
        return None

    @staticmethod
    def _first_existing_column(frame: Any, candidates: tuple[str, ...]) -> str | None:
        for column in candidates:
            if column in frame.columns:
                return column
        return None

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
