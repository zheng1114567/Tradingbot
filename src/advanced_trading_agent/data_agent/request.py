"""Pydantic-validated input boundary for standalone DataAgent runs."""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator


class DataAgentRequest(BaseModel):
    """Input boundary for a standalone data-agent run (Pydantic-validated)."""

    ticker: str
    trade_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    include_market: bool = True
    include_capital_flow: bool = True
    include_news: bool = True
    include_factors: bool = True
    include_risk: bool = True
    include_sector_context: bool = True
    news_keyword: str | None = None
    sector_keyword: str | None = None
    use_llm_news_filter: bool = True
    fetch_news_full_text: bool = True
    news_relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_react_planner: bool = False
    output_dir: str | None = None
    max_news_records: int = Field(default=20, ge=1, le=200)
    max_return_records: int = Field(default=20, ge=1, le=200)
    sector_top_n: int = Field(default=20, ge=1, le=100)

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: str) -> str:
        if not v:
            return v
        if not re.match(r"^\d{6}\.(SH|SZ|BJ)$", v, re.IGNORECASE):
            raise ValueError(f"Invalid ticker format: {v!r}. Expected pattern: 000001.SZ")
        return v.upper()

    @field_validator("trade_date", "start_date", "end_date")
    @classmethod
    def _validate_date(cls, v: str | None) -> str | None:
        if v is None:
            return v
        clean = v.replace("-", "")
        if not re.match(r"^\d{8}$", clean):
            raise ValueError(f"Invalid date format: {v!r}. Expected YYYY-MM-DD or YYYYMMDD")
        return v

    def normalized_trade_date(self) -> str:
        return self.trade_date or date.today().isoformat()

    def normalized_end_date(self) -> str | None:
        if self.end_date:
            return self.end_date
        if self.trade_date:
            return self.trade_date.replace("-", "")
        return None
