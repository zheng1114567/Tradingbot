"""Cleaning pipeline owned by the scan/data collection layer."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from .cleaner import DataCleaner
from .news_text import is_noise_news_record, select_evidence_text
from .sector_context import clean_sector_context


NowFn = Callable[[], str]


def clean_data_agent_raw(
    raw_payload: dict[str, Any],
    *,
    now_fn: NowFn,
) -> dict[str, Any]:
    """Normalize raw scan output into DataAgent's cleaned artifact shape."""
    daily_raw = raw_payload.get("daily")
    if isinstance(daily_raw, list):
        daily_df = DataCleaner.clean_daily(daily_raw)
        daily_df = DataCleaner.detect_limit_up_down(daily_df)
    else:
        daily_df = pd.DataFrame()

    capital_flow_raw = raw_payload.get("capital_flow")
    capital_flow = capital_flow_raw if isinstance(capital_flow_raw, list) else []
    news_raw = raw_payload.get("news")
    news = news_raw if isinstance(news_raw, list) else []
    market_raw = raw_payload.get("market")
    if isinstance(market_raw, list):
        market_df = DataCleaner.clean_daily(market_raw)
    else:
        market_df = pd.DataFrame()
    limit_up_summary = raw_payload.get("limit_up_summary")
    dragon_tiger_raw = raw_payload.get("dragon_tiger")
    market_breadth = raw_payload.get("market_breadth")
    sector_raw = raw_payload.get("sector_context")
    sector_context = clean_sector_context(sector_raw if isinstance(sector_raw, list) else [])
    risk_raw = raw_payload.get("risk", {})
    cleaned_news = clean_news(news)

    return {
        "stage": "cleaned",
        "created_at": now_fn(),
        "market": {
            "record_count": int(len(market_df)),
            "columns": list(market_df.columns),
            "records": records_from_frame(market_df),
        },
        "daily": {
            "record_count": int(len(daily_df)),
            "columns": list(daily_df.columns),
            "records": records_from_frame(daily_df),
        },
        "sector_context": {
            "record_count": len(sector_context),
            "records": sector_context,
        },
        "limit_up_summary": limit_up_summary if isinstance(limit_up_summary, dict) else {},
        "dragon_tiger": {
            "record_count": len(dragon_tiger_raw) if isinstance(dragon_tiger_raw, list) else 0,
            "records": dragon_tiger_raw if isinstance(dragon_tiger_raw, list) else [],
        },
        "market_breadth": market_breadth if isinstance(market_breadth, dict) else {},
        "capital_flow": {
            "record_count": len(capital_flow),
            "records": capital_flow,
        },
        "news": {
            "record_count": len(cleaned_news),
            "records": cleaned_news,
        },
        "risk": risk_raw if isinstance(risk_raw, dict) else {},
    }


def clean_news(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        if is_noise_news_record(record):
            continue
        title = first_present(record, ["title", "标题", "新闻标题", "article_title", "name"]) or ""
        summary = first_present(record, ["summary", "摘要", "内容", "content", "article_content"]) or title
        full_text = first_present(record, ["full_text", "正文", "text", "article_text"])
        evidence_text = select_evidence_text(full_text, summary, title)
        content_cleaning = record.get("content_cleaning")
        source = first_present(record, ["source", "来源", "data_source"]) or record.get("data_source", "")
        event_time = first_present(record, ["time", "时间", "datetime", "发布时间", "date", "日期"])
        url = first_present(record, ["url", "链接", "link"])

        # Rule-based direction detection
        direction = _detect_direction(title, summary, full_text)
        relevance = _compute_relevance(title, summary, record.get("code", ""), record.get("sector_name", ""))

        cleaned.append({
            "event_id": str(record.get("event_id") or f"news_{idx:04d}"),
            "event_type": str(record.get("event_type") or "新闻"),
            "summary": str(summary)[:500],
            "title": str(title or summary)[:200],
            "full_text": str(full_text or "")[:8000],
            "evidence_text": evidence_text,
            "content_status": str(record.get("content_status") or ("full_text" if full_text else "summary_only")),
            "content_cleaning": content_cleaning if isinstance(content_cleaning, dict) else {},
            "content_error": record.get("content_error"),
            "direction": direction,
            "relevance_score": relevance,
            "confidence": float(record.get("confidence") or 0.5),
            "event_time": json_safe_value(event_time),
            "source": source,
            "url": url,
            "raw": record,
        })
    return cleaned


def records_from_frame(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    frame = df if limit is None else df.tail(limit)
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def json_safe_value(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


# ── Rule-based direction / relevance ────────────────────────────

_BULLISH = [
    "涨停", "大涨", "暴涨", "飙升", "拉升", "走强", "走高", "领涨",
    "增长", "超预期", "突破", "创新高", "新高",
    "利好", "回购", "增持", "买入", "评级上调",
    "净流入", "主力加仓", "机构调研", "北上增持",
    "扭亏", "盈利", "业绩预增", "订单", "签约",
]
_BEARISH = [
    "跌停", "大跌", "暴跌", "跳水", "下挫", "走弱", "走低", "领跌",
    "下降", "不及预期", "破位", "创新低", "新低",
    "利空", "减持", "卖出", "评级下调",
    "净流出", "主力减仓", "亏损", "业绩预亏", "下滑",
    "立案", "处罚", "问询函", "监管函", "ST", "退市", "暴雷", "爆雷",
]


def _detect_direction(title: str, summary: str, full_text: Any) -> str:
    text = f"{title} {summary} {full_text or ''}"
    bull_score = sum(1 for kw in _BULLISH if kw in text)
    bear_score = sum(1 for kw in _BEARISH if kw in text)
    if bull_score > bear_score + 1:
        return "利好"
    if bear_score > bull_score + 1:
        return "利空"
    if bull_score > 0 and bull_score == bear_score:
        return "中性"
    if bull_score > 0:
        return "中性偏多"
    if bear_score > 0:
        return "中性偏空"
    return "中性"


def _compute_relevance(title: str, summary: str, code: str, sector: str) -> float:
    """Score how relevant this news is to the target stock (0.1-1.0)."""
    text = f"{title} {summary}"
    score = 0.1

    # Exact ticker match
    if code and code.split(".")[0] in text:
        score = max(score, 0.9)
    # Partial code match (6 digits)
    if code and any(part in text for part in [code[:6], code.split(".")[0]]):
        score = max(score, 0.7)
    # Sector match
    if sector and sector in text:
        score = max(score, 0.4)

    # Length penalty for very short titles (likely low-quality)
    if len(title) < 10:
        score = max(0.1, score - 0.2)

    return round(min(score, 1.0), 2)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None
