"""News filtering and event shaping for DataAgent analysis."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from .request import DataAgentRequest


LLMFactory = Callable[[], Any]


def build_event_records(
    news_records: list[dict[str, Any]],
    request: DataAgentRequest,
) -> list[dict[str, Any]]:
    events = []
    for record in news_records[: request.max_news_records]:
        events.append({
            "event_id": record.get("event_id"),
            "event_type": record.get("event_type", "新闻"),
            "summary": record.get("summary", ""),
            "evidence_text": record.get("evidence_text") or record.get("summary", ""),
            "content_status": record.get("content_status", "summary_only"),
            "content_cleaning": record.get("content_cleaning", {}),
            "content_error": record.get("content_error"),
            "direction": record.get("direction", "中性"),
            "relevance_score": record.get("llm_relevance") or record.get("relevance_score"),
            "confidence": record.get("confidence", 0.5),
            "transmission_path": record.get("transmission_path", "新闻输入"),
            "direct_beneficiaries": record.get("direct_beneficiaries") or [request.sector_keyword or request.ticker],
            "evidence_level": "公开新闻",
            "pricing_status": "未定价",
            "chain_quality": record.get("chain_quality", "weak"),
            "event_time": record.get("event_time"),
            "source": record.get("source", ""),
            "url": record.get("url"),
            "news_scope": record.get("news_scope", "ticker"),
            "sector_name": record.get("sector_name"),
            "llm_reason": record.get("llm_reason", ""),
            "raw": record.get("raw", {}),
        })
    return events


class NewsFilter:
    """Filter relevant news with optional LLM support and deterministic fallback."""

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._llm_factory = llm_factory

    def filter(
        self,
        news_records: list[dict[str, Any]],
        request: DataAgentRequest,
    ) -> dict[str, Any]:
        if not news_records:
            return {
                "records": [],
                "trace": {
                    "mode": "empty",
                    "used_llm": False,
                    "reason": "no news records",
                    "input_count": 0,
                    "output_count": 0,
                },
            }

        if not request.use_llm_news_filter:
            records = filter_news_deterministically(news_records, request)
            return {
                "records": records,
                "trace": {
                    "mode": "deterministic",
                    "used_llm": False,
                    "reason": "LLM news filter disabled",
                    "input_count": len(news_records),
                    "output_count": len(records),
                },
            }

        try:
            llm = self.get_llm()
            if not self.llm_configured(llm):
                raise RuntimeError(f"{getattr(llm, 'provider', 'llm')} API key is not configured")
            decisions = ask_llm_to_filter_news(llm, news_records, request)
            selected: list[dict[str, Any]] = []
            decision_by_id = {str(item.get("event_id")): item for item in decisions}
            decision_trace: list[dict[str, Any]] = []
            for record in news_records:
                decision = decision_by_id.get(str(record.get("event_id")), {})
                try:
                    relevance = float(decision.get("relevance", 0))
                except (TypeError, ValueError):
                    relevance = 0.0
                keep = bool(decision.get("keep"))
                decision_trace.append({
                    "event_id": record.get("event_id"),
                    "title": record.get("title"),
                    "keep": keep,
                    "relevance": relevance,
                    "direction": decision.get("direction"),
                    "confidence": bounded_float(decision.get("confidence"), default=0.0),
                    "reason": str(decision.get("reason", ""))[:300],
                })
                if not keep or relevance < request.news_relevance_threshold:
                    continue
                selected.append({
                    **record,
                    "direction": decision.get("direction") or record.get("direction", "中性"),
                    "confidence": bounded_float(decision.get("confidence"), default=0.5),
                    "llm_relevance": relevance,
                    "llm_reason": str(decision.get("reason", ""))[:300],
                })
            guardrail_records = []
            if not selected:
                guardrail_records = filter_news_deterministically(news_records, request)
                for record in guardrail_records:
                    selected.append({
                        **record,
                        "llm_relevance": max(float(record.get("llm_relevance", 0.5)), request.news_relevance_threshold),
                        "llm_reason": "keyword guardrail after LLM filtered all candidates",
                    })
            return {
                "records": selected[: request.max_news_records],
                "trace": {
                    "mode": "llm" if not guardrail_records else "llm_with_keyword_guardrail",
                    "used_llm": True,
                    "model": getattr(llm, "model", ""),
                    "provider": getattr(llm, "provider", ""),
                    "input_count": len(news_records),
                    "output_count": len(selected[: request.max_news_records]),
                    "threshold": request.news_relevance_threshold,
                    "guardrail_added_count": len(guardrail_records),
                    "decisions": decision_trace,
                },
            }
        except Exception as exc:
            records = filter_news_deterministically(news_records, request)
            return {
                "records": records,
                "trace": {
                    "mode": "fallback",
                    "used_llm": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "input_count": len(news_records),
                    "output_count": len(records),
                    "threshold": request.news_relevance_threshold,
                },
            }

    def get_llm(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        if self._llm_factory is not None:
            return self._llm_factory()
        from ..llm.client import create_llm

        return create_llm()

    def llm_configured(self, llm: Any) -> bool:
        if self._llm_client is not None:
            return True
        return llm_news_filter_configured(llm)


def llm_news_filter_configured(llm: Any) -> bool:
    from ..llm.client import llm_api_key_configured

    return llm_api_key_configured(str(getattr(llm, "provider", "qwen")))


def ask_llm_to_filter_news(
    llm: Any,
    news_records: list[dict[str, Any]],
    request: DataAgentRequest,
) -> list[dict[str, Any]]:
    candidates = [
        {
            "event_id": record.get("event_id"),
            "title": record.get("title"),
            "summary": record.get("summary"),
            "content_status": record.get("content_status"),
            "event_time": record.get("event_time"),
            "source": record.get("source"),
        }
        for record in news_records[: max(request.max_news_records * 2, 15)]
    ]
    prompt = {
        "ticker": request.ticker,
        "trade_date": request.normalized_trade_date(),
        "news_keyword": request.news_keyword,
        "sector_keyword": request.sector_keyword,
        "task": (
            "Select news relevant to this ticker, its company, its business, or its sector for downstream trading agents. "
            "If sector_keyword is present, sector-level news is valid even when it does not mention the ticker directly. "
            "If title or summary directly contains the ticker keyword, company name, sector keyword, or clearly describes this company or its board, keep it unless it is obviously unrelated. "
            "Return one decision for every candidate in the same event_id space. "
            "Return only JSON with key decisions: list of objects containing event_id, keep, "
            "relevance from 0 to 1, direction in 正面/负面/中性, confidence from 0 to 1, and reason."
        ),
        "candidates": candidates,
    }
    response = llm.chat(
        [
            (
                "system",
                "你是量化交易数据管道里的新闻筛选器。只返回 JSON，不要输出解释文字。",
            ),
            ("human", json.dumps(prompt, ensure_ascii=False)),
        ],
        temperature=0,
        max_tokens=4096,
    )
    payload = json.loads(str(response))
    if isinstance(payload, list):
        decisions = payload
    elif isinstance(payload, dict):
        decisions = payload.get("decisions", [])
    else:
        decisions = []
    if not isinstance(decisions, list):
        raise ValueError("LLM news filter response must contain a decisions list")
    return [item for item in decisions if isinstance(item, dict)]


def filter_news_deterministically(
    news_records: list[dict[str, Any]],
    request: DataAgentRequest,
) -> list[dict[str, Any]]:
    keywords = [
        str(request.sector_keyword or "").lower(),
        str(request.news_keyword or "").lower(),
        str(request.ticker or "").lower(),
    ]
    keywords = [keyword for keyword in keywords if keyword]
    selected = []
    for record in news_records:
        haystack = json.dumps(record, ensure_ascii=False).lower()
        if not keywords or any(keyword in haystack for keyword in keywords):
            selected.append({
                **record,
                "llm_relevance": 0.5,
                "llm_reason": "deterministic keyword fallback",
            })
    return selected[: request.max_news_records]


def bounded_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))
