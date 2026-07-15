"""News article text enrichment shared by scan and DataAgent.

The scan stage should do article fetching when it pre-collects ticker news.
DataAgent still uses the same helpers as a standalone fallback.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any, Callable

import requests


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NAVIGATION_NEWS_TITLES = {
    "财经首页",
    "新浪首页",
    "新浪导航",
    "行情中心",
    "意见反馈",
    "资金流向",
    "业绩报表",
    "业绩预告",
    "限售解禁",
    "新股申购",
    "实时大单",
    "基金重仓",
}
_NAVIGATION_URL_FRAGMENTS = (
    "finance.sina.com.cn/",
    "www.sina.com.cn/",
    "news.sina.com.cn/guide/",
    "vip.stock.finance.sina.com.cn/mkt/",
    "gu.sina.cn/pc/feedback/",
    "vip.stock.finance.sina.com.cn/moneyflow/",
    "vFinanceAnalyze",
    "vInvestConsult",
    "vRPD_NewStockIssue",
    "cn_bill_sum.php",
    "fund_center/index.html",
)
_NEWS_NOISE_PATTERNS = (
    re.compile(r"^原标题[:：]"),
    re.compile(r"^来源[:：]"),
    re.compile(r"^文章来源[:：]"),
    re.compile(r"^责任编辑[:：]"),
    re.compile(r"^编辑[:：]"),
    re.compile(r"^校对[:：]"),
    re.compile(r"^作者[:：]\s*$"),
    re.compile(r"^免责声明[:：]"),
    re.compile(r"^风险提示[:：]"),
    re.compile(r"^广告$"),
    re.compile(r"^更多精彩.*"),
    re.compile(r"^下载.*APP.*"),
    re.compile(r"^打开.*APP.*"),
    re.compile(r"^微信扫一扫.*"),
    re.compile(r"^海量资讯.*"),
    re.compile(r"^本文源自[:：]"),
    re.compile(r"^本文来自.*"),
    re.compile(r"^股市有风险.*"),
    re.compile(r"^投资需谨慎.*"),
    re.compile(r"^Copyright\b", re.IGNORECASE),
)


FetchFn = Callable[[str], str]


def select_evidence_text(full_text: Any, summary: Any, title: Any) -> str:
    text = str(full_text or "").strip()
    if len(text) >= 80:
        return text[:3000]
    fallback = str(summary or title or "").strip()
    return fallback[:1500]


def enrich_news_full_text(
    records: list[dict[str, Any]],
    *,
    fetch_fn: FetchFn | None = None,
    source: str = "data_agent",
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    fetch = fetch_fn or fetch_news_full_text
    for record in records:
        item = dict(record)
        title = _first_present(item, ["title", "标题", "新闻标题", "article_title", "name"]) or ""
        summary = _first_present(item, ["summary", "摘要", "内容", "content", "article_content"]) or title
        existing_text = _first_present(item, ["full_text", "正文", "text", "article_text"])
        url = _first_present(item, ["url", "链接", "link"])

        if item.get("full_text_attempted") and not existing_text:
            item["full_text"] = str(item.get("full_text") or "")
            item["content_status"] = item.get("content_status") or "summary_only"
            item["evidence_text"] = item.get("evidence_text") or select_evidence_text("", summary, title)
            enriched.append(item)
            continue

        item["full_text_attempted"] = True
        item["full_text_source"] = item.get("full_text_source") or source

        if existing_text:
            cleaned_text, cleaning_trace = clean_article_text(str(existing_text))
            item["raw_full_text"] = str(existing_text)[:12000]
            item["full_text"] = cleaned_text[:8000]
            item["content_status"] = item.get("content_status") or ("full_text" if cleaned_text else "summary_only")
            item["content_cleaning"] = cleaning_trace
            item["evidence_text"] = select_evidence_text(item["full_text"], summary, title)
            enriched.append(item)
            continue

        if not url:
            item["full_text"] = ""
            item["content_status"] = item.get("content_status") or "summary_only"
            item["content_error"] = item.get("content_error") or "missing_url"
            item["evidence_text"] = select_evidence_text("", summary, title)
            enriched.append(item)
            continue

        try:
            full_text = fetch(str(url))
        except Exception as exc:
            full_text = ""
            item["content_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

        if full_text:
            cleaned_text, cleaning_trace = clean_article_text(full_text)
            item["raw_full_text"] = full_text[:12000]
            item["full_text"] = cleaned_text[:8000]
            item["content_status"] = "full_text" if cleaned_text else "summary_only"
            item["content_cleaning"] = cleaning_trace
            item["content_error"] = item.get("content_error")
        else:
            item["full_text"] = ""
            item["content_status"] = "summary_only"
            item["content_cleaning"] = {
                "status": "skipped",
                "raw_length": 0,
                "cleaned_length": 0,
                "removed_segments": 0,
                "deduplicated_segments": 0,
            }
            item["content_error"] = item.get("content_error") or "extract_empty"
        item["evidence_text"] = select_evidence_text(item.get("full_text"), summary, title)
        enriched.append(item)
    return enriched


def is_noise_news_record(record: dict[str, Any]) -> bool:
    """Return True for navigation/data-entry rows masquerading as news."""
    title = str(_first_present(record, ["title", "标题", "新闻标题", "article_title", "name"]) or "").strip()
    summary = str(_first_present(record, ["summary", "摘要", "内容", "content", "article_content"]) or "").strip()
    url = str(_first_present(record, ["url", "链接", "link"]) or "").strip()
    event_time = _first_present(record, ["time", "时间", "datetime", "发布时间", "date", "日期"])

    if title in _NAVIGATION_NEWS_TITLES:
        return True
    if title and title == summary and not str(event_time or "").strip() and len(title) <= 8:
        return True
    if url and any(fragment in url for fragment in _NAVIGATION_URL_FRAGMENTS):
        return True
    return False

def fetch_news_full_text(url: str) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        },
        timeout=8,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    return extract_article_text(response.text)


def extract_article_text(html: str) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", html or "")
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", text, flags=re.IGNORECASE | re.DOTALL)
    if paragraphs:
        candidates = [_WHITESPACE_RE.sub(" ", _HTML_TAG_RE.sub(" ", p)).strip() for p in paragraphs]
        body = "\n".join(p for p in candidates if len(p) >= 12)
    else:
        body = _HTML_TAG_RE.sub(" ", text)
    body = unescape(body).replace("\u3000", " ").replace("\xa0", " ")
    body = "\n".join(
        _WHITESPACE_RE.sub(" ", line).strip()
        for line in body.splitlines()
        if line.strip()
    )
    if len(body) < 80:
        return ""
    return body[:12000]


def clean_article_text(text: str) -> tuple[str, dict[str, Any]]:
    raw = unescape(text or "").replace("\u3000", " ").replace("\xa0", " ")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw_length = len(raw)

    segments = [
        _WHITESPACE_RE.sub(" ", segment).strip()
        for segment in re.split(r"\n+|(?<=[。！？；])\s+", raw)
    ]
    cleaned_segments: list[str] = []
    seen: set[str] = set()
    removed = 0
    deduped = 0
    for segment in segments:
        if not segment:
            continue
        segment = strip_news_noise(segment)
        if not segment:
            removed += 1
            continue
        if is_noise_segment(segment):
            removed += 1
            continue
        normalized_key = re.sub(r"\W+", "", segment.lower())
        if normalized_key and normalized_key in seen:
            deduped += 1
            continue
        if normalized_key:
            seen.add(normalized_key)
        cleaned_segments.append(segment)

    cleaned = "\n".join(cleaned_segments).strip()
    if len(cleaned) < 80:
        cleaned = _WHITESPACE_RE.sub(" ", raw).strip()
        status = "raw_fallback" if cleaned else "empty"
    else:
        status = "cleaned"
    cleaned = cleaned[:12000]
    return cleaned, {
        "status": status,
        "raw_length": raw_length,
        "cleaned_length": len(cleaned),
        "removed_segments": removed,
        "deduplicated_segments": deduped,
    }


def strip_news_noise(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text).strip()
    text = re.sub(r"（?文章来源[:：].*?）?$", "", text).strip()
    text = re.sub(r"（?责任编辑[:：].*?）?$", "", text).strip()
    text = re.sub(r"（?编辑[:：].*?）?$", "", text).strip()
    text = re.sub(r"（?原标题[:：].*?）?$", "", text).strip()
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_noise_segment(text: str) -> bool:
    if len(text) < 6:
        return True
    if any(pattern.search(text) for pattern in _NEWS_NOISE_PATTERNS):
        return True
    if text.count(" ") > 0 and len(text.split()) <= 2 and len(text) < 16:
        return True
    return False


def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None
