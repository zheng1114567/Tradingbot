"""Sector-first ETF selection.

This module is the seam for the strategy shift from stock picking to
sector picking.  Callers ask for sector candidates and ETF matches; the
implementation can use cached market data, news, and ETF universe details
without exposing those details to LangGraph or the conversation layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable

from .etf_watchlist import (
    ExcludedSectorCandidate,
    SectorCandidatePayload,
    WatchlistETFCandidate,
)
from .scanner import MarketScanner, ScanResult
from .vendor_router import route_to_vendor


RouteFn = Callable[..., Any]


@dataclass
class ETFCandidate:
    """A tradable ETF that maps to one sector candidate."""

    code: str
    name: str
    match_score: float
    liquidity_score: float
    total_score: float
    reason: str
    raw: dict[str, Any] = field(default_factory=dict)
    tracking_purity_score: float = 0.0
    tradable: bool = True
    blocked_reasons: list[str] = field(default_factory=list)
    pre_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_watchlist_candidate(self) -> WatchlistETFCandidate:
        return WatchlistETFCandidate(
            code=self.code,
            name=self.name,
            match_score=self.match_score,
            liquidity_score=self.liquidity_score,
            tracking_purity_score=self.tracking_purity_score,
            total_score=self.total_score,
            reason=self.reason,
            tradable=self.tradable,
            blocked_reasons=self.blocked_reasons,
            pre_rank=self.pre_rank,
            raw=_slim_etf_raw(self.raw),
        )


@dataclass
class SectorCandidate:
    """A sector decision unit for the ETF strategy."""

    sector_name: str
    score: float
    momentum_score: float
    breadth_score: float
    event_score: float
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    etfs: list[ETFCandidate] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_etf(self) -> ETFCandidate | None:
        return self.etfs[0] if self.etfs else None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_etf"] = self.primary_etf.to_dict() if self.primary_etf else None
        return payload

    def to_watchlist_payload(self) -> SectorCandidatePayload:
        return SectorCandidatePayload(
            sector_name=self.sector_name,
            pre_score=self.score,
            momentum_score=self.momentum_score,
            breadth_score=self.breadth_score,
            event_score=self.event_score,
            evidence={
                "momentum": [item for item in self.evidence if "动量" in item],
                "breadth": [item for item in self.evidence if "宽度" in item],
                "events": [item for item in self.evidence if "新闻" in item or "事件" in item],
                "etf": [item for item in self.evidence if "ETF" in item],
            },
            support_evidence=list(self.evidence),
            risk_flags=list(self.risks),
            raw_etf_candidates=[etf.to_watchlist_candidate() for etf in self.etfs],
            raw=_slim_etf_raw(self.raw),
        )


@dataclass
class SectorETFSelection:
    """Batch selection output: roundtable-ready candidates plus pre-roundtable exclusions."""

    trade_date: str
    candidates: list[SectorCandidate] = field(default_factory=list)
    excluded: list[ExcludedSectorCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date,
            "candidates": [payload.model_dump(mode="json") for payload in self.watchlist_payloads()],
            "excluded": [item.model_dump(mode="json") for item in self.excluded],
        }

    def watchlist_payloads(self) -> list[SectorCandidatePayload]:
        return [candidate.to_watchlist_payload() for candidate in self.candidates]


class SectorETFSelector:
    """Build sector-ranked ETF candidates from cached/free data sources."""

    def __init__(
        self,
        *,
        scanner: MarketScanner | None = None,
        route_fn: RouteFn | None = None,
        top_sectors: int = 8,
        top_etfs_per_sector: int = 3,
        auto_refresh_cache: bool = False,
    ) -> None:
        self.scanner = scanner or MarketScanner(
            top_sectors=top_sectors,
            top_n=30,
            auto_refresh_cache=auto_refresh_cache,
        )
        self.route_fn = route_fn or route_to_vendor
        self.top_sectors = top_sectors
        self.top_etfs_per_sector = top_etfs_per_sector

    def select(
        self,
        trade_date: str | None = None,
        *,
        sector_query: str | None = None,
        scan_results: list[ScanResult] | None = None,
    ) -> list[SectorCandidate]:
        """Return ranked sector candidates with ETF matches.

        If *sector_query* is provided, candidates are filtered to matching
        sectors but still retain the same scoring/risk logic.
        """
        td = trade_date or date.today().isoformat()
        results = scan_results if scan_results is not None else self.scanner.scan(td)
        context = getattr(self.scanner, "_last_scan_context", {}) or {}
        sector_rows = self._sector_rows(td, context)
        if sector_query and not any(self._matches(str(row.get("sector_name") or ""), sector_query) for row in sector_rows):
            sector_rows = self._with_requested_sector_rows(td, sector_rows, sector_query)
        etf_rows = self._etf_rows(td)
        news_by_sector = self._sector_news_map(td, sector_rows, sector_query=sector_query)
        grouped = self._group_scan_results(results)

        candidates: list[SectorCandidate] = []
        for idx, row in enumerate(sector_rows[: self.top_sectors * 2], start=1):
            name = str(row.get("sector_name") or row.get("name") or "").strip()
            if not name:
                continue
            if sector_query and not self._matches(name, sector_query):
                continue

            scan_evidence = grouped.get(name, [])
            momentum = self._momentum_score(row, rank=idx)
            breadth = self._breadth_score(scan_evidence)
            events = news_by_sector.get(name, [])
            event_score = min(len(events) * 0.6, 3.0)
            score = round(momentum + breadth + event_score, 2)
            etfs = self._match_etfs(name, etf_rows)

            evidence = [
                f"板块动量={momentum:.1f}",
                f"候选宽度={len(scan_evidence)}",
            ]
            if events:
                evidence.append(f"相关新闻={len(events)}条")
            if etfs:
                evidence.append(f"匹配ETF={etfs[0].code} {etfs[0].name}")

            risks = self._risk_flags(name, score=score, etfs=etfs, scan_evidence=scan_evidence, events=events)
            candidates.append(
                SectorCandidate(
                    sector_name=name,
                    score=score,
                    momentum_score=round(momentum, 2),
                    breadth_score=round(breadth, 2),
                    event_score=round(event_score, 2),
                    evidence=evidence,
                    risks=risks,
                    etfs=etfs[: self.top_etfs_per_sector],
                    raw={"sector": row, "scan_results": [asdict(item) for item in scan_evidence[:10]], "news": events[:10]},
                )
            )

        candidates.sort(key=lambda item: (item.primary_etf is not None, item.score), reverse=True)
        return candidates[: self.top_sectors]

    def select_with_exclusions(
        self,
        trade_date: str | None = None,
        *,
        sector_query: str | None = None,
        scan_results: list[ScanResult] | None = None,
        max_roundtable_sectors: int = 8,
    ) -> SectorETFSelection:
        """Return candidates allowed into the batch roundtable plus excluded sectors.

        Final ETF watchlist reports require a primary ETF for every sector.  Sectors
        with no ETF mapping, untradable ETF rows, or weak ETF liquidity are therefore
        removed before the roundtable and preserved in a compact exclusion list.
        """
        td = trade_date or date.today().isoformat()
        raw_candidates = self.select(td, sector_query=sector_query, scan_results=scan_results)
        candidates: list[SectorCandidate] = []
        excluded: list[ExcludedSectorCandidate] = []
        for candidate in raw_candidates:
            reason = self._pre_roundtable_exclusion_reason(candidate)
            if reason:
                excluded.append(self._excluded_sector(candidate, reason))
                continue
            candidates.append(candidate)
            if len(candidates) >= max_roundtable_sectors:
                break
        return SectorETFSelection(trade_date=td, candidates=candidates, excluded=excluded)

    def explain_sector(
        self,
        sector_name: str,
        trade_date: str | None = None,
    ) -> dict[str, Any]:
        """Return a compact explanation payload for sector Q&A."""
        candidates = self.select(trade_date, sector_query=sector_name)
        if not candidates:
            return {
                "sector_name": sector_name,
                "status": "not_found",
                "verdict": "数据不足",
                "reasons": [f"没有在今日热点板块或缓存板块排名中找到“{sector_name}”。"],
                "risks": ["需要先刷新板块、ETF、新闻缓存后再判断。"],
                "candidate": None,
            }

        candidate = candidates[0]
        verdict = "可关注" if candidate.score >= 6 and candidate.primary_etf else "暂不适合"
        reasons = list(candidate.evidence)
        if verdict == "暂不适合":
            reasons.extend(candidate.risks or ["板块证据强度不足，不能只凭单一热度买入ETF。"])
        return {
            "sector_name": candidate.sector_name,
            "status": "matched",
            "verdict": verdict,
            "score": candidate.score,
            "reasons": reasons,
            "risks": candidate.risks,
            "primary_etf": candidate.primary_etf.to_dict() if candidate.primary_etf else None,
            "candidate": candidate.to_dict(),
        }

    def format_markdown(self, candidates: list[SectorCandidate]) -> str:
        """Render sector ETF candidates as a Markdown table."""
        if not candidates:
            return "未发现可映射到 ETF 的板块候选。"
        lines = [
            "## 板块ETF候选",
            "",
            "| # | 板块 | 评分 | 首选ETF | 证据 | 风险 |",
            "|---|------|------|---------|------|------|",
        ]
        for idx, candidate in enumerate(candidates, start=1):
            etf = candidate.primary_etf
            etf_text = f"{etf.code} {etf.name}" if etf else "无匹配ETF"
            evidence = "；".join(candidate.evidence[:3])
            risks = "；".join(candidate.risks[:2]) if candidate.risks else "-"
            lines.append(
                f"| {idx} | {candidate.sector_name} | {candidate.score:.1f} | "
                f"{etf_text} | {evidence} | {risks} |"
            )
        return "\n".join(lines)

    def _sector_rows(self, trade_date: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        rows = context.get("hot_sectors") or []
        if isinstance(rows, list) and rows:
            return rows
        try:
            data = self.route_fn("get_sector", top_n=self.top_sectors * 2, trade_date=trade_date)
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _with_requested_sector_rows(
        self,
        trade_date: str,
        rows: list[dict[str, Any]],
        sector_query: str,
    ) -> list[dict[str, Any]]:
        """Ensure a user-requested sector can be explained even if not hot."""
        try:
            data = self.route_fn("get_sector", top_n=self.top_sectors * 3, trade_date=trade_date)
        except Exception:
            data = []
        matched = [
            row for row in data
            if isinstance(row, dict) and self._matches(str(row.get("sector_name") or row.get("name") or ""), sector_query)
        ] if isinstance(data, list) else []
        if matched:
            return [*matched, *rows]
        return [
            {
                "sector_name": sector_query,
                "change_pct": 0,
                "strength_score": 0,
                "source": "user_query_unranked",
            },
            *rows,
        ]

    def _etf_rows(self, trade_date: str) -> list[dict[str, Any]]:
        try:
            data = self.route_fn("get_etf_spot", trade_date=trade_date)
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _sector_news_map(
        self,
        trade_date: str,
        sector_rows: list[dict[str, Any]],
        *,
        sector_query: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        targets = [str(row.get("sector_name") or "") for row in sector_rows[: self.top_sectors]]
        if sector_query and sector_query not in targets:
            targets.insert(0, sector_query)
        news_map: dict[str, list[dict[str, Any]]] = {}
        for name in targets:
            if not name:
                continue
            try:
                data = self.route_fn("get_news", sector=name, keyword=name, trade_date=trade_date, limit=8)
            except Exception:
                data = []
            news_map[name] = data if isinstance(data, list) else []
        return news_map

    @staticmethod
    def _group_scan_results(results: list[ScanResult]) -> dict[str, list[ScanResult]]:
        grouped: dict[str, list[ScanResult]] = {}
        for result in results:
            sectors = result.extra.get("all_sectors") if isinstance(result.extra, dict) else None
            names = sectors if isinstance(sectors, list) and sectors else [result.sector]
            for name in names:
                key = str(name or "").strip()
                if key:
                    grouped.setdefault(key, []).append(result)
        return grouped

    @staticmethod
    def _momentum_score(row: dict[str, Any], *, rank: int) -> float:
        change = _float(row.get("change_pct") or row.get("涨跌幅") or row.get("change"))
        strength = _float(row.get("strength_score"))
        if strength is None:
            strength = change or 0.0
        rank_bonus = max(0.0, 2.0 - (rank - 1) * 0.2)
        return max(0.0, min(float(strength) * 0.8 + abs(change or 0.0) * 0.4 + rank_bonus, 7.0))

    @staticmethod
    def _breadth_score(scan_evidence: list[ScanResult]) -> float:
        if not scan_evidence:
            return 0.0
        source_bonus = len({source for item in scan_evidence for source in item.source.split("+")}) * 0.35
        return min(len(scan_evidence) * 0.35 + source_bonus, 3.0)

    def _match_etfs(self, sector_name: str, etf_rows: list[dict[str, Any]]) -> list[ETFCandidate]:
        sector_tokens = self._tokens(sector_name)
        candidates: list[ETFCandidate] = []
        for row in etf_rows:
            name = str(row.get("name") or row.get("名称") or "").strip()
            code = str(row.get("code") or row.get("代码") or "").strip()
            if not name or not code:
                continue
            name_tokens = self._tokens(name)
            overlap = sector_tokens & name_tokens
            contains = any(token and token in name for token in sector_tokens)
            if not overlap and not contains:
                continue
            match_score = 5.0 if contains else 3.0 + len(overlap)
            liquidity = self._liquidity_score(row)
            purity = self._tracking_purity_score(sector_name, name, row)
            blocked = self._etf_blocked_reasons(row)
            total = round(match_score + liquidity + purity, 2)
            candidates.append(
                ETFCandidate(
                    code=code,
                    name=name,
                    match_score=round(match_score, 2),
                    liquidity_score=round(liquidity, 2),
                    total_score=total,
                    reason=f"名称与“{sector_name}”匹配，流动性评分 {liquidity:.1f}，跟踪纯度 {purity:.1f}",
                    raw=dict(row),
                    tracking_purity_score=round(purity, 2),
                    tradable=not blocked,
                    blocked_reasons=blocked,
                )
            )
        candidates.sort(key=lambda item: item.total_score, reverse=True)
        for idx, candidate in enumerate(candidates, start=1):
            candidate.pre_rank = idx
        return candidates

    @staticmethod
    def _liquidity_score(row: dict[str, Any]) -> float:
        amount = _float(row.get("amount") or row.get("成交额") or row.get("turnover"))
        if amount is None:
            return 0.0
        if amount >= 500_000_000:
            return 3.0
        if amount >= 100_000_000:
            return 2.0
        if amount >= 30_000_000:
            return 1.0
        return 0.2

    @staticmethod
    def _tracking_purity_score(sector_name: str, etf_name: str, row: dict[str, Any]) -> float:
        """Score how directly an ETF tracks the requested sector/theme."""
        del row
        broad_terms = ("沪深300", "中证500", "中证1000", "创业板", "上证50", "A500", "红利", "央企")
        if any(term in etf_name for term in broad_terms):
            return 0.2
        if sector_name and sector_name in etf_name:
            return 2.0
        overlap = SectorETFSelector._tokens(sector_name) & SectorETFSelector._tokens(etf_name)
        return min(0.5 + len(overlap) * 0.2, 1.5) if overlap else 0.0

    @staticmethod
    def _etf_blocked_reasons(row: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        name = str(row.get("name") or row.get("名称") or "")
        status = str(row.get("status") or row.get("交易状态") or "")
        pct = _float(row.get("pct_chg") or row.get("涨跌幅"))
        if "停牌" in status or "停牌" in name:
            reasons.append("ETF 停牌")
        if pct is not None and abs(pct) >= 9.8:
            reasons.append("ETF 接近涨跌停，交易可执行性不足")
        return reasons

    @staticmethod
    def _risk_flags(
        sector_name: str,
        *,
        score: float,
        etfs: list[ETFCandidate],
        scan_evidence: list[ScanResult],
        events: list[dict[str, Any]],
    ) -> list[str]:
        risks: list[str] = []
        if not etfs:
            risks.append("未匹配到可交易ETF，不能直接执行板块ETF策略。")
        elif etfs[0].liquidity_score < 1.0:
            risks.append("首选ETF成交额偏低，可能有冲击成本。")
        if score < 6:
            risks.append("综合分低于关注阈值，动量/事件/宽度共振不足。")
        if len(scan_evidence) < 3:
            risks.append("板块内强势样本少，可能只是个别成分股扰动。")
        if not events:
            risks.append("缺少新闻或事件催化证据，持续性需要打折。")
        if any(word in sector_name for word in ("ST", "退市")):
            risks.append("板块名称含高风险标签。")
        return risks

    @staticmethod
    def _pre_roundtable_exclusion_reason(candidate: SectorCandidate) -> str:
        if not candidate.etfs:
            return "no_tradable_etf"
        tradable = [etf for etf in candidate.etfs if etf.tradable]
        if not tradable:
            return "etf_suspended"
        if max(etf.liquidity_score for etf in tradable) < 1.0:
            return "low_etf_liquidity"
        if max(etf.match_score for etf in tradable) < 3.5:
            return "mapping_uncertain"
        return ""

    @staticmethod
    def _excluded_sector(candidate: SectorCandidate, reason: str) -> ExcludedSectorCandidate:
        return ExcludedSectorCandidate(
            sector=candidate.sector_name,
            excluded_reason=reason,  # type: ignore[arg-type]
            brief_evidence=[
                f"板块评分 {candidate.score:.1f}",
                *candidate.evidence[:3],
                *(candidate.risks[:2] if candidate.risks else []),
            ],
            raw={"sector_name": candidate.sector_name, "score": candidate.score, "risks": candidate.risks},
        )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        cleaned = re.sub(r"(ETF|基金|指数|联接|增强|LOF|C|A)$", "", text, flags=re.IGNORECASE)
        chunks = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", cleaned)
        tokens = set()
        for chunk in chunks:
            tokens.add(chunk)
            for size in (2, 3, 4):
                if len(chunk) >= size:
                    tokens.update(chunk[i : i + size] for i in range(0, len(chunk) - size + 1))
        return {token for token in tokens if token}

    @staticmethod
    def _matches(sector_name: str, query: str) -> bool:
        if query in sector_name or sector_name in query:
            return True
        return bool(SectorETFSelector._tokens(sector_name) & SectorETFSelector._tokens(query))


def _slim_etf_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep final watchlist JSON compact while preserving audit-relevant ETF fields."""
    keys = ("code", "raw_code", "name", "latest_price", "change_pct", "amount", "premium_discount", "data_source")
    return {key: raw.get(key) for key in keys if key in raw}


def sector_candidates_to_json(candidates: list[SectorCandidate]) -> str:
    """Serialize sector candidates for prompts/reports."""
    return json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False, indent=2)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("%", "").strip()
    multiplier = 1.0
    if text.endswith("万"):
        multiplier = 10_000.0
        text = text[:-1]
    elif text.endswith("亿"):
        multiplier = 100_000_000.0
        text = text[:-1]
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        return None
    try:
        return float(text) * multiplier
    except ValueError:
        return None
