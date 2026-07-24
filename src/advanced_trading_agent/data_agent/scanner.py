"""Autonomous market scanner — discovers hot sectors and stocks.

Design inspired by stock-money (IndigoBlueInChina) and quant-stock-selector:
1. Find hot sectors via capital flow / price strength
2. Get constituent stocks of hot sectors
3. Cross-reference with limit-up pool, northbound, dragon-tiger
4. Score by cross-channel confidence: stocks appearing in multiple hot
   sectors + limit-up boards + northbound interest = highest confidence

When scan_and_collect() is used, raw data (daily, capital_flow, news) is
fetched for top candidates during the scan pass, eliminating the need for
DataAgent to re-fetch the same data later.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

from ..core.vendor import timed_vendor_call
from .build_cache import ensure_scan_cache
from .scan_cleaning import clean_data_agent_raw as clean_raw_payload
from .news_text import enrich_news_full_text
from .trading_calendar import resolve_market_trade_date
from .vendor_router import ensure_default_vendor_registration, get_vendor_impl, route_to_vendor

logger = logging.getLogger(__name__)


def route_to_local_cache_only(method: str, **kwargs: Any) -> Any:
    """Resolve scan requests strictly from local cache implementations."""
    ensure_default_vendor_registration()
    impl = get_vendor_impl(method, "local_cache")
    if impl is None:
        raise RuntimeError(f"No local_cache implementation for {method}")
    return impl(**kwargs)


@dataclass
class ScanResult:
    ticker: str
    name: str
    source: str          # "hot_sector" | "limit_up" | "northbound" | "dragon_tiger" | "cross_sector"
    sector: str
    score: float
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanBundle:
    """Complete scan results with pre-collected raw data for top candidates.

    Shared data (market, sector_context, risk) is collected once.
    Per-ticker data (daily, capital_flow, news) is collected only for
    candidates that made the top-N cut.
    """

    trade_date: str
    results: list[ScanResult]
    shared_raw: ScanSharedRaw = field(default_factory=dict)
    ticker_data: dict[str, ScanTickerRaw] = field(default_factory=dict)
    route_trace: list[dict[str, Any]] = field(default_factory=list)

    def raw_for_ticker(self, ticker: str) -> dict[str, Any]:
        """Return the canonical raw payload consumed by DataAgent."""
        ticker_raw = self.ticker_data.get(ticker, {})
        return {
            "daily": ticker_raw.get("daily", []),
            "market": self.shared_raw.get("market", []),
            "sector_context": self.shared_raw.get("sector_context", []),
            "limit_up_summary": self.shared_raw.get("limit_up_summary", {}),
            "dragon_tiger": self.shared_raw.get("dragon_tiger", []),
            "market_breadth": self.shared_raw.get("market_breadth", {}),
            "capital_flow": ticker_raw.get("capital_flow", []),
            "news": ticker_raw.get("news", []),
            "risk": self.shared_raw.get("risk", {}),
            "route_trace": self.route_trace,
        }

    def package_for_ticker(self, ticker: str) -> ScanDataPackage:
        """Return scan-owned raw and cleaned payloads for DataAgent processing."""
        raw_payload = self.raw_for_ticker(ticker)
        return ScanDataPackage(
            raw_payload=raw_payload,
            cleaned_payload=MarketScanner.clean_data_agent_raw(raw_payload),
            route_trace=self.route_trace,
        )


@dataclass
class ScanDataPackage:
    """DataAgent input package produced by scan.

    Scan owns vendor fetching and deterministic cleaning. DataAgent consumes
    this package as an auditable processing input and does not reimplement
    either responsibility.
    """

    raw_payload: dict[str, Any]
    cleaned_payload: dict[str, Any]
    route_trace: list[dict[str, Any]] = field(default_factory=list)


class _ScorerEntry(TypedDict):
    """Internal accumulator for multi-channel scoring evidence."""
    name: str
    score: float
    sectors: list[str]
    sources: list[str]
    board_count: int
    reasons: list[str]


class ScanRiskRaw(TypedDict):
    st_status: list[str]
    suspended: list[str]
    delisting: list[str]


class ScanSharedRaw(TypedDict, total=False):
    market: list[dict[str, Any]]
    sector_context: list[dict[str, Any]]
    limit_up_summary: dict[str, Any]
    dragon_tiger: list[dict[str, Any]]
    market_breadth: dict[str, Any]
    risk: ScanRiskRaw


class ScanTickerRaw(TypedDict, total=False):
    daily: list[dict[str, Any]]
    capital_flow: list[dict[str, Any]]
    news: list[dict[str, Any]]


class MarketScanner:
    """Discover trending stocks through multi-channel scanning.

    Channels (in scoring order):
    1. Hot sector constituents (cross-sector = highest confidence)
    2. Limit-up pool details (momentum signal)
    3. Northbound top holdings (smart money signal)
    4. Dragon-tiger list (institutional signal)
    5. Short-term technical signals (MA trend, breakout, momentum — from local cache)
    """

    def __init__(
        self,
        top_sectors: int = 5,
        top_n: int = 20,
        base_candidates: int | None = None,
        per_sector_cap: int | None = None,
        *,
        route_fn: Callable[..., Any] | None = None,
        cache_only: bool = True,
        auto_refresh_cache: bool = False,
        live_news_fallback: bool = False,
    ):
        self.top_sectors = top_sectors
        self.top_n = top_n
        self.base_candidates = min(base_candidates or 12, top_n)
        self.per_sector_cap = per_sector_cap or max(2, math.ceil(top_n / 3))
        self._cache_only = cache_only
        self._auto_refresh_cache = auto_refresh_cache
        self._live_news_fallback = live_news_fallback
        self._route_fn = route_fn or (route_to_local_cache_only if cache_only else route_to_vendor)
        self._last_scan_context: dict[str, Any] = {}

    def scan(self, trade_date: str | None = None) -> list[ScanResult]:
        """Run full market scan and return ranked results."""
        td = resolve_market_trade_date(trade_date)
        self._ensure_cache_ready(td)
        scorer: dict[str, _ScorerEntry] = {}  # ticker -> accumulated evidence
        ctx: dict[str, Any] = {"trade_date": td}

        # Channel 1: Hot sectors → constituents (primary)
        self._scan_hot_sectors(td, scorer, ctx)

        # Channel 2: Limit-up pool
        self._scan_limit_up(td, scorer, ctx)

        # Channel 3: Northbound top holdings
        self._scan_northbound(td, scorer, ctx)

        # Channel 4: Dragon-tiger
        self._scan_dragon_tiger(td, scorer)

        # Channel 5: Short-term technical signals (from local cache, no API)
        self._scan_short_term_signals(td, scorer, ctx)

        self._last_scan_context = ctx

        # Build ranked results
        ranked = self._rank(scorer)
        return self._select_candidates(ranked, ctx)

    # ------------------------------------------------------------------
    # Channel scanners
    # ------------------------------------------------------------------

    def _scan_hot_sectors(self, td: str, scorer: dict[str, _ScorerEntry], ctx: dict[str, Any]) -> None:
        """Find top sectors and their constituent stocks."""
        try:
            sectors = self._route_fn("get_sector", top_n=self.top_sectors * 2, trade_date=td)
        except Exception as exc:
            logger.warning("Sector scan failed: %s", exc)
            return

        if isinstance(sectors, str) or not sectors:
            return
        if isinstance(sectors, dict) and "error" in sectors:
            return

        sector_list = sectors if isinstance(sectors, list) else []
        hot_sectors = sector_list[:self.top_sectors]
        ctx["hot_sectors"] = hot_sectors

        for sector in hot_sectors:
            sector_name = sector.get("sector_name", "")
            strength = float(sector.get("strength_score", 0) or 0)
            change_pct = float(sector.get("change_pct", 0) or 0)

            if not sector_name:
                continue

            try:
                constituents = self._route_fn(
                    "get_sector_constituents",
                    sector_name=sector_name,
                    trade_date=td,
                )
            except Exception:
                continue

            if not isinstance(constituents, list):
                continue

            for stock in constituents:
                code = str(stock.get("code", ""))
                if not code or not code[0].isdigit():
                    continue
                ticker = self._normalize_ticker(code)
                name = str(stock.get("name", ""))
                sector_score = min(strength * 0.3 + abs(change_pct) * 0.1, 4.0)

                if ticker not in scorer:
                    scorer[ticker] = {
                        "name": name,
                        "score": 0.0,
                        "sectors": [],
                        "sources": [],
                        "board_count": 0,
                        "reasons": [],
                    }
                scorer[ticker]["score"] += sector_score
                scorer[ticker]["sectors"].append(sector_name)
                if "hot_sector" not in scorer[ticker]["sources"]:
                    scorer[ticker]["sources"].append("hot_sector")

        # Cross-sector bonus: stocks in 2+ hot sectors get multiplied confidence
        for ticker, info in scorer.items():
            if len(info["sectors"]) >= 2:
                info["score"] *= 1.5
                info["sources"].append("cross_sector")
                info["reasons"].append(f"出现在 {len(info['sectors'])} 个热点板块: {', '.join(info['sectors'][:3])}")

    def _scan_limit_up(self, td: str, scorer: dict[str, _ScorerEntry], ctx: dict[str, Any]) -> None:
        """Score stocks in the limit-up pool."""
        try:
            data = self._route_fn("get_limit_up_tiers", trade_date=td)
        except Exception as exc:
            logger.warning("Limit-up scan failed: %s", exc)
            return

        if not isinstance(data, dict):
            return

        ctx["limit_up_summary"] = {
            "first_board": data.get("first_board", 0),
            "second_board": data.get("second_board", 0),
            "third_plus": data.get("third_plus", 0),
        }

        stocks = data.get("stocks", [])
        for stock in stocks:
            code = str(stock.get("code", ""))
            if not code or not code[0].isdigit():
                continue
            ticker = self._normalize_ticker(code)
            if ticker not in scorer:
                continue  # only bonus hot-sector stocks
            board = int(stock.get("board_count", 1))
            score = min(board * 1.5, 5.0)

            scorer[ticker]["score"] += score
            scorer[ticker]["board_count"] = max(scorer[ticker]["board_count"], board)
            if "limit_up" not in scorer[ticker]["sources"]:
                scorer[ticker]["sources"].append("limit_up")
            if board >= 2:
                scorer[ticker]["reasons"].append(f"{board}连板涨停")

    def _scan_northbound(self, td: str, scorer: dict[str, _ScorerEntry], ctx: dict[str, Any]) -> None:
        """Score stocks with northbound net buying."""
        try:
            top10 = self._route_fn("get_northbound_top10", trade_date=td)
        except Exception as exc:
            ctx["northbound_status"] = {
                "status": "unavailable",
                "reason": str(exc),
            }
            logger.info("Northbound scan unavailable: %s", exc)
            return

        if not isinstance(top10, list):
            ctx["northbound_status"] = {
                "status": "unavailable",
                "reason": "non-list response",
            }
            return

        ctx["northbound_top10"] = top10
        ctx["northbound_status"] = {
            "status": "available",
            "record_count": len(top10),
        }

        for stock in top10:
            code = str(stock.get("code", ""))
            if not code or not code[0].isdigit():
                continue
            ticker = self._normalize_ticker(code)
            if ticker not in scorer:
                continue  # only bonus hot-sector stocks
            net_buy = float(stock.get("net_buy", 0) or 0)
            score = 2.0 if net_buy > 0 else 1.0

            scorer[ticker]["score"] += score
            if "northbound" not in scorer[ticker]["sources"]:
                scorer[ticker]["sources"].append("northbound")
            if net_buy > 0:
                scorer[ticker]["reasons"].append(f"北向净买入 {net_buy/1e8:.1f}亿")

    def _scan_dragon_tiger(self, td: str, scorer: dict[str, _ScorerEntry]) -> None:
        """Score stocks appearing on dragon-tiger list."""
        try:
            dt_list = self._route_fn("get_dragon_tiger", trade_date=td)
        except Exception as exc:
            logger.warning("Dragon-tiger scan failed: %s", exc)
            return

        if not isinstance(dt_list, list):
            return

        for stock in dt_list:
            code = str(stock.get("代码", stock.get("code", "")))
            if not code or not code[0].isdigit():
                continue
            ticker = self._normalize_ticker(code)
            if ticker not in scorer:
                continue  # only bonus hot-sector stocks
            score = 1.0  # supplementary signal only

            scorer[ticker]["score"] += score
            if "dragon_tiger" not in scorer[ticker]["sources"]:
                scorer[ticker]["sources"].append("dragon_tiger")

    def _scan_short_term_signals(self, td: str, scorer: dict[str, _ScorerEntry], ctx: dict[str, Any]) -> None:
        """Score stocks using cached short-term technical signals.

        Reads from short_term_signals_{date}.json (computed by build_cache
        or compute_short_term_signals). Stocks with composite >= 60
        get a confidence bonus; stocks with composite <= 40 get a penalty.

        This channel uses only local cache — no API calls.
        """
        from pathlib import Path
        from ..config import config

        cache_dir = Path(config.get("results_dir")) / "local_cache"
        path = cache_dir / f"short_term_signals_{td}.json"
        if not path.exists():
            logger.debug("Short-term signal cache not found: %s", path)
            return

        try:
            import json
            data = json.loads(path.read_text("utf-8"))
        except Exception as exc:
            logger.warning("Failed to read short-term signals: %s", exc)
            return

        signals_data = data.get("signals", {})
        if not signals_data:
            return

        ctx_signals = {
            "signal_tickers": len(signals_data),
            "signal_top_bullish": [],
        }

        scored = 0
        for ticker, report in signals_data.items():
            composite = float(report.get("composite", 50))
            n_sig = int(report.get("n_signals", 0))

            if ticker not in scorer:
                continue

            if composite >= 60:
                bonus = (composite - 60) / 40 * 3.0  # 0~3分
                scorer[ticker]["score"] += bonus
                scorer[ticker]["sources"].append("short_term_signal")
                scorer[ticker]["reasons"].append(
                    f"短线信号 {composite:.0f}/100 ({n_sig}个信号共振)"
                )
                scored += 1
                if len(ctx_signals["signal_top_bullish"]) < 5:
                    ctx_signals["signal_top_bullish"].append(
                        f"{ticker}({composite:.0f})"
                    )
            elif composite <= 40:
                penalty = (40 - composite) / 40 * 1.5  # 0~1.5分
                scorer[ticker]["score"] = max(0, scorer[ticker]["score"] - penalty)
                scorer[ticker]["reasons"].append(
                    f"短线信号偏弱 {composite:.0f}/100"
                )

        self._last_scan_context["short_term_signals"] = ctx_signals
        ctx["short_term_signals"] = ctx_signals
        logger.debug("Short-term signal scan: scored %d tickers from cache", scored)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(self, scorer: dict[str, _ScorerEntry]) -> list[ScanResult]:
        results: list[ScanResult] = []
        for ticker, info in scorer.items():
            sectors = info.get("sectors", [])
            sources = info.get("sources", [])
            primary_sector = sectors[0] if sectors else ""

            reason_parts: list[str] = []
            if "cross_sector" in sources:
                reason_parts.append(f"跨 {len(sectors)} 个热点板块")
            if "limit_up" in sources:
                reason_parts.append("涨停池")
            if "northbound" in sources:
                reason_parts.append("北向资金关注")
            if "dragon_tiger" in sources:
                reason_parts.append("龙虎榜活跃")
            for r in info.get("reasons", [])[:2]:
                if r not in reason_parts:
                    reason_parts.append(r)

            results.append(ScanResult(
                ticker=ticker,
                name=info.get("name", ""),
                source="+".join(sources) if sources else "unknown",
                sector=primary_sector,
                score=round(info["score"], 1),
                reason="; ".join(reason_parts) if reason_parts else "多信号共振",
                extra={
                    "all_sectors": sectors,
                    "board_count": info.get("board_count", 0),
                },
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _select_candidates(
        self,
        ranked: list[ScanResult],
        ctx: dict[str, Any],
    ) -> list[ScanResult]:
        """Select a dynamic candidate pool with per-sector caps.

        Rules:
        - concentrated tape -> shrink candidate pool
        - broad tape -> widen candidate pool
        - always enforce a per-sector cap to avoid one-theme domination
        - penalize stocks with no daily cache data (BJ/unlisted)
        """
        if not ranked:
            return []

        # 过滤退市股和无格式代码
        filtered: list[ScanResult] = []
        for r in ranked:
            if "退" in r.name:
                continue
            if "." not in r.ticker:
                normalized = self._normalize_ticker(r.ticker)
                if normalized == r.ticker:
                    continue  # still has no suffix, skip
            # Penalize stocks without daily data: -4 points
            if not self._has_daily_cache(r.ticker):
                r.score = max(0, r.score - 4)
                r.reason = f"[无日线数据] {r.reason}"
            filtered.append(r)

        # Re-sort after penalty
        filtered.sort(key=lambda r: r.score, reverse=True)

        dynamic_limit = self._dynamic_candidate_limit(filtered, ctx)
        selected: list[ScanResult] = []
        sector_counts: dict[str, int] = {}

        for result in filtered:
            sector = result.sector or "未识别"
            if sector_counts.get(sector, 0) >= self.per_sector_cap:
                continue
            selected.append(result)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) >= dynamic_limit:
                break

        return selected

    def _dynamic_candidate_limit(
        self,
        ranked: list[ScanResult],
        ctx: dict[str, Any],
    ) -> int:
        """Adjust candidate count based on breadth of leading sectors."""
        if not ranked:
            return 0

        distinct_sectors = {result.sector for result in ranked[: min(len(ranked), self.base_candidates)] if result.sector}
        hot_sectors = ctx.get("hot_sectors", [])
        strong_hot_count = sum(
            1
            for sector in hot_sectors
            if float(sector.get("strength_score", 0) or 0) >= 2.5
            or float(sector.get("change_pct", 0) or 0) >= 2.5
        )

        if len(distinct_sectors) <= 2 or strong_hot_count <= 2:
            return min(self.top_n, max(8, self.base_candidates - 2))
        if len(distinct_sectors) >= 5 or strong_hot_count >= 4:
            return min(self.top_n, self.base_candidates + 3)
        return min(self.top_n, self.base_candidates)

    @staticmethod
    def _normalize_ticker(code: str) -> str:
        """Normalize a raw stock code to 000001.SZ format."""
        code = code.strip().upper()
        if "." in code:
            return code
        if code.startswith("6"):
            return f"{code}.SH"
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
        if code.startswith(("8", "4", "920")):
            return f"{code}.BJ"
        return code

    @staticmethod
    def _has_daily_cache(ticker: str) -> bool:
        """Check if daily parquet cache exists for this ticker."""
        from pathlib import Path
        from ..config import config

        cache_dir = Path(config.get("results_dir")) / "local_cache" / "daily"
        path = cache_dir / f"{ticker.replace('.', '_')}.parquet"
        return path.exists()

    def format_results(self, results: list[ScanResult]) -> str:
        """Render scan results as a readable table."""
        if not results:
            return "未发现符合条件的强势股。"

        lines = [
            "## 市场扫描结果",
            "",
            f"共发现 **{len(results)}** 只候选标的:",
            "",
            "| # | 代码 | 名称 | 来源 | 板块 | 评分 | 理由 |",
            "|---|------|------|------|------|------|------|",
        ]
        for i, r in enumerate(results, 1):
            source_icon = {
                "cross_sector": "跨板",
                "hot_sector": "热板",
                "limit_up": "涨停",
                "northbound": "北向",
                "dragon_tiger": "龙虎",
                "short_term_signal": "信号",
            }
            src = "+".join(source_icon.get(s, s) for s in r.source.split("+")[:2])
            lines.append(f"| {i} | {r.ticker} | {r.name} | {src} | {r.sector[:12]} | {r.score:.1f} | {r.reason[:50]} |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM market summary
    # ------------------------------------------------------------------

    def summarize_with_llm(
        self,
        results: list[ScanResult],
        llm_client: Any | None = None,
    ) -> str:
        """Generate a markdown market analysis report using LLM.

        Uses scan results and the context data captured during scanning
        (hot sectors, limit-up stats, northbound flows) to produce a
        human-readable market narrative.

        Args:
            results: Ranked scan results from self.scan().
            llm_client: Optional pre-configured LLM client. Created on demand.

        Returns:
            Markdown string with sections: 市场情绪, 板块分析, 资金动向, 候选逻辑.
        """
        if not results:
            return ""

        ctx = self._last_scan_context
        prompt = self._build_summary_prompt(results, ctx)

        try:
            if llm_client is None:
                from ..llm.client import create_llm
                llm_client = create_llm()

            response = llm_client.chat(
                [
                    (
                        "system",
                        "你是量化交易系统中的市场分析师。根据扫描数据生成简洁、结构化的市场分析报告。"
                        "只返回纯 Markdown，不要有任何前缀或后缀解释。"
                        "分析要具体，引用实际数据，不要泛泛而谈。"
                        "字数控制在 500 字以内。",
                    ),
                    ("human", prompt),
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            return str(response).strip()
        except Exception as exc:
            logger.warning("LLM market summary failed: %s", exc)
            return self._fallback_summary(results, ctx)

    def _build_summary_prompt(
        self,
        results: list[ScanResult],
        ctx: dict[str, Any],
    ) -> str:
        """Build the LLM prompt from scan results and context."""
        parts: list[str] = []

        # Hot sectors
        hot_sectors = ctx.get("hot_sectors", [])
        if hot_sectors:
            parts.append("## 今日热门板块")
            for s in hot_sectors[:8]:
                name = s.get("sector_name", "")
                pct = s.get("change_pct", 0)
                parts.append(f"- {name}: {pct:+.2f}%")

        # Limit-up summary
        lu = ctx.get("limit_up_summary", {})
        if lu:
            parts.append("")
            parts.append("## 涨停统计")
            parts.append(f"- 首板: {lu.get('first_board', 0)} 只")
            parts.append(f"- 二板: {lu.get('second_board', 0)} 只")
            parts.append(f"- 三板及以上: {lu.get('third_plus', 0)} 只")

        # Northbound top 10
        nb = ctx.get("northbound_top10", [])
        if nb:
            parts.append("")
            parts.append("## 北向资金十大成交")
            for s in nb[:10]:
                name = s.get("name", "")
                code = s.get("code", "")
                net = s.get("net_buy", 0) or 0
                direction = "买入" if net > 0 else "卖出"
                parts.append(f"- {code} {name}: {direction} {abs(net)/1e8:.1f}亿")

        # Top candidates
        parts.append("")
        parts.append("## 扫描候选 (Top 10)")
        for i, r in enumerate(results[:10], 1):
            parts.append(f"{i}. **{r.ticker}** {r.name} | 评分 {r.score:.1f} | {r.reason}")

        data_block = "\n".join(parts)

        return f"""根据以下 A 股市场扫描数据，生成一份简洁的市场分析报告。

{data_block}

请按以下结构输出 Markdown:

### 市场情绪
- 综合涨跌停比、连板高度、北向态度，判断今日市场温度（冰点/偏冷/正常/偏热/过热）
- 一句话总结

### 板块分析
- 今日领涨板块及可能驱动逻辑
- 板块轮动迹象（资金从哪流向哪）
- 是否有板块共振现象

### 资金动向
- 北向资金态度（积极/中性/谨慎）
- 主力资金偏好方向

### 候选逻辑
- 扫描评分体系的信号构成
- Top 候选的共性特征
- 需要关注的风险点

交易日期: {ctx.get('trade_date', 'unknown')}"""

    def _fallback_summary(
        self,
        results: list[ScanResult],
        ctx: dict[str, Any],
    ) -> str:
        """Deterministic fallback when LLM is unavailable."""
        hot_sectors = ctx.get("hot_sectors", [])
        lu = ctx.get("limit_up_summary", {})
        nb = ctx.get("northbound_top10", [])

        lines = [
            "### 市场情绪",
            "",
            "> LLM 不可用，以下为基础数据汇总。",
            "",
        ]

        if hot_sectors:
            lines.append("### 热门板块")
            for s in hot_sectors[:5]:
                lines.append(f"- {s.get('sector_name', '')}: {s.get('change_pct', 0):+.2f}%")

        if lu:
            lines.append("")
            lines.append("### 涨停统计")
            lines.append(f"- 首板 {lu.get('first_board', 0)} / 二板 {lu.get('second_board', 0)} / 三板+ {lu.get('third_plus', 0)}")

        if nb:
            lines.append("")
            lines.append("### 北向资金 Top 5")
            for s in nb[:5]:
                net = s.get("net_buy", 0) or 0
                lines.append(f"- {s.get('name', '')}: 净{'买入' if net > 0 else '卖出'} {abs(net)/1e8:.1f}亿")

        lines.append("")
        lines.append("### Top 候选")
        for i, r in enumerate(results[:10], 1):
            lines.append(f"{i}. {r.ticker} {r.name} ({r.score:.1f}分) - {r.reason}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM review for scan results
    # ------------------------------------------------------------------

    def review_with_llm(
        self,
        results: list[ScanResult],
        llm_client: Any | None = None,
    ) -> str:
        """LLM review of scan results: risk check + quality assessment.

        Flags delisted stocks, sector concentration, signal quality.

        Returns a Markdown paragraph. Falls back to empty string.
        """
        if not results:
            return ""

        prompt = self._build_review_prompt(results)
        try:
            if llm_client is None:
                from ..llm.client import create_llm

                llm_client = create_llm()

            response = llm_client.chat(
                [
                    (
                        "system",
                        "你是量化交易系统里的扫描结果审计员。检查候选列表中的异常："
                        "退市股（名称含'退'）、信号质量、板块集中度。"
                        "只返回纯文本，300字以内，不加前缀。"
                        "如果没有问题可以返回空字符串。",
                    ),
                    ("human", prompt),
                ],
                temperature=0.1,
                max_tokens=800,
            )
            return str(response).strip()
        except Exception as exc:
            logger.debug("LLM review failed: %s", exc)
            return ""

    def _build_review_prompt(self, results: list[ScanResult]) -> str:
        """Build the LLM review prompt from scan results."""
        parts = ["以下为市场扫描候选列表，请检查风险和异常：", ""]
        parts.append(f"## Top 候选 ({len(results)} 只)")
        for i, r in enumerate(results[:15], 1):
            parts.append(f"{i}. {r.ticker} {r.name} 评分={r.score} 来源={r.source} 板块={r.sector} {r.reason}")

        parts.append("")
        parts.append("## 检查清单")
        parts.append("1. 是否有名称含'退'的股票（退市不可交易）？")
        parts.append("2. Top 5 是否集中在单一板块（集中度风险）？")
        parts.append("3. 主要信号来源是啥（板块/涨停/龙虎榜/北向）？信号是否扎实？")
        parts.append("4. 有无明显异常（退市、停牌、成交量极低）？")
        parts.append("5. 综合评估：这批候选质量如何？")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Data collection (combined scan fetch + clean)
    # ------------------------------------------------------------------

    def collect_shared_data(
        self,
        trade_date: str,
        route_trace: list[dict[str, Any]] | None = None,
    ) -> ScanSharedRaw:
        """Collect market-wide data shared across all tickers.

        Returns a dict with keys matching the shared raw ScanDataPackage fields:
        market, sector_context, risk.
        """
        trace: list[dict[str, Any]] = []
        if route_trace is not None:
            trace = route_trace

        market = self._safe_fetch("get_daily", trace, code="000001.SH",
                                  start_date=trade_date, end_date=trade_date)
        sector_context = self._safe_fetch("get_sector", trace, top_n=20, trade_date=trade_date)
        limit_up_summary = self._safe_fetch("get_limit_up_tiers", trace, trade_date=trade_date)
        dragon_tiger = self._safe_fetch("get_dragon_tiger", trace, trade_date=trade_date)
        market_breadth = self._safe_fetch("get_market_breadth", trace, trade_date=trade_date)
        st_status = self._safe_fetch("get_st_status", trace, trade_date=trade_date)
        suspended = self._safe_fetch("get_suspended", trace, trade_date=trade_date)
        delisting = self._safe_fetch("get_delisting", trace, trade_date=trade_date)

        return {
            "market": market if isinstance(market, list) else [],
            "sector_context": sector_context if isinstance(sector_context, list) else [],
            "limit_up_summary": limit_up_summary if isinstance(limit_up_summary, dict) else {},
            "dragon_tiger": dragon_tiger if isinstance(dragon_tiger, list) else [],
            "market_breadth": market_breadth if isinstance(market_breadth, dict) else {},
            "risk": {
                "st_status": st_status if isinstance(st_status, list) else [],
                "suspended": suspended if isinstance(suspended, list) else [],
                "delisting": delisting if isinstance(delisting, list) else [],
            },
        }

    def collect_ticker_data(
        self,
        ticker: str,
        trade_date: str,
        news_keyword: str | None = None,
        route_trace: list[dict[str, Any]] | None = None,
        fetch_news_full_text: bool = True,
        *,
        start_date: str | None = None,
        sector_keyword: str | None = None,
        include_capital_flow: bool = True,
        include_news: bool = True,
        prefer_cached_news: bool = True,
        allow_live_news: bool | None = None,
    ) -> ScanTickerRaw:
        """Collect per-ticker raw data: daily, capital_flow, enriched news.

        Returns the per-ticker raw portion of ScanDataPackage:
        daily, capital_flow, news.
        """
        trace: list[dict[str, Any]] = []
        if route_trace is not None:
            trace = route_trace

        daily = self._safe_fetch("get_daily", trace, code=ticker,
                                 start_date=start_date, end_date=trade_date)
        capital_flow = []
        if include_capital_flow:
            capital_flow = self._safe_fetch("get_capital_flow", trace, code=ticker,
                                            start_date=start_date, end_date=trade_date)
        from .local_cache import get_cached_news

        news = []
        if include_news:
            if prefer_cached_news:
                news = get_cached_news(ticker, trade_date=trade_date)
            if not news and (self._live_news_fallback if allow_live_news is None else allow_live_news):
                from .vendor_router import route_to_vendor
                news_route_fn = route_to_vendor if self._route_fn is route_to_local_cache_only else self._route_fn
                try:
                    news, _elapsed = timed_vendor_call(
                        "get_news",
                        code=ticker,
                        sector=sector_keyword,
                        keyword=news_keyword,
                        trade_date=trade_date,
                        route_fn=news_route_fn,
                        route_trace=trace,
                    )
                except Exception:
                    news = []
                if isinstance(news, dict):
                    news = []
            if not news and not prefer_cached_news:
                news = get_cached_news(ticker, trade_date=trade_date)
            if news_keyword and not news:
                logger.debug("No news for %s/%s during scan collection", ticker, trade_date)
        if fetch_news_full_text and isinstance(news, list) and news:
            news = enrich_news_full_text(news, source="scanner")

        return {
            "daily": daily if isinstance(daily, list) else [],
            "capital_flow": capital_flow if isinstance(capital_flow, list) else [],
            "news": news if isinstance(news, list) else [],
        }

    def collect_data_agent_raw(
        self,
        request: Any,
        route_trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Collect all raw inputs needed by DataAgent, owned by scan layer."""
        trace: list[dict[str, Any]] = []
        if route_trace is not None:
            trace = route_trace

        trade_date = request.normalized_trade_date()
        self._ensure_cache_ready(trade_date)
        end_date = request.normalized_end_date() or trade_date

        shared: ScanSharedRaw = {}
        if request.include_market:
            market = self._safe_fetch(
                "get_daily",
                trace,
                code="000001.SH",
                start_date=request.start_date or end_date,
                end_date=end_date,
            )
            shared["market"] = market if isinstance(market, list) else []
            limit_up_summary = self._safe_fetch("get_limit_up_tiers", trace, trade_date=trade_date)
            dragon_tiger = self._safe_fetch("get_dragon_tiger", trace, trade_date=trade_date)
            market_breadth = self._safe_fetch("get_market_breadth", trace, trade_date=trade_date)
            shared["limit_up_summary"] = limit_up_summary if isinstance(limit_up_summary, dict) else {}
            shared["dragon_tiger"] = dragon_tiger if isinstance(dragon_tiger, list) else []
            shared["market_breadth"] = market_breadth if isinstance(market_breadth, dict) else {}
        else:
            shared["market"] = []
            shared["limit_up_summary"] = {}
            shared["dragon_tiger"] = []
            shared["market_breadth"] = {}

        if request.include_sector_context:
            sector_context = self._safe_fetch(
                "get_sector",
                trace,
                top_n=request.sector_top_n,
                trade_date=trade_date,
            )
            shared["sector_context"] = sector_context if isinstance(sector_context, list) else []
        else:
            shared["sector_context"] = []

        if request.include_risk:
            st_status = self._safe_fetch("get_st_status", trace, trade_date=trade_date)
            suspended = self._safe_fetch("get_suspended", trace, trade_date=trade_date)
            delisting = self._safe_fetch("get_delisting", trace, trade_date=trade_date)
            shared["risk"] = {
                "st_status": self._risk_value("get_st_status", st_status, trace),
                "suspended": self._risk_value("get_suspended", suspended, trace),
                "delisting": self._risk_value("get_delisting", delisting, trace),
            }
        else:
            shared["risk"] = {"st_status": [], "suspended": [], "delisting": []}

        ticker_raw = self.collect_ticker_data(
            request.ticker,
            end_date,
            request.news_keyword,
            trace,
            request.fetch_news_full_text,
            start_date=request.start_date,
            sector_keyword=request.sector_keyword,
            include_capital_flow=request.include_capital_flow,
            include_news=request.include_news,
            prefer_cached_news=False,
            allow_live_news=True,
        )

        return {
            **shared,
            **ticker_raw,
            "route_trace": trace,
        }

    def collect_data_agent_package(
        self,
        request: Any,
        route_trace: list[dict[str, Any]] | None = None,
    ) -> ScanDataPackage:
        """Collect and clean the complete DataAgent input package.

        This is the canonical scan -> DataAgent seam:
        - scan fetches vendor/cache/news inputs
        - scan normalizes and cleans them
        - DataAgent only processes the cleaned structure into factors/events/payloads
        """
        trace: list[dict[str, Any]] = []
        if route_trace is not None:
            trace = route_trace
        raw_payload = self.collect_data_agent_raw(request, trace)
        return ScanDataPackage(
            raw_payload=raw_payload,
            cleaned_payload=self.clean_data_agent_raw(raw_payload),
            route_trace=trace,
        )

    @staticmethod
    def clean_data_agent_raw(raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Clean raw data for DataAgent; scan owns this stage."""
        return clean_raw_payload(raw_payload, now_fn=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def _risk_value(method: str, value: Any, route_trace: list[dict[str, Any]]) -> list[Any] | dict[str, Any]:
        if isinstance(value, list):
            for attempt in reversed(route_trace):
                if attempt.get("method") == method and attempt.get("status") == "error":
                    return {"error": attempt.get("error", "unknown"), "method": method}
            return value
        return value if isinstance(value, dict) else []

    def scan_and_collect(
        self,
        trade_date: str | None = None,
        top_n: int | None = None,
        news_keyword: str | None = None,
        fetch_news_full_text: bool = False,
        live_news: bool = True,
    ) -> ScanBundle:
        """Scan for hot stocks and collect raw data for top candidates in one pass.

        Shared data (market index, sector context, risk lists) is fetched once.
        Per-ticker data (daily OHLCV, capital flow, news) is fetched only for
        the top-N ranked candidates.

        Returns a ScanBundle ready to feed into DataAgent.run_with_raw().
        """
        td = resolve_market_trade_date(trade_date)
        self._ensure_cache_ready(td)
        limit = top_n or self.top_n
        route_trace: list[dict[str, Any]] = []

        # Phase 1: scan (lightweight discovery)
        results = self.scan(td)

        if not results:
            return ScanBundle(trade_date=td, results=[], route_trace=route_trace)

        # Phase 2: collect shared data once
        shared = self.collect_shared_data(td, route_trace)

        # Phase 3: collect per-ticker data for top candidates
        ticker_data: dict[str, ScanTickerRaw] = {}
        for r in results[:limit]:
            logger.info("Collecting data for %s %s", r.ticker, r.name)
            ticker_data[r.ticker] = self.collect_ticker_data(
                r.ticker, td, news_keyword, route_trace, fetch_news_full_text,
                allow_live_news=live_news,
            )

        return ScanBundle(
            trade_date=td,
            results=results,
            shared_raw=shared,
            ticker_data=ticker_data,
            route_trace=route_trace,
        )

    def _ensure_cache_ready(self, trade_date: str) -> None:
        if not self._cache_only or not self._auto_refresh_cache:
            return
        try:
            ensure_scan_cache(trade_date, compute_signals=True)
        except Exception as exc:
            logger.warning("Startup cache refresh failed for %s; scan will use existing local cache only: %s", trade_date, exc)

    def _safe_fetch(
        self,
        method: str,
        route_trace: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Call route_to_vendor with timing and error tracking."""
        try:
            result, elapsed_ms = timed_vendor_call(
                method,
                route_trace=route_trace,
                route_fn=self._route_fn,
                **kwargs,
            )
            record_count = len(result) if isinstance(result, list) else None
            logger.debug("Fetched %s in %.0fms (%d records)", method, elapsed_ms, record_count or 0)
            return result
        except Exception as exc:
            logger.warning("Fetch %s failed: %s", method, exc)
            if route_trace is not None:
                route_trace.append({
                    "method": method,
                    "status": "error",
                    "error": str(exc),
                })
            return []
