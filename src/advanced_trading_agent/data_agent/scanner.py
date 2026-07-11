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
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .vendor_router import get_vendor_chain, route_to_vendor

logger = logging.getLogger(__name__)


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
    shared_raw: dict[str, Any] = field(default_factory=dict)
    ticker_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    route_trace: list[dict[str, Any]] = field(default_factory=list)


class MarketScanner:
    """Discover trending stocks through multi-channel scanning.

    Channels (in scoring order):
    1. Hot sector constituents (cross-sector = highest confidence)
    2. Limit-up pool details (momentum signal)
    3. Northbound top holdings (smart money signal)
    4. Dragon-tiger list (institutional signal)
    """

    def __init__(self, top_sectors: int = 5, top_n: int = 20):
        self.top_sectors = top_sectors
        self.top_n = top_n
        self._last_scan_context: dict[str, Any] = {}

    def scan(self, trade_date: str | None = None) -> list[ScanResult]:
        """Run full market scan and return ranked results."""
        td = trade_date or date.today().isoformat()
        scorer: dict[str, dict[str, Any]] = {}  # ticker -> accumulated evidence
        ctx: dict[str, Any] = {"trade_date": td}

        # Channel 1: Hot sectors → constituents (primary)
        self._scan_hot_sectors(td, scorer, ctx)

        # Channel 2: Limit-up pool
        self._scan_limit_up(td, scorer, ctx)

        # Channel 3: Northbound top holdings
        self._scan_northbound(td, scorer, ctx)

        # Channel 4: Dragon-tiger
        self._scan_dragon_tiger(td, scorer)

        self._last_scan_context = ctx

        # Build ranked results
        results = self._rank(scorer)
        return results[:self.top_n]

    # ------------------------------------------------------------------
    # Channel scanners
    # ------------------------------------------------------------------

    def _scan_hot_sectors(self, td: str, scorer: dict[str, dict[str, Any]], ctx: dict[str, Any]) -> None:
        """Find top sectors and their constituent stocks."""
        try:
            sectors = route_to_vendor("get_sector", top_n=self.top_sectors * 2)
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
                constituents = route_to_vendor("get_sector_constituents", sector_name=sector_name)
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

    def _scan_limit_up(self, td: str, scorer: dict[str, dict[str, Any]], ctx: dict[str, Any]) -> None:
        """Score stocks in the limit-up pool."""
        try:
            data = route_to_vendor("get_limit_up_tiers", trade_date=td)
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
            board = int(stock.get("board_count", 1))
            score = min(board * 1.5, 5.0)

            if ticker not in scorer:
                scorer[ticker] = {
                    "name": str(stock.get("name", "")),
                    "score": 0.0,
                    "sectors": [],
                    "sources": [],
                    "board_count": 0,
                    "reasons": [],
                }
            scorer[ticker]["score"] += score
            scorer[ticker]["board_count"] = max(scorer[ticker]["board_count"], board)
            if "limit_up" not in scorer[ticker]["sources"]:
                scorer[ticker]["sources"].append("limit_up")
            if board >= 2:
                scorer[ticker]["reasons"].append(f"{board}连板涨停")

    def _scan_northbound(self, td: str, scorer: dict[str, dict[str, Any]], ctx: dict[str, Any]) -> None:
        """Score stocks with northbound net buying."""
        try:
            top10 = route_to_vendor("get_northbound_top10", trade_date=td)
        except Exception as exc:
            logger.warning("Northbound scan failed: %s", exc)
            return

        if not isinstance(top10, list):
            return

        ctx["northbound_top10"] = top10

        for stock in top10:
            code = str(stock.get("code", ""))
            if not code or not code[0].isdigit():
                continue
            ticker = self._normalize_ticker(code)
            net_buy = float(stock.get("net_buy", 0) or 0)
            score = 2.0 if net_buy > 0 else 1.0

            if ticker not in scorer:
                scorer[ticker] = {
                    "name": str(stock.get("name", "")),
                    "score": 0.0,
                    "sectors": [],
                    "sources": [],
                    "board_count": 0,
                    "reasons": [],
                }
            scorer[ticker]["score"] += score
            if "northbound" not in scorer[ticker]["sources"]:
                scorer[ticker]["sources"].append("northbound")
            if net_buy > 0:
                scorer[ticker]["reasons"].append(f"北向净买入 {net_buy/1e8:.1f}亿")

    def _scan_dragon_tiger(self, td: str, scorer: dict[str, dict[str, Any]]) -> None:
        """Score stocks appearing on dragon-tiger list."""
        try:
            dt_list = route_to_vendor("get_dragon_tiger", trade_date=td)
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
            score = 1.0  # supplementary signal only

            if ticker not in scorer:
                scorer[ticker] = {
                    "name": str(stock.get("名称", stock.get("name", ""))),
                    "score": 0.0,
                    "sectors": [],
                    "sources": [],
                    "board_count": 0,
                    "reasons": [],
                }
            scorer[ticker]["score"] += score
            if "dragon_tiger" not in scorer[ticker]["sources"]:
                scorer[ticker]["sources"].append("dragon_tiger")

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(self, scorer: dict[str, dict[str, Any]]) -> list[ScanResult]:
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
        if code.startswith(("8", "4")):
            return f"{code}.BJ"
        return code

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
    # Data collection (combined scan + fetch)
    # ------------------------------------------------------------------

    def collect_shared_data(
        self,
        trade_date: str,
        route_trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Collect market-wide data shared across all tickers.

        Returns a dict with keys matching DataAgent._collect_raw shared fields:
        market, sector_context, risk.
        """
        trace: list[dict[str, Any]] = []
        if route_trace is not None:
            trace = route_trace

        market = self._safe_fetch("get_daily", trace, code="000001.SH",
                                  start_date=trade_date, end_date=trade_date)
        sector_context = self._safe_fetch("get_sector", trace, top_n=20)
        st_status = self._safe_fetch("get_st_status", trace)
        suspended = self._safe_fetch("get_suspended", trace, trade_date=trade_date)
        delisting = self._safe_fetch("get_delisting", trace)

        return {
            "market": market if isinstance(market, list) else [],
            "sector_context": sector_context if isinstance(sector_context, list) else [],
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
    ) -> dict[str, Any]:
        """Collect per-ticker raw data: daily, capital_flow, news.

        Returns a dict with keys matching the per-ticker portion of
        DataAgent._collect_raw: daily, capital_flow, news.
        """
        trace: list[dict[str, Any]] = []
        if route_trace is not None:
            trace = route_trace

        daily = self._safe_fetch("get_daily", trace, code=ticker,
                                 start_date=None, end_date=trade_date)
        capital_flow = self._safe_fetch("get_capital_flow", trace, code=ticker,
                                        start_date=None, end_date=trade_date)
        news = self._safe_fetch("get_news", trace, code=ticker, keyword=news_keyword)

        return {
            "daily": daily if isinstance(daily, list) else [],
            "capital_flow": capital_flow if isinstance(capital_flow, list) else [],
            "news": news if isinstance(news, list) else [],
        }

    def scan_and_collect(
        self,
        trade_date: str | None = None,
        top_n: int | None = None,
        news_keyword: str | None = None,
    ) -> ScanBundle:
        """Scan for hot stocks and collect raw data for top candidates in one pass.

        Shared data (market index, sector context, risk lists) is fetched once.
        Per-ticker data (daily OHLCV, capital flow, news) is fetched only for
        the top-N ranked candidates.

        Returns a ScanBundle ready to feed into DataAgent.run_with_raw().
        """
        td = trade_date or date.today().isoformat()
        limit = top_n or self.top_n
        route_trace: list[dict[str, Any]] = []

        # Phase 1: scan (lightweight discovery)
        results = self.scan(td)

        if not results:
            return ScanBundle(trade_date=td, results=[], route_trace=route_trace)

        # Phase 2: collect shared data once
        shared = self.collect_shared_data(td, route_trace)

        # Phase 3: collect per-ticker data for top candidates
        ticker_data: dict[str, dict[str, Any]] = {}
        for r in results[:limit]:
            logger.info("Collecting data for %s %s", r.ticker, r.name)
            ticker_data[r.ticker] = self.collect_ticker_data(
                r.ticker, td, news_keyword, route_trace,
            )

        return ScanBundle(
            trade_date=td,
            results=results,
            shared_raw=shared,
            ticker_data=ticker_data,
            route_trace=route_trace,
        )

    @staticmethod
    def _safe_fetch(
        method: str,
        route_trace: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Call route_to_vendor with timing and error tracking."""
        vendor_chain = get_vendor_chain(method)
        try:
            start = time.perf_counter()
            result = route_to_vendor(method, _route_trace=route_trace, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            record_count = len(result) if isinstance(result, list) else None
            logger.debug("Fetched %s in %.0fms (%d records)", method, elapsed_ms, record_count or 0)
            return result
        except Exception as exc:
            logger.warning("Fetch %s failed: %s", method, exc)
            route_trace.append({
                "method": method,
                "vendor": vendor_chain[0] if vendor_chain else "unknown",
                "status": "error",
                "error": str(exc),
            })
            return []
