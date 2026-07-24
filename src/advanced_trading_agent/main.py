"""
多智能体量化交易分析系统 — 主入口

命令行用法:
  python -m advanced_trading_agent.main --ticker 000001.SZ --date 2026-07-10
  python -m advanced_trading_agent.main --batch stocks.txt

借鉴 TradingAgents' main.py + cli/main.py 的 CLI 模式
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import config
from .core.atomic_write import atomic_write_json, atomic_write_text
from .data_agent.trading_calendar import resolve_market_trade_date
from .strategy_rules import load_strategy_proposals, review_strategy_proposal

if TYPE_CHECKING:
    from .data_agent.scanner import ScanBundle

# 日志: 控制台 + 文件 (results_dir/runtime.log)
_log_dir = Path(config.get("results_dir"))
_log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_dir / "runtime.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def analyze_single(ticker: str, trade_date: str | None = None,
                   debug: bool = False, skip_backtest: bool = False) -> str:
    """分析单个标的

    Args:
        ticker: 股票代码 (如 "000001.SZ")
        trade_date: 交易日, 默认今天
        debug: 是否打印调试信息
        skip_backtest: 是否跳过回测审查

    Returns:
        Markdown 格式的分析报告
    """
    from .graph.workflow import TradingSystem

    trade_date = resolve_market_trade_date(trade_date)
    logger.info("Analyzing %s on %s...", ticker, trade_date)

    system = TradingSystem(debug=debug)
    final_state, report = system.analyze(
        ticker,
        trade_date,
        skip_backtest=skip_backtest,
    )

    return report


def analyze_batch(tickers_file: str, debug: bool = False,
                  max_workers: int = 4, skip_backtest: bool = False) -> list[str]:
    """批量分析 (并发执行)

    Args:
        tickers_file: 每行一个股票代码的文件
        debug: 是否打印调试信息
        max_workers: 最大并发数 (默认 4)
        skip_backtest: 是否跳过回测审查
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open(tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    results: list[tuple[int, str]] = []
    logger.info("Batch analyzing %d tickers (max_workers=%d)...", len(tickers), max_workers)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_map = {
            executor.submit(
                analyze_single,
                ticker,
                debug=debug,
                skip_backtest=skip_backtest,
            ): (idx, ticker)
            for idx, ticker in enumerate(tickers)
        }
        for future in as_completed(future_map):
            idx, ticker = future_map[future]
            try:
                report = future.result()
                results.append((idx, report))
                logger.info("=== Done %s ===", ticker)
            except Exception as e:
                logger.error("Failed %s: %s", ticker, e)
                results.append((idx, f"# {ticker}\n\n**失败**: {e}"))

    # 按原始 ticker 顺序排序
    reports = [report for _, report in sorted(results, key=lambda item: item[0])]

    # 保存汇总
    summary_path = Path(config.get("results_dir")) / "batch_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(summary_path, "\n\n---\n\n".join(reports))
    logger.info("Batch summary saved to %s", summary_path)

    return reports


def review_memory(price_file: str | None = None, as_of: str | None = None) -> str:
    """Resolve pending decisions and print alpha-source performance advice.

    price_file 可选，格式为 CSV，至少包含 code/trade_date/open/close/volume/amount。
    若不传 price_file，则只汇总已有 resolved 复盘记录。
    """
    import pandas as pd

    from .agents.memory_agent import MemoryStore
    from .backtest.review import ReviewEngine

    store = MemoryStore()
    reviewer = ReviewEngine()

    if price_file:
        price_df = pd.read_csv(price_file)
        if "trade_date" in price_df.columns:
            price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])

        def price_loader(ticker: str, signal_date: str):
            if "code" not in price_df.columns:
                return price_df
            return price_df[price_df["code"] == ticker].copy()

        reviewer.resolve_due(store, price_loader=price_loader, as_of=as_of)

    summary = reviewer.summarize_entries(store.load_entries())
    report = reviewer.format_summary(summary)
    review_path = Path(config.get("results_dir")) / "review_summary.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(review_path, report)
    return report


def backtest_portfolio(signals_file: str, price_file: str) -> str:
    """Run observation-pool portfolio backtest from signal and price CSV files."""
    import pandas as pd

    from .backtest.portfolio import ObservationPortfolioBacktester

    signals = pd.read_csv(signals_file)
    prices = pd.read_csv(price_file)
    result = ObservationPortfolioBacktester().run(signals, prices)
    lines = [
        "# 观察池组合回测",
        "",
        f"- 总收益: {result.summary.get('total_return', 0):+.2%}",
        f"- 最大回撤: {result.summary.get('max_drawdown', 0):.2%}",
        f"- 交易数: {result.summary.get('trade_count', 0)}",
        f"- 胜率: {result.summary.get('win_rate', 0):.1%}",
        f"- 单笔平均收益: {result.summary.get('avg_trade_return', 0):+.2%}",
    ]
    report = "\n".join(lines)
    results_dir = Path(config.get("results_dir"))
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(results_dir / "portfolio_backtest_summary.md", report)
    result.nav.to_csv(results_dir / "portfolio_nav.csv", index=False)
    result.trades.to_csv(results_dir / "portfolio_trades.csv", index=False)
    return report


def run_standalone_data_agent(
    ticker: str,
    *,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str | None = None,
    use_react_planner: bool = False,
    news_keyword: str | None = None,
    use_llm_news_filter: bool = True,
    use_llm_data_review: bool = False,
    fetch_news_full_text: bool = True,
) -> str:
    """Run data collection, cleaning, analysis, and layered persistence only."""
    from .data_agent.data_agent import DataAgent, DataAgentRequest

    result = DataAgent(results_dir=output_dir).run(
        DataAgentRequest(
            ticker=ticker,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            use_react_planner=use_react_planner,
            news_keyword=news_keyword,
            use_llm_news_filter=use_llm_news_filter,
            use_llm_data_review=use_llm_data_review,
            fetch_news_full_text=fetch_news_full_text,
        )
    )
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)



def run_full_data_analysis(
    ticker: str,
    *,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str | None = None,
    use_react_planner: bool = True,
    news_keyword: str | None = None,
    sector_keyword: str | None = None,
    use_llm_news_filter: bool = True,
    use_llm_data_review: bool = False,
    fetch_news_full_text: bool = True,
    skip_backtest: bool = False,
    lookback_days: int = 90,
    store_memory: bool = False,
    compact: bool = True,
) -> str:
    """采集数据并运行完整 LangGraph 分析工作流。"""
    from .pipeline import run_full_analysis
    from .data_agent.cli import summarize_run

    result = run_full_analysis(
        ticker=ticker,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        use_react_planner=use_react_planner,
        news_keyword=news_keyword,
        sector_keyword=sector_keyword,
        use_llm_news_filter=use_llm_news_filter,
        use_llm_data_review=use_llm_data_review,
        fetch_news_full_text=fetch_news_full_text,
        skip_backtest=skip_backtest,
        lookback_days=lookback_days,
    )
    payload = result.to_dict()
    output = summarize_run(payload) if compact else payload
    return json.dumps(output, ensure_ascii=False, indent=2, default=str)

def list_strategy_audit_queue() -> str:
    """Return pending and reviewed strategy change proposals as JSON."""
    return json.dumps(load_strategy_proposals(), ensure_ascii=False, indent=2)


def refresh_local_cache(
    trade_date: str | None = None,
    *,
    output_dir: str | None = None,
    cache_days: int = 60,
    force_news: bool = False,
) -> str:
    """Refresh the daily cache without running the full agent workflow."""
    from .data_agent.build_cache import build as build_local_cache

    requested_date = trade_date or str(date.today())
    effective_trade_date = resolve_market_trade_date(trade_date)
    cache_dir = build_local_cache(
        trade_date=effective_trade_date,
        output_dir=output_dir,
        days_back=cache_days,
        compute_signals=True,
        refresh_news=True,
        force_news=force_news,
    )
    payload = {
        "requested_date": requested_date,
        "effective_trade_date": effective_trade_date,
        "cache_dir": str(cache_dir),
        "cache_days": cache_days,
        "force_news": force_news,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def refresh_etf_data_cache(
    trade_date: str | None = None,
    *,
    etf_codes: list[str] | None = None,
    daily_limit: int = 20,
) -> str:
    """Refresh ETF spot/daily caches without running any agent workflow."""
    from .data_agent.data_health import refresh_etf_cache

    payload = refresh_etf_cache(
        trade_date=trade_date,
        etf_codes=etf_codes,
        daily_limit=daily_limit,
    )
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def data_source_health_report(trade_date: str | None = None) -> str:
    """Probe required data sources and return an auditable JSON health report."""
    from .data_agent.data_health import run_data_source_health

    payload = run_data_source_health(trade_date=trade_date)
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def scan_sector_etfs(
    trade_date: str | None = None,
    *,
    sector: str | None = None,
    top_n: int = 8,
    force: bool = False,
    refresh_cache: bool = False,
) -> str:
    """Scan sectors first and return JSON-first ETF watchlist candidates."""
    from .data_agent.etf_watchlist import build_watchlist_report, render_watchlist_markdown
    from .data_agent.sector_etf import SectorETFSelector

    trade_date = resolve_market_trade_date(trade_date)
    results_dir = Path(config.get("results_dir"))
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = sector.replace("/", "_") if sector else "all"
    report_path = results_dir / f"sector_etf_report_{trade_date}_{suffix}.md"
    json_path = results_dir / f"sector_etf_report_{trade_date}_{suffix}.json"

    if report_path.exists() and json_path.exists() and not force:
        return report_path.read_text(encoding="utf-8")

    selector = SectorETFSelector(top_sectors=top_n, auto_refresh_cache=refresh_cache)
    selection = selector.select_with_exclusions(
        trade_date,
        sector_query=sector,
        max_roundtable_sectors=top_n,
    )
    report = build_watchlist_report(
        trade_date=trade_date,
        candidates=selection.watchlist_payloads(),
        excluded=selection.excluded,
    )
    markdown = render_watchlist_markdown(report)
    atomic_write_text(report_path, markdown)
    atomic_write_json(json_path, report.model_dump(mode="json"))
    return markdown


def analyze_sector_etf(
    sector: str,
    trade_date: str | None = None,
    *,
    question: str | None = None,
    store_memory: bool = True,
    refresh_cache: bool = False,
) -> str:
    """Run the full LangGraph sector ETF pipeline."""
    from .graph.sector_etf_workflow import SectorETFTradingSystem

    trade_date = resolve_market_trade_date(trade_date)
    _state, report = SectorETFTradingSystem(refresh_cache=refresh_cache).analyze(
        sector,
        question=question,
        trade_date=trade_date,
        store_memory=store_memory,
    )
    return report


def analyze_sector_etf_watchlist(
    trade_date: str | None = None,
    *,
    max_roundtable_sectors: int = 5,
    store_memory: bool = True,
    force: bool = False,
    json_output: bool = False,
    refresh_cache: bool = False,
) -> str:
    """Run the batch sector ETF observation-pool workflow with AutoGen roundtable."""
    from .data_agent.etf_watchlist import build_watchlist_report, render_watchlist_markdown
    from .data_agent.sector_etf import SectorETFSelector
    from .roundtable.etf_watchlist_autogen import ETFWatchlistAutoGenRoundtable

    trade_date = resolve_market_trade_date(trade_date)
    results_dir = Path(config.get("results_dir"))
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"sector_etf_watchlist_{trade_date}.md"
    json_path = results_dir / f"sector_etf_watchlist_{trade_date}.json"

    if report_path.exists() and json_path.exists() and not force:
        return json.dumps(json.loads(json_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2) if json_output else report_path.read_text(encoding="utf-8")

    started = time.perf_counter()
    selector = SectorETFSelector(
        top_sectors=max_roundtable_sectors,
        auto_refresh_cache=refresh_cache,
    )
    selection = selector.select_with_exclusions(
        trade_date,
        max_roundtable_sectors=max_roundtable_sectors,
    )
    select_seconds = time.perf_counter() - started

    roundtable_started = time.perf_counter()
    try:
        autogen_result = ETFWatchlistAutoGenRoundtable().run(
            trade_date=trade_date,
            candidates=selection.watchlist_payloads(),
            max_final_decisions=3,
        )
        roundtable_summary = autogen_result.to_summary_dict()
    except Exception as exc:
        logger.warning("AutoGen roundtable failed, falling back: %s", _compact_exception(exc))
        roundtable_summary = {
            "provider": "deterministic_fallback",
            "mode": "no_roundtable",
            "autogen_requested": True,
            "fallback_reason": _compact_exception(exc),
            "note": "AutoGen roundtable failed; sector selector scoring used as fallback.",
        }

    report = build_watchlist_report(
        trade_date=trade_date,
        candidates=selection.watchlist_payloads(),
        excluded=selection.excluded,
        roundtable_summary=roundtable_summary,
    )
    roundtable_seconds = time.perf_counter() - roundtable_started
    payload = report.model_dump(mode="json")
    payload["roundtable_summary"]["timings"] = {
        "select_and_process_seconds": round(select_seconds, 3),
        "autogen_roundtable_seconds": round(roundtable_seconds, 3),
        "total_pre_render_seconds": round(time.perf_counter() - started, 3),
    }
    markdown = render_watchlist_markdown(report)

    if store_memory:
        from .agents.conversation_memory import ConversationEntry, ConversationMemoryStore

        summary = ", ".join(
            f"{d.sector}:{d.status}:{d.primary_etf.code}"
            for d in report.decisions[:8]
        )
        ConversationMemoryStore().append(ConversationEntry(
            question="每日板块 ETF 观察池",
            answer=summary,
            trade_date=report.trade_date,
            target_type="sector_etf_watchlist",
            target="batch",
            evidence={
                "run_id": report.run_id,
                "decision_count": len(report.decisions),
                "excluded_count": len(report.excluded_sector_candidates),
            },
        ))

    atomic_write_text(report_path, markdown)
    atomic_write_json(json_path, payload)
    return json.dumps(payload, ensure_ascii=False, indent=2) if json_output else markdown


def _compact_exception(exc: Exception, *, max_length: int = 1000) -> str:
    """Return an audit-friendly exception string without leaking configured secrets."""
    text = f"{type(exc).__name__}: {exc}"
    for key in (
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "KIMI_API_KEY",
        "MOONSHOT_API_KEY",
        "GLM_API_KEY",
        "ZHIPU_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        secret = os.environ.get(key)
        if secret:
            text = text.replace(secret, "***")
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def audit_strategy_proposal(
    proposal_id: str,
    *,
    action: str,
    reviewer: str,
    comment: str = "",
) -> str:
    """Approve/reject a strategy change proposal without mutating live config."""
    record = review_strategy_proposal(
        proposal_id,
        action=action,
        reviewer=reviewer,
        comment=comment,
    )
    return json.dumps(record, ensure_ascii=False, indent=2)


def scan_and_analyze(top_n: int = 10, trade_date: str | None = None,
                    debug: bool = False, skip_backtest: bool = False,
                    force: bool = False, refresh_cache: bool = True) -> str:
    """Scan for hot stocks, collect data during scan, then analyze top candidates.

    Uses the combined scan+collect flow: MarketScanner.scan_and_collect()
    fetches raw data for top candidates in the same pass, eliminating
    redundant vendor calls when DataAgent runs later.

    An LLM-generated market summary is included at the top of the report.
    Results are cached by trade_date — re-running the same date returns
    the cached report immediately unless `force=True`.

    Returns a consolidated Markdown report.
    """
    from .data_agent.scanner import MarketScanner

    trade_date = resolve_market_trade_date(trade_date)
    results_dir = Path(config.get("results_dir"))
    results_dir.mkdir(parents=True, exist_ok=True)
    scan_path = results_dir / f"scan_report_{trade_date}.md"

    # Cache hit: return existing report for the same trade date
    if scan_path.exists() and not force:
        logger.info("Scan report already exists for %s, returning cached. Use --force to re-scan.", trade_date)
        return scan_path.read_text(encoding="utf-8")

    scanner = MarketScanner(top_sectors=5, top_n=top_n, auto_refresh_cache=refresh_cache)
    bundle = scanner.scan_and_collect(trade_date, top_n=top_n)

    if not bundle.results:
        return "# Market Scan\n\nNo hot stocks found."

    # LLM market summary (always on)
    llm_summary = scanner.summarize_with_llm(bundle.results)

    lines = [
        "# 市场扫描与深度分析报告",
        "",
        f"**交易日期**: {trade_date}",
        "",
        "---",
        "",
        "## 市场综述 (AI)",
        "",
        llm_summary,
        "",
        "---",
        "",
        scanner.format_results(bundle.results),
        "",
        "---",
        "",
        f"## 逐标深度分析 (Top {min(len(bundle.results), top_n)})",
        "",
    ]

    for i, r in enumerate(bundle.results[:top_n], 1):
        logger.info("Analyzing %d/%d: %s %s", i, len(bundle.results[:top_n]), r.ticker, r.name)
        lines.append(f"### {i}. {r.ticker} {r.name} (score: {r.score:.1f})")
        lines.append(f"*{r.reason}*")
        lines.append("")
        try:
            report = _analyze_from_bundle(r.ticker, trade_date, bundle, debug=debug, skip_backtest=skip_backtest)
            lines.append(report)
        except Exception as exc:
            logger.error("Analysis failed for %s: %s", r.ticker, exc)
            lines.append(f"**Analysis failed**: {exc}")
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)
    atomic_write_text(scan_path, report)
    logger.info("Scan report saved to %s", scan_path)

    return report


def _analyze_from_bundle(
    ticker: str,
    trade_date: str,
    bundle: "ScanBundle",
    debug: bool = False,
    skip_backtest: bool = False,
    analysis_mode: str = "workflow",
    lookback_days: int = 90,
    store_memory: bool = False,
) -> str:
    """Run DataAgent with pre-collected data, then feed tier1/tier2 to TradingSystem.

    Passes the scan-owned raw+clean package from ScanBundle into DataAgent,
    so DataAgent only performs structured processing.
    """
    from .data_agent.data_agent import DataAgent, DataAgentRequest
    from .graph.workflow import TradingSystem


    da_result = DataAgent().run(
        DataAgentRequest(
            ticker=ticker,
            trade_date=trade_date,
            start_date=trade_date,
            end_date=trade_date,
            include_market=True,
            include_capital_flow=True,
            include_news=True,
            include_factors=True,
            include_risk=True,
            use_react_planner=False,
        ),
        scan_package=bundle.package_for_ticker(ticker),

    )

    payload = da_result.final_data.get("analysis", {}).get("agent_payload", {})
    tier1 = payload.get("tier1_data", {})
    tier2 = payload.get("tier2_data", {})

    system = TradingSystem(debug=debug)
    _, report = system.analyze(
        ticker,
        trade_date,
        tier1_data=tier1,
        tier2_data=tier2,
        skip_backtest=skip_backtest,
    )
    return report


def ask_llm(
    question: str,
    trade_date: str | None = None,
    debug: bool = False,
    *,
    store_memory: bool = True,
) -> str:
    """自然语言问答：根据问题自动分析对应标的或板块并回答。

    1. LLM 解析问题 → 提取 ticker/板块/意图
    2. 扫描 + DataAgent 采集数据
    3. 把 tier1/tier2 + 问题发给 LLM → 生成分析回答
    """
    trade_date = resolve_market_trade_date(trade_date)
    from .agents.conversation_memory import ConversationEntry, ConversationMemoryStore
    from .data_agent.data_agent import DataAgent, DataAgentRequest
    from .data_agent.scanner import MarketScanner
    from .graph.sector_etf_workflow import SectorETFTradingSystem

    # Step 1: 先做确定性意图识别，板块问答不需要先消耗 LLM。
    inferred_sector = _infer_sector_from_question(question)
    if inferred_sector:
        target_type = "sector"
        target = inferred_sector
        llm = None
    else:
        llm = _get_llm()
        parse_prompt = (
            f"你是一个量化交易系统的意图识别器。根据用户的提问，判断目标对象。"
            f"只返回 JSON，格式: {{'type': 'ticker'|'sector'|'market'|'unknown', "
            f"'target': '股票代码或板块名称或null', 'reason': '简短判断理由'}}\n"
            f"如果提到具体公司名（如'茅台'），转换为 ticker（如 '600519.SH'）。"
            f"如果提到板块（如'新能源车'、'白酒'），type 为 sector，target 为板块名。\n"
            f"问题: {question}"
        )
        try:
            parse_result = json.loads(str(llm.chat([("system", "只返回JSON，不要解释"), ("human", parse_prompt)], temperature=0, max_tokens=500)))
            target_type = parse_result.get("type", "unknown")
            target = parse_result.get("target", "")
        except Exception as exc:
            logger.warning("LLM question parsing failed: %s", exc)
            target_type = "unknown"
            target = ""

    if debug:
        print(f"[Ask] Parsed: type={target_type}, target={target}")

    memory_store = ConversationMemoryStore()

    # Step 2: 采集数据
    tier1_data: dict[str, Any] = {}
    tier2_data: dict[str, Any] = {}
    collected_context: str = ""

    if target_type == "ticker" and target:
        # 单个标的分析
        try:
            scanner = MarketScanner(top_sectors=5, top_n=10, auto_refresh_cache=True)
            bundle = scanner.scan_and_collect(trade_date, top_n=10)
            run = DataAgent().run(
                DataAgentRequest(
                    ticker=target,
                    trade_date=trade_date,
                    include_market=True,
                    include_capital_flow=True,
                    include_news=True,
                    include_factors=True,
                    include_risk=True,
                    use_react_planner=False,
                    use_llm_news_filter=True,
                    fetch_news_full_text=False,
                ),
                scan_package=bundle.package_for_ticker(target),

            )
            ap = run.final_data.get("analysis", {}).get("agent_payload", {})
            tier1_data = ap.get("tier1_data", {})
            tier2_data = ap.get("tier2_data", {})
            collected_context = f"标的 {target} 的分析数据"
        except Exception as exc:
            logger.error("Ticker data collection failed for %s: %s", target, exc)
            collected_context = f"数据采集失败: {exc}"

    elif target_type == "sector" and target:
        _state, report = SectorETFTradingSystem(memory_store=memory_store).analyze(
            target,
            question=question,
            trade_date=trade_date,
            store_memory=store_memory,
        )
        return report

    else:
        # 市场全局或未知 → 扫描大盘
        try:
            scanner = MarketScanner(top_sectors=5, top_n=5, auto_refresh_cache=True)
            results = scanner.scan(trade_date)
            collected_context = json.dumps({
                "top_sectors": [(s.get("sector_name"), s.get("change_pct")) for s in scanner._last_scan_context.get("hot_sectors", [])[:8]],
                "limit_up": scanner._last_scan_context.get("limit_up_summary", {}),
                "top_candidates": [(r.ticker, r.name, r.sector, r.score) for r in results[:10]],
            }, ensure_ascii=False)
        except Exception as exc:
            logger.error("Market scan failed: %s", exc)
            collected_context = f"市场扫描失败: {exc}"

    # Step 3: LLM 综合分析
    if llm is None:
        llm = _get_llm()
    answer_prompt = (
        f"你是一个A股量化交易分析师。用户提问: {question}\n\n"
        f"以下是系统采集到的结构化数据，请根据这些数据给出分析回答:\n\n"
        f"{collected_context}\n\n"
    )

    if tier1_data:
        answer_prompt += f"## Tier 1 (市场摘要)\n{json.dumps(tier1_data, ensure_ascii=False, indent=2)}\n\n"
    if tier2_data:
        answer_prompt += "## Tier 2 (详细数据)\n"
        answer_prompt += f"- 日线记录: {len(tier2_data.get('price_data', []))} 条\n"
        answer_prompt += f"- 因子记录: {len(tier2_data.get('factors', []))} 条\n"
        answer_prompt += f"- 事件记录: {len(tier2_data.get('events', []))} 条\n"

    answer_prompt += (
        "\n请从以下角度回答:\n"
        "1. 市场背景（大盘情绪、板块状态）\n"
        "2. 标/板块的量化特征（因子、资金、技术面）\n"
        "3. 风险和注意事项\n"
        "4. 结论和建议\n"
        "用中文回答，500字以内，Markdown格式。"
    )

    try:
        answer = str(llm.chat(
            [("system", "你是一个严谨的A股量化分析师。基于数据回答，不臆测。"), ("human", answer_prompt)],
            temperature=0.3,
            max_tokens=2000,
        ))
    except Exception as exc:
        answer = f"⚠️ 分析生成失败: {exc}"

    if store_memory:
        memory_store.append(ConversationEntry(
            question=question,
            answer=answer,
            trade_date=trade_date,
            target_type=str(target_type),
            target=str(target),
            evidence={"context": collected_context[:4000]},
        ))

    return f"# 智能问答\n\n**问题**: {question}\n\n---\n\n{answer}\n"


def _infer_sector_from_question(question: str) -> str:
    """Best-effort non-LLM sector extraction for CLI robustness."""
    text = question.strip()
    for marker in ("板块", "行业", "赛道"):
        if marker in text:
            before = text.split(marker, 1)[0]
            token = before.split()[-1] if before.split() else before
            token = token.strip("，。！？? 为什么为啥是否看好不好能买吗买不买")
            if token:
                return token.removesuffix(marker)
    return ""


def _get_llm():
    """Get or create LLM client, trying configured provider then deepseek."""
    from .llm.client import create_llm
    try:
        return create_llm()
    except Exception:
        return create_llm(provider="deepseek")


def main():
    parser = argparse.ArgumentParser(description="板块ETF观察池分析系统")
    parser.add_argument("--date", "-d", help="交易日 (默认今天)")
    parser.add_argument("--review", action="store_true", help="运行复盘汇总")
    parser.add_argument("--daily-review", action="store_true", help="运行每日复盘任务")
    parser.add_argument("--refresh-cache", action="store_true", help="刷新当日本地数据缓存")
    parser.add_argument("--refresh-etf-cache", action="store_true", help="刷新 ETF 现货/日线缓存")
    parser.add_argument("--data-health", action="store_true", help="检查行情、板块、涨停、ETF 数据源可用性")
    parser.add_argument("--sector-etf-scan", action="store_true", help="扫描板块并映射相关ETF")
    parser.add_argument("--sector-etf-analyze", action="store_true", help="运行板块ETF 圆桌决策流水线")
    parser.add_argument("--sector", help="指定板块名称，如 半导体、机器人、创新药")
    parser.add_argument("--no-conversation-memory", action="store_true", help="板块ETF分析不写入对话记忆")
    parser.add_argument("--etf-code", action="append", default=[], help="刷新指定 ETF 代码，可重复传入")
    parser.add_argument("--etf-daily-limit", type=int, default=20, help="未指定 ETF 时，按现货成交额刷新前 N 只 ETF 日线")
    parser.add_argument("--force-news", action="store_true", help="刷新缓存时强制重拉当天新闻")
    parser.add_argument("--cache-days", type=int, default=60, help="刷新本地行情缓存的回看天数，默认 60")
    parser.add_argument("--portfolio-backtest", action="store_true", help="运行观察池组合回测")
    parser.add_argument("--full-analysis", action="store_true", help="按指定日期采集数据、处理并自动分析，返回 JSON")
    parser.add_argument("--lookback-days", type=int, default=90, help="full-analysis 未指定 start-date 时的日线回看天数")
    parser.add_argument("--store-memory", action="store_true", help="full-analysis rules 模式写入交易 MemoryStore")
    parser.add_argument("--react-planner", action="store_true", help="DataAgent 运行前启用 ReAct Planner")
    parser.add_argument("--strategy-audit", action="store_true", help="查看策略规则变更审计队列")
    parser.add_argument("--audit-proposal-id", help="要审批的策略变更 proposal_id")
    parser.add_argument("--audit-action", choices=["approve", "reject"], help="策略变更审批动作")
    parser.add_argument("--reviewer", default="human_required", help="审批人")
    parser.add_argument("--audit-comment", default="", help="审批备注")
    parser.add_argument("--price-file", help="复盘用行情 CSV")
    parser.add_argument("--signals-file", help="观察池信号 CSV")
    parser.add_argument("--start-date", help="DataAgent 起始日期, 如 20260101")
    parser.add_argument("--end-date", help="DataAgent 结束日期, 如 20260710")
    parser.add_argument("--output-dir", help="DataAgent 输出目录")
    parser.add_argument("--news-keyword", help="DataAgent 新闻关键词过滤")
    parser.add_argument("--no-llm-news-filter", action="store_true", help="DataAgent 新闻筛选不调用 LLM")
    parser.add_argument("--llm-data-review", action="store_true", help="DataAgent 使用 LLM 做数据质量复核")
    parser.add_argument("--no-news-full-text", action="store_true", help="DataAgent 不抓取新闻正文")
    parser.add_argument("--skip-backtest", action="store_true", help="跳过回测")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--scan-top-n", type=int, default=10, help="扫描 Top N 板块")
    parser.add_argument("--no-refresh-cache", action="store_true", help="不刷新扫描缓存")
    parser.add_argument("--refresh-scan-cache", action="store_true", help="扫描前刷新行情缓存")
    parser.add_argument("--force", "-f", action="store_true", help="强制重新扫描，忽略缓存")
    parser.add_argument("--ask", "-a", help="自然语言提问，如 '新能源汽车板块怎么样'")

    args = parser.parse_args()

    if args.ask:
        print(ask_llm(
            args.ask,
            trade_date=args.date,
            debug=args.debug,
            store_memory=not args.no_conversation_memory,
        ))
        return

    if args.review:
        print(review_memory(price_file=args.price_file, as_of=args.date))
        return

    if args.daily_review:
        from .backtest.scheduler import run_daily_review

        result = run_daily_review(price_file=args.price_file, as_of=args.date)
        print(result.report)
        return

    if args.data_health:
        print(data_source_health_report(args.date))
        return

    if args.sector_etf_scan:
        print(scan_sector_etfs(
            args.date,
            sector=args.sector,
            top_n=args.scan_top_n,
            force=args.force,
            refresh_cache=args.refresh_scan_cache,
        ))
        return

    if args.sector_etf_analyze:
        if args.sector:
            print(analyze_sector_etf(
                args.sector,
                args.date,
                question=f"{args.sector}板块是否适合买ETF？",
                store_memory=not args.no_conversation_memory,
                refresh_cache=args.refresh_scan_cache,
            ))
        else:
            print(analyze_sector_etf_watchlist(
                args.date,
                max_roundtable_sectors=min(args.scan_top_n, 8),
                store_memory=not args.no_conversation_memory,
                force=args.force,
                json_output=args.json,
                refresh_cache=args.refresh_scan_cache,
            ))
        return

    if args.refresh_cache:
        print(refresh_local_cache(
            args.date,
            output_dir=args.output_dir,
            cache_days=args.cache_days,
            force_news=args.force_news,
        ))
        return

    if args.refresh_etf_cache:
        print(refresh_etf_data_cache(
            args.date,
            etf_codes=args.etf_code or None,
            daily_limit=args.etf_daily_limit,
        ))
        return

    if args.portfolio_backtest:
        if not args.signals_file or not args.price_file:
            parser.error("--portfolio-backtest requires --signals-file and --price-file")
        print(backtest_portfolio(args.signals_file, args.price_file))
        return

    if args.full_analysis:
        if not args.ticker:
            parser.error("--full-analysis requires --ticker")
        print(run_full_data_analysis(
            args.ticker,
            trade_date=args.date,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            use_react_planner=args.react_planner,
            news_keyword=args.news_keyword,
            sector_keyword=args.sector,
            use_llm_news_filter=not args.no_llm_news_filter,
            use_llm_data_review=args.llm_data_review,
            fetch_news_full_text=not args.no_news_full_text,
            skip_backtest=args.skip_backtest,
            lookback_days=args.lookback_days,
            compact=not args.json,
        ))
        return

    if args.strategy_audit:
        if args.audit_proposal_id:
            if not args.audit_action:
                parser.error("--audit-proposal-id requires --audit-action")
            print(audit_strategy_proposal(
                args.audit_proposal_id,
                action=args.audit_action,
                reviewer=args.reviewer,
                comment=args.audit_comment,
            ))
        else:
            print(list_strategy_audit_queue())
        return

    print(analyze_sector_etf_watchlist(
        args.date,
        max_roundtable_sectors=8,
        store_memory=not args.no_conversation_memory,
        force=args.force,
        json_output=args.json,
        refresh_cache=args.refresh_scan_cache,
    ))


if __name__ == "__main__":
    main()
