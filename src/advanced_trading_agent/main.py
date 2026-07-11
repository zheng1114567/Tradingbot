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
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .agents.memory_agent import MemoryStore
from .backtest.portfolio import ObservationPortfolioBacktester
from .backtest.review import ReviewEngine
from .backtest.scheduler import run_daily_review
from .config import config
from .core.atomic_write import atomic_write_text
from .data_agent.data_agent import DataAgent, DataAgentRequest
from .data_agent.scanner import MarketScanner, ScanBundle
from .graph.workflow import TradingSystem
from .strategy_rules import load_strategy_proposals, review_strategy_proposal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    trade_date = trade_date or str(date.today())
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
    summary_path = Path(config.get("results_dir", "data/results")) / "batch_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(summary_path, "\n\n---\n\n".join(reports))
    logger.info("Batch summary saved to %s", summary_path)

    return reports


def review_memory(price_file: str | None = None, as_of: str | None = None) -> str:
    """Resolve pending decisions and print alpha-source performance advice.

    price_file 可选，格式为 CSV，至少包含 code/trade_date/open/close/volume/amount。
    若不传 price_file，则只汇总已有 resolved 复盘记录。
    """
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
    review_path = Path(config.get("results_dir", "data/results")) / "review_summary.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(review_path, report)
    return report


def backtest_portfolio(signals_file: str, price_file: str) -> str:
    """Run observation-pool portfolio backtest from signal and price CSV files."""
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
    results_dir = Path(config.get("results_dir", "data/results"))
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
    fetch_news_full_text: bool = True,
) -> str:
    """Run data collection, cleaning, analysis, and layered persistence only."""
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
            fetch_news_full_text=fetch_news_full_text,
        )
    )
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def list_strategy_audit_queue() -> str:
    """Return pending and reviewed strategy change proposals as JSON."""
    return json.dumps(load_strategy_proposals(), ensure_ascii=False, indent=2)


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
                    force: bool = False) -> str:
    """Scan for hot stocks, collect data during scan, then analyze top candidates.

    Uses the combined scan+collect flow: MarketScanner.scan_and_collect()
    fetches raw data for top candidates in the same pass, eliminating
    redundant vendor calls when DataAgent runs later.

    An LLM-generated market summary is included at the top of the report.
    Results are cached by trade_date — re-running the same date returns
    the cached report immediately unless `force=True`.

    Returns a consolidated Markdown report.
    """
    trade_date = trade_date or str(date.today())
    results_dir = Path(config.get("results_dir", "data/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    scan_path = results_dir / f"scan_report_{trade_date}.md"

    # Cache hit: return existing report for the same trade date
    if scan_path.exists() and not force:
        logger.info("Scan report already exists for %s, returning cached. Use --force to re-scan.", trade_date)
        return scan_path.read_text(encoding="utf-8")

    scanner = MarketScanner(top_sectors=5, top_n=top_n)
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
) -> str:
    """Run DataAgent with pre-collected data, then feed tier1/tier2 to TradingSystem.

    Assembles the raw_data payload from the ScanBundle's shared and
    per-ticker portions so DataAgent skips _collect_raw entirely.
    """
    from .data_agent.data_agent import DataAgent, DataAgentRequest

    shared = bundle.shared_raw
    ticker_raw = bundle.ticker_data.get(ticker, {})

    raw_data = {
        "daily": ticker_raw.get("daily", []),
        "market": shared.get("market", []),
        "sector_context": shared.get("sector_context", []),
        "capital_flow": ticker_raw.get("capital_flow", []),
        "news": ticker_raw.get("news", []),
        "risk": shared.get("risk", {}),
        "route_trace": bundle.route_trace,
    }

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
        raw_data=raw_data,
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


def main():
    parser = argparse.ArgumentParser(description="多智能体量化交易分析系统")
    parser.add_argument("--ticker", "-t", help="股票代码 (如 000001.SZ)")
    parser.add_argument("--date", "-d", help="交易日 (默认今天)")
    parser.add_argument("--batch", "-b", help="批量分析文件路径")
    parser.add_argument("--review", action="store_true", help="运行复盘汇总")
    parser.add_argument("--daily-review", action="store_true", help="运行每日复盘任务")
    parser.add_argument("--portfolio-backtest", action="store_true", help="运行观察池组合回测")
    parser.add_argument("--data-agent", action="store_true", help="单独运行 DataAgent 并分层保存数据")
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
    parser.add_argument("--news-keyword", help="DataAgent 新闻关键词过滤, 可选")
    parser.add_argument("--no-llm-news-filter", action="store_true", help="DataAgent 新闻筛选不调用 LLM, 仅使用规则兜底")
    parser.add_argument("--no-news-full-text", action="store_true", help="DataAgent 不抓取新闻 URL 正文, 仅保留摘要")
    parser.add_argument("--workers", type=int, default=4, help="批量并发数 (默认 4)")
    parser.add_argument("--skip-backtest", action="store_true", help="跳过 Backtest Agent，仅保留可审计占位报告")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--scan", action="store_true", help="扫描热点板块和强势股，自动分析 Top-N")
    parser.add_argument("--scan-top-n", type=int, default=10, help="扫描后分析 Top N 只股票 (默认 10)")
    parser.add_argument("--force", "-f", action="store_true", help="强制重新扫描，忽略缓存")

    args = parser.parse_args()

    if args.review:
        print(review_memory(price_file=args.price_file, as_of=args.date))
        return

    if args.daily_review:
        result = run_daily_review(price_file=args.price_file, as_of=args.date)
        print(result.report)
        return

    if args.portfolio_backtest:
        if not args.signals_file or not args.price_file:
            parser.error("--portfolio-backtest requires --signals-file and --price-file")
        print(backtest_portfolio(args.signals_file, args.price_file))
        return

    if args.data_agent:
        if not args.ticker:
            parser.error("--data-agent requires --ticker")
        print(run_standalone_data_agent(
            args.ticker,
            trade_date=args.date,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            use_react_planner=args.react_planner,
            news_keyword=args.news_keyword,
            use_llm_news_filter=not args.no_llm_news_filter,
            fetch_news_full_text=not args.no_news_full_text,
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

    if args.scan:
        print(scan_and_analyze(
            top_n=args.scan_top_n,
            trade_date=args.date,
            debug=args.debug,
            skip_backtest=args.skip_backtest,
            force=args.force,
        ))
        return

    if args.batch:
        reports = analyze_batch(
            args.batch,
            debug=args.debug,
            max_workers=args.workers,
            skip_backtest=args.skip_backtest,
        )
        print(f"Batch complete: {len(reports)} tickers analyzed")
        return

    if not args.ticker:
        parser.print_help()
        sys.exit(1)

    report = analyze_single(
        args.ticker,
        args.date,
        debug=args.debug,
        skip_backtest=args.skip_backtest,
    )

    if args.json:
        # 提取 JSON 部分
        from .agents.memory_agent import MemoryStore
        store = MemoryStore()
        entries = store.load_entries()
        print(json.dumps(entries, ensure_ascii=False, indent=2))
    else:
        print(report)


if __name__ == "__main__":
    main()
