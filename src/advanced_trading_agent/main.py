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
from .data_agent.data_agent import DataAgent, DataAgentRequest
from .graph.workflow import TradingSystem
from .strategy_rules import load_strategy_proposals, review_strategy_proposal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_single(ticker: str, trade_date: str | None = None,
                   debug: bool = False) -> str:
    """分析单个标的

    Args:
        ticker: 股票代码 (如 "000001.SZ")
        trade_date: 交易日, 默认今天
        debug: 是否打印调试信息

    Returns:
        Markdown 格式的分析报告
    """
    trade_date = trade_date or str(date.today())
    logger.info("Analyzing %s on %s...", ticker, trade_date)

    system = TradingSystem(debug=debug)
    final_state, report = system.analyze(ticker, trade_date)

    return report


def analyze_batch(tickers_file: str, debug: bool = False,
                  max_workers: int = 4) -> list[str]:
    """批量分析 (并发执行)

    Args:
        tickers_file: 每行一个股票代码的文件
        debug: 是否打印调试信息
        max_workers: 最大并发数 (默认 4)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open(tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    results: list[tuple[int, str]] = []
    logger.info("Batch analyzing %d tickers (max_workers=%d)...", len(tickers), max_workers)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_map = {
            executor.submit(analyze_single, ticker, debug=debug): (idx, ticker)
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
    summary_path.write_text("\n\n---\n\n".join(reports), encoding="utf-8")
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
    review_path.write_text(report, encoding="utf-8")
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
    (results_dir / "portfolio_backtest_summary.md").write_text(report, encoding="utf-8")
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
    parser.add_argument("--workers", type=int, default=4, help="批量并发数 (默认 4)")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--json", action="store_true", help="JSON 输出")

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

    if args.batch:
        reports = analyze_batch(args.batch, debug=args.debug, max_workers=args.workers)
        print(f"Batch complete: {len(reports)} tickers analyzed")
        return

    if not args.ticker:
        parser.print_help()
        sys.exit(1)

    report = analyze_single(args.ticker, args.date, debug=args.debug)

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
