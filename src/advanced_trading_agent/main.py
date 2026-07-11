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

from .config import config
from .graph.workflow import TradingSystem

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


def main():
    parser = argparse.ArgumentParser(description="多智能体量化交易分析系统")
    parser.add_argument("--ticker", "-t", help="股票代码 (如 000001.SZ)")
    parser.add_argument("--date", "-d", help="交易日 (默认今天)")
    parser.add_argument("--batch", "-b", help="批量分析文件路径")
    parser.add_argument("--workers", type=int, default=4, help="批量并发数 (默认 4)")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--json", action="store_true", help="JSON 输出")

    args = parser.parse_args()

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
