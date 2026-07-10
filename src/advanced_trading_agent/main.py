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

    # 保存决策到 Memory
    decision_obj = final_state.get("system_decision_obj")
    from .agents.memory_agent import MemoryStore
    if decision_obj:
        store = MemoryStore()
        store.store_decision(ticker, trade_date, decision_obj)

    return report


def analyze_batch(tickers_file: str, debug: bool = False) -> list[str]:
    """批量分析

    Args:
        tickers_file: 每行一个股票代码的文件
        debug: 是否打印调试信息
    """
    with open(tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    reports = []
    for ticker in tickers:
        logger.info("=== Analyzing %s ===", ticker)
        try:
            report = analyze_single(ticker, debug=debug)
            reports.append(report)
            logger.info("=== Done %s ===", ticker)
        except Exception as e:
            logger.error("Failed %s: %s", ticker, e)
            reports.append(f"# {ticker}\n\n**失败**: {e}")

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
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--json", action="store_true", help="JSON 输出")

    args = parser.parse_args()

    if args.batch:
        reports = analyze_batch(args.batch, debug=args.debug)
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
