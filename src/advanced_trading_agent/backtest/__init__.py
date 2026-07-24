"""
回测包初始化 — 保留实际使用的模块
"""
from .engine import BacktestEngine, BacktestResult
from .data_qa import DataQAIssue, DataQAReport, DataQualityGate
from .metrics import PerformanceMetrics
from .portfolio import ObservationPortfolioBacktester, PortfolioBacktestResult
from .review import AlphaPerformance, ReviewEngine
from .scheduler import DailyReviewResult, run_daily_review
from .sector_backtest import (
    SectorBacktestEntry,
    SectorBacktestSummary,
    SectorETFBacktester,
    format_sector_backtest_summary,
)
from ..strategy_rules import (
    StrategyChangeProposal,
    StrategyRulebook,
    current_rulebook,
    enqueue_strategy_proposals,
    load_strategy_proposals,
    review_strategy_proposal,
)

__all__ = [
    "AlphaPerformance",
    "BacktestEngine",
    "BacktestResult",
    "DataQAIssue",
    "DataQAReport",
    "DataQualityGate",
    "DailyReviewResult",
    "ObservationPortfolioBacktester",
    "PerformanceMetrics",
    "PortfolioBacktestResult",
    "ReviewEngine",
    "SectorBacktestEntry",
    "SectorBacktestSummary",
    "SectorETFBacktester",
    "StrategyChangeProposal",
    "StrategyRulebook",
    "current_rulebook",
    "enqueue_strategy_proposals",
    "format_sector_backtest_summary",
    "load_strategy_proposals",
    "review_strategy_proposal",
    "run_daily_review",
]
