"""Post-trade review tests."""
from __future__ import annotations

import pandas as pd

from advanced_trading_agent.agents.memory_agent import MemoryStore
from advanced_trading_agent.agents.schemas import DecisionType, RiskVerdict, SystemDecision
from advanced_trading_agent.backtest.scheduler import run_daily_review
from advanced_trading_agent.backtest.review import ReviewEngine
from advanced_trading_agent.strategy_rules import (
    enqueue_strategy_proposals,
    load_strategy_proposals,
    review_strategy_proposal,
)


def _decision(source: str = "factor") -> SystemDecision:
    return SystemDecision(
        decision=DecisionType.RECOMMEND,
        position=0.1,
        alpha_source=[source],
        horizon_days=5,
        reasons=["factor strength"],
        objections=[],
        risk_verdict=RiskVerdict.PASS,
        reasoning="test",
    )


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2026-07-10", periods=8, freq="B")
    return pd.DataFrame({
        "code": ["000001.SZ"] * len(dates),
        "trade_date": dates,
        "open": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4],
        "close": [10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5],
        "bench_close": [100, 100, 100, 100, 100, 100, 100, 100],
        "volume": [1e7] * len(dates),
        "amount": [1e8] * len(dates),
        "is_limit_up": [False] * len(dates),
        "is_limit_down": [False] * len(dates),
    })


def test_review_resolves_pending_from_prices(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-10", _decision())
    reviewer = ReviewEngine(review_config={"review_horizons": [5]})

    resolved = reviewer.resolve_due(
        store,
        price_loader=lambda ticker, signal_date: _prices(),
        as_of="2026-07-20",
    )

    assert len(resolved) == 1
    assert resolved[0]["pending"] is False
    assert resolved[0]["outcome"]["horizon_days"] == 5
    assert resolved[0]["outcome"]["absolute_return"] is not None


def test_review_waits_until_horizon_is_available(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-10", _decision())
    short_prices = _prices().head(3)
    reviewer = ReviewEngine(review_config={"review_horizons": [5]})

    resolved = reviewer.resolve_due(
        store,
        price_loader=lambda ticker, signal_date: short_prices,
        as_of="2026-07-14",
    )

    assert resolved == []
    assert store.load_entries()[0]["pending"] is True


def test_daily_review_writes_summary(tmp_path):
    path = tmp_path / "memory.md"
    queue_path = tmp_path / "audit.jsonl"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-10", _decision())

    result = run_daily_review(
        price_file=None,
        as_of="2026-07-20",
        store=store,
        reviewer=ReviewEngine(),
        audit_queue_path=str(queue_path),
    )

    assert result.pending_count == 1
    assert result.rulebook["version"]
    assert "复盘绩效" in result.report


def test_review_summarizes_alpha_and_downweights_poor_sample():
    reviewer = ReviewEngine(review_config={
        "min_samples_to_adjust": 2,
        "min_hit_rate": 0.45,
        "pause_hit_rate": 0.1,
        "pause_avg_excess_return": -0.05,
    })
    entries = [
        {
            "pending": False,
            "decision_data": {"alpha_source": ["event"]},
            "rulebook": {"version": "rules-test"},
            "outcome": {"absolute_return": -0.01, "excess_return": -0.01, "tradable": True},
        },
        {
            "pending": False,
            "decision_data": {"alpha_source": ["event"]},
            "rulebook": {"version": "rules-test"},
            "outcome": {"absolute_return": 0.01, "excess_return": 0.005, "tradable": True},
        },
    ]

    summary = reviewer.summarize_entries(entries)

    assert summary[0].alpha_source == "event"
    assert summary[0].rule_version == "rules-test"
    assert summary[0].recommendation == "DOWNWEIGHT"
    assert summary[0].avg_excess_return < 0


def test_review_groups_same_alpha_by_rule_version():
    reviewer = ReviewEngine(review_config={"min_samples_to_adjust": 1})
    entries = [
        {
            "pending": False,
            "decision_data": {"alpha_source": ["factor"]},
            "rulebook": {"version": "rules-v1"},
            "outcome": {"absolute_return": 0.01, "excess_return": 0.01, "tradable": True},
        },
        {
            "pending": False,
            "decision_data": {"alpha_source": ["factor"]},
            "rulebook": {"version": "rules-v2"},
            "outcome": {"absolute_return": -0.01, "excess_return": -0.01, "tradable": True},
        },
    ]

    summary = reviewer.summarize_entries(entries)

    assert [item.rule_version for item in summary] == ["rules-v1", "rules-v2"]


def test_strategy_change_proposals_require_human_audit(tmp_path):
    queue_path = tmp_path / "strategy_audit.jsonl"
    reviewer = ReviewEngine(review_config={
        "min_samples_to_adjust": 1,
        "pause_hit_rate": 0.1,
        "pause_avg_excess_return": -0.05,
    })
    summary = reviewer.summarize_entries([{
        "pending": False,
        "decision_data": {"alpha_source": ["event"]},
        "rulebook": {"version": "rules-test"},
        "outcome": {"absolute_return": -0.01, "excess_return": -0.01, "tradable": True},
    }])

    proposals = enqueue_strategy_proposals(summary, path=str(queue_path))

    assert len(proposals) == 1
    assert proposals[0]["status"] == "pending_human_review"
    assert proposals[0]["proposed_action"] == "pause_alpha_source"
    assert load_strategy_proposals(str(queue_path))[0]["status"] == "pending_human_review"

    approved = review_strategy_proposal(
        proposals[0]["proposal_id"],
        action="approve",
        reviewer="tester",
        comment="accept",
        path=str(queue_path),
    )

    assert approved["status"] == "approved"
    assert approved["reviewer"] == "tester"
    assert approved["reviewer_comment"] == "accept"
