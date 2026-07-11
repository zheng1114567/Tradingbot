"""Tests for LLM cost tracker."""
from __future__ import annotations

from advanced_trading_agent.llm.cost_tracker import CostRecord, CostTracker


class TestCostRecord:
    def test_creation(self):
        record = CostRecord(
            agent="Market Agent",
            model="deepseek-chat",
            input_tokens=1000,
            output_tokens=500,
            duration_sec=1.5,
            cost_cny=0.002,
        )
        assert record.agent == "Market Agent"
        assert record.model == "deepseek-chat"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.duration_sec == 1.5
        assert record.cost_cny == 0.002

    def test_defaults(self):
        record = CostRecord(agent="Test", model="test-model")
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.cost_cny == 0.0
        assert record.timestamp


class TestCostTracker:
    def test_record_and_cost(self):
        tracker = CostTracker(model="deepseek-chat")
        tracker.record("Market Agent", 1000, 500)
        assert len(tracker.records) == 1
        # 1000/1M * 1.0 + 500/1M * 2.0 = 0.001 + 0.001 = 0.002
        assert tracker.total_cost == 0.002

    def test_total_cost_accumulates(self):
        tracker = CostTracker(model="deepseek-chat")
        tracker.record("Market Agent", 1_000_000, 500_000)
        # 1M * 1.0 + 0.5M * 2.0 = 1.0 + 1.0 = 2.0 CNY
        assert tracker.total_cost == 2.0

    def test_total_tokens(self):
        tracker = CostTracker(model="deepseek-chat")
        tracker.record("Agent1", 100, 50)
        tracker.record("Agent2", 200, 100)
        assert tracker.total_tokens == 450

    def test_summary_format(self):
        tracker = CostTracker(model="deepseek-chat")
        tracker.record("Market Agent", 1000, 500)
        summary = tracker.summary()
        assert "Market Agent" in summary
        assert "¥" in summary

    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
        assert len(tracker.records) == 0
        assert "No LLM calls" in tracker.summary()

    def test_warning_threshold_configurable(self):
        tracker = CostTracker(model="deepseek-chat")
        tracker.warning_threshold = 0.0001  # very low threshold
        # Should not crash, just log a warning
        tracker.record("Agent", 1000, 500)
        assert len(tracker.records) == 1

    def test_custom_model_pricing(self):
        tracker = CostTracker(model="gpt-4o")
        tracker.record("Agent", 1_000_000, 1_000_000)
        # 1M * 15 + 1M * 60 = 75 CNY
        assert tracker.total_cost == 75.0
