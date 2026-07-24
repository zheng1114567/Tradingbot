"""Tests for data quality checker."""
from __future__ import annotations

from datetime import date

from advanced_trading_agent.core.data_quality import DataQualityChecker, DataQualityReport


class TestDataQualityReport:
    def test_grade_a_perfect(self):
        report = DataQualityReport(passed=True, grade="A")
        assert report.grade == "A"
        assert report.passed is True
        assert report.critical_missing == []

    def test_grade_f_broken(self):
        report = DataQualityReport(
            passed=False,
            critical_missing=["market.index_close", "market.index_change_pct"],
            grade="F",
        )
        assert report.grade == "F"
        assert report.passed is False
        assert len(report.critical_missing) == 2

    def test_to_prompt_f_grade(self):
        report = DataQualityReport(
            passed=False,
            critical_missing=["market.index_close"],
            grade="F",
        )
        prompt = report.to_prompt()
        assert "失败" in prompt
        assert "market.index_close" in prompt

    def test_to_prompt_a_grade(self):
        report = DataQualityReport(passed=True, grade="A")
        prompt = report.to_prompt()
        assert "A" in prompt

    def test_warnings_in_prompt(self):
        report = DataQualityReport(
            passed=True,
            warnings=["情绪数据缺失"],
            stale_fields=["行情感 2026-01-01 早于 2026-07-10"],
            grade="B",
        )
        prompt = report.to_prompt()
        assert "情绪数据缺失" in prompt


class TestDataQualityChecker:
    def test_check_tier1_all_present(self):
        tier1 = {
            "market": {
                "index_close": 3500.0,
                "index_change_pct": 0.5,
                "advance_count": 2000,
                "decline_count": 1500,
                "limit_up_count": 50,
                "limit_down_count": 10,
            },
            "sentiment": {"sentiment": "normal"},
        }
        result = DataQualityChecker.check_tier1(tier1)
        assert result.grade == "A"
        assert result.passed is True

    def test_check_tier1_missing_critical(self):
        tier1 = {
            "market": {},
            "sentiment": {},
        }
        result = DataQualityChecker.check_tier1(tier1)
        assert result.grade == "F"
        assert result.passed is False
        assert len(result.critical_missing) >= 1

    def test_check_tier1_zero_index_close_is_critical(self):
        tier1 = {
            "market": {
                "index_close": 0,
                "index_change_pct": 0,
                "advance_count": 0,
                "decline_count": 0,
                "limit_up_count": 0,
                "limit_down_count": 0,
            },
            "sentiment": {"sentiment": "未知"},
        }
        result = DataQualityChecker.check_tier1(tier1)
        assert result.grade == "F"
        assert result.passed is False
        assert "market.index_close" in result.critical_missing

    def test_check_tier1_zero_breadth_warns(self):
        tier1 = {
            "market": {
                "index_close": 3500.0,
                "index_change_pct": 0,
                "advance_count": 0,
                "decline_count": 0,
                "limit_up_count": 0,
                "limit_down_count": 0,
            },
            "sentiment": {"sentiment": "正常"},
        }
        result = DataQualityChecker.check_tier1(tier1)
        assert result.passed is True
        assert any("涨跌家数" in warning for warning in result.warnings)

    def test_check_tier1_with_trade_date_stale(self):
        tier1 = {
            "market": {
                "index_close": 3500.0,
                "index_change_pct": 0.5,
                "advance_count": 2000,
                "decline_count": 1500,
                "limit_up_count": 50,
                "limit_down_count": 10,
                "as_of_date": "2026-01-01",
            },
            "sentiment": {"sentiment": "normal"},
        }
        result = DataQualityChecker.check_tier1(tier1, trade_date=date(2026, 7, 10))
        assert len(result.stale_fields) > 0

    def test_check_tier1_with_current_date_not_stale(self):
        tier1 = {
            "market": {
                "index_close": 3500.0,
                "index_change_pct": 0.5,
                "advance_count": 2000,
                "decline_count": 1500,
                "limit_up_count": 50,
                "limit_down_count": 10,
                "as_of_date": "2026-07-10",
            },
            "sentiment": {"sentiment": "normal"},
        }
        result = DataQualityChecker.check_tier1(tier1, trade_date=date(2026, 7, 10))
        assert len(result.stale_fields) == 0

    def test_check_event_data(self):
        events = [
            {"event_type": "policy", "summary": "test"},
            {"event_type": "earnings", "summary": "test2"},
        ]
        result = DataQualityChecker.check_event_data(events)
        assert result.passed is True

    def test_check_event_data_empty(self):
        result = DataQualityChecker.check_event_data([])
        assert "无可用事件数据" in result.warnings

    def test_check_event_data_missing_summary(self):
        events = [{"event_type": "policy"}]
        result = DataQualityChecker.check_event_data(events)
        assert len(result.warnings) > 0

    def test_check_factor_data(self):
        factors = [
            {"code": "000001.SZ", "composite_score": 7.5},
            {"code": "000002.SZ", "composite_score": 6.0},
        ]
        result = DataQualityChecker.check_factor_data(factors)
        assert result.passed is True

    def test_check_factor_data_empty(self):
        result = DataQualityChecker.check_factor_data([])
        assert result.grade == "F"
        assert result.passed is False
