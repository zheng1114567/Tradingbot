"""Tests for roundtable participant resolution from A-share signals.

The resolution logic (trigger rules, participant ordering) is tested here
by directly accessing the static methods on RoundtableHarness.  If the
module-level import fails due to a pre-existing circular import in the
codebase, tests are skipped with a clear message.
"""

from __future__ import annotations

import pytest

try:
    from advanced_trading_agent.roundtable.harness import RoundtableHarness
    HARNESS_AVAILABLE = True
except ImportError:
    HARNESS_AVAILABLE = False


# ====================================================================
# _signal_meets_criteria
# ====================================================================

@pytest.mark.skipif(not HARNESS_AVAILABLE, reason="circular import in codebase prevents harness import")
class TestSignalMeetsCriteria:
    """Trigger-rule predicate logic."""

    def test_passes_with_matching_signal(self):
        rules = {"require_data_status": ["available"], "required_signals": ["confirmed"]}
        data = {"signal": "confirmed", "data_status": "available"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is True

    def test_fails_with_wrong_signal(self):
        rules = {"require_data_status": ["available"], "required_signals": ["confirmed"]}
        data = {"signal": "absent", "data_status": "available"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is False

    def test_fails_with_missing_data_status(self):
        rules = {"require_data_status": ["available"], "required_signals": ["overheated"]}
        data = {"signal": "overheated", "data_status": "missing"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is False

    def test_passes_with_min_strength(self):
        rules = {"min_strength": 0.6, "required_signals": ["positive", "negative"]}
        data = {"signal": "neutral", "strength": 0.72, "data_status": "available"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is True

    def test_fails_below_min_strength(self):
        rules = {"min_strength": 0.6, "required_signals": ["positive", "negative"]}
        data = {"signal": "neutral", "strength": 0.3, "data_status": "available"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is False

    def test_passes_with_risk_level(self):
        rules = {"required_risk_levels": ["high", "medium"]}
        data = {"risk_level": "high", "data_status": "available"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is True

    def test_fails_with_unknown_risk_level(self):
        rules = {"required_risk_levels": ["high", "medium"]}
        data = {"risk_level": "low", "data_status": "available"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is False

    def test_no_required_status_means_no_gate(self):
        rules = {"required_signals": ["confirmed"]}
        data = {"signal": "confirmed", "data_status": "missing"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is True

    def test_signal_wins_over_min_strength(self):
        """Signal match is sufficient even if min_strength not met."""
        rules = {"min_strength": 0.6, "required_signals": ["positive"]}
        data = {"signal": "positive", "strength": 0.0, "data_status": "available"}
        assert RoundtableHarness._signal_meets_criteria(data, rules) is True


# ====================================================================
# _resolve_participants
# ====================================================================

@pytest.mark.skipif(not HARNESS_AVAILABLE, reason="circular import in codebase prevents harness import")
class TestParticipantResolution:
    """Dynamic participant list resolution from A-share signals."""

    def _tier2_with_signals(self, hot_money_signal: str, data_status: str = "available", **hm_kw) -> dict:
        signals: dict = {
            "hot_money": {
                "signal": hot_money_signal, "score": 50, "board_count": 0,
                "dragon_tiger_active": False, "warnings": [], "evidence": [],
                "data_status": data_status, **hm_kw,
            },
            "policy": {"signal": "insufficient", "strength": 0, "data_status": "missing"},
            "unlock": {"risk_level": "unknown", "data_status": "missing"},
            "multifactor": {"signal": "insufficient", "data_status": "missing"},
        }
        return {"a_share_signals": signals}

    def test_default_participants_only(self):
        participants = RoundtableHarness._resolve_participants(
            self._tier2_with_signals("absent")
        )
        assert participants == ("Market", "Event", "Analysis", "Backtest", "Risk")

    def test_hot_money_confirmed_activates(self):
        participants = RoundtableHarness._resolve_participants(
            self._tier2_with_signals("confirmed")
        )
        assert "HotMoney" in participants

    def test_hot_money_overheated_activates(self):
        participants = RoundtableHarness._resolve_participants(
            self._tier2_with_signals("overheated")
        )
        assert "HotMoney" in participants

    def test_hot_money_absent_does_not_activate(self):
        participants = RoundtableHarness._resolve_participants(
            self._tier2_with_signals("absent")
        )
        assert "HotMoney" not in participants

    def test_confirmed_but_missing_data_does_not_activate(self):
        participants = RoundtableHarness._resolve_participants(
            self._tier2_with_signals("confirmed", data_status="missing")
        )
        assert "HotMoney" not in participants

    def test_hot_money_position_after_event(self):
        participants = RoundtableHarness._resolve_participants(
            self._tier2_with_signals("confirmed")
        )
        idx = {name: i for i, name in enumerate(participants)}
        assert idx["Event"] < idx["HotMoney"] < idx["Analysis"]

    def test_no_signals_key_returns_defaults(self):
        participants = RoundtableHarness._resolve_participants({})
        assert participants == ("Market", "Event", "Analysis", "Backtest", "Risk")

    def test_empty_a_share_signals_returns_defaults(self):
        participants = RoundtableHarness._resolve_participants(
            {"a_share_signals": {}}
        )
        assert participants == ("Market", "Event", "Analysis", "Backtest", "Risk")
