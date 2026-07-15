"""Tests for A-share specialist signals (Phase 0.5: HotMoney).

These tests only depend on AShareSignalBuilder which has no
circular-import risk.  Roundtable participant-resolution tests
live in test_roundtable_specialists.py.
"""

from __future__ import annotations

import pytest

from advanced_trading_agent.data_agent.a_share_signals import AShareSignalBuilder


# ====================================================================
# AShareSignalBuilder — HotMoney
# ====================================================================

class TestHotMoneySignal:
    """HotMoney signal computation under various data conditions."""

    def _tier2(self, **overrides) -> dict:
        """Build a minimal tier2 dict with sensible defaults."""
        base: dict = {
            "limit_up_summary": {
                "first_board": 5,
                "second_board": 2,
                "third_plus": 1,
                "stocks": [],
            },
            "dragon_tiger": [
                {"code": "000001", "name": "平安银行", "net_buy": 5000000},
            ],
            "data_summary": {"ticker": "000001"},
        }
        base.update(overrides)
        return base

    # ── Normal cases ───────────────────────────────────────────

    def test_hot_money_absent(self):
        """No limit-up stocks, no dragon-tiger → signal=absent."""
        tier2 = self._tier2(
            limit_up_summary={
                "first_board": 0, "second_board": 0, "third_plus": 0, "stocks": [],
            },
            dragon_tiger=[],
        )
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        assert hm["signal"] == "absent"
        assert hm["data_status"] == "available"
        assert hm["score"] < 20

    def test_hot_money_confirmed_limit_up(self):
        """Target ticker is on limit-up list with 1 board → confirmed."""
        tier2 = self._tier2(
            limit_up_summary={
                "first_board": 1, "second_board": 0, "third_plus": 0,
                "stocks": [{"code": "000001", "name": "平安银行", "board_count": 1}],
            },
            dragon_tiger=[{"code": "000001", "net_buy": 3000000}],
        )
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        assert hm["signal"] == "confirmed"
        assert hm["board_count"] == 1

    def test_hot_money_speculative_two_board(self):
        """Target ticker with 2 boards → speculative + warning."""
        tier2 = self._tier2(
            limit_up_summary={
                "first_board": 0, "second_board": 1, "third_plus": 0,
                "stocks": [{"code": "000001", "board_count": 2}],
            },
        )
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        assert hm["signal"] == "speculative"
        assert hm["board_count"] == 2
        assert len(hm["warnings"]) >= 1

    def test_hot_money_overheated_three_board(self):
        """Target ticker with 3+ boards → overheated + dragon_tiger warning."""
        tier2 = self._tier2(
            limit_up_summary={
                "first_board": 0, "second_board": 0, "third_plus": 1,
                "stocks": [{"code": "000001", "board_count": 4}],
            },
            dragon_tiger=[{"code": "000001", "net_buy": 10000000}],
        )
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        assert hm["signal"] == "overheated"
        assert hm["board_count"] == 4
        assert hm["score"] >= 80
        assert any("连板" in w for w in hm["warnings"])

    def test_dragon_tiger_net_buy_boosts_absent(self):
        """Dragon-tiger net buy > 0 promotes absent → confirmed, even when <2 records."""
        tier2 = self._tier2(
            limit_up_summary={
                "first_board": 5, "second_board": 1, "third_plus": 0, "stocks": [],
            },
            # Only 1 dragon-tiger record → dt_active=False (<2),
            # but net_buy > 0 should still boost signal
            dragon_tiger=[{"code": "000001", "net_buy": 8000000}],
        )
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        assert hm["signal"] in ("confirmed",)
        assert hm["dragon_tiger_active"] is False
        assert hm["score"] >= 50

    # ── Edge cases ─────────────────────────────────────────────

    def test_hot_money_insufficient_missing_keys(self):
        """No limit_up_summary or dragon_tiger keys → hot_money still in result with missing."""
        tier2 = {"data_summary": {"ticker": "000001"}}
        result = AShareSignalBuilder.build(tier2)
        assert "hot_money" in result
        hm = result["hot_money"]
        assert hm["data_status"] == "missing"
        assert hm["signal"] == "insufficient"

    def test_hot_money_market_overheat_warning(self):
        """Market-wide limit-up > 50 adds overheat warning."""
        tier2 = self._tier2(
            limit_up_summary={
                "first_board": 30, "second_board": 15, "third_plus": 10,
                "stocks": [],
            },
            dragon_tiger=[],
        )
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        assert any("涨停" in w for w in hm["warnings"])

    def test_hot_money_dragon_tiger_no_net_buy(self):
        """Dragon-tiger records without net_buy → no crash, no boost."""
        tier2 = self._tier2(
            limit_up_summary={
                "first_board": 1, "second_board": 0, "third_plus": 0, "stocks": [],
            },
            dragon_tiger=[{"code": "000001", "name": "平安银行"}],
        )
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        assert hm["signal"] in ("absent",)

    def test_hot_money_empty_limit_up_dict(self):
        """Empty dict limit_up_summary → no crash, signal=absent (data available)."""
        tier2 = self._tier2(limit_up_summary={}, dragon_tiger=[])
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        # Both keys exist but are empty → data_status=available, signal=absent
        assert hm["data_status"] == "available"
        assert hm["signal"] == "absent"

    def test_hot_money_target_not_in_stocks(self):
        """Ticker not found in limit-up stocks → board_count=0."""
        tier2 = self._tier2(
            limit_up_summary={
                "first_board": 3, "second_board": 0, "third_plus": 0,
                "stocks": [{"code": "600000", "board_count": 1}],
            },
            data_summary={"ticker": "000001"},
        )
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        assert hm["board_count"] == 0

    def test_hot_money_all_fields_present(self):
        """All expected output keys exist."""
        tier2 = self._tier2()
        result = AShareSignalBuilder.build(tier2)
        hm = result["hot_money"]
        expected_keys = {
            "signal", "score", "limit_up_count", "board_count",
            "dragon_tiger_active", "warnings", "evidence", "data_status",
        }
        assert expected_keys.issubset(hm.keys())
        assert isinstance(hm["warnings"], list)
        assert isinstance(hm["evidence"], list)
        assert isinstance(hm["score"], float)

    # ── Placeholder signals ────────────────────────────────────

    def test_policy_placeholder(self):
        """Policy signal is a safe placeholder in Phase 0.5."""
        result = AShareSignalBuilder.build(self._tier2())
        assert result["policy"]["data_status"] == "missing"
        assert result["policy"]["signal"] == "insufficient"

    def test_unlock_placeholder(self):
        """Unlock signal is a safe placeholder."""
        result = AShareSignalBuilder.build(self._tier2())
        assert result["unlock"]["data_status"] == "missing"
        assert "解禁" in result["unlock"]["warnings"][0]

    def test_multifactor_placeholder(self):
        """Multifactor signal is a safe placeholder."""
        result = AShareSignalBuilder.build(self._tier2())
        assert result["multifactor"]["data_status"] == "missing"
