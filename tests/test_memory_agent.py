"""Memory Agent deferred reflection tests."""
from __future__ import annotations

from advanced_trading_agent.agents.memory_agent import MemoryStore
from advanced_trading_agent.agents.schemas import (
    DecisionType,
    RiskVerdict,
    SystemDecision,
)


def _decision(decision: DecisionType = DecisionType.RECOMMEND) -> SystemDecision:
    return SystemDecision(
        decision=decision,
        position=0.1 if decision == DecisionType.RECOMMEND else 0,
        alpha_source=["factor"],
        horizon_days=5,
        reasons=["factor strength"],
        objections=[],
        risk_verdict=RiskVerdict.PASS,
        reasoning="test",
    )


def test_resolve_pending_marks_entry_resolved_and_reflects_hit(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-10", _decision())

    resolved = store.resolve_pending(
        outcomes={
            "2026-07-10|000001.SZ": {
                "horizon_days": 5,
                "absolute_return": 0.04,
                "excess_return": 0.02,
            }
        },
        as_of="2026-07-17",
    )

    assert len(resolved) == 1
    assert resolved[0]["pending"] is False
    assert resolved[0]["reflection"]["hit"] is True
    assert "resolved" in path.read_text(encoding="utf-8")
    assert "pending" not in path.read_text(encoding="utf-8")


def test_resolve_pending_leaves_unmatched_entries_pending(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-10", _decision())

    resolved = store.resolve_pending(outcomes={}, as_of="2026-07-17")

    assert resolved == []
    entries = store.load_entries()
    assert len(entries) == 1
    assert entries[0]["pending"] is True


def test_load_entries_preserves_resolved_decision_data(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-10", _decision())
    store.resolve_pending(
        outcomes={"000001.SZ": {"horizon_days": 5, "excess_return": -0.01}},
        as_of="2026-07-17",
    )

    entries = store.load_entries()

    assert entries[0]["pending"] is False
    assert entries[0]["decision_data"]["decision"] == "推荐"
    assert entries[0]["decision_data"]["alpha_source"] == ["factor"]
    assert entries[0]["rulebook"]["version"]
    assert entries[0]["rulebook"]["fingerprint"]
