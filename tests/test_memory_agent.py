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


def test_memory_store_derives_index_from_configured_log_path(tmp_path, monkeypatch):
    from advanced_trading_agent.config import config

    previous_log = config.get("memory_log_path")
    previous_index = config.get("memory_index_path")
    monkeypatch.delenv("ATA_MEMORY_INDEX_PATH", raising=False)
    try:
        configured_log = tmp_path / "configured_memory.md"
        config.update({
            "memory_log_path": str(configured_log),
            "memory_index_path": str(tmp_path / "should_not_be_used.jsonl"),
        })

        store = MemoryStore()

        assert store._log_path == configured_log
        assert store._index_path == configured_log.with_suffix(".jsonl")
    finally:
        config.update({
            "memory_log_path": previous_log,
            "memory_index_path": previous_index,
        })


def test_get_context_ignores_future_resolved_entries(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-01", _decision())
    store.resolve_pending(
        outcomes={"000001.SZ": {"horizon_days": 5, "excess_return": -0.10}},
        as_of="2026-07-20",
    )

    context = store.get_context("000001.SZ", as_of="2026-07-10")

    assert context == ""


def test_get_context_includes_entries_resolved_by_as_of(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-01", _decision())
    store.resolve_pending(
        outcomes={"000001.SZ": {"horizon_days": 5, "excess_return": 0.02}},
        as_of="2026-07-10",
    )

    context = store.get_context("000001.SZ", as_of="2026-07-10")

    assert "Past analyses" in context
    assert "factor strength" in context


def test_get_context_ignores_resolved_entries_without_as_of_when_filtering(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-01", _decision())
    store.resolve_pending(
        outcomes={"000001.SZ": {"horizon_days": 5, "excess_return": 0.02}},
        as_of="2026-07-10",
    )
    entries = store.load_entries()
    entries[0]["reflection"].pop("as_of", None)
    store._write_index_entries(entries)
    store._write_markdown_entries(entries)

    assert store.get_context("000001.SZ", as_of="2026-07-10") == ""


def test_get_context_accepts_compact_resolution_dates(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-01", _decision())
    store.resolve_pending(
        outcomes={"000001.SZ": {"horizon_days": 5, "excess_return": 0.02}},
        as_of="2026-07-10",
    )
    entries = store.load_entries()
    entries[0]["reflection"]["as_of"] = "20260710"
    store._write_index_entries(entries)
    store._write_markdown_entries(entries)

    context = store.get_context("000001.SZ", as_of="2026-07-10")

    assert "Past analyses" in context


def test_get_context_ignores_invalid_resolution_dates(tmp_path):
    path = tmp_path / "memory.md"
    store = MemoryStore(log_path=str(path))
    store.store_decision("000001.SZ", "2026-07-01", _decision())
    store.resolve_pending(
        outcomes={"000001.SZ": {"horizon_days": 5, "excess_return": 0.02}},
        as_of="2026-07-10",
    )
    entries = store.load_entries()
    entries[0]["reflection"]["as_of"] = "not-a-date"
    store._write_index_entries(entries)
    store._write_markdown_entries(entries)

    assert store.get_context("000001.SZ", as_of="2026-07-10") == ""
