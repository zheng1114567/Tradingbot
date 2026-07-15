from __future__ import annotations

from advanced_trading_agent.agents.conversation_memory import ConversationEntry, ConversationMemoryStore


def test_conversation_memory_is_separate_jsonl_store(tmp_path):
    path = tmp_path / "conversation.jsonl"
    store = ConversationMemoryStore(path=str(path))

    store.append(ConversationEntry(
        question="半导体板块为什么不好？",
        answer="ETF流动性不足。",
        trade_date="2026-07-15",
        target_type="sector",
        target="半导体",
        evidence={"score": 4.2},
    ))

    rows = store.load()
    assert rows[0]["target"] == "半导体"
    assert rows[0]["evidence"]["score"] == 4.2
    assert "仅用于上下文" in store.recall("半导体")
