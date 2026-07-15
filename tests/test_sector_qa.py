from __future__ import annotations

from advanced_trading_agent.roundtable import sector_qa


class FakeSelector:
    def explain_sector(self, sector_name, trade_date):
        return {
            "sector_name": sector_name,
            "status": "matched",
            "verdict": "暂不适合",
            "score": 3.5,
            "reasons": ["板块动量=1.0"],
            "risks": ["缺少新闻或事件催化证据，持续性需要打折。"],
            "primary_etf": None,
            "candidate": {"raw": {"news": []}},
        }


def test_sector_qa_fallback_gives_bad_sector_reasons():
    payload = sector_qa.answer_sector_question_with_roundtable(
        "半导体板块为什么不好？",
        sector_name="半导体",
        trade_date="2026-07-15",
        selector=FakeSelector(),
        use_autogen=False,
    )

    assert payload["provider"] == "deterministic"
    assert "为什么不好" in payload["answer"]
    assert "缺少新闻" in payload["answer"]
