from __future__ import annotations

from advanced_trading_agent.data_agent.etf_watchlist import SectorCandidatePayload, WatchlistETFCandidate
from advanced_trading_agent.roundtable.etf_watchlist_autogen import ETFWatchlistAutoGenRoundtable


def test_etf_autogen_result_builds_dialogue_and_agent_outputs():
    candidate = SectorCandidatePayload(
        sector_name="半导体",
        pre_score=9.0,
        raw_etf_candidates=[
            WatchlistETFCandidate(code="512480.SH", name="半导体ETF", total_score=9.5)
        ],
    )
    result = ETFWatchlistAutoGenRoundtable._to_result(
        [
            {"source": "Market_Agent", "content": "半导体动量支持，但注意宽度。"},
            {"source": "Risk_Agent", "content": "风险：流动性可接受，不允许自动交易。"},
            {"source": "System_Moderator", "content": "最终保留半导体，首选 512480.SH。TERMINATE"},
        ],
        candidates=[candidate],
    )

    payload = result.to_summary_dict()
    assert payload["provider"] == "autogen"
    assert payload["mode"] == "autogen_batch_roundtable"
    assert len(payload["dialogue_records"]) == 3
    assert payload["dialogue_records"][0]["speaker"] == "Market"
    assert payload["round_history"][0]["turn_count"] == 3
    assert {item["agent"] for item in payload["agent_outputs"]} == {"Market", "Risk"}
    assert "512480.SH" in payload["summary"]


def test_etf_autogen_result_extracts_moderator_json():
    candidate = SectorCandidatePayload(
        sector_name="半导体",
        pre_score=9.0,
        raw_etf_candidates=[
            WatchlistETFCandidate(code="512480.SH", name="半导体ETF", total_score=9.5)
        ],
    )
    result = ETFWatchlistAutoGenRoundtable._to_result(
        [
            {
                "source": "System_Moderator",
                "content": """
最终裁决如下。
```json
{
  "final_decisions": [
    {
      "sector": "半导体",
      "status": "active",
      "primary_etf_code": "512480.SH",
      "support_reasons": ["动量最强", "ETF流动性合格"],
      "objections": ["事件催化不足"],
      "confidence": "medium"
    }
  ],
  "excluded_by_roundtable": [
    {"sector": "机器人", "reason": "ETF映射不纯"}
  ]
}
```
TERMINATE
""",
            },
        ],
        candidates=[candidate],
    )

    payload = result.to_summary_dict()
    assert payload["final_decisions"][0]["sector"] == "半导体"
    assert payload["final_decisions"][0]["primary_etf_code"] == "512480.SH"
    assert payload["excluded_by_roundtable"][0]["sector"] == "机器人"


def test_json_direct_result_extracts_structured_agent_dialogue():
    candidate = SectorCandidatePayload(
        sector_name="半导体",
        pre_score=9.0,
        raw_etf_candidates=[
            WatchlistETFCandidate(code="512480.SH", name="半导体ETF", total_score=9.5)
        ],
    )
    result = ETFWatchlistAutoGenRoundtable._to_result(
        [
            {
                "source": "System_Moderator",
                "content": """
{
  "agent_outputs": [
    {
      "agent": "Market",
      "sector": "batch",
      "stance": "support",
      "summary": "半导体动量领先。",
      "evidence": ["momentum_score=4.0"],
      "objections": []
    },
    {
      "agent": "Risk",
      "sector": "batch",
      "stance": "caution",
      "summary": "事件催化不足，需要审批。",
      "evidence": ["risk_flags=事件催化不足"],
      "objections": ["不允许自动交易"]
    }
  ],
  "dialogue_records": [
    {"round": 1, "sector": "batch", "speaker": "Market", "message": "半导体动量领先。", "references": ["半导体"]},
    {"round": 1, "sector": "batch", "speaker": "Risk", "message": "事件催化不足，需要审批。", "references": ["半导体"]},
    {"round": 1, "sector": "batch", "speaker": "Moderator", "message": "保留观察。", "references": ["512480.SH"]}
  ],
  "final_decisions": [
    {
      "sector": "半导体",
      "status": "monitor",
      "primary_etf_code": "512480.SH",
      "support_reasons": ["动量领先"],
      "objections": ["事件催化不足"],
      "confidence": "medium"
    }
  ],
  "excluded_by_roundtable": []
}
""",
            }
        ],
        candidates=[candidate],
    )

    payload = result.to_summary_dict()
    assert {item["agent"] for item in payload["agent_outputs"]} == {"Market", "Risk"}
    assert [turn["speaker"] for turn in payload["dialogue_records"]] == ["Market", "Risk", "Moderator"]
    assert payload["final_decisions"][0]["status"] == "monitor"


def test_qwen_direct_roundtable_prompt_is_readable_chinese():
    candidate = SectorCandidatePayload(
        sector_name="半导体",
        pre_score=9.0,
        momentum_score=4.0,
        breadth_score=2.0,
        event_score=1.0,
        support_evidence=["板块动量强"],
        risk_flags=["事件催化不足"],
        raw_etf_candidates=[
            WatchlistETFCandidate(code="512480.SH", name="半导体ETF", total_score=9.5)
        ],
    )

    prompt = ETFWatchlistAutoGenRoundtable._build_direct_json_task(
        trade_date="2026-07-10",
        candidates=[candidate],
        max_final_decisions=3,
    )

    assert "交易日: 2026-07-10" in prompt
    assert "严格 JSON" in prompt
    assert "???" not in prompt
