from __future__ import annotations

import json

from advanced_trading_agent.agents.memory_agent import MemoryStore
from advanced_trading_agent.agents.schemas import DecisionType, RiskVerdict, SystemDecision
from advanced_trading_agent.data_agent.data_agent import DataAgentArtifact, DataAgentRun
from advanced_trading_agent.pipeline import run_full_analysis, workflow_payload_from_data_run


def _fake_run(
    *,
    event_direction: str = "中性",
    factor_score: float = 5,
    backtest_samples: list[dict] | None = None,
    price_data: list[dict] | None = None,
) -> DataAgentRun:
    price_data = price_data or [{"trade_date": "2026-07-10", "close": 10}]
    final_data = {
        "manifest": {"fields": {"stock.daily": {"available": True}}},
        "cleaned": {"daily": {"record_count": 2}, "news": {"record_count": 1}},
        "analysis": {
            "summary": {"latest": {"code": "000001.SZ", "close": 10}},
            "market": {"index_change_pct": 0.1},
            "sector": {"matched_sector": "银行"},
            "capital": {"confirmation": "资金确认"},
            "risk": {"risk_data_available": True},
            "data_quality": {"daily_consistency": {"status": "single_source"}},
            "agent_payload": {
                "tier1_data": {
                    "market": {"index_close": 3000, "index_change_pct": 0.1},
                    "sentiment": {"sentiment": "正常"},
                    "capital": {"confirmation": "资金确认"},
                    "risk": {"risk_data_available": True, "daily_volume": 20000000},
                },
                "tier2_data": {
                    "price_data": price_data,
                    "events": [{"title": "测试新闻", "direction": event_direction}],
                    "factors": [{"code": "000001.SZ", "composite_score": factor_score}],
                    "backtest_samples": backtest_samples or [],
                    "data_quality": {"daily_consistency": {"status": "single_source"}},
                },
            },
        },
        "errors": [],
    }
    return DataAgentRun(
        run_id="run-1",
        request={"ticker": "000001.SZ", "trade_date": "2026-07-10"},
        artifacts={"final": DataAgentArtifact(stage="final", path="out/response.json")},
        manifest_path="out/manifest.json",
        response_path="out/response.json",
        final_data=final_data,
        collection_summary={"categories_with_data": 5, "categories_failed": 0, "categories_empty": 1},
        audit_trail=[{"stage": "data", "message": "ok"}],
    )


class FakeDataAgent:
    def __init__(self, run: DataAgentRun | None = None):
        self.calls = []
        self._run = run

    def run(self, request):
        self.calls.append(request)
        return self._run or _fake_run()


class FakeTradingSystem:
    def __init__(self):
        self.calls = []

    def analyze(self, ticker, trade_date, *, tier1_data=None, tier2_data=None, skip_backtest=False):
        self.calls.append({
            "ticker": ticker,
            "trade_date": trade_date,
            "tier1_data": tier1_data,
            "tier2_data": tier2_data,
            "skip_backtest": skip_backtest,
        })
        return {
            "execution_allowed": False,
            "round2_state": {"final_pressure": "downgrade"},
            "risk_check_1": {"verdict": "PASS"},
            "audit_trace": {"report_path": "out/report.md"},
            "audit_trace_path": "out/audit.json",
        }, "# report"


def test_workflow_payload_from_data_run_attaches_metadata():
    tier1, tier2 = workflow_payload_from_data_run(_fake_run())

    assert tier1["_data_manifest_path"] == "out/manifest.json"
    assert tier1["_collection_summary"]["categories_with_data"] == 5
    assert tier1["_audit_trail"][0]["message"] == "ok"
    assert tier2["price_data"][0]["close"] == 10


def test_run_full_analysis_rules_mode_is_fast_and_writes_report(tmp_path):
    data_agent = FakeDataAgent()
    trading_system = FakeTradingSystem()

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        start_date="20260701",
        end_date="20260710",
        output_dir=str(tmp_path),
        skip_backtest=True,
        data_agent=data_agent,
        trading_system=trading_system,
    )
    payload = result.to_dict()

    assert len(data_agent.calls) == 1
    assert len(trading_system.calls) == 0
    assert payload["analysis_mode"] == "rules"
    assert payload["stage"] == "full_analysis"
    assert payload["analysis"]["final_report_path"].endswith("report_2026-07-10_rules.md")
    assert payload["data_agent"]["collection_summary"]["categories_failed"] == 0
    assert data_agent.calls[0].start_date == "20260701"
    assert payload["data"]["cleaned_counts"]["daily"] == 2
    assert payload["data"]["latest"]["code"] == "000001.SZ"


def test_run_full_analysis_defaults_to_lookback_window(tmp_path):
    data_agent = FakeDataAgent()

    run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        end_date="20260710",
        output_dir=str(tmp_path),
        data_agent=data_agent,
        lookback_days=30,
    )

    assert data_agent.calls[0].start_date == "20260610"
    assert data_agent.calls[0].end_date == "20260710"


def test_run_full_analysis_rules_mode_can_store_memory(tmp_path):
    data_agent = FakeDataAgent()
    store = MemoryStore(log_path=str(tmp_path / "memory.md"))

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        output_dir=str(tmp_path),
        data_agent=data_agent,
        store_memory=True,
        memory_store=store,
    )

    payload = result.to_dict()
    assert result.final_state["audit_trace"]["memory_status"]["stored"] is True
    assert payload["analysis"]["rules_diagnostics"]["memory_status"]["stored"] is True
    audit_payload = json.loads((tmp_path / "000001_SZ" / "audit_2026-07-10_rules.json").read_text(encoding="utf-8"))
    assert audit_payload["rules_diagnostics"]["memory_status"]["stored"] is True
    entries = store.load_entries()
    assert len(entries) == 1
    assert entries[0]["ticker"] == "000001.SZ"


def test_run_full_analysis_rules_mode_downgrades_on_weak_backtest(tmp_path):
    data_agent = FakeDataAgent(_fake_run(
        event_direction="利好",
        factor_score=7,
        backtest_samples=[{
            "sample_size": 12,
            "win_rate": 0.35,
            "avg_excess_return": -0.02,
        }],
    ))

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        output_dir=str(tmp_path),
        data_agent=data_agent,
    )
    payload = result.to_dict()

    assert payload["analysis"]["system_decision"]["decision"] == "观察"
    assert payload["analysis"]["round2_state"]["final_pressure"] == "downgrade"
    diagnostics = payload["analysis"]["rules_diagnostics"]
    assert diagnostics["backtest_evidence"]["gate"] == "downgrade"
    assert "历史平均超额为负" in " ".join(diagnostics["soft_objections"])
    roundtable = payload["analysis"]["round2_state"]["rules_roundtable"]
    assert roundtable["mode"] == "rules_roundtable"
    assert roundtable["agent_outputs"]
    assert any(item["agent"] == "Risk" for item in roundtable["agent_outputs"])
    assert roundtable["final_pressure"] == "downgrade"
    assert roundtable["risk_focus"]
    market_output = next(item for item in roundtable["agent_outputs"] if item["agent"] == "Market")
    event_output = next(item for item in roundtable["agent_outputs"] if item["agent"] == "Event")
    assert "资金确认" in market_output["reasoning"]
    assert event_output["pressure"] == "upgrade"
    assert "利好/中性偏多 1 条" in event_output["reasoning"]


def test_run_full_analysis_rules_roundtable_includes_a_share_specialists(tmp_path):
    run = _fake_run(event_direction="利好", factor_score=7)
    run.final_data["analysis"]["agent_payload"]["tier2_data"]["a_share_signals"] = {
        "hot_money": {
            "signal": "confirmed",
            "score": 65,
            "warnings": ["全市场涨停过热"],
            "evidence": ["limit_up_count=60"],
            "data_status": "available",
        },
        "policy": {
            "signal": "positive",
            "score": 70,
            "warnings": [],
            "evidence": ["policy_strength=0.7"],
            "data_status": "available",
        },
        "unlock": {
            "signal": "insufficient",
            "risk_level": "unavailable",
            "warnings": ["无解禁数据"],
            "evidence": [],
            "data_status": "missing",
        },
    }
    data_agent = FakeDataAgent(run)

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        output_dir=str(tmp_path),
        data_agent=data_agent,
    )
    roundtable = result.to_dict()["analysis"]["round2_state"]["rules_roundtable"]
    agents = {item["agent"] for item in roundtable["agent_outputs"]}

    assert {"Market", "Event", "Analysis", "Backtest", "Risk", "HotMoney", "Policy"}.issubset(agents)
    assert "Unlock" not in agents
    assert roundtable["dialogue_records"]


def test_run_full_analysis_rules_mode_downgrades_on_memory_underperformance(tmp_path):
    data_agent = FakeDataAgent(_fake_run(event_direction="利好", factor_score=7))
    store = MemoryStore(log_path=str(tmp_path / "memory.md"))
    for idx in range(3):
        decision = SystemDecision(
            decision=DecisionType.RECOMMEND,
            position=0.1,
            alpha_source=["rules_data_summary"],
            reasons=[f"old signal {idx}"],
            objections=[],
            risk_verdict=RiskVerdict.PASS,
            reasoning="test",
        )
        trade_date = f"2026-07-0{idx + 1}"
        store.store_decision("000001.SZ", trade_date, decision)
        store.resolve_pending(
            outcomes={
                f"{trade_date}|000001.SZ": {
                    "horizon_days": 5,
                    "absolute_return": -0.01,
                    "excess_return": -0.02,
                }
            },
            as_of="2026-07-10",
        )

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        output_dir=str(tmp_path),
        data_agent=data_agent,
        memory_store=store,
    )
    payload = result.to_dict()

    assert payload["analysis"]["system_decision"]["decision"] == "观察"
    diagnostics = payload["analysis"]["rules_diagnostics"]
    assert diagnostics["memory_summary"]["gate"] == "downgrade"
    assert diagnostics["memory_summary"]["resolved_count"] == 3



def test_run_full_analysis_memory_summary_is_point_in_time(tmp_path):
    data_agent = FakeDataAgent(_fake_run(event_direction="利好", factor_score=7))
    store = MemoryStore(log_path=str(tmp_path / "memory.md"))
    decision = SystemDecision(
        decision=DecisionType.RECOMMEND,
        position=0.1,
        alpha_source=["rules_data_summary"],
        reasons=["future-resolved sample"],
        objections=[],
        risk_verdict=RiskVerdict.PASS,
        reasoning="test",
    )
    store.store_decision("000001.SZ", "2026-07-01", decision)
    store.resolve_pending(
        outcomes={
            "2026-07-01|000001.SZ": {
                "horizon_days": 5,
                "absolute_return": -0.10,
                "excess_return": -0.10,
            }
        },
        as_of="2026-07-20",
    )

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        output_dir=str(tmp_path),
        data_agent=data_agent,
        memory_store=store,
    )

    diagnostics = result.to_dict()["analysis"]["rules_diagnostics"]
    assert diagnostics["memory_summary"]["status"] == "empty"
    assert diagnostics["memory_summary"]["resolved_count"] == 0
    assert diagnostics["memory_summary"].get("ignored_future_count") == 1


def test_run_full_analysis_memory_summary_ignores_unknown_resolution_date(tmp_path):
    data_agent = FakeDataAgent(_fake_run(event_direction="利好", factor_score=7))
    store = MemoryStore(log_path=str(tmp_path / "memory.md"))
    decision = SystemDecision(
        decision=DecisionType.RECOMMEND,
        position=0.1,
        alpha_source=["rules_data_summary"],
        reasons=["legacy resolved sample"],
        objections=[],
        risk_verdict=RiskVerdict.PASS,
        reasoning="test",
    )
    store.store_decision("000001.SZ", "2026-07-01", decision)
    store.resolve_pending(
        outcomes={"000001.SZ": {"horizon_days": 5, "absolute_return": -0.10, "excess_return": -0.10}},
        as_of="2026-07-10",
    )
    entries = store.load_entries()
    entries[0]["reflection"].pop("as_of", None)
    store._write_index_entries(entries)
    store._write_markdown_entries(entries)

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        output_dir=str(tmp_path),
        data_agent=data_agent,
        memory_store=store,
    )

    diagnostics = result.to_dict()["analysis"]["rules_diagnostics"]
    assert diagnostics["memory_summary"]["status"] == "empty"
    assert diagnostics["memory_summary"].get("ignored_unknown_as_of_count") == 1

def test_run_full_analysis_resolves_pending_memory_from_price_window(tmp_path):
    price_data = [
        {"trade_date": f"2026-07-{day:02d}", "open": 10 + day / 10, "close": 10 + day / 10, "volume": 1000000, "amount": 20000000}
        for day in range(1, 11)
    ]
    data_agent = FakeDataAgent(_fake_run(price_data=price_data))
    store = MemoryStore(log_path=str(tmp_path / "memory.md"))
    old_decision = SystemDecision(
        decision=DecisionType.RECOMMEND,
        position=0.1,
        alpha_source=["rules_data_summary"],
        reasons=["old signal"],
        objections=[],
        risk_verdict=RiskVerdict.PASS,
        reasoning="test",
    )
    store.store_decision("000001.SZ", "2026-07-01", old_decision)

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        output_dir=str(tmp_path),
        data_agent=data_agent,
        memory_store=store,
    )
    payload = result.to_dict()

    diagnostics = payload["analysis"]["rules_diagnostics"]
    assert diagnostics["memory_resolution"]["resolved_count"] == 1
    entries = store.load_entries()
    assert entries[0]["pending"] is False
    assert entries[0]["reflection"]["hit"] is True


def test_run_full_analysis_memory_resolution_respects_trade_date_as_of(tmp_path):
    price_data = [
        {"trade_date": f"2026-07-{day:02d}", "open": 10 + day / 10, "close": 10 + day / 10, "volume": 1000000, "amount": 20000000}
        for day in range(10, 21)
    ]
    data_agent = FakeDataAgent(_fake_run(price_data=price_data))
    store = MemoryStore(log_path=str(tmp_path / "memory.md"))
    old_decision = SystemDecision(
        decision=DecisionType.RECOMMEND,
        position=0.1,
        alpha_source=["rules_data_summary"],
        reasons=["old signal"],
        objections=[],
        risk_verdict=RiskVerdict.PASS,
        reasoning="test",
    )
    store.store_decision("000001.SZ", "2026-07-10", old_decision)

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-14",
        output_dir=str(tmp_path),
        data_agent=data_agent,
        memory_store=store,
    )

    diagnostics = result.to_dict()["analysis"]["rules_diagnostics"]
    assert diagnostics["memory_resolution"]["resolved_count"] == 0
    assert store.load_entries()[0]["pending"] is True

def test_run_full_analysis_workflow_mode_reuses_dataagent_payload():
    data_agent = FakeDataAgent()
    trading_system = FakeTradingSystem()

    result = run_full_analysis(
        "000001.SZ",
        trade_date="2026-07-10",
        start_date="20260701",
        end_date="20260710",
        skip_backtest=True,
        analysis_mode="workflow",
        data_agent=data_agent,
        trading_system=trading_system,
    )
    payload = result.to_dict()

    assert len(data_agent.calls) == 1
    assert data_agent.calls[0].start_date == "20260701"
    assert len(trading_system.calls) == 1
    assert trading_system.calls[0]["tier1_data"]["market"]["index_close"] == 3000
    assert trading_system.calls[0]["tier2_data"]["events"][0]["title"] == "测试新闻"
    assert trading_system.calls[0]["skip_backtest"] is True
    assert payload["analysis_mode"] == "workflow"
    assert payload["analysis"]["final_report_path"] == "out/report.md"




def test_main_full_data_analysis_defaults_to_compact(monkeypatch):
    import json
    import advanced_trading_agent.main as main_mod

    class Run:
        def to_dict(self):
            return {
                "stage": "full_analysis",
                "analysis_mode": "rules",
                "ticker": "000001.SZ",
                "trade_date": "2026-07-10",
                "data_agent": {
                    "run_id": "run-main",
                    "response_path": "out/response.json",
                    "manifest_path": "out/manifest.json",
                    "collection_summary": {"categories_with_data": 5, "categories_failed": 0, "categories_empty": 0},
                    "errors": [],
                },
                "analysis": {
                    "final_report_path": "out/report.md",
                    "audit_trace_path": "out/audit.json",
                    "execution_allowed": False,
                    "round2_state": {"final_pressure": "neutral"},
                    "final_report": "large report body",
                },
                "agent_payload": {"large": [1, 2, 3]},
            }

    def fake_run_full_analysis(**kwargs):
        return Run()

    monkeypatch.setattr("advanced_trading_agent.pipeline.run_full_analysis", fake_run_full_analysis)

    compact = json.loads(main_mod.run_full_data_analysis("000001.SZ", trade_date="2026-07-10"))
    full = json.loads(main_mod.run_full_data_analysis("000001.SZ", trade_date="2026-07-10", compact=False))

    assert compact["run_id"] == "run-main"
    assert "agent_payload" not in compact
    assert full["agent_payload"]["large"] == [1, 2, 3]

