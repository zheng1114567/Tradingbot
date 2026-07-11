"""Standalone DataAgent tests."""

import json

from advanced_trading_agent.data_agent.data_agent import DataAgent, DataAgentRequest
from advanced_trading_agent.data_agent.planner import DataAgentPlanner


class FakeLLM:
    provider = "fake"
    model = "fake-news-filter"

    def __init__(self, response: str | Exception):
        self.response = response
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_data_agent_persists_layered_trace(tmp_path):
    def fake_route(method, **kwargs):
        if method == "get_daily":
            return [
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260709",
                    "open": 10.0,
                    "high": 10.6,
                    "low": 9.8,
                    "close": 10.4,
                    "pre_close": 10.0,
                    "pct_chg": 4.0,
                    "vol": 1000,
                    "amount": 10400,
                },
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260710",
                    "open": 10.4,
                    "high": 11.5,
                    "low": 10.4,
                    "close": 11.44,
                    "pre_close": 10.4,
                    "pct_chg": 10.0,
                    "vol": 1200,
                    "amount": 13728,
                },
            ]
        if method == "get_capital_flow":
            return [{"ts_code": kwargs["code"], "trade_date": "20260710", "net_mf_amount": 88.0}]
        if method == "get_news":
            return [
                {
                    "标题": "平安银行发布经营动态",
                    "内容": "平安银行零售业务保持稳定。",
                    "发布时间": "2026-07-10 09:30:00",
                    "来源": "akshare",
                }
            ]
        if method == "get_sector":
            return [{"板块名称": "银行", "涨跌幅": 1.2, "rank": 1, "data_source": "test"}]
        if method in {"get_st_status", "get_suspended", "get_delisting"}:
            return []
        raise AssertionError(f"unexpected method: {method}")

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path)).run(
        DataAgentRequest(
            ticker="000001.SZ",
            trade_date="2026-07-10",
            start_date="20260701",
            end_date="20260710",
            use_llm_news_filter=False,
        )
    )

    payload = result.to_dict()
    assert set(payload["artifacts"]) == {"input", "raw", "cleaned", "analysis", "news_events", "agent_payload", "final"}
    for artifact in payload["artifacts"].values():
        assert str(tmp_path) in artifact["path"]

    final_path = result.artifacts["final"].path
    final_payload = json.loads(open(final_path, encoding="utf-8").read())
    assert final_payload["input"]["request"]["ticker"] == "000001.SZ"
    assert final_payload["cleaned"]["market"]["record_count"] == 2
    assert final_payload["cleaned"]["daily"]["record_count"] == 2
    assert final_payload["analysis"]["summary"]["latest"]["code"] == "000001.SZ"
    assert final_payload["agent_payload"]["tier1_data"]["risk"]["risk_data_available"] is True
    assert final_payload["agent_payload"]["tier2_data"]["price_data"]
    assert final_payload["cleaned"]["sector_context"]["record_count"] == 1
    assert final_payload["analysis"]["sector"]["matched_sector"] == "银行"
    assert final_payload["agent_payload"]["tier1_data"]["sector"]["matched_sector"] == "银行"
    assert final_payload["agent_payload"]["tier2_data"]["sector_context"]["matched_sector"] == "银行"
    assert final_payload["cleaned"]["news"]["record_count"] == 1
    assert final_payload["agent_payload"]["tier2_data"]["events"][0]["summary"] == "平安银行零售业务保持稳定。"
    assert final_payload["agent_payload"]["tier2_data"]["events"][0]["evidence_text"] == "平安银行零售业务保持稳定。"
    assert final_payload["agent_payload"]["tier2_data"]["events"][0]["content_status"] == "summary_only"
    assert final_payload["analysis"]["events"]["filter"]["mode"] == "deterministic"
    assert final_payload["analysis"]["data_quality"]["daily_consistency"]["status"] == "single_source"
    assert final_payload["agent_payload"]["tier2_data"]["data_quality"]["daily_consistency"]["confidence_score"] == 0.7
    assert final_payload["vendor_health"]["attempt_count"] > 0
    assert "custom_route_fn" in final_payload["vendor_health"]["vendors"]
    assert final_payload["analysis"]["agent_payload"]["tier1_data"]["risk"]["risk_data_available"] is True
    assert final_payload["manifest"]["fields"]["stock.daily"]["available"] is True
    assert result.artifacts["agent_payload"].path.endswith("05_agent_payload\\agent_payload.json") or result.artifacts["agent_payload"].path.endswith("05_agent_payload/agent_payload.json")
    assert result.artifacts["news_events"].path.endswith("04_analysis\\news_events.json") or result.artifacts["news_events"].path.endswith("04_analysis/news_events.json")


def test_data_agent_auto_resolves_stock_profile_keywords(tmp_path):
    calls = []

    def fake_route(method, **kwargs):
        calls.append((method, kwargs))
        if method == "get_daily":
            return [
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260710",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.9,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "pct_chg": 2.0,
                    "vol": 1000,
                    "amount": 10200,
                }
            ]
        if method == "get_capital_flow":
            return []
        if method == "get_news":
            return [
                {
                    "title": "平安银行经营稳定",
                    "summary": "零售业务保持稳定。",
                    "source": "test",
                }
            ]
        if method == "get_sector":
            return [{"sector_name": "银行", "change_pct": 1.2, "rank": 1, "data_source": "test"}]
        if method in {"get_st_status", "get_suspended", "get_delisting"}:
            return []
        raise AssertionError(f"unexpected method: {method}")

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path)).run(
        DataAgentRequest(
            ticker="000001.SZ",
            trade_date="2026-07-10",
            include_factors=False,
            use_llm_news_filter=False,
        )
    )

    news_calls = [kwargs for method, kwargs in calls if method == "get_news"]
    assert news_calls[0]["keyword"] == "平安银行"

    final_payload = result.final_data
    assert final_payload["input"]["request"]["news_keyword"] == "平安银行"
    assert final_payload["input"]["request"]["sector_keyword"] == "银行"
    assert final_payload["input"]["stock_profile"]["company_name"] == "平安银行"
    assert final_payload["input"]["stock_profile"]["applied_fields"] == ["news_keyword", "sector_keyword"]
    assert final_payload["analysis"]["sector"]["matched_sector"] == "银行"


def test_data_agent_react_planner_persists_plan(tmp_path):
    calls = []

    def fake_route(method, **kwargs):
        calls.append((method, kwargs))
        if method == "get_daily":
            return [
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260710",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.9,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "pct_chg": 2.0,
                    "vol": 1000,
                    "amount": 10200,
                }
            ]
        if method in {"get_st_status", "get_suspended", "get_delisting"}:
            return []
        if method == "get_news":
            return []
        if method == "get_sector":
            return [{"sector_name": "银行", "change_pct": 1.2, "rank": 1, "data_source": "test"}]
        raise AssertionError(f"unexpected method: {method}")

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path)).run(
        DataAgentRequest(
            ticker="000001.SZ",
            trade_date="2026-07-10",
            include_capital_flow=False,
            include_factors=False,
            use_react_planner=True,
        )
    )

    assert "planner" in result.artifacts
    assert result.plan is not None
    assert result.plan["required_methods"] == [
        "get_daily",
        "get_daily:index",
        "get_sector",
        "get_news",
        "filter_news:llm",
        "get_st_status",
        "get_suspended",
        "get_delisting",
    ]
    assert result.request["end_date"] == "20260710"
    assert calls[0][0] == "get_daily"

    final_payload = json.loads(open(result.response_path, encoding="utf-8").read())
    assert final_payload["planner"]["trace"][0]["action"] == "inspect_request"
    assert final_payload["planner"]["trace"][0]["reason"]
    assert "thought" not in final_payload["planner"]["trace"][0]
    assert final_payload["planner"]["next_actions"][0]["name"] == "collect_daily_bars"
    assert final_payload["planner"]["decision_summary"]
    assert final_payload["planner"]["clarification_questions"] == []
    assert final_payload["agent_payload"]["tier2_data"]["sector_context"]["matched_sector"] == "银行"
    assert final_payload["input"]["planner"]["skipped_methods"] == [
        "get_capital_flow",
        "compute_factors",
    ]


def test_data_agent_planner_returns_user_clarification_questions():
    request = DataAgentRequest(
        ticker="",
        trade_date=None,
        include_news=True,
        news_keyword=None,
    )

    questions = DataAgentPlanner.clarification_questions(request)

    assert questions[0]["id"] == "ticker"
    assert questions[0]["required"] is True
    assert "哪只股票" in questions[0]["question"]
    assert any(item["id"] == "trade_date" for item in questions)
    assert any(item["id"] == "news_keyword" for item in questions)


def test_data_agent_marks_risk_errors_unavailable(tmp_path):
    def fake_route(method, **kwargs):
        if method == "get_daily":
            return [
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260710",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.9,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "pct_chg": 2.0,
                    "vol": 1000,
                    "amount": 10200,
                }
            ]
        if method == "get_capital_flow":
            return []
        if method == "get_news":
            return []
        if method == "get_st_status":
            raise RuntimeError("risk endpoint failed")
        if method in {"get_suspended", "get_delisting"}:
            return []
        raise AssertionError(f"unexpected method: {method}")

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path)).run(
        DataAgentRequest(
            ticker="000001.SZ",
            trade_date="2026-07-10",
            include_factors=False,
            use_llm_news_filter=False,
        )
    )

    risk = result.final_data["agent_payload"]["tier1_data"]["risk"]
    assert risk["risk_data_available"] is False
    assert risk["risk_data_errors"] == ["st_status: risk endpoint failed"]


def test_data_agent_uses_llm_to_filter_news(tmp_path):
    def fake_route(method, **kwargs):
        if method == "get_daily":
            return []
        if method == "get_capital_flow":
            return []
        if method == "get_news":
            return [
                {"title": "Ping An Bank operating update", "summary": "Retail business stable", "source": "test"},
                {"title": "Unrelated sports headline", "summary": "A match result", "source": "test"},
            ]
        if method in {"get_st_status", "get_suspended", "get_delisting"}:
            return []
        raise AssertionError(method)

    llm = FakeLLM(json.dumps({
        "decisions": [
            {
                "event_id": "news_0001",
                "keep": True,
                "relevance": 0.92,
                "direction": "正面",
                "confidence": 0.8,
                "reason": "Directly related to the ticker.",
            },
            {
                "event_id": "news_0002",
                "keep": False,
                "relevance": 0.05,
                "direction": "中性",
                "confidence": 0.7,
                "reason": "Unrelated.",
            },
        ]
    }, ensure_ascii=False))

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path), llm_client=llm).run(
        DataAgentRequest(ticker="000001.SZ", trade_date="2026-07-10", news_keyword="Ping An")
    )

    events = result.final_data["agent_payload"]["tier2_data"]["events"]
    assert len(events) == 1
    assert events[0]["summary"] == "Retail business stable"
    assert events[0]["direction"] == "正面"
    assert events[0]["confidence"] == 0.8
    assert result.final_data["analysis"]["events"]["filter"]["used_llm"] is True
    assert llm.calls


def test_data_agent_reports_cross_source_daily_consistency(tmp_path):
    def fake_route(method, **kwargs):
        if method == "get_daily":
            return [
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260710",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.9,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "pct_chg": 2.0,
                    "vol": 1000,
                    "amount": 10200,
                    "data_source": "akshare",
                },
                {
                    "ts_code": kwargs["code"],
                    "trade_date": "20260710",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.9,
                    "close": 10.2005,
                    "pre_close": 10.0,
                    "pct_chg": 2.0005,
                    "vol": 1000,
                    "amount": 10200,
                    "data_source": "baostock",
                },
            ]
        if method == "get_news":
            return []
        raise AssertionError(method)

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path)).run(
        DataAgentRequest(
            ticker="000001.SZ",
            trade_date="2026-07-10",
            include_market=False,
            include_capital_flow=False,
            include_factors=False,
            include_risk=False,
            use_llm_news_filter=False,
        )
    )

    consistency = result.final_data["analysis"]["data_quality"]["daily_consistency"]
    assert consistency["status"] == "consistent"
    assert consistency["sources"] == ["akshare", "baostock"]
    assert consistency["differences"] == []


def test_data_agent_news_filter_falls_back_when_llm_fails(tmp_path):
    def fake_route(method, **kwargs):
        if method == "get_daily":
            return []
        if method == "get_capital_flow":
            return []
        if method == "get_news":
            return [
                {"title": "Ping An Bank operating update", "summary": "Retail business stable", "source": "test"},
                {"title": "Unrelated sports headline", "summary": "A match result", "source": "test"},
            ]
        if method in {"get_st_status", "get_suspended", "get_delisting"}:
            return []
        raise AssertionError(method)

    llm = FakeLLM(RuntimeError("LLM unavailable"))
    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path), llm_client=llm).run(
        DataAgentRequest(ticker="000001.SZ", trade_date="2026-07-10", news_keyword="Ping An")
    )

    events = result.final_data["agent_payload"]["tier2_data"]["events"]
    assert len(events) == 1
    assert events[0]["summary"] == "Retail business stable"
    assert result.final_data["analysis"]["events"]["filter"]["mode"] == "fallback"


def test_data_agent_llm_filter_uses_keyword_guardrail(tmp_path):
    def fake_route(method, **kwargs):
        if method == "get_daily":
            return []
        if method == "get_capital_flow":
            return []
        if method == "get_news":
            return [
                {"title": "Ping An Bank launches AI card", "summary": "Ping An Bank product update", "source": "test"},
                {"title": "Unrelated sports headline", "summary": "A match result", "source": "test"},
            ]
        if method in {"get_st_status", "get_suspended", "get_delisting"}:
            return []
        raise AssertionError(method)

    llm = FakeLLM(json.dumps({
        "decisions": [
            {
                "event_id": "news_0001",
                "keep": False,
                "relevance": 0.0,
                "direction": "中性",
                "confidence": 0.1,
                "reason": "overly strict model decision",
            },
            {
                "event_id": "news_0002",
                "keep": False,
                "relevance": 0.0,
                "direction": "中性",
                "confidence": 0.1,
                "reason": "unrelated",
            },
        ]
    }, ensure_ascii=False))

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path), llm_client=llm).run(
        DataAgentRequest(ticker="000001.SZ", trade_date="2026-07-10", news_keyword="Ping An")
    )

    events = result.final_data["agent_payload"]["tier2_data"]["events"]
    trace = result.final_data["analysis"]["events"]["filter"]
    assert trace["mode"] == "llm_with_keyword_guardrail"
    assert trace["used_llm"] is True
    assert trace["guardrail_added_count"] == 1
    assert len(trace["decisions"]) == 2
    assert len(events) == 1
    assert events[0]["summary"] == "Ping An Bank product update"


def test_data_agent_fetches_full_news_text(tmp_path, monkeypatch):
    class FakeResponse:
        status_code = 200
        apparent_encoding = "utf-8"
        encoding = "utf-8"
        text = (
            "<html><body>"
            "<p>原标题：noise headline</p>"
            "<p>Ping An Bank released a detailed operating update with retail banking evidence.</p>"
            "<p>Ping An Bank released a detailed operating update with retail banking evidence.</p>"
            "<p>The full article contains enough context for downstream trading agents.</p>"
            "<p>责任编辑：someone</p>"
            "<p>投资需谨慎。</p>"
            "</body></html>"
        )

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        assert url == "https://example.test/news"
        return FakeResponse()

    monkeypatch.setattr("advanced_trading_agent.data_agent.data_agent.requests.get", fake_get)

    def fake_route(method, **kwargs):
        if method == "get_daily":
            return []
        if method == "get_capital_flow":
            return []
        if method == "get_news":
            return [{
                "title": "Ping An Bank operating update",
                "summary": "Short summary",
                "source": "test",
                "url": "https://example.test/news",
            }]
        if method in {"get_st_status", "get_suspended", "get_delisting"}:
            return []
        raise AssertionError(method)

    result = DataAgent(route_fn=fake_route, results_dir=str(tmp_path)).run(
        DataAgentRequest(
            ticker="000001.SZ",
            trade_date="2026-07-10",
            news_keyword="Ping An",
            use_llm_news_filter=False,
        )
    )

    raw_news = result.final_data["raw"]["news"][0]
    event = result.final_data["agent_payload"]["tier2_data"]["events"][0]
    assert raw_news["content_status"] == "full_text"
    assert "full article contains enough context" in raw_news["full_text"]
    assert "责任编辑" not in raw_news["full_text"]
    assert "原标题" not in raw_news["full_text"]
    assert raw_news["full_text"].count("retail banking evidence") == 1
    assert raw_news["content_cleaning"]["status"] == "cleaned"
    assert raw_news["content_cleaning"]["removed_segments"] >= 2
    assert raw_news["content_cleaning"]["deduplicated_segments"] == 1
    assert event["content_status"] == "full_text"
    assert event["content_cleaning"]["status"] == "cleaned"
    assert "downstream trading agents" in event["evidence_text"]
