"""Tests for MarketScanner and ScanBundle."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from advanced_trading_agent.data_agent.scanner import MarketScanner, ScanBundle, ScanResult


class TestScanResult:
    def test_creation(self):
        r = ScanResult(
            ticker="000001.SZ",
            name="平安银行",
            source="hot_sector",
            sector="银行",
            score=7.5,
            reason="跨 2 个热点板块",
        )
        assert r.ticker == "000001.SZ"
        assert r.score == 7.5
        assert "热点板块" in r.reason


class TestScanBundle:
    def test_creation_empty(self):
        bundle = ScanBundle(trade_date="2026-07-10", results=[])
        assert bundle.trade_date == "2026-07-10"
        assert bundle.results == []
        assert bundle.shared_raw == {}
        assert bundle.ticker_data == {}
        assert bundle.route_trace == []

    def test_creation_with_data(self):
        results = [
            ScanResult(ticker="000001.SZ", name="平安银行", source="hot_sector",
                       sector="银行", score=8.0, reason="test"),
        ]
        shared = {"market": [], "sector_context": [], "risk": {}}
        ticker_data = {"000001.SZ": {"daily": [{"close": 10.5}], "capital_flow": [], "news": []}}
        trace = [{"method": "get_daily", "vendor": "akshare", "status": "success"}]

        bundle = ScanBundle(
            trade_date="2026-07-10",
            results=results,
            shared_raw=shared,
            ticker_data=ticker_data,
            route_trace=trace,
        )
        assert len(bundle.results) == 1
        assert "000001.SZ" in bundle.ticker_data
        assert bundle.ticker_data["000001.SZ"]["daily"][0]["close"] == 10.5
        assert len(bundle.route_trace) == 1


class TestMarketScanner:
    def test_normalize_ticker_sh(self):
        assert MarketScanner._normalize_ticker("600000") == "600000.SH"

    def test_normalize_ticker_sz(self):
        assert MarketScanner._normalize_ticker("000001") == "000001.SZ"

    def test_normalize_ticker_sz_3xx(self):
        assert MarketScanner._normalize_ticker("300750") == "300750.SZ"

    def test_normalize_ticker_bj(self):
        assert MarketScanner._normalize_ticker("830799") == "830799.BJ"

    def test_normalize_already_formatted(self):
        assert MarketScanner._normalize_ticker("000001.SZ") == "000001.SZ"

    def test_format_results_empty(self):
        scanner = MarketScanner()
        assert "未发现" in scanner.format_results([])

    def test_format_results_with_data(self):
        scanner = MarketScanner()
        results = [
            ScanResult(
                ticker="000001.SZ",
                name="平安银行",
                source="hot_sector+limit_up",
                sector="银行",
                score=8.5,
                reason="涨停池; 跨 2 个热点板块",
            ),
            ScanResult(
                ticker="600000.SH",
                name="浦发银行",
                source="northbound",
                sector="银行",
                score=4.0,
                reason="北向资金关注",
            ),
        ]
        output = scanner.format_results(results)
        assert "000001.SZ" in output
        assert "600000.SH" in output
        assert "8.5" in output
        assert "平安银行" in output

    def test_scan_with_no_data_does_not_crash(self):
        """Scanner should handle the case where vendor data is unavailable gracefully."""
        scanner = MarketScanner(top_sectors=3, top_n=10)
        results = scanner.scan()
        assert isinstance(results, list)

    def test_select_candidates_enforces_sector_cap(self):
        scanner = MarketScanner(top_n=10, base_candidates=10, per_sector_cap=2)
        ranked = [
            ScanResult(ticker="000001.SZ", name="A", source="hot_sector", sector="银行", score=10.0, reason=""),
            ScanResult(ticker="000002.SZ", name="B", source="hot_sector", sector="银行", score=9.8, reason=""),
            ScanResult(ticker="000003.SZ", name="C", source="hot_sector", sector="银行", score=9.6, reason=""),
            ScanResult(ticker="000004.SZ", name="D", source="hot_sector", sector="券商", score=9.4, reason=""),
            ScanResult(ticker="000005.SZ", name="E", source="hot_sector", sector="券商", score=9.2, reason=""),
            ScanResult(ticker="000006.SZ", name="F", source="hot_sector", sector="科技", score=9.0, reason=""),
        ]
        ctx = {"hot_sectors": [{"sector_name": "银行", "strength_score": 3.0, "change_pct": 2.0}]}

        selected = scanner._select_candidates(ranked, ctx)

        assert len([r for r in selected if r.sector == "银行"]) == 2
        assert selected[0].ticker == "000001.SZ"
        assert selected[1].ticker == "000002.SZ"
        assert "000003.SZ" not in {r.ticker for r in selected}

    def test_dynamic_candidate_limit_shrinks_when_sectors_concentrated(self):
        scanner = MarketScanner(top_n=15, base_candidates=12, per_sector_cap=5)
        ranked = [
            ScanResult(ticker=f"00000{i}.SZ", name=str(i), source="hot_sector", sector="银行", score=10 - i * 0.1, reason="")
            for i in range(1, 10)
        ]
        ranked += [
            ScanResult(ticker="600001.SH", name="X", source="hot_sector", sector="券商", score=8.0, reason=""),
            ScanResult(ticker="600002.SH", name="Y", source="hot_sector", sector="券商", score=7.9, reason=""),
        ]
        ctx = {
            "hot_sectors": [
                {"sector_name": "银行", "strength_score": 3.5, "change_pct": 3.2},
                {"sector_name": "券商", "strength_score": 2.0, "change_pct": 1.0},
            ]
        }

        limit = scanner._dynamic_candidate_limit(ranked, ctx)
        assert limit == 10

    def test_dynamic_candidate_limit_widens_when_breadth_is_high(self):
        scanner = MarketScanner(top_n=15, base_candidates=12, per_sector_cap=5)
        sectors = ["银行", "券商", "科技", "军工", "电力", "医药"]
        ranked = [
            ScanResult(ticker=f"{i:06d}.SZ", name=str(i), source="hot_sector", sector=sector, score=10 - i * 0.1, reason="")
            for i, sector in enumerate(sectors, start=1)
        ]
        ctx = {
            "hot_sectors": [
                {"sector_name": "银行", "strength_score": 3.5, "change_pct": 3.2},
                {"sector_name": "券商", "strength_score": 3.2, "change_pct": 2.9},
                {"sector_name": "科技", "strength_score": 3.0, "change_pct": 2.8},
                {"sector_name": "军工", "strength_score": 2.7, "change_pct": 2.6},
            ]
        }

        limit = scanner._dynamic_candidate_limit(ranked, ctx)
        assert limit == 15

    # -- collect_shared_data --

    def test_collect_shared_data_returns_expected_keys(self):
        scanner = MarketScanner()
        shared = scanner.collect_shared_data("2026-07-10")
        assert "market" in shared
        assert "sector_context" in shared
        assert "risk" in shared
        assert "st_status" in shared["risk"]
        assert "suspended" in shared["risk"]
        assert "delisting" in shared["risk"]

    def test_collect_shared_data_all_values_are_lists_or_dicts(self):
        scanner = MarketScanner()
        shared = scanner.collect_shared_data("2026-07-10")
        assert isinstance(shared["market"], list)
        assert isinstance(shared["sector_context"], list)
        assert isinstance(shared["risk"], dict)

    # -- collect_ticker_data --

    def test_collect_ticker_data_returns_expected_keys(self):
        scanner = MarketScanner()
        data = scanner.collect_ticker_data("000001.SZ", "2026-07-10")
        assert "daily" in data
        assert "capital_flow" in data
        assert "news" in data

    def test_collect_ticker_data_all_values_are_lists(self):
        scanner = MarketScanner()
        data = scanner.collect_ticker_data("000001.SZ", "2026-07-10")
        assert isinstance(data["daily"], list)
        assert isinstance(data["capital_flow"], list)
        assert isinstance(data["news"], list)

    def test_collect_ticker_data_enriches_cached_news_full_text(self, monkeypatch):
        class FakeResponse:
            status_code = 200
            apparent_encoding = "utf-8"
            encoding = "utf-8"
            text = (
                "<html><body>"
                "<p>Ping An Bank released a detailed operating update with retail banking evidence.</p>"
                "<p>The full article contains enough context for downstream trading agents.</p>"
                "</body></html>"
            )

            def raise_for_status(self):
                return None

        def fake_route(method, **kwargs):
            return []

        def fake_news(ticker, *, trade_date=None):
            return [{
                "title": "Ping An Bank operating update",
                "summary": "Short summary",
                "url": "https://example.test/news",
            }]

        def fake_get(url, **kwargs):
            assert url == "https://example.test/news"
            return FakeResponse()

        monkeypatch.setattr("advanced_trading_agent.data_agent.local_cache.get_cached_news", fake_news)
        monkeypatch.setattr("advanced_trading_agent.data_agent.news_text.requests.get", fake_get)

        scanner = MarketScanner(route_fn=fake_route)
        data = scanner.collect_ticker_data("000001.SZ", "2026-07-10")

        news = data["news"][0]
        assert news["full_text_source"] == "scanner"
        assert news["full_text_attempted"] is True
        assert news["content_status"] == "full_text"
        assert "downstream trading agents" in news["evidence_text"]

    # -- scan_and_collect --

    def test_scan_and_collect_returns_scan_bundle(self):
        scanner = MarketScanner(top_sectors=3, top_n=5)
        bundle = scanner.scan_and_collect("2026-07-10", top_n=3)
        assert isinstance(bundle, ScanBundle)
        assert bundle.trade_date == "2026-07-10"
        assert isinstance(bundle.results, list)
        assert isinstance(bundle.shared_raw, dict)
        assert isinstance(bundle.ticker_data, dict)
        assert isinstance(bundle.route_trace, list)

    def test_scan_and_collect_only_collects_top_n(self):
        scanner = MarketScanner(top_sectors=3, top_n=10)
        bundle = scanner.scan_and_collect("2026-07-10", top_n=3)
        # ticker_data should have at most top_n entries
        assert len(bundle.ticker_data) <= 3
        # each entry in ticker_data should correspond to a scan result
        result_tickers = {r.ticker for r in bundle.results[:3]}
        for ticker in bundle.ticker_data:
            assert ticker in result_tickers
    def test_scan_bundle_raw_for_ticker_returns_canonical_payload(self):
        bundle = ScanBundle(
            trade_date="2026-07-10",
            results=[],
            shared_raw={
                "market": [{"close": 3000}],
                "sector_context": [{"sector_name": "银行"}],
                "limit_up_summary": {"first_board": 1},
                "dragon_tiger": [{"code": "000001"}],
                "market_breadth": {"advance_count": 10},
                "risk": {"st_status": [], "suspended": [], "delisting": []},
            },
            ticker_data={
                "000001.SZ": {
                    "daily": [{"close": 10}],
                    "capital_flow": [{"net_mf_amount": 1}],
                    "news": [{"title": "news"}],
                }
            },
            route_trace=[{"method": "get_daily", "status": "success"}],
        )

        payload = bundle.raw_for_ticker("000001.SZ")

        assert payload["daily"] == [{"close": 10}]
        assert payload["market"] == [{"close": 3000}]
        assert payload["market_breadth"] == {"advance_count": 10}
        assert payload["news"] == [{"title": "news"}]
        assert payload["route_trace"] == bundle.route_trace

    def test_scan_and_collect_empty_results(self):
        """When scan finds nothing, scan_and_collect returns empty bundle."""
        scanner = MarketScanner(top_sectors=0, top_n=5)
        with patch.object(scanner, "scan", return_value=[]):
            bundle = scanner.scan_and_collect("2026-07-10", top_n=3)
        assert isinstance(bundle, ScanBundle)
        assert bundle.results == []
        assert bundle.ticker_data == {}

    # -- _safe_fetch --

    def test_safe_fetch_returns_list_on_error(self):
        """_safe_fetch should return empty list when vendor fails."""
        scanner = MarketScanner()
        trace: list = []
        result = scanner._safe_fetch("nonexistent_method", trace)
        assert isinstance(result, list)
        assert result == []

    def test_safe_fetch_records_error_in_trace(self):
        scanner = MarketScanner()
        trace: list = []
        scanner._safe_fetch("nonexistent_method", trace)
        errors = [t for t in trace if t.get("status") == "error"]
        assert len(errors) >= 1

    def test_safe_fetch_success(self):
        scanner = MarketScanner()
        mock_route = MagicMock(return_value=[{"close": 10.0}])
        scanner._route_fn = mock_route
        trace: list = []
        result = scanner._safe_fetch("get_daily", trace, code="000001.SZ")
        assert result == [{"close": 10.0}]
