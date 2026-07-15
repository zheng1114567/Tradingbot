"""Tests for explicit akshare data adapters."""

from __future__ import annotations

import pandas as pd

from advanced_trading_agent.data_agent import collector


def test_get_daily_akshare_normalizes_rows_and_writes_cache(monkeypatch):
    calls: dict[str, object] = {}

    class FakeAk:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            calls["hist_kwargs"] = kwargs
            return pd.DataFrame([
                {
                    "日期": "2026-07-10",
                    "开盘": "10.0",
                    "收盘": "10.5",
                    "最高": "10.8",
                    "最低": "9.9",
                    "成交量": "1000",
                    "成交额": "10500",
                    "涨跌幅": "5.0",
                    "换手率": "1.2",
                }
            ])

    saved: dict[str, object] = {}

    monkeypatch.setattr(collector, "_get_akshare", lambda: FakeAk)
    monkeypatch.setattr(collector, "_vendor_jitter", lambda *args, **kwargs: None)
    monkeypatch.setattr(collector, "call_with_vendor_guard", lambda _vendor, fn: fn())
    monkeypatch.setattr(
        "advanced_trading_agent.data_agent.local_cache.save_cached_daily",
        lambda ticker, records: saved.update({"ticker": ticker, "records": records}) or "cached",
    )

    rows = collector.get_daily_akshare("000001.SZ", start_date="2026-07-01", end_date="2026-07-10")

    assert calls["hist_kwargs"]["symbol"] == "000001"
    assert calls["hist_kwargs"]["start_date"] == "20260701"
    assert rows[0]["code"] == "000001.SZ"
    assert rows[0]["close"] == 10.5
    assert rows[0]["data_source"] == "akshare"
    assert saved["ticker"] == "000001.SZ"


def test_get_etf_spot_akshare_normalizes_rows_and_writes_cache(monkeypatch):
    class FakeAk:
        @staticmethod
        def fund_etf_spot_em():
            return pd.DataFrame([
                {
                    "代码": "512480",
                    "名称": "半导体ETF",
                    "最新价": "1.234",
                    "涨跌幅": "2.5",
                    "成交额": "50000000",
                    "折溢价率": "0.15",
                }
            ])

    saved: dict[str, object] = {}

    monkeypatch.setattr(collector, "_get_akshare", lambda: FakeAk)
    monkeypatch.setattr(collector, "_vendor_jitter", lambda *args, **kwargs: None)
    monkeypatch.setattr(collector, "call_with_vendor_guard", lambda _vendor, fn: fn())
    monkeypatch.setattr(
        "advanced_trading_agent.data_agent.local_cache.save_cached_etf_spot",
        lambda records, trade_date=None: saved.update({"records": records, "trade_date": trade_date}) or "cached",
    )

    rows = collector.get_etf_spot_akshare(trade_date="2026-07-10")

    assert rows[0]["code"] == "512480.SH"
    assert rows[0]["name"] == "半导体ETF"
    assert rows[0]["latest_price"] == 1.234
    assert rows[0]["amount"] == 50_000_000
    assert rows[0]["data_source"] == "akshare"
    assert saved["trade_date"] == "2026-07-10"


def test_get_etf_daily_eastmoney_normalizes_kline_and_writes_cache(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "klines": [
                        "2026-07-10,1.00,1.05,1.08,0.99,1000,10500,9.0,5.0,0.05,1.2"
                    ]
                }
            }

    calls: dict[str, object] = {}
    saved: dict[str, object] = {}

    def fake_get(url, **kwargs):
        calls["url"] = url
        calls["params"] = kwargs["params"]
        return FakeResponse()

    monkeypatch.setattr(collector, "_vendor_jitter", lambda *args, **kwargs: None)
    monkeypatch.setattr(collector.requests, "get", fake_get)
    monkeypatch.setattr(collector, "call_with_vendor_guard", lambda _vendor, fn: fn())
    monkeypatch.setattr(
        "advanced_trading_agent.data_agent.local_cache.save_cached_etf_daily",
        lambda code, records: saved.update({"code": code, "records": records}) or "cached",
    )

    rows = collector.get_etf_daily_eastmoney("512480.SH", start_date="2026-07-10", end_date="2026-07-10")

    assert calls["params"]["secid"] == "1.512480"
    assert rows[0]["code"] == "512480.SH"
    assert rows[0]["close"] == 1.05
    assert rows[0]["amount"] == 10500
    assert rows[0]["data_source"] == "eastmoney"
    assert saved["code"] == "512480.SH"


def test_get_etf_daily_sina_filters_dates_and_writes_cache(monkeypatch):
    class FakeAk:
        @staticmethod
        def fund_etf_hist_sina(symbol):
            assert symbol == "sh512480"
            return pd.DataFrame([
                {"date": "2026-07-09", "open": 0.9, "high": 1.0, "low": 0.8, "close": 0.95, "volume": 100, "amount": 95},
                {"date": "2026-07-10", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000, "amount": 1050},
            ])

    saved: dict[str, object] = {}

    monkeypatch.setattr(collector, "_get_akshare", lambda: FakeAk)
    monkeypatch.setattr(collector, "_vendor_jitter", lambda *args, **kwargs: None)
    monkeypatch.setattr(collector, "call_with_vendor_guard", lambda _vendor, fn: fn())
    monkeypatch.setattr(
        "advanced_trading_agent.data_agent.local_cache.save_cached_etf_daily",
        lambda code, records: saved.update({"code": code, "records": records}) or "cached",
    )

    rows = collector.get_etf_daily_sina("512480.SH", start_date="2026-07-10", end_date="2026-07-10")

    assert len(rows) == 1
    assert rows[0]["code"] == "512480.SH"
    assert rows[0]["close"] == 1.05
    assert rows[0]["data_source"] == "sina"
    assert saved["code"] == "512480.SH"
