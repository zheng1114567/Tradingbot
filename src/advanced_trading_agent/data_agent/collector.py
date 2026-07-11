"""Free-first data collection adapters.

Default vendor order uses API-key-free data sources:
akshare -> baostock -> yfinance.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .vendor_router import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
    register_vendor_impl,
)

logger = logging.getLogger(__name__)


def _fmt_yyyymmdd(value: str | None, default: date) -> str:
    return (value or default.strftime("%Y%m%d")).replace("-", "")


def _fmt_iso(value: str | None, default: date) -> str:
    raw = value or default.isoformat()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw


def _digits(code: str) -> str:
    return code.split(".")[0].replace("sh", "").replace("sz", "")


def _market_suffix(code: str) -> str:
    upper = code.upper()
    if upper.endswith(".SH") or upper.startswith("SH"):
        return "sh"
    if upper.endswith(".SZ") or upper.startswith("SZ"):
        return "sz"
    if upper.endswith(".BJ") or upper.startswith("BJ"):
        return "bj"
    digits = _digits(code)
    if digits.startswith(("5", "6", "9")):
        return "sh"
    return "sz"


def _baostock_code(code: str) -> str:
    return f"{_market_suffix(code)}.{_digits(code)}"


def _ak_index_symbol(code: str) -> str | None:
    digits = _digits(code)
    if digits in {"000001", "000300", "000905", "000852"} and _market_suffix(code) == "sh":
        return f"sh{digits}"
    if digits.startswith(("399", "159")) and _market_suffix(code) == "sz":
        return f"sz{digits}"
    return None


def _with_source(records: list[dict[str, Any]], source: str, code: str = "") -> list[dict[str, Any]]:
    for record in records:
        record.setdefault("data_source", source)
        if code:
            record.setdefault("code", code)
    return records


def _get_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError as exc:
        raise VendorNotConfiguredError("akshare not installed (pip install akshare)", vendor="akshare") from exc


def _get_baostock():
    try:
        import baostock as bs
        return bs
    except ImportError as exc:
        raise VendorNotConfiguredError("baostock not installed (pip install baostock)", vendor="baostock") from exc


def _get_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError as exc:
        raise VendorNotConfiguredError("yfinance not installed (pip install yfinance)", vendor="yfinance") from exc


def get_daily_akshare(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Daily OHLCV from AkShare for A-share stocks and major indexes."""

    ak = _get_akshare()
    end = _fmt_yyyymmdd(end_date, date.today())
    start = _fmt_yyyymmdd(start_date, date.today() - timedelta(days=365))
    symbol = _digits(code)
    try:
        index_symbol = _ak_index_symbol(code)
        if index_symbol:
            df = ak.stock_zh_index_daily(symbol=index_symbol)
            if df is not None and not df.empty:
                df = df.rename(columns={"date": "trade_date"})
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                start_ts = pd.to_datetime(start)
                end_ts = pd.to_datetime(end)
                df = df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)]
                if "amount" not in df.columns and {"close", "volume"}.issubset(df.columns):
                    df["amount"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(
                        df["volume"], errors="coerce"
                    )
                records = df.to_dict("records")
                if records:
                    return _with_source(records, "akshare", code)

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df is None or df.empty:
            raise NoMarketDataError(f"No daily data for {code}", symbol=code, vendor="akshare")
        return _with_source(df.to_dict("records"), "akshare", code)
    except NoMarketDataError:
        raise
    except Exception as exc:
        logger.warning("akshare get_daily failed for %s: %s", code, exc)
        raise NoMarketDataError(str(exc), symbol=code, vendor="akshare") from exc


def get_daily_baostock(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Daily OHLCV from BaoStock as a free A-share fallback."""

    bs = _get_baostock()
    start = _fmt_iso(start_date, date.today() - timedelta(days=365))
    end = _fmt_iso(end_date, date.today())
    login_result = bs.login()
    if getattr(login_result, "error_code", "0") != "0":
        raise VendorRateLimitError(getattr(login_result, "error_msg", "baostock login failed"), vendor="baostock")
    try:
        rs = bs.query_history_k_data_plus(
            _baostock_code(code),
            "date,code,open,high,low,close,preclose,volume,amount,pctChg,turn",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",
        )
        rows: list[dict[str, Any]] = []
        while getattr(rs, "error_code", "0") == "0" and rs.next():
            rows.append(dict(zip(rs.fields, rs.get_row_data())))
        if not rows:
            raise NoMarketDataError(f"No daily data for {code}", symbol=code, vendor="baostock")
        return _with_source(rows, "baostock", code)
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def get_daily_yfinance(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Daily OHLCV from Yahoo Finance for cross-market fallback."""

    yf = _get_yfinance()
    start = _fmt_iso(start_date, date.today() - timedelta(days=365))
    end = _fmt_iso(end_date, date.today() + timedelta(days=1))
    try:
        df = yf.download(code, start=start, end=end, progress=False, auto_adjust=False)
        if df is None or df.empty:
            raise NoMarketDataError(f"No daily data for {code}", symbol=code, vendor="yfinance")
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        if "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])
        if {"Close", "Volume"}.issubset(df.columns):
            df["amount"] = pd.to_numeric(df["Close"], errors="coerce") * pd.to_numeric(
                df["Volume"], errors="coerce"
            )
            df["amount_estimated"] = True
        return _with_source(df.to_dict("records"), "yfinance", code)
    except NoMarketDataError:
        raise
    except Exception as exc:
        logger.warning("yfinance get_daily failed for %s: %s", code, exc)
        raise NoMarketDataError(str(exc), symbol=code, vendor="yfinance") from exc


def get_capital_flow_akshare(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    trade_date: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Individual A-share capital flow from AkShare."""

    if not code:
        return {
            "net_inflow_main": 0,
            "confirmation": "未知",
            "data_source": "akshare",
            "note": "capital flow requires a stock code",
        }
    ak = _get_akshare()
    symbol = _digits(code)
    try:
        df = ak.stock_individual_fund_flow(stock=symbol, market="sh" if _market_suffix(code) == "sh" else "sz")
        if df is None or df.empty:
            raise NoMarketDataError(f"No capital flow for {code}", symbol=code, vendor="akshare")
        records = _with_source(df.to_dict("records"), "akshare", code)
        return records
    except NoMarketDataError:
        raise
    except Exception as exc:
        logger.warning("akshare capital_flow failed for %s: %s", code, exc)
        raise NoMarketDataError(str(exc), symbol=code, vendor="akshare") from exc


def get_news_akshare(sector: str | None = None, keyword: str | None = None) -> list[dict[str, Any]]:
    ak = _get_akshare()
    try:
        df = ak.stock_info_global()
        if df is not None and not df.empty:
            records = _with_source(df.to_dict("records"), "akshare")
            if keyword:
                records = [record for record in records if keyword in str(record)]
            return records
    except Exception as exc:
        logger.warning("akshare news failed: %s", exc)
    return []


def get_sector_akshare(top_n: int = 10) -> list[dict[str, Any]]:
    ak = _get_akshare()
    try:
        df = ak.stock_board_concept_name_em()
        if df is None or df.empty:
            return []
        records = []
        for idx, row in enumerate(df.head(top_n).to_dict("records"), start=1):
            change = row.get("涨跌幅", row.get("change_pct", 0))
            try:
                change_pct = float(change)
            except (TypeError, ValueError):
                change_pct = 0.0
            records.append({
                **row,
                "rank": idx,
                "sector_name": row.get("板块名称", row.get("sector_name", "")),
                "change_pct": change_pct,
                "strength_score": change_pct,
                "data_source": "akshare",
            })
        return records
    except Exception as exc:
        logger.warning("akshare sector failed: %s", exc)
        return []


def get_financial_akshare(code: str) -> list[dict[str, Any]]:
    """Best-effort free financial indicators from AkShare."""

    ak = _get_akshare()
    symbol = _digits(code)
    for fn_name in ("stock_financial_analysis_indicator", "stock_financial_abstract"):
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            df = fn(symbol=symbol)
            if df is not None and not df.empty:
                return _with_source(df.to_dict("records"), "akshare", code)
        except Exception as exc:
            logger.debug("akshare %s failed for %s: %s", fn_name, code, exc)
    raise NoMarketDataError(f"No financial data for {code}", symbol=code, vendor="akshare")


def get_st_status_akshare() -> list[str]:
    ak = _get_akshare()
    try:
        fn = getattr(ak, "stock_zh_a_st_em", None)
        if fn is None:
            return []
        df = fn()
        if df is None or df.empty:
            return []
        code_col = "代码" if "代码" in df.columns else "code"
        return [str(code) for code in df[code_col].dropna().tolist()]
    except Exception as exc:
        logger.warning("akshare ST list failed: %s", exc)
        return []


def get_suspended_baostock(trade_date: str | None = None) -> list[str]:
    bs = _get_baostock()
    day = _fmt_iso(trade_date, date.today())
    login_result = bs.login()
    if getattr(login_result, "error_code", "0") != "0":
        raise VendorRateLimitError(getattr(login_result, "error_msg", "baostock login failed"), vendor="baostock")
    try:
        rs = bs.query_all_stock(day=day)
        suspended: list[str] = []
        while getattr(rs, "error_code", "0") == "0" and rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            status = row.get("tradeStatus", row.get("tradestatus", "1"))
            if status not in {"1", "交易"}:
                suspended.append(str(row.get("code", "")))
        return suspended
    except Exception as exc:
        logger.warning("baostock suspended list failed: %s", exc)
        return []
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def get_suspended_akshare(trade_date: str | None = None) -> list[str]:
    """AkShare does not expose one stable suspended-list endpoint; return empty best effort."""

    return []


def get_delisting_akshare() -> list[str]:
    """Best-effort delisting risk list using free AkShare ST data."""

    return get_st_status_akshare()


def get_northbound_flow_akshare(trade_date: str | None = None) -> dict[str, Any]:
    ak = _get_akshare()
    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df is not None and not df.empty:
            record = df.to_dict("records")[-1]
            record["data_source"] = "akshare"
            return record
    except Exception as exc:
        logger.warning("akshare northbound failed: %s", exc)
    return {"net_inflow": 0, "data_source": "akshare", "note": "northbound data unavailable"}


def get_limit_up_tiers_akshare(trade_date: str | None = None) -> dict[str, int]:
    td = _fmt_yyyymmdd(trade_date, date.today())
    try:
        ak = _get_akshare()
        df = ak.stock_zt_pool_em(date=td)
        if df is None or df.empty:
            return {"first_board": 0, "second_board": 0, "third_plus": 0}

        first_board = 0
        second_board = 0
        third_plus = 0
        for record in df.to_dict("records"):
            board_count = record.get("连板数", record.get("连续涨停", 1))
            try:
                board_count = int(board_count)
            except (TypeError, ValueError):
                board_count = 1
            if board_count <= 1:
                first_board += 1
            elif board_count == 2:
                second_board += 1
            else:
                third_plus += 1
        return {
            "first_board": first_board,
            "second_board": second_board,
            "third_plus": third_plus,
            "data_source": "akshare",
        }
    except Exception as exc:
        logger.warning("limit-up tiers failed: %s", exc)
        return {"first_board": 0, "second_board": 0, "third_plus": 0, "data_source": "akshare"}


def get_dragon_tiger_akshare(trade_date: str | None = None) -> list[dict[str, Any]]:
    ak = _get_akshare()
    td = _fmt_yyyymmdd(trade_date, date.today())
    for fn_name in ("stock_lhb_detail_em", "stock_lhb_stock_statistic_em"):
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            df = fn(date=td)
            if df is not None and not df.empty:
                return _with_source(df.head(20).to_dict("records"), "akshare")
        except Exception as exc:
            logger.debug("akshare %s failed: %s", fn_name, exc)
    return []


def get_margin_akshare(trade_date: str | None = None) -> list[dict[str, Any]]:
    ak = _get_akshare()
    try:
        df = ak.stock_margin_detail_sse(date=_fmt_yyyymmdd(trade_date, date.today()))
        if df is not None and not df.empty:
            return _with_source(df.head(20).to_dict("records"), "akshare")
    except Exception as exc:
        logger.warning("akshare margin failed: %s", exc)
    return []


def get_factors_computed(code: str = "", sector: str = "") -> list[dict[str, Any]]:
    if not code:
        return []

    from .cleaner import DataCleaner
    from .factors import FactorCalculator
    from .vendor_router import route_to_vendor

    try:
        daily = route_to_vendor("get_daily", code=code)
        if isinstance(daily, str) or not daily:
            return []
        df = DataCleaner.clean_daily(daily)
        if df.empty:
            return []
        df = FactorCalculator.run_all(df)
        latest = df.iloc[-1].to_dict()
        return [{
            "code": code,
            "name": latest.get("name", ""),
            "sector": sector,
            "quality_score": latest.get("roe"),
            "growth_score": latest.get("revenue_growth"),
            "valuation_score": latest.get("pe_quantile", latest.get("pb_quantile")),
            "momentum_score": latest.get("momentum_20d"),
            "volatility_score": latest.get("volatility"),
            "liquidity_score": latest.get("amihud"),
            "composite_score": latest.get("composite_score"),
            "factor_warning": None,
            "data_source": latest.get("data_source", ""),
        }]
    except Exception as exc:
        logger.warning("factor calculation failed for %s: %s", code, exc)
        return []


def check_crowding_stub(sector: str = "") -> dict[str, Any]:
    return {"is_crowded": False, "warnings": [], "data_source": "deterministic_stub"}


def find_similar_stub(sentiment: str = "", sector: str = "", event_type: str = "") -> dict[str, Any]:
    return {
        "sample_size": 0,
        "win_rate": 0,
        "avg_excess_return": 0,
        "confidence": "low",
        "data_source": "deterministic_stub",
    }


def register_all_vendors() -> None:
    """Register free vendor adapters."""

    register_vendor_impl("get_daily", "akshare", get_daily_akshare)
    register_vendor_impl("get_daily", "baostock", get_daily_baostock)
    register_vendor_impl("get_daily", "yfinance", get_daily_yfinance)

    register_vendor_impl("get_capital_flow", "akshare", get_capital_flow_akshare)
    register_vendor_impl("get_news", "akshare", get_news_akshare)
    register_vendor_impl("get_sector", "akshare", get_sector_akshare)
    register_vendor_impl("get_financial", "akshare", get_financial_akshare)

    register_vendor_impl("get_suspended", "akshare", get_suspended_akshare)
    register_vendor_impl("get_suspended", "baostock", get_suspended_baostock)
    register_vendor_impl("get_st_status", "akshare", get_st_status_akshare)
    register_vendor_impl("get_delisting", "akshare", get_delisting_akshare)

    register_vendor_impl("get_northbound_flow", "akshare", get_northbound_flow_akshare)
    register_vendor_impl("get_limit_up_tiers", "akshare", get_limit_up_tiers_akshare)
    register_vendor_impl("get_dragon_tiger", "akshare", get_dragon_tiger_akshare)
    register_vendor_impl("get_margin", "akshare", get_margin_akshare)

    register_vendor_impl("get_factors", "akshare", get_factors_computed)
    register_vendor_impl("get_factors", "baostock", get_factors_computed)
    register_vendor_impl("get_factors", "yfinance", get_factors_computed)
    register_vendor_impl("check_crowding", "akshare", check_crowding_stub)
    register_vendor_impl("find_similar", "akshare", find_similar_stub)


register_all_vendors()
