"""Free-first data collection adapters.

Default vendor order uses API-key-free A-share data sources:
akshare -> baostock -> sina/eastmoney fallbacks.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

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


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", "").replace("%", "").strip()
        if value in {"", "-", "--", "None", "nan"}:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value or "")
    return re.sub(r"\s+", " ", text).strip()


def _http_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }


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


def get_news_akshare(
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    days: int = 2,
    limit: int = 50,
    include_announcements: bool = True,
) -> list[dict[str, Any]]:
    ak = _get_akshare()
    code6 = _digits(code or keyword or sector or "")
    records: list[dict[str, Any]] = []
    if not code6:
        raise NoMarketDataError("AkShare news requires a stock code", symbol=code or "", vendor="akshare")

    try:
        df = ak.stock_news_em(symbol=code6)
        if df is not None and not df.empty:
            for row in df.head(limit).to_dict("records"):
                records.append({
                    "title": row.get("新闻标题") or row.get("标题") or row.get("title") or "",
                    "summary": row.get("新闻内容") or row.get("内容") or row.get("summary") or row.get("新闻标题") or "",
                    "source": row.get("文章来源") or row.get("来源") or row.get("source") or "akshare",
                    "time": row.get("发布时间") or row.get("time") or "",
                    "url": row.get("新闻链接") or row.get("url") or "",
                    "type": "news",
                    "code": code,
                    "data_source": "akshare",
                })
    except Exception as exc:
        logger.warning("akshare stock_news_em failed for %s: %s", code6, exc)

    if include_announcements and len(records) < limit:
        try:
            fn = (
                getattr(ak, "stock_announcement_em", None)
                or getattr(ak, "stock_individual_notice_report", None)
                or getattr(ak, "stock_notice_report", None)
            )
            if fn is None:
                df = None
            elif getattr(fn, "__name__", "") == "stock_individual_notice_report":
                df = fn(security=code6, symbol="全部")
            elif getattr(fn, "__name__", "") == "stock_notice_report":
                df = fn(symbol="全部")
            else:
                df = fn(symbol=code6)
            if df is not None and not df.empty:
                for row in df.head(max(0, limit - len(records))).to_dict("records"):
                    records.append({
                        "title": (
                            row.get("公告标题")
                            or row.get("title")
                            or row.get("公告名称")
                            or row.get("title_ch")
                            or ""
                        ),
                        "summary": row.get("公告标题") or row.get("summary") or row.get("公告名称") or "",
                        "source": "akshare",
                        "time": row.get("公告时间") or row.get("time") or row.get("公告日期") or "",
                        "url": row.get("公告链接") or row.get("url") or row.get("网址") or "",
                        "type": "announcement",
                        "code": code,
                        "data_source": "akshare",
                    })
        except Exception as exc:
            logger.warning("akshare stock_announcement_em failed for %s: %s", code6, exc)

    if keyword:
        records = [record for record in records if keyword.lower() in json.dumps(record, ensure_ascii=False).lower()]
    if not records:
        raise NoMarketDataError(f"No AkShare news for {code or keyword or sector}", symbol=code or "", vendor="akshare")
    return records[:limit]


def get_news_sina(
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    days: int = 2,
    limit: int = 50,
    include_announcements: bool = True,
) -> list[dict[str, Any]]:
    if not code:
        raise NoMarketDataError("Sina news requires a stock code", symbol="", vendor="sina")

    symbol = f"{_market_suffix(code)}{_digits(code)}"
    urls = [
        (
            "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php",
            {"symbol": symbol},
        ),
        (
            f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{symbol}.phtml",
            None,
        ),
    ]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_error: Exception | None = None
    for url, params in urls:
        try:
            response = requests.get(url, params=params, headers=_http_headers(), timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding or "gb18030"
            text = response.text
        except Exception as exc:
            last_error = exc
            logger.warning("sina news failed for %s: %s", symbol, exc)
            continue

        for match in re.finditer(
            r"<a[^>]+href=[\"'](?P<url>https?://[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            title = _strip_html(match.group("title"))
            link = match.group("url").strip()
            if len(title) < 4 or link in seen:
                continue
            candidate = {
                "title": title,
                "summary": title,
                "source": "sina",
                "time": "",
                "url": link,
                "type": "news",
                "code": code,
                "data_source": "sina",
            }
            if keyword and keyword.lower() not in json.dumps(candidate, ensure_ascii=False).lower():
                continue
            seen.add(link)
            records.append(candidate)
            if len(records) >= limit:
                return records

    if not records:
        detail = f": {last_error}" if last_error else ""
        raise NoMarketDataError(f"No Sina news for {code}{detail}", symbol=code, vendor="sina")
    return records[:limit]


def get_sector_akshare(top_n: int = 10) -> list[dict[str, Any]]:
    ak = _get_akshare()
    try:
        df = ak.stock_board_concept_name_em()
        if df is None or df.empty:
            raise NoMarketDataError("No AkShare sector data", vendor="akshare")
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
        if not records:
            raise NoMarketDataError("No AkShare sector data", vendor="akshare")
        return records
    except Exception as exc:
        logger.warning("akshare sector failed: %s", exc)
        raise NoMarketDataError(str(exc), vendor="akshare") from exc


def get_sector_eastmoney(top_n: int = 10) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    specs = [
        ("industry", "m:90+t:2+f:!50"),
        ("concept", "m:90+t:3+f:!50"),
    ]
    last_error: Exception | None = None
    for sector_type, fs in specs:
        try:
            response = requests.get(
                "https://push2.eastmoney.com/api/qt/clist/get",
                params={
                    "pn": 1,
                    "pz": max(top_n, 20),
                    "po": 1,
                    "np": 1,
                    "fltt": 2,
                    "invt": 2,
                    "fid": "f3",
                    "fs": fs,
                    "fields": "f12,f14,f3,f62,f128,f136,f152",
                    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                },
                headers=_http_headers(),
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            rows = (payload.get("data") or {}).get("diff") or []
            for row in rows:
                change_pct = _float_or_none(row.get("f3")) or 0.0
                records.append({
                    "rank": len(records) + 1,
                    "sector_code": row.get("f12", ""),
                    "sector_name": row.get("f14", ""),
                    "sector_type": sector_type,
                    "change_pct": change_pct,
                    "strength_score": change_pct,
                    "net_inflow_main": _float_or_none(row.get("f62")),
                    "data_source": "eastmoney",
                })
        except Exception as exc:
            last_error = exc
            logger.warning("eastmoney sector failed for %s: %s", sector_type, exc)
            continue

    if not records:
        detail = f": {last_error}" if last_error else ""
        raise NoMarketDataError(f"No Eastmoney sector data{detail}", vendor="eastmoney")
    return records[:top_n]


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
        fn = (
            getattr(ak, "stock_hsgt_north_net_flow_in_em", None)
            or getattr(ak, "stock_hsgt_hist_em", None)
            or getattr(ak, "stock_hsgt_fund_flow_summary_em", None)
        )
        if fn is None:
            return {"net_inflow": 0, "data_source": "akshare", "note": "northbound function unavailable"}
        if getattr(fn, "__name__", "") == "stock_hsgt_hist_em":
            df = fn(symbol="北向资金")
        else:
            df = fn()
        if df is not None and not df.empty:
            record = df.to_dict("records")[-1]
            record["data_source"] = "akshare"
            return record
    except Exception as exc:
        logger.warning("akshare northbound failed: %s", exc)
    return {"net_inflow": 0, "data_source": "akshare", "note": "northbound data unavailable"}


def get_limit_up_tiers_akshare(trade_date: str | None = None) -> dict[str, Any]:
    td = _fmt_yyyymmdd(trade_date, date.today())
    try:
        ak = _get_akshare()
        df = ak.stock_zt_pool_em(date=td)
        if df is None or df.empty:
            return {"first_board": 0, "second_board": 0, "third_plus": 0, "stocks": []}

        first_board = 0
        second_board = 0
        third_plus = 0
        stocks: list[dict[str, Any]] = []
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
            stocks.append({
                "code": str(record.get("代码", "")),
                "name": str(record.get("名称", "")),
                "board_count": board_count,
                "turnover": _float_or_none(record.get("换手率", record.get("换手", 0))),
                "change_pct": _float_or_none(record.get("涨跌幅", 0)),
            })
        return {
            "first_board": first_board,
            "second_board": second_board,
            "third_plus": third_plus,
            "stocks": stocks,
            "data_source": "akshare",
        }
    except Exception as exc:
        logger.warning("limit-up tiers failed: %s", exc)
        return {"first_board": 0, "second_board": 0, "third_plus": 0, "stocks": [], "data_source": "akshare"}


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


def get_northbound_top10_akshare(trade_date: str | None = None) -> list[dict[str, Any]]:
    """Top 10 northbound-bought stocks via akshare (hsgt_top10_em)."""
    ak = _get_akshare()
    try:
        fn = getattr(ak, "stock_hsgt_top10_em", None)
        if fn is None:
            return []
        td = _fmt_yyyymmdd(trade_date, date.today())
        df = fn(date=td)
        if df is not None and not df.empty:
            results: list[dict[str, Any]] = []
            for record in df.head(20).to_dict("records"):
                results.append({
                    "code": str(record.get("代码", "")),
                    "name": str(record.get("名称", "")),
                    "net_buy": _float_or_none(record.get("净买入", record.get("净买入额", 0))),
                    "change_pct": _float_or_none(record.get("涨跌幅", 0)),
                })
            return results
    except Exception as exc:
        logger.warning("akshare northbound top10 failed: %s", exc)
    return []


def get_sector_constituents_akshare(sector_name: str = "") -> list[dict[str, Any]]:
    """Get constituent stocks of a concept/industry board via akshare."""
    if not sector_name:
        return []
    ak = _get_akshare()
    for fn_name in ("stock_board_concept_cons_em", "stock_board_industry_cons_em"):
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            df = fn(symbol=sector_name)
            if df is not None and not df.empty:
                results: list[dict[str, Any]] = []
                for record in df.head(30).to_dict("records"):
                    results.append({
                        "code": str(record.get("代码", "")),
                        "name": str(record.get("名称", "")),
                        "sector": sector_name,
                    })
                return results
        except Exception as exc:
            logger.debug("akshare %s(%s) failed: %s", fn_name, sector_name, exc)
    return []


# ------------------------------------------------------------------
# efinance-based adapters (fallback when eastmoney push2 endpoints are blocked)
# ------------------------------------------------------------------

# Representative A-share stocks used to build sector rankings via
# get_belong_board aggregation. One stock from each major sector.
_EFINANCE_PROBE_STOCKS: list[str] = []


def _get_probe_stocks() -> list[str]:
    """Lazy-init a diverse set of probe stocks for sector discovery."""
    global _EFINANCE_PROBE_STOCKS
    if _EFINANCE_PROBE_STOCKS:
        return _EFINANCE_PROBE_STOCKS
    _EFINANCE_PROBE_STOCKS = [
        '000001', '000002', '000858', '002415', '300750', '600519', '601398',
        '600036', '000333', '002594', '300059', '600030', '601857', '688981',
        '000725', '002230', '600276', '601318', '000651', '603259', '600900',
        '300124', '601012', '600104', '000568', '002475', '300433', '600809',
        '002714', '300498', '600585', '000063', '688111', '601899', '600050',
        '002352', '300015', '600887', '601088', '600028', '000792', '600941',
    ]
    return _EFINANCE_PROBE_STOCKS


def get_sector_efinance(top_n: int = 10) -> list[dict[str, Any]]:
    """Sector ranking via efinance board membership aggregation.

    Calls get_belong_board() for a diverse set of probe stocks and
    aggregates board change percentages. Uses the push2 slist endpoint
    which is typically not blocked by DPI (unlike clist).
    """
    try:
        import efinance as ef
    except ImportError:
        raise VendorNotConfiguredError("efinance not installed (pip install efinance)", vendor="efinance")

    from collections import defaultdict

    board_changes: dict[str, list[float]] = defaultdict(list)
    for code in _get_probe_stocks():
        try:
            df = ef.stock.get_belong_board(code)
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                board_name = str(row.get("板块名称", ""))
                chg = row.get("板块涨幅", None)
                if board_name and chg is not None:
                    try:
                        board_changes[board_name].append(float(chg))
                    except (TypeError, ValueError):
                        pass
        except Exception:
            continue

    rankings: list[dict[str, Any]] = []
    for name, changes in board_changes.items():
        if len(changes) >= 2:
            avg = sum(changes) / len(changes)
            rankings.append({
                "sector_name": name,
                "change_pct": round(avg, 2),
                "strength_score": round(avg, 2),
                "data_source": "efinance",
                "rank": 0,
            })

    rankings.sort(key=lambda x: x["change_pct"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    if not rankings:
        raise NoMarketDataError("No sector data from efinance probe stocks", vendor="efinance")
    return rankings[:top_n]


def get_dragon_tiger_efinance(trade_date: str | None = None) -> list[dict[str, Any]]:
    """Dragon-tiger list via efinance (uses datacenter API, not push2)."""
    try:
        import efinance as ef
    except ImportError:
        raise VendorNotConfiguredError("efinance not installed (pip install efinance)", vendor="efinance")

    td = trade_date or date.today().isoformat()
    try:
        df = ef.stock.get_daily_billboard(start_date=td, end_date=td)
        if df is None or df.empty:
            return []
        results: list[dict[str, Any]] = []
        for record in df.head(30).to_dict("records"):
            results.append({
                "code": str(record.get("股票代码", record.get("code", ""))),
                "name": str(record.get("股票名称", record.get("name", ""))),
                "close": _float_or_none(record.get("收盘价", record.get("close", 0))),
                "change_pct": _float_or_none(record.get("涨跌幅", 0)),
                "turnover": _float_or_none(record.get("换手率", 0)),
                "net_buy": _float_or_none(record.get("龙虎榜净买额", 0)),
                "reason": str(record.get("上榜原因", "")),
                "data_source": "efinance",
            })
        return results
    except Exception as exc:
        logger.warning("efinance dragon_tiger failed: %s", exc)
        raise NoMarketDataError(str(exc), vendor="efinance") from exc


# Cache for efinance reverse board index (board_name → [{code, name}, ...])
_EFINANCE_BOARD_INDEX: dict[str, list[dict[str, str]]] | None = None


def _build_board_index() -> dict[str, list[dict[str, str]]]:
    """Build board→stocks reverse index from efinance get_belong_board.

    Queries a broad set of A-share stocks and aggregates which stocks
    belong to each board. Cached in memory for the session lifetime.
    """
    global _EFINANCE_BOARD_INDEX
    if _EFINANCE_BOARD_INDEX is not None:
        return _EFINANCE_BOARD_INDEX

    try:
        import efinance as ef
    except ImportError:
        _EFINANCE_BOARD_INDEX = {}
        return _EFINANCE_BOARD_INDEX

    from collections import defaultdict

    # Use probe stocks + expand with more for broader coverage
    codes = list(_get_probe_stocks())
    # Add more stocks for better sector coverage
    extra = [
        '600000', '600016', '600031', '600048', '600050', '600085', '600111',
        '600150', '600196', '600309', '600406', '600436', '600438', '600519',
        '600570', '600588', '600690', '600703', '600745', '600809', '600837',
        '600887', '600893', '600900', '601006', '601012', '601088', '601111',
        '601166', '601318', '601328', '601398', '601668', '601857', '601939',
        '603160', '603259', '603288', '603501', '603986', '688008', '688012',
        '688036', '688111', '688126', '688169', '688187', '688981',
        '000063', '000066', '000333', '000568', '000596', '000651', '000725',
        '000858', '000876', '000895', '002007', '002049', '002142', '002230',
        '002241', '002271', '002304', '002352', '002415', '002460', '002475',
        '002594', '002714', '300003', '300015', '300059', '300124', '300347',
        '300433', '300498', '300750', '300760',
    ]
    codes.extend(extra)
    codes = list(dict.fromkeys(codes))  # deduplicate

    board_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for code in codes:
        try:
            df = ef.stock.get_belong_board(code)
            if df is None or df.empty:
                continue
            name = str(df.iloc[0].get("股票名称", "")) if len(df) > 0 else ""
            for _, row in df.iterrows():
                board_name = str(row.get("板块名称", ""))
                if board_name:
                    board_index[board_name].append({
                        "code": code,
                        "name": name,
                        "sector": board_name,
                    })
        except Exception:
            continue

    _EFINANCE_BOARD_INDEX = dict(board_index)
    logger.info("efinance board index: %d boards from %d stocks", len(_EFINANCE_BOARD_INDEX), len(codes))
    return _EFINANCE_BOARD_INDEX


def get_sector_constituents_efinance(sector_name: str = "") -> list[dict[str, Any]]:
    """Get board members via efinance reverse index.

    Builds a board→stocks mapping by calling get_belong_board() for
    ~100 representative A-share stocks and caching the result.
    """
    if not sector_name:
        return []

    try:
        import efinance as ef
    except ImportError:
        raise VendorNotConfiguredError("efinance not installed (pip install efinance)", vendor="efinance")

    board_index = _build_board_index()
    constituents = board_index.get(sector_name, [])

    if not constituents:
        # Fuzzy match: try partial name matching
        for bname, stocks in board_index.items():
            if sector_name in bname or bname in sector_name:
                constituents = stocks
                break

    if not constituents:
        raise NoMarketDataError(
            f"No constituents for board '{sector_name}' in efinance index",
            vendor="efinance",
        )

    return [dict(c) for c in constituents]


# ------------------------------------------------------------------
# local_cache adapters (baostock-based, zero eastmoney dependency)
# ------------------------------------------------------------------


def get_sector_local(top_n: int = 10) -> list[dict[str, Any]]:
    """Sector ranking from local baostock cache."""
    from .local_cache import get_cached_sector_data

    try:
        data = get_cached_sector_data(top_n=top_n)
        if not data:
            raise NoMarketDataError("Local sector cache is empty — run build_full_cache first", vendor="local_cache")
        return data
    except ImportError:
        raise VendorNotConfiguredError("local_cache requires baostock", vendor="local_cache")


def get_sector_constituents_local(sector_name: str = "") -> list[dict[str, Any]]:
    """Sector constituents from local baostock cache."""
    from .local_cache import get_cached_sector_constituents

    try:
        data = get_cached_sector_constituents(sector_name)
        if not data:
            raise NoMarketDataError(f"No cached constituents for {sector_name}", vendor="local_cache")
        return data
    except ImportError:
        raise VendorNotConfiguredError("local_cache requires baostock", vendor="local_cache")


def get_dragon_tiger_local(trade_date: str | None = None) -> list[dict[str, Any]]:
    """Dragon-tiger from local cache."""
    from .local_cache import get_cached_dragon_tiger

    data = get_cached_dragon_tiger(trade_date)
    if not data:
        raise NoMarketDataError("No cached dragon-tiger data — run build_cache first", vendor="local_cache")
    return data


def get_limit_up_tiers_local(trade_date: str | None = None) -> dict[str, Any]:
    """Limit-up tiers from local cache."""
    from .local_cache import get_cached_limit_up

    data = get_cached_limit_up(trade_date)
    if not data or not data.get("stocks"):
        raise NoMarketDataError("No cached limit-up data — run build_cache first", vendor="local_cache")
    return data
    """Daily OHLCV from local baostock parquet cache."""
    from .local_cache import get_cached_daily

    try:
        data = get_cached_daily(code, start_date, end_date)
        if not data:
            raise NoMarketDataError(f"No cached daily data for {code}", symbol=code, vendor="local_cache")
        return data
    except ImportError:
        raise VendorNotConfiguredError("local_cache requires baostock", vendor="local_cache")


def register_all_vendors() -> None:
    """Register free vendor adapters."""

    register_vendor_impl("get_daily", "akshare", get_daily_akshare)
    register_vendor_impl("get_daily", "baostock", get_daily_baostock)
    register_vendor_impl("get_daily", "local_cache", get_daily_local)

    register_vendor_impl("get_capital_flow", "akshare", get_capital_flow_akshare)
    register_vendor_impl("get_news", "akshare", get_news_akshare)
    register_vendor_impl("get_news", "sina", get_news_sina)
    register_vendor_impl("get_sector", "akshare", get_sector_akshare)
    register_vendor_impl("get_sector", "eastmoney", get_sector_eastmoney)
    register_vendor_impl("get_sector", "efinance", get_sector_efinance)
    register_vendor_impl("get_sector", "local_cache", get_sector_local)
    register_vendor_impl("get_financial", "akshare", get_financial_akshare)

    register_vendor_impl("get_suspended", "akshare", get_suspended_akshare)
    register_vendor_impl("get_suspended", "baostock", get_suspended_baostock)
    register_vendor_impl("get_st_status", "akshare", get_st_status_akshare)
    register_vendor_impl("get_delisting", "akshare", get_delisting_akshare)

    register_vendor_impl("get_northbound_flow", "akshare", get_northbound_flow_akshare)
    register_vendor_impl("get_northbound_top10", "akshare", get_northbound_top10_akshare)
    register_vendor_impl("get_limit_up_tiers", "akshare", get_limit_up_tiers_akshare)
    register_vendor_impl("get_limit_up_tiers", "local_cache", get_limit_up_tiers_local)
    register_vendor_impl("get_dragon_tiger", "akshare", get_dragon_tiger_akshare)
    register_vendor_impl("get_dragon_tiger", "efinance", get_dragon_tiger_efinance)
    register_vendor_impl("get_dragon_tiger", "local_cache", get_dragon_tiger_local)
    register_vendor_impl("get_margin", "akshare", get_margin_akshare)
    register_vendor_impl("get_sector_constituents", "akshare", get_sector_constituents_akshare)
    register_vendor_impl("get_sector_constituents", "efinance", get_sector_constituents_efinance)
    register_vendor_impl("get_sector_constituents", "local_cache", get_sector_constituents_local)

    register_vendor_impl("get_factors", "akshare", get_factors_computed)
    register_vendor_impl("get_factors", "baostock", get_factors_computed)
    register_vendor_impl("check_crowding", "akshare", check_crowding_stub)
    register_vendor_impl("find_similar", "akshare", find_similar_stub)


register_all_vendors()
