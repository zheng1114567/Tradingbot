"""Free-first data collection adapters.

Default vendor order prefers lower-ban-risk A-share sources:
mootdx -> akshare -> baostock -> local cache, with HTTP sources as fallbacks.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
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


def _vendor_jitter(min_seconds: float = 0.1, max_seconds: float = 0.5) -> None:
    """Best-effort jitter to reduce bursty free-endpoint access."""
    time.sleep(random.uniform(min_seconds, max_seconds))


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


def _normalize_baostock_code(code: str) -> str:
    raw = str(code or "")
    if "." not in raw:
        return raw
    prefix, digits = raw.split(".", 1)
    return f"{digits}.{prefix.upper()}"


def _quarter_candidates(trade_date: str | None = None, limit: int = 6) -> list[tuple[str, str]]:
    raw = (trade_date or date.today().isoformat()).replace("-", "")
    base = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    year = base.year
    quarter = ((base.month - 1) // 3) + 1
    pairs: list[tuple[str, str]] = []
    for _ in range(limit):
        pairs.append((str(year), str(quarter)))
        quarter -= 1
        if quarter == 0:
            year -= 1
            quarter = 4
    return pairs



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




def _get_baostock():
    try:
        import baostock as bs
        return bs
    except ImportError as exc:
        raise VendorNotConfiguredError("baostock not installed (pip install baostock)", vendor="baostock") from exc


def _get_mootdx():
    try:
        from mootdx.quotes import Quotes
        return Quotes.factory(market="std")
    except ImportError as exc:
        raise VendorNotConfiguredError("mootdx not installed (pip install mootdx)", vendor="mootdx") from exc


def _read_baostock_rows(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while getattr(result, "error_code", "0") == "0" and result.next():
        rows.append(dict(zip(result.fields, result.get_row_data())))
    return rows


def get_daily_mootdx(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Daily OHLCV via mootdx TCP protocol, avoiding HTTP anti-bot limits."""

    client = _get_mootdx()
    symbol = _digits(code)
    start = pd.to_datetime(_fmt_iso(start_date, date.today() - timedelta(days=365)))
    end = pd.to_datetime(_fmt_iso(end_date, date.today()))
    try:
        df = client.bars(symbol=symbol, frequency=9, offset=400)
        if df is None or df.empty:
            raise NoMarketDataError(f"No mootdx daily data for {code}", symbol=code, vendor="mootdx")
        df = df.copy()
        date_col = "datetime" if "datetime" in df.columns else "date"
        if date_col not in df.columns:
            raise NoMarketDataError(f"mootdx schema missing date field for {code}", symbol=code, vendor="mootdx")
        df["trade_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if df.empty:
            raise NoMarketDataError(f"No mootdx daily data in range for {code}", symbol=code, vendor="mootdx")
        if "vol" in df.columns and "volume" not in df.columns:
            df["volume"] = pd.to_numeric(df["vol"], errors="coerce")
        if "turnover" in df.columns and "turn" not in df.columns:
            df["turn"] = pd.to_numeric(df["turnover"], errors="coerce")
        if "turn" in df.columns and "turnover_rate" not in df.columns:
            df["turnover_rate"] = pd.to_numeric(df["turn"], errors="coerce")
        if "amount" not in df.columns and {"close", "volume"}.issubset(df.columns):
            df["amount"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
        return _with_source(df.to_dict("records"), "mootdx", code)
    except NoMarketDataError:
        raise
    except Exception as exc:
        logger.warning("mootdx get_daily failed for %s: %s", code, exc)
        raise NoMarketDataError(str(exc), symbol=code, vendor="mootdx") from exc


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


def get_financial_baostock(
    code: str,
    trade_date: str | None = None,
) -> list[dict[str, Any]]:
    """Quarterly financial snapshots from BaoStock, merged across available tables."""

    bs = _get_baostock()
    login_result = bs.login()
    if getattr(login_result, "error_code", "0") != "0":
        raise VendorRateLimitError(getattr(login_result, "error_msg", "baostock login failed"), vendor="baostock")

    datasets = [
        "query_profit_data",
        "query_operation_data",
        "query_growth_data",
        "query_balance_data",
        "query_cash_flow_data",
        "query_dupont_data",
    ]
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for year, quarter in _quarter_candidates(trade_date):
            found_any = False
            for fn_name in datasets:
                fn = getattr(bs, fn_name, None)
                if fn is None:
                    continue
                try:
                    result = fn(code=_baostock_code(code), year=year, quarter=quarter)
                except TypeError:
                    continue
                rows = _read_baostock_rows(result)
                if not rows:
                    continue
                found_any = True
                for row in rows:
                    stat_date = str(row.get("statDate") or row.get("stat_date") or f"{year}Q{quarter}")
                    pub_date = str(row.get("pubDate") or row.get("pub_date") or "")
                    record = merged.setdefault(
                        (stat_date, pub_date),
                        {
                            "code": code,
                            "statDate": stat_date,
                            "pubDate": pub_date,
                            "year": int(year),
                            "quarter": int(quarter),
                            "data_source": "baostock",
                        },
                    )
                    for key, value in row.items():
                        if key == "code":
                            record[key] = _normalize_baostock_code(value)
                        elif key not in {"statDate", "pubDate"}:
                            record[key] = value
            if merged and not found_any:
                # Once we have recent quarter data and then hit an empty quarter,
                # stop scanning older periods to keep this cheap.
                break
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    records = sorted(
        merged.values(),
        key=lambda item: (str(item.get("statDate", "")), str(item.get("pubDate", ""))),
        reverse=True,
    )
    if not records:
        raise NoMarketDataError(f"No baostock financial data for {code}", symbol=code, vendor="baostock")

    from .local_cache import save_cached_financial

    save_cached_financial(code, records)
    return _with_source(records, "baostock", code)


def get_financial_local(code: str, trade_date: str | None = None) -> list[dict[str, Any]]:
    """Financial snapshots from local cache."""
    del trade_date
    from .local_cache import get_cached_financial

    records = get_cached_financial(code)
    if not records:
        raise NoMarketDataError(f"No cached financial data for {code}", symbol=code, vendor="local_cache")
    return _with_source(records, "local_cache", code)


def get_news_local(
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    trade_date: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Ticker/sector news from local cache before hitting online sources."""
    del kwargs
    from .local_cache import get_cached_news, get_cached_sector_news

    records: list[dict[str, Any]] = []
    if code:
        records.extend(get_cached_news(code, trade_date=trade_date))
    if sector:
        records.extend(get_cached_sector_news(sector, trade_date=trade_date))
    if keyword and not records:
        records.extend(get_cached_sector_news(keyword, trade_date=trade_date))
    if not records:
        raise NoMarketDataError("No cached news data", symbol=code or keyword or sector or "", vendor="local_cache")
    return _with_source(records, "local_cache", code or "")




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
            _vendor_jitter()
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


def get_news_cls(
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    days: int = 2,
    limit: int = 50,
    include_announcements: bool = True,
) -> list[dict[str, Any]]:
    """财联社快讯 fallback, filtered locally by ticker/sector keyword."""
    del days, include_announcements
    query = str(keyword or sector or code or "").strip()
    if not query:
        raise NoMarketDataError("CLS news requires a keyword, sector, or code", symbol=code or "", vendor="cls")

    try:
        _vendor_jitter()
        response = requests.get(
            "https://www.cls.cn/nodeapi/telegraphList",
            params={
                "app": "CailianpressWeb",
                "os": "web",
                "sv": "8.4.6",
                "sign": "9f8797a1f4de66c2370f7a03990d2737",
            },
            headers=_http_headers(),
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("cls news failed for %s: %s", query, exc)
        raise NoMarketDataError(str(exc), symbol=code or query, vendor="cls") from exc

    rows = (
        payload.get("data", {}).get("roll_data", [])
        or payload.get("data", {}).get("data", [])
        or []
    )
    query_lower = query.lower()
    records: list[dict[str, Any]] = []
    for row in rows:
        haystack = json.dumps(row, ensure_ascii=False).lower()
        if query_lower not in haystack:
            continue
        title = str(row.get("title") or row.get("brief") or row.get("content") or "").strip()
        summary = str(row.get("content") or row.get("brief") or title).strip()
        records.append({
            "title": title,
            "summary": summary,
            "source": "cls",
            "time": row.get("ctime") or row.get("created_at") or row.get("time") or "",
            "url": row.get("share_url") or row.get("url") or "",
            "type": "telegraph",
            "code": code,
            "sector_name": sector,
            "data_source": "cls",
        })
        if len(records) >= limit:
            break

    if not records:
        raise NoMarketDataError(f"No CLS news for {query}", symbol=code or query, vendor="cls")
    return records


def get_sector_eastmoney(top_n: int = 10) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    specs = [
        ("industry", "m:90+t:2+f:!50"),
        ("concept", "m:90+t:3+f:!50"),
    ]
    last_error: Exception | None = None
    for sector_type, fs in specs:
        try:
            _vendor_jitter()
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




def get_st_status_local(trade_date: str | None = None) -> list[str]:
    """Read cached ST list from local risk snapshot."""
    from .local_cache import get_cached_risk_snapshot

    snapshot = get_cached_risk_snapshot(trade_date)
    value = snapshot.get("st_status", [])
    return value if isinstance(value, list) else []


def get_suspended_local(trade_date: str | None = None) -> list[str]:
    """Read cached suspended list from local risk snapshot."""
    from .local_cache import get_cached_risk_snapshot

    snapshot = get_cached_risk_snapshot(trade_date)
    value = snapshot.get("suspended", [])
    return value if isinstance(value, list) else []


def get_delisting_local(trade_date: str | None = None) -> list[str]:
    """Read cached delisting list from local risk snapshot."""
    from .local_cache import get_cached_risk_snapshot

    snapshot = get_cached_risk_snapshot(trade_date)
    value = snapshot.get("delisting", [])
    return value if isinstance(value, list) else []


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
            _vendor_jitter()
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
        _vendor_jitter()
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
            _vendor_jitter()
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


def get_sector_local(top_n: int = 10, trade_date: str | None = None) -> list[dict[str, Any]]:
    """Sector ranking from local baostock cache."""
    from .local_cache import get_cached_sector_data

    try:
        data = get_cached_sector_data(trade_date=trade_date, top_n=top_n)
        if not data:
            raise NoMarketDataError("Local sector cache is empty — run build_full_cache first", vendor="local_cache")
        return data
    except ImportError:
        raise VendorNotConfiguredError("local_cache requires baostock", vendor="local_cache")


def get_sector_constituents_local(
    sector_name: str = "",
    trade_date: str | None = None,
) -> list[dict[str, Any]]:
    """Sector constituents from local baostock cache."""
    from .local_cache import get_cached_sector_constituents

    try:
        data = get_cached_sector_constituents(sector_name, trade_date=trade_date)
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


def get_daily_local(code: str = "", start_date: str | None = None,
                    end_date: str | None = None) -> list[dict[str, Any]]:
    """Daily OHLCV from local parquet cache."""
    from .local_cache import get_cached_daily

    try:
        data = get_cached_daily(code, start_date, end_date)
        if not data:
            raise NoMarketDataError(f"No cached daily data for {code}", symbol=code, vendor="local_cache")
        return data
    except ImportError:
        raise VendorNotConfiguredError("local_cache requires baostock", vendor="local_cache")


def get_capital_flow_local(code: str = "", start_date: str | None = None,
                           end_date: str | None = None,
                           trade_date: str | None = None) -> list[dict[str, Any]]:
    """Compute capital flow proxy from cached daily OHLCV data.

    Uses amount (成交额) deviation from 5-day MA as a proxy for
    capital flow direction. Works fully offline from parquet cache.
    """
    import pandas as pd
    from .local_cache import get_cached_daily

    try:
        data = get_cached_daily(code, start_date, end_date)
        if not data:
            raise NoMarketDataError(f"No cached daily data for {code}", symbol=code, vendor="local_cache")
    except ImportError:
        raise VendorNotConfiguredError("local_cache requires baostock", vendor="local_cache")

    df = pd.DataFrame(data)
    date_col = "datetime" if "datetime" in df.columns else "trade_date"
    if date_col not in df.columns and "date" in df.columns:
        date_col = "date"
    if date_col not in df.columns:
        return []

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    amount_col = "amount" if "amount" in df.columns else None
    if not amount_col:
        return []

    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df["amount_ma5"] = df[amount_col].rolling(5).mean()
    df["net_inflow_main"] = df[amount_col] - df["amount_ma5"]
    df["data_source"] = "local_computed"

    records = df.tail(5).to_dict("records")
    for r in records:
        r["confirmation"] = "资金确认" if r.get("net_inflow_main", 0) > 0 else "资金背离"
    return records


def get_market_breadth_local(trade_date: str | None = None) -> dict[str, Any]:
    """Market breadth proxy from cached daily parquet universe."""
    from .local_cache import get_cached_market_breadth

    data = get_cached_market_breadth(trade_date=trade_date)
    if not data:
        raise NoMarketDataError("No cached market breadth data", vendor="local_cache")
    return data


def get_northbound_top10_local(trade_date: str | None = None) -> list[dict[str, Any]]:
    """Northbound top-10 turnover from local cache, if prebuilt."""
    from .local_cache import get_cached_northbound_top10

    data = get_cached_northbound_top10(trade_date)
    if not data:
        raise NoMarketDataError("No cached northbound top10 data", vendor="local_cache")
    return _with_source(data, "local_cache")


def get_northbound_flow_local(trade_date: str | None = None) -> dict[str, Any]:
    """Northbound flow proxy from cached top-10 records."""
    rows = get_northbound_top10_local(trade_date)
    net_buy = sum(float(item.get("net_buy", 0) or 0) for item in rows)
    return {
        "trade_date": trade_date,
        "net_buy": net_buy,
        "record_count": len(rows),
        "data_source": "local_cache",
        "coverage_note": "proxy_from_cached_northbound_top10",
    }


# ------------------------------------------------------------------
# Tencent Finance real-time snapshot (PE, PB, turnover, market cap)
# ------------------------------------------------------------------

# Typical qt.gtimg.cn response field positions (varies by exchange/version).
# We return all parsed fields as a dict so callers can pick what they need.
_TENCENT_FIELDS: dict[int, str] = {
    1: "market",      # 1=SH, 2=SZ
    2: "name",
    3: "code",
    4: "price",
    5: "pre_close",
    6: "open",
    7: "high",
    8: "low",
}


def _parse_tencent_response(text: str, code: str) -> dict[str, Any]:
    """Parse qt.gtimg.cn pipe-delimited response into a dict."""
    match = re.search(r'"(.*)"', text)
    if not match:
        raise NoMarketDataError(f"Tencent snapshot: unexpected response for {code}", symbol=code, vendor="tencent")
    parts = match.group(1).split("~")
    result: dict[str, Any] = {"data_source": "tencent", "code": code}
    for idx, value in enumerate(parts):
        label = _TENCENT_FIELDS.get(idx) or f"field_{idx}"
        result[label] = value.strip() if isinstance(value, str) else value
    return result


def get_snapshot_tencent(code: str, **kwargs: Any) -> dict[str, Any]:
    """Real-time snapshot from Tencent Finance (PE/PB/turnover/market cap).

    Returns a dict with all parsed fields. Key fields:
    price, pre_close, open, high, low, name, code, pe (field_38/39),
    turnover_rate, total_market_cap, amplitude, change_pct.

    Positions 38+ vary between exchanges; callers should verify field meaning.
    """
    symbol = f"{_market_suffix(code)}{_digits(code)}"
    try:
        response = requests.get(
            f"https://qt.gtimg.cn/q={symbol}",
            headers=_http_headers(),
            timeout=10,
        )
        response.raise_for_status()
        response.encoding = "gbk"
        return _parse_tencent_response(response.text.strip(), code)
    except NoMarketDataError:
        raise
    except Exception as exc:
        raise NoMarketDataError(f"Tencent snapshot failed for {code}: {exc}", symbol=code, vendor="tencent") from exc


def get_snapshot_tencent_batch(codes: list[str]) -> list[dict[str, Any]]:
    """Batch real-time snapshot from Tencent Finance (multiple codes per call)."""
    query = ",".join(f"{_market_suffix(c)}{_digits(c)}" for c in codes)
    try:
        response = requests.get(
            f"https://qt.gtimg.cn/q={query}",
            headers=_http_headers(),
            timeout=15,
        )
        response.raise_for_status()
        response.encoding = "gbk"
        text = response.text.strip()
    except Exception as exc:
        raise NoMarketDataError(f"Tencent batch snapshot failed: {exc}", vendor="tencent") from exc

    results: list[dict[str, Any]] = []
    for line in text.splitlines():
        parsed = _parse_tencent_response(line, "")
        if parsed.get("code"):
            results.append(parsed)
    return results


def register_all_vendors() -> None:
    """Register free vendor adapters (no akshare dependency)."""

    register_vendor_impl("get_daily", "local_cache", get_daily_local)
    register_vendor_impl("get_daily", "mootdx", get_daily_mootdx)
    register_vendor_impl("get_daily", "baostock", get_daily_baostock)
    register_vendor_impl("get_market_breadth", "local_cache", get_market_breadth_local)

    register_vendor_impl("get_financial", "local_cache", get_financial_local)
    register_vendor_impl("get_financial", "baostock", get_financial_baostock)
    register_vendor_impl("get_capital_flow", "local_cache", get_capital_flow_local)
    register_vendor_impl("get_news", "local_cache", get_news_local)
    register_vendor_impl("get_news", "sina", get_news_sina)
    register_vendor_impl("get_news", "cls", get_news_cls)
    register_vendor_impl("get_sector", "eastmoney", get_sector_eastmoney)
    register_vendor_impl("get_sector", "efinance", get_sector_efinance)
    register_vendor_impl("get_sector", "local_cache", get_sector_local)

    register_vendor_impl("get_suspended", "local_cache", get_suspended_local)
    register_vendor_impl("get_suspended", "baostock", get_suspended_baostock)
    register_vendor_impl("get_st_status", "local_cache", get_st_status_local)
    register_vendor_impl("get_delisting", "local_cache", get_delisting_local)

    register_vendor_impl("get_limit_up_tiers", "local_cache", get_limit_up_tiers_local)
    register_vendor_impl("get_northbound_flow", "local_cache", get_northbound_flow_local)
    register_vendor_impl("get_northbound_top10", "local_cache", get_northbound_top10_local)
    register_vendor_impl("get_dragon_tiger", "efinance", get_dragon_tiger_efinance)
    register_vendor_impl("get_dragon_tiger", "local_cache", get_dragon_tiger_local)
    register_vendor_impl("get_sector_constituents", "efinance", get_sector_constituents_efinance)
    register_vendor_impl("get_sector_constituents", "local_cache", get_sector_constituents_local)
    register_vendor_impl("get_snapshot", "tencent", get_snapshot_tencent)

    register_vendor_impl("get_factors", "local_cache", get_factors_computed)


register_all_vendors()
