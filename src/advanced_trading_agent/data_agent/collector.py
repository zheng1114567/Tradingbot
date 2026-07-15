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
from .vendor_throttle import call_with_vendor_guard

logger = logging.getLogger(__name__)


def _vendor_jitter(min_seconds: float = 0.8, max_seconds: float = 2.0) -> None:
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


def _write_through_news_cache(
    records: list[dict[str, Any]],
    *,
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    trade_date: str | None = None,
) -> None:
    """Best-effort news write-through cache for ticker and sector scopes."""
    if not records:
        return
    try:
        if code:
            from .local_cache import save_cached_news

            save_cached_news(code, records, trade_date=trade_date)
        elif sector or keyword:
            from .local_cache import save_cached_sector_news

            sector_name = str(sector or keyword or "").strip()
            if sector_name:
                save_cached_sector_news(sector_name, records, trade_date=trade_date)
    except Exception as cache_exc:
        logger.debug(
            "news write-through cache failed for %s: %s",
            code or sector or keyword or "unknown",
            cache_exc,
        )


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


def _get_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError as exc:
        raise VendorNotConfiguredError("akshare not installed (pip install akshare)", vendor="akshare") from exc


def _read_baostock_rows(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while getattr(result, "error_code", "0") == "0" and result.next():
        rows.append(dict(zip(result.fields, result.get_row_data())))
    return rows


def _normalize_ticker_code(code: str) -> str:
    """Normalize six-digit A-share / ETF codes to ticker suffix notation."""
    digits = _digits(code)
    if len(digits) != 6:
        return code
    if digits.startswith(("5", "6", "9")):
        return f"{digits}.SH"
    if digits.startswith(("0", "1", "2", "3")):
        return f"{digits}.SZ"
    return digits


def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_akshare_daily_frame(df: pd.DataFrame, code: str, source: str) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    frame = df.copy()
    rename = {
        "date": "trade_date",
        "日期": "trade_date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover_rate",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in ["open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover_rate"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["code"] = code
    frame["data_source"] = source
    return frame.to_dict("records")


def _eastmoney_secid(code: str) -> str:
    """Return Eastmoney secid prefix for A-share/ETF symbols."""
    digits = _digits(code)
    suffix = _market_suffix(code)
    market_id = "1" if suffix == "sh" else "0"
    return f"{market_id}.{digits}"


def _normalize_eastmoney_kline_rows(rows: list[str], code: str, source: str) -> list[dict[str, Any]]:
    """Normalize Eastmoney kline comma rows into OHLCV records."""
    records: list[dict[str, Any]] = []
    for raw in rows:
        parts = str(raw).split(",")
        if len(parts) < 7:
            continue
        records.append({
            "trade_date": parts[0],
            "open": _float_or_none(parts[1]),
            "close": _float_or_none(parts[2]),
            "high": _float_or_none(parts[3]),
            "low": _float_or_none(parts[4]),
            "volume": _float_or_none(parts[5]),
            "amount": _float_or_none(parts[6]),
            "amplitude": _float_or_none(parts[7]) if len(parts) > 7 else None,
            "pct_chg": _float_or_none(parts[8]) if len(parts) > 8 else None,
            "change": _float_or_none(parts[9]) if len(parts) > 9 else None,
            "turnover_rate": _float_or_none(parts[10]) if len(parts) > 10 else None,
            "code": code,
            "data_source": source,
        })
    return records


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
        records = _with_source(df.to_dict("records"), "mootdx", code)
        try:
            from .local_cache import save_cached_daily

            save_cached_daily(code, records)
        except Exception as cache_exc:
            logger.debug("daily write-through cache failed for %s via mootdx: %s", code, cache_exc)
        return records
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
        records = _with_source(rows, "baostock", code)
        try:
            from .local_cache import save_cached_daily

            save_cached_daily(code, records)
        except Exception as cache_exc:
            logger.debug("daily write-through cache failed for %s via baostock: %s", code, cache_exc)
        return records
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def get_daily_akshare(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Daily A-share OHLCV from akshare, with local write-through cache."""
    ak = _get_akshare()
    start = _fmt_yyyymmdd(start_date, date.today() - timedelta(days=365))
    end = _fmt_yyyymmdd(end_date, date.today())
    digits = _digits(code)
    try:
        _vendor_jitter()
        df = call_with_vendor_guard(
            "akshare",
            lambda: ak.stock_zh_a_hist(
                symbol=digits,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            ),
        )
    except Exception as exc:
        raise NoMarketDataError(f"akshare daily failed for {code}: {exc}", symbol=code, vendor="akshare") from exc
    records = _normalize_akshare_daily_frame(df, _normalize_ticker_code(code), "akshare")
    if not records:
        raise NoMarketDataError(f"No akshare daily data for {code}", symbol=code, vendor="akshare")
    try:
        from .local_cache import save_cached_daily

        save_cached_daily(_normalize_ticker_code(code), records)
    except Exception as cache_exc:
        logger.debug("daily write-through cache failed for %s via akshare: %s", code, cache_exc)
    return records


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
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if not code:
        raise NoMarketDataError("Sina news requires a stock code", symbol="", vendor="sina")

    symbol = f"{_market_suffix(code)}{_digits(code)}"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        _vendor_jitter()
        response = requests.get(
            "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php",
            {"symbol": symbol},
            headers=_http_headers(),
            timeout=3,
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding or "gb18030"
        text = response.text
    except Exception as exc:
        raise NoMarketDataError(f"Sina news failed for {symbol}: {exc}", symbol=code, vendor="sina")

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
        from .news_text import is_noise_news_record
        if is_noise_news_record(candidate):
            continue
        seen.add(link)
        records.append(candidate)
        if len(records) >= limit:
            break

    if not records:
        raise NoMarketDataError(f"No Sina news for {code}", symbol=code, vendor="sina")
    result = records[:limit]
    _write_through_news_cache(result, code=code, sector=sector, keyword=keyword, trade_date=kwargs.get("trade_date"))
    return result


def get_news_akshare(
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    days: int = 2,
    limit: int = 50,
    include_announcements: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Ticker news via akshare stock_news_em, cached on success."""
    del sector, days, include_announcements
    if not code:
        raise NoMarketDataError("akshare stock news requires a stock code", symbol="", vendor="akshare")
    ak = _get_akshare()
    trade_date = kwargs.get("trade_date")
    digits = _digits(code)
    try:
        _vendor_jitter()
        df = call_with_vendor_guard("akshare", lambda: ak.stock_news_em(symbol=digits))
    except Exception as exc:
        raise NoMarketDataError(f"akshare stock news failed for {code}: {exc}", symbol=code, vendor="akshare") from exc
    if df is None or df.empty:
        raise NoMarketDataError(f"No akshare news for {code}", symbol=code, vendor="akshare")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        title = str(_first_present(raw, ["新闻标题", "标题", "title"]) or "")
        if not title:
            continue
        candidate = {
            "title": title,
            "summary": str(_first_present(raw, ["新闻内容", "摘要", "内容", "summary"]) or title),
            "source": "akshare",
            "time": str(_first_present(raw, ["发布时间", "时间", "time"]) or ""),
            "url": str(_first_present(raw, ["新闻链接", "链接", "url"]) or ""),
            "type": "news",
            "code": code,
            "data_source": "akshare",
        }
        if keyword and keyword.lower() not in json.dumps(candidate, ensure_ascii=False).lower():
            continue
        from .news_text import is_noise_news_record
        if is_noise_news_record(candidate):
            continue
        records.append(candidate)
        if len(records) >= limit:
            break

    if not records:
        raise NoMarketDataError(f"No akshare news for {code}", symbol=code, vendor="akshare")
    _write_through_news_cache(records, code=code, keyword=keyword, trade_date=trade_date)
    return records


def get_news_eastmoney(
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    days: int = 2,
    limit: int = 50,
    include_announcements: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """个股新闻 via akshare stock_news_em (东方财富)."""
    del days, include_announcements
    trade_date = kwargs.get("trade_date")
    digits = _digits(code or "") if code else ""
    if not digits:
        raise NoMarketDataError("Eastmoney news requires a stock code", symbol=code or "", vendor="eastmoney")

    try:
        import akshare as ak
    except ImportError:
        raise VendorNotConfiguredError("akshare not installed (pip install akshare)", vendor="eastmoney")

    _vendor_jitter()
    try:
        df = call_with_vendor_guard("eastmoney", lambda: ak.stock_news_em(symbol=digits))
    except Exception as exc:
        raise NoMarketDataError(str(exc), symbol=code, vendor="eastmoney") from exc

    if df is None or df.empty:
        raise NoMarketDataError(f"No eastmoney news for {code}", symbol=code, vendor="eastmoney")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        title = str(row.get("新闻标题", ""))
        if not title:
            continue
        candidate = {
            "title": title,
            "summary": str(row.get("新闻内容", title)),
            "source": "eastmoney",
            "time": str(row.get("发布时间", "")),
            "url": str(row.get("新闻链接", "")),
            "type": "news",
            "code": code,
            "data_source": "eastmoney",
        }
        if keyword and keyword.lower() not in json.dumps(candidate, ensure_ascii=False).lower():
            continue
        from .news_text import is_noise_news_record
        if is_noise_news_record(candidate):
            continue
        records.append(candidate)
        if len(records) >= limit:
            break

    if not records:
        raise NoMarketDataError(f"No eastmoney news for {code}", symbol=code, vendor="eastmoney")
    _write_through_news_cache(records, code=code, sector=sector, keyword=keyword, trade_date=trade_date)
    return records


def get_news_eastmoney_global(
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    days: int = 2,
    limit: int = 50,
    include_announcements: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """全局财经快讯 via akshare stock_info_global_em，关键词过滤."""
    del days, include_announcements
    trade_date = kwargs.get("trade_date")
    query = str(keyword or code or "").strip()
    if not query:
        raise NoMarketDataError("global news requires a keyword or code", symbol=code or "", vendor="eastmoney_global")

    try:
        import akshare as ak
    except ImportError:
        raise VendorNotConfiguredError("akshare not installed (pip install akshare)", vendor="eastmoney_global")

    _vendor_jitter()
    try:
        df = call_with_vendor_guard("eastmoney_global", lambda: ak.stock_info_global_em())
    except Exception as exc:
        raise NoMarketDataError(str(exc), symbol=code, vendor="eastmoney_global") from exc

    if df is None or df.empty:
        raise NoMarketDataError("No global news available", vendor="eastmoney_global")

    import re
    keywords = re.split(r"[|,;]", query)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        title = str(row.get("标题", ""))
        if not title:
            continue
        text = f"{title} {row.get('摘要', '')}"
        if not any(kw.lower() in text.lower() for kw in keywords if kw.strip()):
            continue
        candidate = {
            "title": title,
            "summary": str(row.get("摘要", title)),
            "source": "eastmoney_global",
            "time": str(row.get("发布时间", "")),
            "url": str(row.get("链接", "")),
            "type": "news",
            "code": code,
            "data_source": "eastmoney_global",
        }
        from .news_text import is_noise_news_record
        if is_noise_news_record(candidate):
            continue
        records.append(candidate)
        if len(records) >= limit:
            break

    if not records:
        raise NoMarketDataError(f"No global news matching {query}", vendor="eastmoney_global")
    _write_through_news_cache(records, code=code, sector=sector, keyword=keyword, trade_date=trade_date)
    return records


def get_news_cls(
    code: str | None = None,
    sector: str | None = None,
    keyword: str | None = None,
    days: int = 2,
    limit: int = 50,
    include_announcements: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """财联社快讯 fallback, filtered locally by ticker/sector keyword."""
    del days, include_announcements
    trade_date = kwargs.get("trade_date")
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
    _write_through_news_cache(records, code=code, sector=sector, keyword=keyword, trade_date=trade_date)
    return records


def get_sector_eastmoney(top_n: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
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


def _normalize_akshare_sector_frame(df: pd.DataFrame, sector_type: str, top_n: int) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(df.head(max(top_n, 20)).to_dict("records"), start=1):
        name = str(_first_present(row, ["板块名称", "行业名称", "概念名称", "名称", "name"]) or "")
        if not name:
            continue
        change_pct = _float_or_none(_first_present(row, ["涨跌幅", "涨跌幅%", "涨幅", "change_pct"])) or 0.0
        records.append({
            "rank": int(_first_present(row, ["排名", "rank"]) or idx),
            "sector_code": str(_first_present(row, ["板块代码", "代码", "code"]) or ""),
            "sector_name": name,
            "sector_type": sector_type,
            "change_pct": change_pct,
            "strength_score": change_pct,
            "net_inflow_main": _float_or_none(_first_present(row, ["主力净流入", "净流入", "资金净流入"])),
            "leading_stock": str(_first_present(row, ["领涨股票", "领涨股"]) or ""),
            "data_source": "akshare",
            "raw": row,
        })
    return records[:top_n]


def get_sector_akshare(top_n: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
    """Sector ranking via akshare industry/concept board endpoints."""
    del kwargs
    ak = _get_akshare()
    records: list[dict[str, Any]] = []
    last_error: Exception | None = None
    endpoints = [
        ("industry", "stock_board_industry_name_em"),
        ("concept", "stock_board_concept_name_em"),
    ]
    for sector_type, fn_name in endpoints:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            _vendor_jitter()
            df = call_with_vendor_guard("akshare", fn)
            records.extend(_normalize_akshare_sector_frame(df, sector_type, top_n))
        except Exception as exc:
            last_error = exc
            logger.warning("akshare sector failed for %s: %s", sector_type, exc)
    if not records:
        detail = f": {last_error}" if last_error else ""
        raise NoMarketDataError(f"No akshare sector data{detail}", vendor="akshare")
    records.sort(key=lambda item: float(item.get("strength_score") or 0), reverse=True)
    for idx, record in enumerate(records, start=1):
        record["rank"] = idx
    return records[:top_n]


def get_sector_constituents_akshare(sector_name: str = "", trade_date: str | None = None) -> list[dict[str, Any]]:
    """Sector constituents via akshare industry/concept board endpoints."""
    del trade_date
    if not sector_name:
        return []
    ak = _get_akshare()
    endpoints = [
        ("industry", "stock_board_industry_cons_em"),
        ("concept", "stock_board_concept_cons_em"),
    ]
    last_error: Exception | None = None
    for sector_type, fn_name in endpoints:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        try:
            _vendor_jitter()
            df = call_with_vendor_guard("akshare", lambda fn=fn: fn(symbol=sector_name))
        except Exception as exc:
            last_error = exc
            continue
        if df is None or df.empty:
            continue
        records: list[dict[str, Any]] = []
        for row in df.to_dict("records"):
            code = str(_first_present(row, ["代码", "股票代码", "code"]) or "")
            name = str(_first_present(row, ["名称", "股票名称", "name"]) or "")
            if not code:
                continue
            records.append({
                "code": _normalize_ticker_code(code),
                "name": name,
                "sector": sector_name,
                "sector_type": sector_type,
                "data_source": "akshare",
                "raw": row,
            })
        if records:
            return records
    detail = f": {last_error}" if last_error else ""
    raise NoMarketDataError(f"No akshare constituents for {sector_name}{detail}", vendor="akshare")


_EASTMONEY_API = "http://push2.eastmoney.com/api/qt/clist/get"
_EM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
}


def _eastmoney_diff(
    fs: str,
    fields: str,
    pz: int = 5000,
    fid: str = "f12",
    po: int = 0,
) -> list[dict[str, Any]]:
    """Fetch diff rows from eastmoney push2 HTTP API."""
    try:
        response = call_with_vendor_guard(
            "eastmoney",
            lambda: requests.get(
                _EASTMONEY_API,
                params={
                    "pn": 1, "pz": pz, "po": po, "np": 1,
                    "fltt": 2, "invt": 2,
                    "fid": fid, "fs": fs,
                    "fields": fields,
                    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                },
                headers=_EM_HEADERS,
                timeout=10,
            ),
        )
        response.raise_for_status()
        payload = response.json()
        return (payload.get("data") or {}).get("diff") or []
    except Exception as exc:
        raise NoMarketDataError(f"Eastmoney diff failed: {exc}", vendor="eastmoney") from exc


def get_limit_up_tiers_eastmoney(**kwargs: Any) -> dict[str, Any]:
    """Limit-up tiers from eastmoney realtime stock list."""
    rows = _eastmoney_diff(
        fs="m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        fields="f12,f14,f3,f2,f4,f15,f16,f17,f20,f184,f185",
        po=1,
        fid="f3",
    )
    limit_up: list[dict[str, Any]] = []
    for r in rows:
        pct = r.get("f3")
        if pct is None or not isinstance(pct, (int, float)):
            continue
        if pct < 9.5:
            continue
        limit_up.append({
            "ticker": _digits_to_ticker(str(r.get("f12", ""))),
            "code": str(r.get("f12", "")),
            "name": str(r.get("f14", "")),
            "change_pct": float(pct),
            "price": r.get("f2"),
            "high": r.get("f15"),
            "low": r.get("f16"),
            "open": r.get("f17"),
            "volume": r.get("f20"),
        })

    first_board = sum(1 for s in limit_up if _is_first_limit_up(s))
    second_board = sum(1 for s in limit_up if _is_second_limit_up(s))
    third_plus = max(0, len(limit_up) - first_board - second_board)
    return {
        "first_board": first_board,
        "second_board": second_board,
        "third_plus": third_plus,
        "stocks": limit_up[:200],
        "data_source": "eastmoney",
    }


def get_limit_up_tiers_akshare(trade_date: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Limit-up pool via akshare stock_zt_pool_em."""
    del kwargs
    ak = _get_akshare()
    td = _fmt_yyyymmdd(trade_date, date.today())
    try:
        _vendor_jitter()
        df = call_with_vendor_guard("akshare", lambda: ak.stock_zt_pool_em(date=td))
    except TypeError:
        try:
            df = call_with_vendor_guard("akshare", ak.stock_zt_pool_em)
        except Exception as exc:
            raise NoMarketDataError(f"akshare limit-up failed: {exc}", vendor="akshare") from exc
    except Exception as exc:
        raise NoMarketDataError(f"akshare limit-up failed: {exc}", vendor="akshare") from exc
    if df is None or df.empty:
        raise NoMarketDataError(f"No akshare limit-up data for {td}", vendor="akshare")

    stocks: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        code = str(_first_present(row, ["代码", "股票代码", "code"]) or "")
        if not code:
            continue
        board_count = int(_float_or_none(_first_present(row, ["连板数", "几天几板", "board_count"])) or 1)
        stocks.append({
            "ticker": _normalize_ticker_code(code),
            "code": code,
            "name": str(_first_present(row, ["名称", "股票名称", "name"]) or ""),
            "change_pct": _float_or_none(_first_present(row, ["涨跌幅", "涨幅", "change_pct"])),
            "price": _float_or_none(_first_present(row, ["最新价", "收盘价", "price"])),
            "volume": _float_or_none(_first_present(row, ["成交量", "volume"])),
            "amount": _float_or_none(_first_present(row, ["成交额", "amount"])),
            "board_count": board_count,
            "reason": str(_first_present(row, ["涨停原因类别", "涨停原因", "原因"]) or ""),
            "data_source": "akshare",
            "raw": row,
        })

    if not stocks:
        raise NoMarketDataError(f"No akshare limit-up stocks for {td}", vendor="akshare")
    first_board = sum(1 for item in stocks if int(item.get("board_count") or 1) <= 1)
    second_board = sum(1 for item in stocks if int(item.get("board_count") or 1) == 2)
    third_plus = sum(1 for item in stocks if int(item.get("board_count") or 1) >= 3)
    return {
        "first_board": first_board,
        "second_board": second_board,
        "third_plus": third_plus,
        "stocks": stocks[:300],
        "data_source": "akshare",
        "trade_date": td,
    }


def _digits_to_ticker(code: str) -> str:
    if len(code) != 6:
        return code
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3", "2")):
        return f"{code}.SZ"
    return code


def _is_first_limit_up(stock: dict[str, Any]) -> bool:
    """Heuristic: first board typically has lower volume surge."""
    vol = stock.get("volume") or 0
    return vol < 5000000


def _is_second_limit_up(stock: dict[str, Any]) -> bool:
    vol = stock.get("volume") or 0
    return 5000000 <= vol < 20000000


def get_market_breadth_eastmoney(**kwargs: Any) -> dict[str, Any]:
    """Market breadth (advance/decline) from eastmoney stock list."""
    rows = _eastmoney_diff(
        fs="m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        fields="f12,f14,f3",
        fid="f3",
    )
    advance = sum(1 for r in rows if isinstance(r.get("f3"), (int, float)) and r["f3"] > 0)
    decline = sum(1 for r in rows if isinstance(r.get("f3"), (int, float)) and r["f3"] < 0)
    flat = sum(1 for r in rows if isinstance(r.get("f3"), (int, float)) and r["f3"] == 0)
    return {
        "advance_count": advance,
        "decline_count": decline,
        "flat_count": flat,
        "total": advance + decline + flat,
        "data_source": "eastmoney",
    }


def get_st_status_eastmoney(**kwargs: Any) -> list[str]:
    """ST/*ST stock codes from eastmoney realtime list."""
    rows = _eastmoney_diff(
        fs="m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        fields="f12,f14",
    )
    st_codes: list[str] = []
    for r in rows:
        name = str(r.get("f14", ""))
        if "ST" in name.upper():
            code = str(r.get("f12", ""))
            st_codes.append(_digits_to_ticker(code))
    return st_codes


def get_suspended_eastmoney(**kwargs: Any) -> list[str]:
    """Suspended stock codes from eastmoney (f20 volume = 0 on trading day)."""
    rows = _eastmoney_diff(
        fs="m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        fields="f12,f14,f20",
    )
    suspended: list[str] = []
    for r in rows:
        vol = r.get("f20")
        name = str(r.get("f14", ""))
        if vol is not None and (isinstance(vol, (int, float)) and vol == 0) and "退" in name:
            code = str(r.get("f12", ""))
            suspended.append(_digits_to_ticker(code))
    return suspended


def get_delisting_eastmoney(**kwargs: Any) -> list[str]:
    """Delisting stock codes from eastmoney (name includes 退)."""
    rows = _eastmoney_diff(
        fs="m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        fields="f12,f14",
    )
    delisted: list[str] = []
    for r in rows:
        name = str(r.get("f14", ""))
        if "退" in name:
            code = str(r.get("f12", ""))
            delisted.append(_digits_to_ticker(code))
    return delisted


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
                suspended.append(_normalize_baostock_code(str(row.get("code", ""))))
        return suspended
    except Exception as exc:
        logger.warning("baostock suspended list failed: %s", exc)
        return []
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def get_st_status_baostock(trade_date: str | None = None) -> list[str]:
    """ST/*ST stock codes from baostock query_all_stock."""
    bs = _get_baostock()
    day = _fmt_iso(trade_date, date.today())
    login_result = bs.login()
    if getattr(login_result, "error_code", "0") != "0":
        raise VendorRateLimitError(getattr(login_result, "error_msg", "baostock login failed"), vendor="baostock")
    try:
        rs = bs.query_all_stock(day=day)
        st_codes: list[str] = []
        while getattr(rs, "error_code", "0") == "0" and rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            name = row.get("code_name", row.get("stock_name", ""))
            if "ST" in str(name).upper():
                st_codes.append(_normalize_baostock_code(str(row.get("code", ""))))
        return st_codes
    except Exception as exc:
        logger.warning("baostock ST list failed: %s", exc)
        return []
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def get_delisting_baostock(trade_date: str | None = None) -> list[str]:
    """Delisting-risk stock codes from baostock (name contains 退)."""
    bs = _get_baostock()
    day = _fmt_iso(trade_date, date.today())
    login_result = bs.login()
    if getattr(login_result, "error_code", "0") != "0":
        raise VendorRateLimitError(getattr(login_result, "error_msg", "baostock login failed"), vendor="baostock")
    try:
        rs = bs.query_all_stock(day=day)
        delisted: list[str] = []
        while getattr(rs, "error_code", "0") == "0" and rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            name = row.get("code_name", row.get("stock_name", ""))
            if "退" in str(name):
                delisted.append(_normalize_baostock_code(str(row.get("code", ""))))
        return delisted
    except Exception as exc:
        logger.warning("baostock delisting list failed: %s", exc)
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


def get_sector_efinance(top_n: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
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
        data = get_cached_daily(code, start_date, end_date, allow_online_repair=False)
        if not data:
            raise NoMarketDataError(f"No cached daily data for {code}", symbol=code, vendor="local_cache")
        return data
    except ImportError:
        raise VendorNotConfiguredError("local_cache requires baostock", vendor="local_cache")


def get_etf_spot_local(trade_date: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """ETF spot rows from local cache."""
    del kwargs
    from .local_cache import get_cached_etf_spot

    data = get_cached_etf_spot(trade_date=trade_date)
    if not data:
        raise NoMarketDataError("No cached ETF spot data", vendor="local_cache")
    return data


def get_etf_universe_local(trade_date: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """ETF universe from cached spot rows."""
    return get_etf_spot_local(trade_date=trade_date, **kwargs)


def get_etf_daily_local(
    code: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """ETF daily OHLCV from local parquet cache."""
    del kwargs
    from .local_cache import get_cached_etf_daily

    data = get_cached_etf_daily(code, start_date=start_date, end_date=end_date)
    if not data:
        raise NoMarketDataError(f"No cached ETF daily data for {code}", symbol=code, vendor="local_cache")
    return data


def _normalize_etf_spot_frame(df: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    records: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        code = str(_first_present(row, ["代码", "基金代码", "symbol", "code"]) or "")
        name = str(_first_present(row, ["名称", "基金简称", "基金名称", "name"]) or "")
        if not code:
            continue
        records.append({
            "code": _normalize_ticker_code(code),
            "raw_code": code,
            "name": name,
            "latest_price": _float_or_none(_first_present(row, ["最新价", "现价", "price", "最新"])),
            "change_pct": _float_or_none(_first_present(row, ["涨跌幅", "涨幅", "change_pct"])),
            "volume": _float_or_none(_first_present(row, ["成交量", "volume"])),
            "amount": _float_or_none(_first_present(row, ["成交额", "amount"])),
            "premium_discount": _float_or_none(_first_present(row, ["折价率", "溢价率", "折溢价率", "premium_discount"])),
            "data_source": source,
            "raw": row,
        })
    return records


def get_etf_spot_akshare(trade_date: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """ETF spot/universe rows via akshare fund_etf_spot_em."""
    del kwargs
    ak = _get_akshare()
    try:
        _vendor_jitter()
        df = call_with_vendor_guard("akshare", ak.fund_etf_spot_em)
    except Exception as exc:
        raise NoMarketDataError(f"akshare ETF spot failed: {exc}", vendor="akshare") from exc
    records = _normalize_etf_spot_frame(df, "akshare")
    if not records:
        raise NoMarketDataError("No akshare ETF spot data", vendor="akshare")
    try:
        from .local_cache import save_cached_etf_spot

        save_cached_etf_spot(records, trade_date=trade_date)
    except Exception as cache_exc:
        logger.debug("ETF spot write-through cache failed via akshare: %s", cache_exc)
    return records


def get_etf_universe_akshare(trade_date: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """ETF universe via akshare spot endpoint."""
    return get_etf_spot_akshare(trade_date=trade_date, **kwargs)


def get_etf_daily_akshare(
    code: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """ETF daily OHLCV via akshare fund_etf_hist_em."""
    del kwargs
    if not code:
        raise NoMarketDataError("ETF daily requires code", vendor="akshare")
    ak = _get_akshare()
    start = _fmt_yyyymmdd(start_date, date.today() - timedelta(days=365))
    end = _fmt_yyyymmdd(end_date, date.today())
    digits = _digits(code)
    try:
        _vendor_jitter()
        df = call_with_vendor_guard(
            "akshare",
            lambda: ak.fund_etf_hist_em(
                symbol=digits,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            ),
        )
    except Exception as exc:
        raise NoMarketDataError(f"akshare ETF daily failed for {code}: {exc}", symbol=code, vendor="akshare") from exc
    etf_code = _normalize_ticker_code(code)
    records = _normalize_akshare_daily_frame(df, etf_code, "akshare")
    if not records:
        raise NoMarketDataError(f"No akshare ETF daily data for {code}", symbol=code, vendor="akshare")
    try:
        from .local_cache import save_cached_etf_daily

        save_cached_etf_daily(etf_code, records)
    except Exception as cache_exc:
        logger.debug("ETF daily write-through cache failed for %s via akshare: %s", code, cache_exc)
    return records


def get_etf_daily_eastmoney(
    code: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """ETF daily OHLCV via direct Eastmoney kline endpoint."""
    del kwargs
    if not code:
        raise NoMarketDataError("ETF daily requires code", vendor="eastmoney")
    etf_code = _normalize_ticker_code(code)
    start = _fmt_yyyymmdd(start_date, date.today() - timedelta(days=365))
    end = _fmt_yyyymmdd(end_date, date.today())
    try:
        _vendor_jitter()
        response = call_with_vendor_guard(
            "eastmoney",
            lambda: requests.get(
                "http://push2his.eastmoney.com/api/qt/stock/kline/get",
                params={
                    "secid": _eastmoney_secid(code),
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                    "ut": "7eea3edcaed734bea9cbfc24409ed989",
                    "klt": 101,
                    "fqt": 1,
                    "beg": start,
                    "end": end,
                },
                headers=_http_headers(),
                timeout=10,
            ),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise NoMarketDataError(f"Eastmoney ETF daily failed for {code}: {exc}", symbol=code, vendor="eastmoney") from exc

    klines = (payload.get("data") or {}).get("klines") or []
    records = _normalize_eastmoney_kline_rows(klines, etf_code, "eastmoney")
    if not records:
        raise NoMarketDataError(f"No Eastmoney ETF daily data for {code}", symbol=code, vendor="eastmoney")
    try:
        from .local_cache import save_cached_etf_daily

        save_cached_etf_daily(etf_code, records)
    except Exception as cache_exc:
        logger.debug("ETF daily write-through cache failed for %s via eastmoney: %s", code, cache_exc)
    return records


def get_etf_daily_sina(
    code: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """ETF daily OHLCV via Sina, accessed through akshare's Sina adapter."""
    del kwargs
    if not code:
        raise NoMarketDataError("ETF daily requires code", vendor="sina")
    ak = _get_akshare()
    etf_code = _normalize_ticker_code(code)
    symbol = f"{_market_suffix(code)}{_digits(code)}"
    try:
        _vendor_jitter()
        df = call_with_vendor_guard("sina", lambda: ak.fund_etf_hist_sina(symbol=symbol))
    except Exception as exc:
        raise NoMarketDataError(f"Sina ETF daily failed for {code}: {exc}", symbol=code, vendor="sina") from exc
    if df is None or df.empty:
        raise NoMarketDataError(f"No Sina ETF daily data for {code}", symbol=code, vendor="sina")

    frame = df.copy()
    date_col = "date" if "date" in frame.columns else "trade_date" if "trade_date" in frame.columns else None
    if date_col:
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        if start_date:
            frame = frame[frame[date_col].dt.date >= pd.to_datetime(_fmt_iso(start_date, date.today())).date()]
        if end_date:
            frame = frame[frame[date_col].dt.date <= pd.to_datetime(_fmt_iso(end_date, date.today())).date()]
    records = _normalize_akshare_daily_frame(frame, etf_code, "sina")
    if not records:
        raise NoMarketDataError(f"No Sina ETF daily data for {code} in requested range", symbol=code, vendor="sina")
    try:
        from .local_cache import save_cached_etf_daily

        save_cached_etf_daily(etf_code, records)
    except Exception as cache_exc:
        logger.debug("ETF daily write-through cache failed for %s via sina: %s", code, cache_exc)
    return records


def get_etf_info_akshare(code: str = "", **kwargs: Any) -> dict[str, Any]:
    """ETF info via akshare fund_etf_fund_info_em when available."""
    del kwargs
    if not code:
        raise NoMarketDataError("ETF info requires code", vendor="akshare")
    ak = _get_akshare()
    digits = _digits(code)
    try:
        _vendor_jitter()
        df = call_with_vendor_guard("akshare", lambda: ak.fund_etf_fund_info_em(fund=digits))
    except Exception as exc:
        raise NoMarketDataError(f"akshare ETF info failed for {code}: {exc}", symbol=code, vendor="akshare") from exc
    if df is None or df.empty:
        raise NoMarketDataError(f"No akshare ETF info for {code}", symbol=code, vendor="akshare")
    return {
        "code": _normalize_ticker_code(code),
        "records": df.to_dict("records"),
        "data_source": "akshare",
    }


def get_etf_premium_discount_akshare(code: str = "", trade_date: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Best-effort ETF premium/discount lookup from akshare spot rows."""
    rows = get_etf_spot_akshare(trade_date=trade_date, **kwargs)
    normalized = _normalize_ticker_code(code)
    for row in rows:
        if row.get("code") == normalized or str(row.get("raw_code")) == _digits(code):
            return {
                "code": normalized,
                "premium_discount": row.get("premium_discount"),
                "latest_price": row.get("latest_price"),
                "data_source": row.get("data_source", "akshare"),
            }
    raise NoMarketDataError(f"No akshare ETF premium/discount row for {code}", symbol=code, vendor="akshare")


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
    """Register free vendor adapters."""

    register_vendor_impl("get_daily", "local_cache", get_daily_local)
    register_vendor_impl("get_daily", "akshare", get_daily_akshare)
    register_vendor_impl("get_daily", "mootdx", get_daily_mootdx)
    register_vendor_impl("get_daily", "baostock", get_daily_baostock)
    register_vendor_impl("get_market_breadth", "local_cache", get_market_breadth_local)
    register_vendor_impl("get_market_breadth", "eastmoney", get_market_breadth_eastmoney)

    register_vendor_impl("get_financial", "local_cache", get_financial_local)
    register_vendor_impl("get_financial", "baostock", get_financial_baostock)
    register_vendor_impl("get_capital_flow", "local_cache", get_capital_flow_local)
    register_vendor_impl("get_news", "akshare", get_news_akshare)
    register_vendor_impl("get_news", "eastmoney", get_news_eastmoney)
    register_vendor_impl("get_news", "eastmoney_global", get_news_eastmoney_global)
    register_vendor_impl("get_news", "local_cache", get_news_local)
    register_vendor_impl("get_news", "sina", get_news_sina)
    register_vendor_impl("get_news", "cls", get_news_cls)
    register_vendor_impl("get_sector", "akshare", get_sector_akshare)
    register_vendor_impl("get_sector", "eastmoney", get_sector_eastmoney)
    register_vendor_impl("get_sector", "efinance", get_sector_efinance)
    register_vendor_impl("get_sector", "local_cache", get_sector_local)

    register_vendor_impl("get_suspended", "local_cache", get_suspended_local)
    register_vendor_impl("get_suspended", "baostock", get_suspended_baostock)
    register_vendor_impl("get_suspended", "eastmoney", get_suspended_eastmoney)
    register_vendor_impl("get_st_status", "local_cache", get_st_status_local)
    register_vendor_impl("get_st_status", "baostock", get_st_status_baostock)
    register_vendor_impl("get_st_status", "eastmoney", get_st_status_eastmoney)
    register_vendor_impl("get_delisting", "local_cache", get_delisting_local)
    register_vendor_impl("get_delisting", "baostock", get_delisting_baostock)
    register_vendor_impl("get_delisting", "eastmoney", get_delisting_eastmoney)

    register_vendor_impl("get_limit_up_tiers", "local_cache", get_limit_up_tiers_local)
    register_vendor_impl("get_limit_up_tiers", "akshare", get_limit_up_tiers_akshare)
    register_vendor_impl("get_limit_up_tiers", "eastmoney", get_limit_up_tiers_eastmoney)
    register_vendor_impl("get_northbound_flow", "local_cache", get_northbound_flow_local)
    register_vendor_impl("get_northbound_top10", "local_cache", get_northbound_top10_local)
    register_vendor_impl("get_dragon_tiger", "efinance", get_dragon_tiger_efinance)
    register_vendor_impl("get_dragon_tiger", "local_cache", get_dragon_tiger_local)
    register_vendor_impl("get_sector_constituents", "efinance", get_sector_constituents_efinance)
    register_vendor_impl("get_sector_constituents", "local_cache", get_sector_constituents_local)
    register_vendor_impl("get_sector_constituents", "akshare", get_sector_constituents_akshare)
    register_vendor_impl("get_snapshot", "tencent", get_snapshot_tencent)

    register_vendor_impl("get_factors", "local_cache", get_factors_computed)

    register_vendor_impl("get_etf_universe", "local_cache", get_etf_universe_local)
    register_vendor_impl("get_etf_universe", "akshare", get_etf_universe_akshare)
    register_vendor_impl("get_etf_spot", "local_cache", get_etf_spot_local)
    register_vendor_impl("get_etf_spot", "akshare", get_etf_spot_akshare)
    register_vendor_impl("get_etf_daily", "local_cache", get_etf_daily_local)
    register_vendor_impl("get_etf_daily", "akshare", get_etf_daily_akshare)
    register_vendor_impl("get_etf_daily", "sina", get_etf_daily_sina)
    register_vendor_impl("get_etf_daily", "eastmoney", get_etf_daily_eastmoney)
    register_vendor_impl("get_etf_info", "akshare", get_etf_info_akshare)
    register_vendor_impl("get_etf_premium_discount", "akshare", get_etf_premium_discount_akshare)


register_all_vendors()

