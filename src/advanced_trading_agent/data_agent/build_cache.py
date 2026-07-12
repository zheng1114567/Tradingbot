#!/usr/bin/env python
"""One-shot: build local data cache from efinance + baostock.

Saves to data/results/local_cache/ so the scanner works offline.
Run once when network is available, then scan without API calls.

Usage:
  python -m advanced_trading_agent.data_agent.build_cache
  python -m advanced_trading_agent.data_agent.build_cache --date 2026-07-10
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ..config import config
from ..core.atomic_write import atomic_write_text
from .vendor_router import ensure_default_vendor_registration, get_vendor_impl

logger = logging.getLogger(__name__)
_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+")


def _vendor_jitter(min_seconds: float = 0.1, max_seconds: float = 0.5) -> None:
    import random
    time.sleep(random.uniform(min_seconds, max_seconds))


def build(trade_date: str | None = None, output_dir: str | None = None,
          days_back: int = 60, compute_signals: bool = True) -> Path:
    """Build complete local cache. Returns path to cache directory."""
    td = trade_date or date.today().isoformat()
    cache_dir = Path(output_dir or config.get("results_dir")) / "local_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _cache_ready(cache_dir, td):
        logger.info("=== Local cache hit for %s → %s ===", td, cache_dir)
        _print_cache_gaps(cache_dir, td)
        _print_summary(cache_dir)
        if compute_signals:
            _run_signal_computation(cache_dir, td)
        return cache_dir

    logger.info("=== Building local cache for %s → %s ===", td, cache_dir)
    gaps = _cache_gaps(cache_dir, td)
    if gaps:
        logger.info("Cache gaps before build: %s", ", ".join(gaps))

    # 1. Board index (sector → stocks) via efinance
    board_index = _cache_board_index(cache_dir)
    logger.info("Board index: %d sectors, %d stock-sector pairs",
                len(board_index), sum(len(v) for v in board_index.values()))

    # 2. Sector ranking via efinance probe aggregation
    sector_ranking = _cache_sector_ranking(cache_dir, board_index, td)
    logger.info("Sector ranking: %d sectors ranked", len(sector_ranking))

    # 2.5 Hot sector constituents via efinance (for sector_resonance signal matching)
    _cache_hot_sector_constituents(cache_dir, td)

    # 3. Industry map via baostock (best-effort)
    _cache_industry_map(cache_dir)

    # 4. Daily data snapshot for top stocks via baostock (best-effort)
    _cache_daily_snapshot(cache_dir, board_index, td, days_back=days_back)

    # 5. Dragon-tiger via efinance
    _cache_dragon_tiger(cache_dir, td)

    # 6. Limit-up pool via akshare
    _cache_limit_up(cache_dir, td)

    # 7. Risk snapshot. This is cheap and makes offline scans auditable.
    cache_risk_snapshot(td, output_dir=str(cache_dir.parent))

    # 8. Compute short-term signals from cached data (batch mode)
    if compute_signals:
        _run_signal_computation(cache_dir, td)

    logger.info("=== Cache complete: %s ===", cache_dir)
    _print_cache_gaps(cache_dir, td)
    _print_summary(cache_dir)

    return cache_dir


def cache_news_for_ticker(
    ticker: str,
    trade_date: str | None = None,
    *,
    output_dir: str | None = None,
    keyword: str | None = None,
    limit: int = 50,
) -> Path:
    """Download and cache news for one ticker using free sources."""
    td = trade_date or date.today().isoformat()
    cache_dir = Path(output_dir or config.get("results_dir")) / "local_cache"
    news_dir = cache_dir / "news" / td
    news_dir.mkdir(parents=True, exist_ok=True)
    path = news_dir / f"{ticker.replace('.', '_')}.json"

    if path.exists():
        return path

    records = _fetch_news_records(ticker, keyword=keyword, limit=limit)
    atomic_write_text(path, json.dumps(records, ensure_ascii=False, indent=2))
    logger.info("News cache saved: %s (%d records)", path, len(records))
    return path


def cache_news_for_sector(
    sector_name: str,
    trade_date: str | None = None,
    *,
    output_dir: str | None = None,
    constituent_tickers: list[str] | None = None,
    constituent_limit: int = 5,
    per_ticker_limit: int = 12,
) -> Path:
    """Download and cache sector-related news by aggregating constituent ticker news."""
    td = trade_date or date.today().isoformat()
    cache_dir = Path(output_dir or config.get("results_dir")) / "local_cache"
    news_dir = cache_dir / "sector_news" / td
    news_dir.mkdir(parents=True, exist_ok=True)
    path = news_dir / f"{_safe_path_part(sector_name, 'sector')}.json"

    if path.exists():
        return path

    if constituent_tickers is None:
        board_path = cache_dir / "board_index.json"
        if board_path.exists():
            board_index = json.loads(board_path.read_text("utf-8"))
            constituent_tickers = [
                str(item.get("code", ""))
                for item in board_index.get(sector_name, [])[:constituent_limit]
                if item.get("code")
            ]
        else:
            constituent_tickers = []

    constituent_tickers = list(dict.fromkeys((constituent_tickers or [])[:constituent_limit]))
    aggregated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ticker in constituent_tickers:
        records = _fetch_news_records(ticker, keyword=None, limit=per_ticker_limit)
        for record in records:
            enriched = {
                **record,
                "sector_name": sector_name,
                "related_ticker": ticker,
                "news_scope": "sector",
                "event_type": record.get("event_type", "板块新闻"),
                "direct_beneficiaries": [sector_name],
            }
            key = str(enriched.get("url") or enriched.get("title") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            text = json.dumps(enriched, ensure_ascii=False).lower()
            if sector_name and sector_name.lower() not in text:
                enriched["sector_match_mode"] = "constituent_proxy"
            else:
                enriched["sector_match_mode"] = "direct_keyword"
            aggregated.append(enriched)

    atomic_write_text(path, json.dumps(aggregated, ensure_ascii=False, indent=2))
    logger.info(
        "Sector news cache saved: %s (%d records, %d constituents)",
        path,
        len(aggregated),
        len(constituent_tickers),
    )
    return path


def cache_risk_snapshot(
    trade_date: str | None = None,
    *,
    output_dir: str | None = None,
) -> Path:
    """Download and cache one daily risk snapshot using free sources."""
    td = trade_date or date.today().isoformat()
    cache_dir = Path(output_dir or config.get("results_dir")) / "local_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"risk_{td}.json"

    if path.exists():
        return path

    payload = {
        "st_status": _fetch_first_available("get_st_status", trade_date=td),
        "suspended": _fetch_first_available("get_suspended", trade_date=td),
        "delisting": _fetch_first_available("get_delisting", trade_date=td),
    }
    previous = _load_latest_risk_snapshot(cache_dir, exclude_trade_date=td)
    if previous:
        if not payload["st_status"]:
            payload["st_status"] = previous.get("st_status", [])
        if not payload["suspended"]:
            payload["suspended"] = previous.get("suspended", [])
        if not payload["delisting"]:
            payload["delisting"] = previous.get("delisting", [])
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info(
        "Risk cache saved: %s (st=%d suspended=%d delisting=%d)",
        path,
        len(payload["st_status"]),
        len(payload["suspended"]),
        len(payload["delisting"]),
    )
    return path


def ensure_candidate_daily_cache(
    tickers: list[str],
    trade_date: str,
    *,
    output_dir: str | None = None,
    days_back: int = 90,
    min_sleep_seconds: float = 0.2,
    max_sleep_seconds: float = 0.6,
) -> list[dict[str, Any]]:
    """Conservatively fill daily cache for shortlisted scan candidates.

    Order:
      1. Use project cache if it already covers trade_date.
      2. Copy from the user's global ATA cache if available.
      3. Fetch one ticker at a time via low-ban-risk free adapters.

    This intentionally handles only a small candidate list, not the full market.
    """
    cache_dir = Path(output_dir or config.get("results_dir")) / "local_cache"
    daily_dir = cache_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    global_daily_dir = Path.home() / ".advanced_trading_agent" / "results" / "local_cache" / "daily"
    start_date = (date.fromisoformat(trade_date) - timedelta(days=days_back)).isoformat()

    statuses: list[dict[str, Any]] = []
    for ticker in list(dict.fromkeys(tickers)):
        target = daily_dir / f"{ticker.replace('.', '_')}.parquet"
        status: dict[str, Any] = {
            "ticker": ticker,
            "path": str(target),
            "status": "missing",
            "source": None,
            "record_count": 0,
        }

        if _daily_file_covers(target, trade_date):
            status.update({"status": "cache_hit", "source": "project_cache", "record_count": _daily_record_count(target)})
            statuses.append(status)
            continue

        global_source = global_daily_dir / target.name
        if _daily_file_covers(global_source, trade_date):
            shutil.copy2(global_source, target)
            status.update({"status": "copied", "source": "global_cache", "record_count": _daily_record_count(target)})
            statuses.append(status)
            continue

        fetched = _fetch_candidate_daily(ticker, start_date=start_date, end_date=trade_date)
        if fetched:
            import pandas as pd

            pd.DataFrame(fetched).to_parquet(target, index=False)
            status.update({
                "status": "fetched",
                "source": str(fetched[0].get("data_source") or "vendor"),
                "record_count": len(fetched),
            })
        else:
            status.update({"status": "unavailable", "error": "no vendor returned daily data"})
        statuses.append(status)
        _vendor_jitter(min_sleep_seconds, max_sleep_seconds)

    return statuses


# ------------------------------------------------------------------
# Cache builders
# ------------------------------------------------------------------


def _cache_board_index(cache_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Build and save board→stocks reverse index from efinance."""
    path = cache_dir / "board_index.json"

    try:
        import efinance as ef
    except ImportError:
        logger.warning("efinance not installed, skipping board index")
        return {}

    # Probe stocks — diverse set across sectors
    codes = [
        '000001', '000002', '000858', '002415', '300750', '600519', '601398',
        '600036', '000333', '002594', '300059', '600030', '601857', '688981',
        '000725', '002230', '600276', '601318', '000651', '603259', '600900',
        '300124', '601012', '600104', '000568', '002475', '300433', '600809',
        '002714', '300498', '600585', '000063', '688111', '601899', '600050',
        '002352', '300015', '600887', '601088', '600028', '000792', '600941',
        '600000', '600016', '600031', '600048', '600085', '600111', '600150',
        '600196', '600309', '600406', '600436', '600438', '600570', '600588',
        '600690', '600703', '600745', '600837', '600893', '601006', '601111',
        '601166', '601328', '601668', '601939', '603160', '603288', '603501',
        '603986', '688008', '688012', '688036', '688126', '688169', '688187',
        '000066', '000876', '000895', '002007', '002049', '002142', '002241',
        '002271', '002304', '002460', '300003', '300347', '300760',
    ]
    codes = list(dict.fromkeys(codes))

    board_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    done = 0
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
            done += 1
        except Exception:
            continue
        _vendor_jitter()

    # Convert to regular dict for JSON
    result = {k: v for k, v in board_index.items()}
    atomic_write_text(path, json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("Board index saved: %d boards from %d stocks queried", len(result), done)
    return result


def _cache_sector_ranking(
    cache_dir: Path,
    board_index: dict[str, list[dict[str, str]]],
    trade_date: str,
) -> list[dict[str, Any]]:
    """Compute and save sector ranking from board index + efinance changes."""
    path = cache_dir / f"sector_ranking_{trade_date}.json"

    try:
        import efinance as ef
    except ImportError:
        logger.warning("efinance not installed, skipping sector ranking")
        return []

    # Get board change % from probe stocks
    probe_codes = list(dict.fromkeys(
        c for stocks in board_index.values() for c in [s["code"] for s in stocks[:1]]
    ))[:60]

    board_changes: dict[str, list[float]] = defaultdict(list)
    for code in probe_codes:
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
        _vendor_jitter()

    rankings: list[dict[str, Any]] = []
    for name, changes in board_changes.items():
        if len(changes) >= 2:
            avg = sum(changes) / len(changes)
            rankings.append({
                "sector_name": name,
                "change_pct": round(avg, 2),
                "strength_score": round(avg, 2),
                "data_source": "efinance_cached",
                "rank": 0,
            })

    rankings.sort(key=lambda x: x["change_pct"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    atomic_write_text(path, json.dumps(rankings, ensure_ascii=False, indent=2))
    return rankings


def _cache_industry_map(cache_dir: Path) -> dict[str, str]:
    """Best-effort: download stock→industry mapping from baostock."""
    path = cache_dir / "industry_map.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))

    try:
        import baostock as bs
    except ImportError:
        logger.warning("baostock not installed, skipping industry map")
        return {}

    bs.login()
    try:
        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            logger.warning("baostock industry query failed: %s", rs.error_msg)
            return {}

        mapping: dict[str, str] = {}
        while rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            code = row.get("code", "")
            industry = row.get("industry", "")
            if code and industry:
                parts = code.split(".")
                if len(parts) == 2:
                    ticker = f"{parts[1]}.{parts[0].upper()}"
                    mapping[ticker] = industry

        atomic_write_text(path, json.dumps(mapping, ensure_ascii=False))
        logger.info("Industry map: %d stocks → %d industries", len(mapping), len(set(mapping.values())))
        return mapping
    except Exception as exc:
        logger.warning("baostock industry map failed: %s", exc)
        return {}
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def _cache_daily_snapshot(
    cache_dir: Path,
    board_index: dict[str, list[dict[str, str]]],
    trade_date: str,
    days_back: int = 60,
) -> int:
    """Download recent daily data via mootdx (TCP, bypasses DPI).

    Falls back to baostock if mootdx is not installed.
    """
    daily_dir = cache_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    # Collect all tickers from board index
    all_tickers: set[str] = set()
    for stocks in board_index.values():
        for s in stocks[:5]:
            all_tickers.add(s["code"])

    # Try to load industry_map for full coverage
    im_path = cache_dir / "industry_map.json"
    if im_path.exists():
        industry_map = json.loads(im_path.read_text("utf-8"))
        all_tickers.update(industry_map.keys())

    # Limit to 500 to control download time
    tickers = sorted(all_tickers)[:500]

    # Try mootdx first (TCP, no DPI issues)
    try:
        from mootdx.quotes import Quotes
        client = Quotes.factory(market="std")
        cached = _cache_via_mootdx(client, tickers, daily_dir)
        if cached > 0:
            logger.info("Daily data (mootdx): %d stocks cached", cached)
            return cached
    except ImportError:
        logger.debug("mootdx not installed, trying baostock")

    # Fallback to baostock
    return _cache_via_baostock(tickers, daily_dir, trade_date, days_back=days_back)


def _cache_via_mootdx(client: Any, tickers: list[str], daily_dir: Path) -> int:
    """Download daily data via mootdx TCP protocol."""
    import pandas as pd

    cached = 0
    for ticker in tickers:
        cache_path = daily_dir / f"{ticker.replace('.', '_')}.parquet"
        if cache_path.exists():
            cached += 1
            continue

        try:
            code = ticker.split(".")[0]
            df = client.bars(symbol=code, frequency=9, offset=60)
            if df is not None and len(df) > 0:
                df["code"] = ticker
                df["data_source"] = "mootdx"
                df.to_parquet(cache_path, index=False)
                cached += 1
        except Exception:
            continue
        _vendor_jitter()

    return cached


def _cache_via_baostock(
    tickers: list[str], daily_dir: Path, trade_date: str,
    days_back: int = 60,
) -> int:
    """Download daily data via baostock (fallback)."""
    try:
        import baostock as bs
        import pandas as pd
        import datetime as dt
    except ImportError:
        return 0

    end_dt = date.fromisoformat(trade_date)
    start_dt = end_dt - dt.timedelta(days=days_back)
    cached = 0

    bs.login()
    try:
        for ticker in tickers:
            cache_path = daily_dir / f"{ticker.replace('.', '_')}.parquet"
            if cache_path.exists():
                cached += 1
                continue

            try:
                parts = ticker.split(".")
                code = f"{parts[1].lower()}.{parts[0]}"
                rs = bs.query_history_k_data_plus(
                    code, "date,code,open,high,low,close,preclose,volume,amount,pctChg,turn",
                    start_date=start_dt.isoformat(),
                    end_date=end_dt.isoformat(),
                    frequency="d", adjustflag="2",
                )
                if rs.error_code != "0":
                    continue
                rows = []
                while rs.next():
                    rows.append(dict(zip(rs.fields, rs.get_row_data())))
                if rows:
                    df = pd.DataFrame(rows)
                    df["data_source"] = "baostock_cached"
                    df["code"] = ticker
                    df.to_parquet(cache_path, index=False)
                    cached += 1
            except Exception:
                continue
            _vendor_jitter(0.2, 0.6)
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    return cached


def _fetch_candidate_daily(ticker: str, *, start_date: str, end_date: str) -> list[dict[str, Any]]:
    from .collector import get_daily_baostock, get_daily_mootdx

    last_error: Exception | None = None
    for fetcher in (get_daily_mootdx, get_daily_baostock):
        try:
            rows = fetcher(ticker, start_date=start_date, end_date=end_date)
            if rows:
                return rows
        except Exception as exc:
            last_error = exc
            logger.warning("Candidate daily fetch failed for %s via %s: %s", ticker, fetcher.__name__, exc)
    if last_error:
        logger.debug("No candidate daily data for %s; last error=%s", ticker, last_error)
    return []


def _daily_file_covers(path: Path, trade_date: str) -> bool:
    if not path.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        if df.empty:
            return False
        date_col = _daily_date_column(df)
        if date_col is None:
            return True
        target = pd.to_datetime(trade_date).date()
        values = pd.to_datetime(df[date_col], errors="coerce").dt.date
        return bool((values == target).any())
    except Exception:
        return False


def _daily_record_count(path: Path) -> int:
    try:
        import pandas as pd

        return int(len(pd.read_parquet(path)))
    except Exception:
        return 0


def _daily_date_column(df: Any) -> str | None:
    for candidate in ("trade_date", "datetime", "date"):
        if candidate in df.columns:
            return candidate
    return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _cache_dragon_tiger(cache_dir: Path, trade_date: str) -> int:
    """Download dragon-tiger board data via efinance."""
    path = cache_dir / f"dragon_tiger_{trade_date}.json"
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        logger.info("Dragon-tiger: %d records (cached)", len(data))
        return len(data)

    try:
        import efinance as ef
    except ImportError:
        logger.warning("efinance not installed, skipping dragon-tiger")
        return 0

    try:
        df = ef.stock.get_daily_billboard(start_date=trade_date, end_date=trade_date)
        if df is None or df.empty:
            logger.warning("Dragon-tiger: no data for %s", trade_date)
            return 0

        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            records.append({
                "code": str(row.get("股票代码", "")),
                "name": str(row.get("股票名称", "")),
                "date": str(row.get("上榜日期", "")),
                "close": float(row.get("收盘价", 0) or 0),
                "change_pct": float(row.get("涨跌幅", 0) or 0),
                "turnover": float(row.get("换手率", 0) or 0),
                "net_buy": float(row.get("龙虎榜净买额", 0) or 0),
                "reason": str(row.get("上榜原因", "")),
            })

        atomic_write_text(path, json.dumps(records, ensure_ascii=False))
        logger.info("Dragon-tiger: %d records saved", len(records))
        return len(records)
    except Exception as exc:
        logger.warning("Dragon-tiger failed: %s", exc)
        return 0


def _cache_limit_up(cache_dir: Path, trade_date: str) -> dict[str, Any]:
    """Download limit-up pool via vendor router (akshare)."""
    path = cache_dir / f"limit_up_{trade_date}.json"
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        logger.info("Limit-up: %d stocks (cached)", len(data.get("stocks", [])))
        return data

    try:
        from .vendor_router import route_to_vendor

        data = route_to_vendor("get_limit_up_tiers", trade_date=trade_date)
        if isinstance(data, dict):
            atomic_write_text(path, json.dumps(data, ensure_ascii=False, default=str))
            logger.info("Limit-up: %d first, %d second, %d third+, %d stocks",
                        data.get("first_board", 0), data.get("second_board", 0),
                        data.get("third_plus", 0), len(data.get("stocks", [])))
            return data
    except Exception as exc:
        logger.warning("Limit-up failed: %s", exc)

    fallback = {
        "first_board": 0,
        "second_board": 0,
        "third_plus": 0,
        "stocks": [],
        "data_source": "empty_fallback",
        "note": "limit_up data unavailable during cache build",
    }
    atomic_write_text(path, json.dumps(fallback, ensure_ascii=False, indent=2))
    logger.warning("Limit-up unavailable; wrote empty fallback cache: %s", path)
    return fallback


def _cache_hot_sector_constituents(cache_dir: Path, trade_date: str) -> dict[str, list[str]]:
    """Build hot sector→tickers mapping from get_belong_board queries.

    Iterates through all daily cache tickers, queries get_belong_board
    for each, and builds a reverse index of board_name → [tickers].
    Only keeps top-10 hot sectors (matching sector_ranking).

    This provides direct 同花顺命名体系 的板块→股票映射，
    解决 industry_map（证监会行业）与 sector_ranking（概念板块）命名不匹配的问题。
    """
    path = cache_dir / f"hot_sector_constituents_{trade_date}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text("utf-8"))
            logger.info("Hot sector constituents: %d sectors (cached)", len(data))
            return data
        except Exception:
            pass

    try:
        import efinance as ef
    except ImportError:
        logger.warning("efinance not installed, skipping hot sector constituents")
        return {}

    # 1. Collect all daily tickers from parquet files
    daily_dir = cache_dir / "daily"
    if not daily_dir.exists():
        logger.warning("No daily cache dir, skipping hot sector constituents")
        return {}

    all_tickers: list[str] = []
    for parquet_path in sorted(daily_dir.glob("*.parquet")):
        ticker = parquet_path.stem.replace("_", ".")
        parts = ticker.split(".")
        if len(parts) == 2 and len(parts[0]) == 6 and parts[1] in ("SH", "SZ", "BJ"):
            all_tickers.append(ticker)
    all_tickers = sorted(set(all_tickers))
    logger.info("Building hot sector map from %d daily tickers...", len(all_tickers))

    # 2. Read sector ranking to know which sectors are "hot"
    ranking_path = cache_dir / f"sector_ranking_{trade_date}.json"
    if not ranking_path.exists():
        logger.warning("No sector ranking for %s", trade_date)
        return {}
    try:
        ranking = json.loads(ranking_path.read_text("utf-8"))
        hot_sector_names = {s["sector_name"] for s in ranking[:10] if s.get("sector_name")}
    except Exception as exc:
        logger.warning("Failed to parse sector ranking: %s", exc)
        return {}

    if not hot_sector_names:
        return {}

    # 3. Build board_name → [tickers] mapping via get_belong_board
    reverse: dict[str, list[str]] = {}
    processed = 0
    for ticker in all_tickers:
        digits = ticker.split(".")[0]
        try:
            df = ef.stock.get_belong_board(digits)
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                bname = str(row.get("板块名称", ""))
                if bname in hot_sector_names:
                    if bname not in reverse:
                        reverse[bname] = []
                    reverse[bname].append(ticker)
            processed += 1
        except Exception:
            continue
        _vendor_jitter(0.02, 0.08)  # minimal jitter for local endpoint

    # 4. Deduplicate
    for bname in reverse:
        reverse[bname] = sorted(set(reverse[bname]))

    # 5. Save
    atomic_write_text(path, json.dumps(reverse, ensure_ascii=False, indent=2))
    total = sum(len(v) for v in reverse.values())
    logger.info("Hot sector constituents saved: %d/%d sectors, %d total tickers (from %d queries)",
                len(reverse), len(hot_sector_names), total, processed)
    return reverse


def _print_summary(cache_dir: Path) -> None:
    """Print cache summary."""
    files = list(cache_dir.glob("**/*")) if cache_dir.exists() else []
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"\nCache directory: {cache_dir}")
    print(f"Files: {len(files)}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    for f in sorted(files):
        if f.is_file():
            print(f"  {f.relative_to(cache_dir)} ({f.stat().st_size / 1024:.0f} KB)")


def _cache_ready(cache_dir: Path, trade_date: str) -> bool:
    return not _cache_gaps(cache_dir, trade_date)


def _cache_gaps(cache_dir: Path, trade_date: str, *, require_news: bool = False) -> list[str]:
    required = [
        ("board_index", cache_dir / "board_index.json"),
        ("sector_ranking", cache_dir / f"sector_ranking_{trade_date}.json"),
        ("dragon_tiger", cache_dir / f"dragon_tiger_{trade_date}.json"),
        ("limit_up", cache_dir / f"limit_up_{trade_date}.json"),
        ("risk_snapshot", cache_dir / f"risk_{trade_date}.json"),
    ]
    daily_dir = cache_dir / "daily"
    gaps = [name for name, path in required if not path.exists()]
    if not daily_dir.exists() or not any(daily_dir.glob("*.parquet")):
        gaps.append("daily_parquet")
    if require_news:
        news_dir = cache_dir / "news" / trade_date
        sector_news_dir = cache_dir / "sector_news" / trade_date
        has_news = (news_dir.exists() and any(news_dir.glob("*.json"))) or (
            sector_news_dir.exists() and any(sector_news_dir.glob("*.json"))
        )
        if not has_news:
            gaps.append("news_optional")
    return gaps


def _print_cache_gaps(cache_dir: Path, trade_date: str) -> None:
    gaps = _cache_gaps(cache_dir, trade_date)
    optional_news = _cache_gaps(cache_dir, trade_date, require_news=True)
    optional_only = [gap for gap in optional_news if gap not in gaps]
    if gaps:
        print(f"\nCache gaps for {trade_date}: {', '.join(gaps)}")
    else:
        print(f"\nCache gaps for {trade_date}: none")
    if optional_only:
        print(f"Optional cache gaps: {', '.join(optional_only)}")


def _load_latest_risk_snapshot(cache_dir: Path, *, exclude_trade_date: str) -> dict[str, Any] | None:
    snapshots = sorted(cache_dir.glob("risk_*.json"), reverse=True)
    for path in snapshots:
        if path.stem == f"risk_{exclude_trade_date}":
            continue
        try:
            payload = json.loads(path.read_text("utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _safe_path_part(value: str, fallback: str) -> str:
    safe = _SAFE_PATH_PART.sub("_", value).strip("._")
    return safe or fallback


def _fetch_news_records(
    ticker: str,
    *,
    keyword: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    ensure_default_vendor_registration()
    attempts = [
        ("sina", {"code": ticker, "keyword": keyword, "limit": limit}),
        ("cls", {"code": ticker, "keyword": keyword, "limit": limit}),
        ("akshare", {"code": ticker, "keyword": keyword, "limit": limit}),
    ]
    for vendor, kwargs in attempts:
        impl = get_vendor_impl("get_news", vendor)
        if impl is None:
            continue
        try:
            data = impl(**kwargs)
            if isinstance(data, list) and data:
                return data
        except Exception as exc:
            logger.warning("news cache fetch failed for %s via %s: %s", ticker, vendor, exc)
            continue
    return []


def _fetch_first_available(method: str, **kwargs: Any) -> list[Any]:
    ensure_default_vendor_registration()
    vendor_order = {
        "get_st_status": ["local_cache"],
        "get_suspended": ["local_cache", "baostock"],
        "get_delisting": ["local_cache"],
    }.get(method, [])
    for vendor in vendor_order:
        impl = get_vendor_impl(method, vendor)
        if impl is None:
            continue
        try:
            data = impl(**kwargs)
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("risk cache fetch failed for %s via %s: %s", method, vendor, exc)
            continue
    return []


# ------------------------------------------------------------------
# Short-term signal computation (batch from cache)
# ------------------------------------------------------------------


def _run_signal_computation(cache_dir: Path, trade_date: str) -> None:
    """Compute short-term signals from cached data and save results."""
    try:
        from .short_term_signals import compute_short_term_signals

        logger.info("=== Computing short-term signals for %s ===", trade_date)
        results = compute_short_term_signals(
            trade_date=trade_date,
            cache_dir=cache_dir,
            save=True,
        )
        if results:
            bullish = sum(1 for r in results.values() if r.composite >= 55)
            bearish = sum(1 for r in results.values() if r.composite <= 45)
            logger.info(
                "Signals computed: %d tickers (%d bullish, %d bearish, %d neutral)",
                len(results), bullish, bearish,
                len(results) - bullish - bearish,
            )
        else:
            logger.warning("No signals computed (cache data may be insufficient)")
    except ImportError:
        logger.warning("short_term_signals module not available, skipping signal computation")
    except Exception as exc:
        logger.warning("Signal computation failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Build local data cache for offline use")
    parser.add_argument("--date", "-d", help="Trade date (default: today)")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--days-back", type=int, default=60,
                        help="Days of daily data to cache (default: 60, for signal computation)")
    parser.add_argument("--compute-signals", action="store_true", default=True,
                        help="Compute short-term signals after caching (default: True)")
    parser.add_argument("--no-signals", action="store_false", dest="compute_signals",
                        help="Skip signal computation")
    args = parser.parse_args()

    build(trade_date=args.date, output_dir=args.output_dir,
          days_back=args.days_back, compute_signals=args.compute_signals)


if __name__ == "__main__":
    main()
