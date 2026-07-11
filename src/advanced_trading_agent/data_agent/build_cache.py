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
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from ..config import config

logger = logging.getLogger(__name__)


def build(trade_date: str | None = None, output_dir: str | None = None) -> Path:
    """Build complete local cache. Returns path to cache directory."""
    td = trade_date or date.today().isoformat()
    cache_dir = Path(output_dir or config.get("results_dir", "data/results")) / "local_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Building local cache for %s → %s ===", td, cache_dir)

    # 1. Board index (sector → stocks) via efinance
    board_index = _cache_board_index(cache_dir)
    logger.info("Board index: %d sectors, %d stock-sector pairs",
                len(board_index), sum(len(v) for v in board_index.values()))

    # 2. Sector ranking via efinance probe aggregation
    sector_ranking = _cache_sector_ranking(cache_dir, board_index, td)
    logger.info("Sector ranking: %d sectors ranked", len(sector_ranking))

    # 3. Industry map via baostock (best-effort)
    _cache_industry_map(cache_dir)

    # 4. Daily data snapshot for top stocks via baostock (best-effort)
    _cache_daily_snapshot(cache_dir, board_index, td)

    # 5. Dragon-tiger via efinance
    _cache_dragon_tiger(cache_dir, td)

    # 6. Limit-up pool via akshare
    _cache_limit_up(cache_dir, td)

    logger.info("=== Cache complete: %s ===", cache_dir)
    _print_summary(cache_dir)

    return cache_dir


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
        time.sleep(0.3)

    # Convert to regular dict for JSON
    result = {k: v for k, v in board_index.items()}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
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
        time.sleep(0.3)

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

    path.write_text(json.dumps(rankings, ensure_ascii=False, indent=2), "utf-8")
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

        path.write_text(json.dumps(mapping, ensure_ascii=False), "utf-8")
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
    return _cache_via_baostock(tickers, daily_dir, trade_date)


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
        time.sleep(0.3)

    return cached


def _cache_via_baostock(
    tickers: list[str], daily_dir: Path, trade_date: str,
) -> int:
    """Download daily data via baostock (fallback)."""
    try:
        import baostock as bs
        import pandas as pd
        import datetime as dt
    except ImportError:
        return 0

    end_dt = date.fromisoformat(trade_date)
    start_dt = end_dt - dt.timedelta(days=30)
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
            time.sleep(0.5)
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    return cached


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

        path.write_text(json.dumps(records, ensure_ascii=False), "utf-8")
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
            path.write_text(json.dumps(data, ensure_ascii=False, default=str), "utf-8")
            logger.info("Limit-up: %d first, %d second, %d third+, %d stocks",
                        data.get("first_board", 0), data.get("second_board", 0),
                        data.get("third_plus", 0), len(data.get("stocks", [])))
            return data
    except Exception as exc:
        logger.warning("Limit-up failed: %s", exc)

    return {}


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Build local data cache for offline use")
    parser.add_argument("--date", "-d", help="Trade date (default: today)")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    args = parser.parse_args()

    build(trade_date=args.date, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
