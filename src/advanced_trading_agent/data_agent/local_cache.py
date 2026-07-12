"""Local data cache built from baostock — no eastmoney dependency.

Downloads and caches daily OHLCV data + industry classifications from
baostock, then computes sector rankings locally. When eastmoney push2
endpoints are blocked, the scanner falls back to this cache.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import config
from ..core.atomic_write import atomic_write_text
from .cache_manifest import CacheManifest

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(config.get("results_dir")) / "local_cache"
_SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+")


def _workspace_cache_dir() -> Path:
    return Path(config.get("project_dir", ".")) / "local_cache"


@dataclass
class LocalCache:
    """Local data cache independent of eastmoney push2 endpoints."""

    days_back: int = 30
    cache_dir: Path = field(default_factory=lambda: _CACHE_DIR)

    def __post_init__(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self.cache_dir = _workspace_cache_dir()
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_sector_data(self, trade_date: str | None = None) -> dict[str, Any]:
        """Ensure sector ranking + constituents cache exists and is fresh.

        Returns a dict with:
          - sectors: list of {sector_name, change_pct, strength_score}
          - constituents: {sector_name: [{code, name}, ...]}
        """
        td = trade_date or date.today().isoformat()
        cache_path = self.cache_dir / f"sector_cache_{td}.json"

        if cache_path.exists():
            logger.info("Sector cache hit for %s", td)
            return json.loads(cache_path.read_text("utf-8"))

        logger.info("Building sector cache for %s from baostock...", td)
        data = self._build_sector_cache(td)
        atomic_write_text(cache_path, json.dumps(data, ensure_ascii=False, indent=2))
        return data

    def ensure_daily_data(
        self, ticker: str, start_date: str | None = None, end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get daily OHLCV data, incrementally repairing local parquet gaps.

        The local cache is still preferred. If the requested date range is not
        covered, only that requested range is fetched from BaoStock and merged
        into the existing parquet file. When online repair fails, existing
        cached rows are returned with ``_cache_status=stale_cache_used``.
        """
        end = _normalize_date(end_date or date.today().isoformat())
        start = _normalize_date(start_date or (date.fromisoformat(end) - timedelta(days=self.days_back)).isoformat())
        cache_path = self.cache_dir / "daily" / f"{ticker.replace('.', '_')}.parquet"
        manifest = CacheManifest(self.cache_dir)

        existing = pd.DataFrame()
        if cache_path.exists():
            existing = self._read_daily_cache(cache_path)
            if self._covers_range(existing, start, end):
                self._update_daily_manifest(manifest, ticker, cache_path, existing, status="cache_hit")
                return self._records_with_cache_status(
                    self._filter_daily_frame(existing, start, end),
                    "cache_hit",
                )

        try:
            fetched = self._fetch_daily_baostock(ticker, start, end)
        except Exception as exc:
            logger.warning("Daily cache repair failed for %s: %s", ticker, exc)
            if not existing.empty:
                self._update_daily_manifest(
                    manifest,
                    ticker,
                    cache_path,
                    existing,
                    status="stale_cache_used",
                    notes=[str(exc)],
                )
                return self._records_with_cache_status(
                    self._filter_daily_frame(existing, start, end),
                    "stale_cache_used",
                )
            return []

        if fetched.empty:
            if not existing.empty:
                self._update_daily_manifest(
                    manifest,
                    ticker,
                    cache_path,
                    existing,
                    status="cache_partial",
                    notes=["vendor_fetch_returned_empty"],
                )
                return self._records_with_cache_status(
                    self._filter_daily_frame(existing, start, end),
                    "cache_partial",
                )
            return []

        merged = self._merge_daily_frames(existing, fetched)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path, index=False)
        self._update_daily_manifest(manifest, ticker, cache_path, merged, status="vendor_fetch")
        return self._records_with_cache_status(
            self._filter_daily_frame(merged, start, end),
            "vendor_fetch",
        )

    def _read_daily_cache(self, cache_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(cache_path)
        date_col = _daily_date_column(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df

    def _fetch_daily_baostock(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError("baostock not installed") from exc

        login_result = bs.login()
        if getattr(login_result, "error_code", "0") != "0":
            raise RuntimeError(getattr(login_result, "error_msg", "baostock login failed"))
        try:
            rs = bs.query_history_k_data_plus(
                self._to_baostock_code(ticker),
                "date,code,open,high,low,close,preclose,volume,amount,pctChg,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            rows: list[dict[str, Any]] = []
            while getattr(rs, "error_code", "0") == "0" and rs.next():
                row = dict(zip(rs.fields, rs.get_row_data()))
                row["data_source"] = "baostock"
                row["code"] = ticker
                rows.append(row)
            return pd.DataFrame(rows)
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    def _merge_daily_frames(self, existing: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
        frames = [df for df in (existing, fetched) if df is not None and not df.empty]
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True, sort=False)
        date_col = _daily_date_column(merged)
        if date_col:
            merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
            dedupe_cols = [date_col]
            if "code" in merged.columns:
                dedupe_cols.insert(0, "code")
            merged = merged.dropna(subset=[date_col]).drop_duplicates(subset=dedupe_cols, keep="last")
            merged = merged.sort_values(date_col)
        return merged

    def _filter_daily_frame(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if df.empty:
            return df
        frame = df.copy()
        date_col = _daily_date_column(frame)
        if not date_col:
            return frame
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        dates = frame[date_col].dt.date
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        return frame[(dates >= start) & (dates <= end)]

    def _covers_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        bounds = self._daily_bounds(df)
        if not bounds:
            return False
        observed_start, observed_end = bounds
        return observed_start <= pd.to_datetime(start_date).date() and observed_end >= pd.to_datetime(end_date).date()

    def _daily_bounds(self, df: pd.DataFrame) -> tuple[date, date] | None:
        if df.empty:
            return None
        date_col = _daily_date_column(df)
        if not date_col:
            return None
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.date
        if dates.empty:
            return None
        return dates.min(), dates.max()

    def _update_daily_manifest(
        self,
        manifest: CacheManifest,
        ticker: str,
        cache_path: Path,
        df: pd.DataFrame,
        *,
        status: str,
        notes: list[str] | None = None,
    ) -> None:
        bounds = self._daily_bounds(df)
        source = self._source_from_frame(df)
        manifest.update_daily(
            ticker=ticker,
            path=cache_path,
            start_date=str(bounds[0]) if bounds else None,
            end_date=str(bounds[1]) if bounds else None,
            source=source,
            row_count=int(len(df)),
            status=status,
            notes=notes,
        )

    @staticmethod
    def _source_from_frame(df: pd.DataFrame) -> str:
        if "data_source" not in df.columns:
            return "local_cache"
        sources = sorted({str(value) for value in df["data_source"].dropna().unique() if str(value)})
        return ",".join(sources) if sources else "local_cache"

    @staticmethod
    def _records_with_cache_status(df: pd.DataFrame, status: str) -> list[dict[str, Any]]:
        if df.empty:
            return []
        records = df.to_dict("records")
        for record in records:
            record.setdefault("data_source", "local_cache")
            record["_cache_status"] = status
        return records

    def build_full_cache(self, trade_date: str | None = None) -> str:
        """Build complete local cache: industry index + sector rankings.

        Downloads data from baostock for all A-share stocks and caches
        to local parquet files. One-time operation, takes ~5-10 min.
        """
        td = trade_date or date.today().isoformat()
        logger.info("Building full local cache for %s...", td)

        # 1. Industry index (stock → industry mapping)
        industry_map = self._fetch_industry_map()
        ind_path = self.cache_dir / "industry_map.json"
        atomic_write_text(ind_path, json.dumps(industry_map, ensure_ascii=False))
        logger.info("Industry map: %d stocks → %d industries cached", len(industry_map), len(set(industry_map.values())))

        # 2. Build sector ranking from recent daily data
        sector_data = self._build_sector_cache(td)
        sec_path = self.cache_dir / f"sector_cache_{td}.json"
        atomic_write_text(sec_path, json.dumps(sector_data, ensure_ascii=False, indent=2))
        logger.info("Sector cache: %d sectors ranked", len(sector_data.get("sectors", [])))

        # 3. Compute and cache daily data for top stocks
        daily_dir = self.cache_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        cached_count = self._cache_daily_batch(td, industry_map)
        logger.info("Daily data: %d stocks cached", cached_count)

        return str(self.cache_dir)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _fetch_industry_map(self) -> dict[str, str]:
        """Query baostock for stock→industry mapping."""
        try:
            import baostock as bs
        except ImportError:
            logger.warning("baostock not installed")
            return {}

        bs.login()
        try:
            rs = bs.query_stock_industry()
            mapping: dict[str, str] = {}
            while rs.next():
                row = dict(zip(rs.fields, rs.get_row_data()))
                code = row.get("code", "")
                industry = row.get("industry", "")
                if code and industry:
                    # Normalize: sz.000001 → 000001.SZ
                    parts = code.split(".")
                    if len(parts) == 2:
                        suffix = parts[0].upper()
                        digits = parts[1]
                        ticker = f"{digits}.{suffix}"
                        mapping[ticker] = industry
            return mapping
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    def _build_sector_cache(self, trade_date: str) -> dict[str, Any]:
        """Compute sector rankings from recent daily data.

        1. Download daily data for representative stocks from each industry
        2. Aggregate by industry to compute avg change %
        3. Rank industries by performance
        """
        # Load industry map
        industry_map = self._load_industry_map()
        if not industry_map:
            industry_map = self._fetch_industry_map()

        if not industry_map:
            return {"sectors": [], "constituents": {}, "source": "baostock", "note": "no industry data available"}

        # Get representative stocks per industry (first 10 per industry)
        ind_stocks: dict[str, list[str]] = defaultdict(list)
        for ticker, ind in industry_map.items():
            if len(ind_stocks[ind]) < 10:
                ind_stocks[ind].append(ticker)

        # Download recent daily data for these stocks
        end_dt = date.fromisoformat(trade_date)
        start_dt = end_dt - timedelta(days=self.days_back)

        import baostock as bs
        bs.login()
        try:
            ind_performance: dict[str, list[float]] = defaultdict(list)

            for ind, tickers in ind_stocks.items():
                for ticker in tickers[:5]:  # First 5 stocks per industry
                    try:
                        code = self._to_baostock_code(ticker)
                        rs = bs.query_history_k_data_plus(
                            code, "date,pctChg",
                            start_date=start_dt.isoformat(),
                            end_date=end_dt.isoformat(),
                            frequency="d", adjustflag="2",
                        )
                        while rs.next():
                            row = dict(zip(rs.fields, rs.get_row_data()))
                            pct = float(row.get("pctChg", 0) or 0)
                            ind_performance[ind].append(pct)
                    except Exception:
                        continue

            # Rank by average performance
            rankings: list[dict[str, Any]] = []
            for ind, changes in ind_performance.items():
                if not changes:
                    continue
                avg_pct = sum(changes) / len(changes)
                rankings.append({
                    "sector_name": ind,
                    "change_pct": round(avg_pct, 2),
                    "strength_score": round(avg_pct, 2),
                    "data_source": "baostock_computed",
                })

            rankings.sort(key=lambda x: x["change_pct"], reverse=True)

            # Build constituents
            constituents: dict[str, list[dict[str, str]]] = {}
            for ind, tickers in ind_stocks.items():
                constituents[ind] = [{"code": t, "name": "", "sector": ind} for t in tickers[:30]]

            return {
                "sectors": rankings,
                "constituents": constituents,
                "source": "baostock_computed",
                "trade_date": trade_date,
            }
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    def _cache_daily_batch(self, trade_date: str, industry_map: dict[str, str]) -> int:
        """Cache daily data for top stocks to local parquet."""
        import baostock as bs

        daily_dir = self.cache_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        end_dt = date.fromisoformat(trade_date)
        start_dt = end_dt - timedelta(days=self.days_back)
        cached = 0

        bs.login()
        try:
            for ticker in list(industry_map.keys())[:500]:  # Top 500 stocks
                cache_path = daily_dir / f"{ticker.replace('.', '_')}.parquet"
                if cache_path.exists():
                    cached += 1
                    continue

                try:
                    code = self._to_baostock_code(ticker)
                    rs = bs.query_history_k_data_plus(
                        code, "date,code,open,high,low,close,preclose,volume,amount,pctChg,turn",
                        start_date=start_dt.isoformat(),
                        end_date=end_dt.isoformat(),
                        frequency="d", adjustflag="2",
                    )
                    rows = []
                    while rs.next():
                        rows.append(dict(zip(rs.fields, rs.get_row_data())))
                    if rows:
                        df = pd.DataFrame(rows)
                        df["data_source"] = "baostock"
                        df["code"] = ticker
                        df.to_parquet(cache_path, index=False)
                        cached += 1
                except Exception:
                    continue
        finally:
            try:
                bs.logout()
            except Exception:
                pass

        return cached

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_industry_map(self) -> dict[str, str]:
        """Load cached industry map if it exists."""
        path = self.cache_dir / "industry_map.json"
        if path.exists():
            return json.loads(path.read_text("utf-8"))
        return {}

    @staticmethod
    def _to_baostock_code(ticker: str) -> str:
        """Convert 000001.SZ → sz.000001."""
        parts = ticker.split(".")
        if len(parts) == 2:
            return f"{parts[1].lower()}.{parts[0]}"
        return ticker


def get_cached_sector_data(trade_date: str | None = None, top_n: int = 10) -> list[dict[str, Any]]:
    """Convenience: get sector ranking from local cache.

    Reads from sector_ranking_{date}.json or board_index.json created
    by build_cache.py. Falls back to baostock-based computation if no
    cache exists.
    """
    td = trade_date or date.today().isoformat()
    cache = LocalCache()

    # Prefer efinance-based ranking cache (from build_cache.py)
    ranking_path = cache.cache_dir / f"sector_ranking_{td}.json"
    if ranking_path.exists():
        data = json.loads(ranking_path.read_text("utf-8"))
        return data[:top_n]

    # Fall back to baostock-computed cache
    data = cache.ensure_sector_data(td)
    sectors = data.get("sectors", [])
    return sectors[:top_n]


def get_cached_sector_constituents(
    sector_name: str,
    trade_date: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience: get constituents for a sector from local cache.

    Reads from board_index.json created by build_cache.py.
    """
    cache = LocalCache()

    # Prefer efinance-based board index (from build_cache.py)
    idx_path = cache.cache_dir / "board_index.json"
    if idx_path.exists():
        board_index = json.loads(idx_path.read_text("utf-8"))
        constituents = board_index.get(sector_name, [])
        if constituents:
            return constituents
        # Fuzzy match
        for bname, stocks in board_index.items():
            if sector_name in bname or bname in sector_name:
                return stocks

    # Fall back to baostock-computed cache
    data = cache.ensure_sector_data(trade_date)
    constituents = data.get("constituents", {})
    return constituents.get(sector_name, [])


def get_cached_daily(ticker: str, start_date: str | None = None,
                     end_date: str | None = None) -> list[dict[str, Any]]:
    """Convenience: get daily data from local parquet cache.

    Uses ``LocalCache.ensure_daily_data`` so callers get incremental cache
    repair and manifest updates instead of a raw parquet read.
    """
    cache = LocalCache()
    return cache.ensure_daily_data(ticker, start_date, end_date)


def get_cached_market_breadth(trade_date: str | None = None) -> dict[str, Any]:
    """Compute a market-breadth proxy from cached daily parquet files."""
    td = trade_date or date.today().isoformat()
    cache = LocalCache()
    daily_dir = cache.cache_dir / "daily"
    if not daily_dir.exists():
        return {}

    advance_count = 0
    decline_count = 0
    flat_count = 0
    sample_size = 0
    matched_rows = 0
    target = pd.to_datetime(td).date()

    for path in daily_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty:
            continue
        date_col = None
        for candidate in ("trade_date", "datetime", "date"):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            continue
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        matched = df[df[date_col].dt.date == target]
        if matched.empty:
            continue
        row = matched.iloc[-1]
        row_position = int(matched.index[-1])
        pct = None
        for candidate in ("pct_chg", "pctChg", "change_pct"):
            if candidate in row.index:
                try:
                    pct = float(row[candidate])
                    break
                except (TypeError, ValueError):
                    pct = None
        if pct is None and "close" in df.columns:
            close_series = pd.to_numeric(df["close"], errors="coerce")
            pre_close = None
            for candidate in ("preclose", "pre_close"):
                if candidate in df.columns:
                    pre_close_series = pd.to_numeric(df[candidate], errors="coerce")
                    try:
                        pre_close = float(pre_close_series.loc[row_position])
                    except (KeyError, TypeError, ValueError):
                        pre_close = None
                    break
            if pre_close is None:
                positions = list(df.index)
                ordinal = positions.index(row_position) if row_position in positions else -1
                if ordinal > 0:
                    try:
                        pre_close = float(close_series.loc[positions[ordinal - 1]])
                    except (KeyError, TypeError, ValueError):
                        pre_close = None
            try:
                last_close = float(close_series.loc[row_position])
                if pre_close:
                    pct = (last_close / pre_close - 1.0) * 100
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pct = None
        if pct is None:
            continue
        matched_rows += 1
        if pct > 0:
            advance_count += 1
        elif pct < 0:
            decline_count += 1
        else:
            flat_count += 1
        sample_size += 1

    if sample_size == 0:
        return {}
    return {
        "trade_date": td,
        "advance_count": advance_count,
        "decline_count": decline_count,
        "flat_count": flat_count,
        "sample_size": sample_size,
        "coverage_note": "proxy_from_cached_daily_universe",
        "matched_rows": matched_rows,
    }


def get_cached_dragon_tiger(trade_date: str | None = None) -> list[dict[str, Any]]:
    """Read cached dragon-tiger data."""
    td = trade_date or date.today().isoformat()
    cache = LocalCache()
    path = cache.cache_dir / f"dragon_tiger_{td}.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    return []


def get_cached_limit_up(trade_date: str | None = None) -> dict[str, Any]:
    """Read cached limit-up pool data."""
    td = trade_date or date.today().isoformat()
    cache = LocalCache()
    path = cache.cache_dir / f"limit_up_{td}.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    return {}


def get_cached_news(ticker: str, trade_date: str | None = None) -> list[dict[str, Any]]:
    """Read cached ticker news from local storage."""
    td = trade_date or date.today().isoformat()
    cache = LocalCache()
    path = cache.cache_dir / "news" / td / f"{ticker.replace('.', '_')}.json"
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        if isinstance(data, list):
            return data
    return []


def get_cached_northbound_top10(trade_date: str | None = None) -> list[dict[str, Any]]:
    """Read cached northbound top-10 turnover data."""
    td = trade_date or date.today().isoformat()
    cache = LocalCache()
    candidates = [
        cache.cache_dir / f"northbound_top10_{td}.json",
        cache.cache_dir / f"northbound_{td}.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                rows = data.get("top10") or data.get("records") or data.get("data")
                if isinstance(rows, list):
                    return rows
    return []


def get_cached_sector_news(sector_name: str, trade_date: str | None = None) -> list[dict[str, Any]]:
    """Read cached sector news from local storage."""
    td = trade_date or date.today().isoformat()
    cache = LocalCache()
    path = cache.cache_dir / "sector_news" / td / f"{_safe_path_part(sector_name, 'sector')}.json"
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        if isinstance(data, list):
            return data
    return []


def get_cached_financial(ticker: str) -> list[dict[str, Any]]:
    """Read cached financial records for one ticker."""
    candidates = [
        LocalCache().cache_dir / "financial" / f"{ticker.replace('.', '_')}.json",
        _workspace_cache_dir() / "financial" / f"{ticker.replace('.', '_')}.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, list):
                return data
    return []


def save_cached_financial(ticker: str, records: list[dict[str, Any]]) -> str:
    """Persist financial records for one ticker into local cache."""
    candidates = [
        LocalCache().cache_dir / "financial",
        _workspace_cache_dir() / "financial",
    ]
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    last_error: Exception | None = None
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{ticker.replace('.', '_')}.json"
            atomic_write_text(path, payload)
            return str(path)
        except PermissionError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise PermissionError("failed to persist cached financial records")


def get_cached_risk_blacklist() -> list[dict[str, str]]:
    """Read cached risk blacklist (delisted/ST/suspended)."""
    cache = LocalCache()
    path = cache.cache_dir / "risk_blacklist.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    return []


def get_cached_risk_snapshot(trade_date: str | None = None) -> dict[str, Any]:
    """Read cached daily risk snapshot."""
    td = trade_date or date.today().isoformat()
    cache = LocalCache()
    path = cache.cache_dir / f"risk_{td}.json"
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        if isinstance(data, dict):
            return data
    return {
        "st_status": [],
        "suspended": [],
        "delisting": [],
    }


def _safe_path_part(value: str, fallback: str) -> str:
    safe = _SAFE_PATH_PART.sub("_", value).strip("._")
    return safe or fallback


def _daily_date_column(df: pd.DataFrame) -> str | None:
    for candidate in ("trade_date", "datetime", "date"):
        if candidate in df.columns:
            return candidate
    return None


def _normalize_date(value: str) -> str:
    raw = str(value)
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw
