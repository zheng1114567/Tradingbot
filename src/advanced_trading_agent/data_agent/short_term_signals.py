"""短线交易信号 — 完全基于本地缓存日线数据

核心原则:
1. 只读 local_cache，不触发任何 API 调用
2. 批量加载 → 批量计算 → 批量输出
3. 每个信号独立计算，互不影响
4. 数据不足时优雅降级（返回 neutral / 低置信度）

信号清单:
  - ma_trend        均线多头排列 (-100~100)
  - volume_breakout 量价突破 (0~100)
  - momentum_accel  动量加速度 (-100~100)
  - limit_up_premium 涨停溢价 (0~100)
  - sector_resonance 板块共振 (0~100)
  - composite       综合短线评分 (0~100, >50 偏多)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import config

logger = logging.getLogger(__name__)

# 各信号在综合评分中的权重
_SIGNAL_WEIGHTS: dict[str, float] = {
    "ma_trend": 0.25,
    "volume_breakout": 0.20,
    "momentum_accel": 0.20,
    "limit_up_premium": 0.20,
    "sector_resonance": 0.15,
}


# ====================================================================
# Data containers
# ====================================================================

@dataclass
class ShortTermSignal:
    """单个信号结果"""
    name: str
    ticker: str
    score: float              # -100~100 (directional) 或 0~100 (magnitude-only)
    direction: str            # "bullish" | "bearish" | "neutral"
    confidence: float         # 0~1
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalReport:
    """单个 ticker 的全量信号报告"""
    ticker: str
    trade_date: str
    signals: dict[str, ShortTermSignal] = field(default_factory=dict)
    composite: float = 50.0   # 0~100, >50 偏多
    n_signals: int = 0

    def to_dict(self) -> dict[str, Any]:
        def _native(v: Any) -> Any:
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, dict):
                return {k: _native(vv) for k, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [_native(x) for x in v]
            return v
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "signals": {k: {
                "name": v.name,
                "score": _native(v.score),
                "direction": v.direction,
                "confidence": _native(v.confidence),
                "details": _native(v.details),
            } for k, v in self.signals.items()},
            "composite": _native(self.composite),
            "n_signals": _native(self.n_signals),
        }


# ====================================================================
# Signal engine
# ====================================================================

class ShortTermSignalEngine:
    """短线信号引擎 — 从本地缓存批量读取数据，批量计算信号"""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir or config.get("results_dir")) / "local_cache"
        self._daily_cache: dict[str, pd.DataFrame] = {}
        self._sector_ranking: list[dict[str, Any]] = []
        self._board_index: dict[str, list[dict[str, Any]]] = {}
        self._hot_sector_constituents: dict[str, list[str]] = {}
        self._limit_up_data: dict[str, Any] = {}
        self._trade_date: str = ""
        self._data_loaded = False

    # ---------------------------------------------------------------
    # Batch preload — 一次性把所有数据读入内存
    # ---------------------------------------------------------------

    def preload(self, trade_date: str | None = None) -> bool:
        """预加载所有需要的数据到内存。

        一次性读取:
          - daily/*.parquet → self._daily_cache
          - sector_ranking_{date}.json → self._sector_ranking
          - board_index.json → self._board_index
          - limit_up_{date}.json → self._limit_up_data
        """
        td = trade_date or date.today().isoformat()
        self._trade_date = td

        loaded = 0
        loaded += self._load_daily_all()
        loaded += self._load_sector_ranking(td)
        loaded += self._load_board_index()
        loaded += self._load_hot_sector_constituents(td)
        loaded += self._load_limit_up(td)

        self._data_loaded = loaded > 0
        if not self._data_loaded:
            logger.warning("ShortTermSignalEngine: no cache data loaded for %s", td)

        return self._data_loaded

    def _load_daily_all(self) -> int:
        """读取所有 daily parquet 到内存。"""
        daily_dir = self.cache_dir / "daily"
        if not daily_dir.exists():
            logger.warning("Daily cache dir not found: %s", daily_dir)
            return 0

        count = 0
        for path in sorted(daily_dir.glob("*.parquet")):
            # 文件名: 000001_SZ.parquet → 000001.SZ
            ticker = path.stem.replace("_", ".")
            # 统一格式: 6位数字带后缀
            if not self._valid_ticker(ticker):
                continue
            try:
                df = pd.read_parquet(path)
                if df.empty:
                    continue
                df = self._normalize_daily(df, ticker)
                self._daily_cache[ticker] = df
                count += 1
            except Exception as exc:
                logger.debug("Failed to load daily %s: %s", path.name, exc)
                continue

        logger.info("Loaded %d tickers from daily cache", count)
        return count

    def _load_sector_ranking(self, trade_date: str) -> int:
        """读取板块排名缓存"""
        path = self.cache_dir / f"sector_ranking_{trade_date}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, list):
                    self._sector_ranking = data
                    return len(data)
            except Exception:
                pass
        # Fallback: try sector_cache
        path2 = self.cache_dir / f"sector_cache_{trade_date}.json"
        if path2.exists():
            try:
                data = json.loads(path2.read_text("utf-8"))
                if isinstance(data, dict):
                    self._sector_ranking = data.get("sectors", [])
                    return len(self._sector_ranking)
            except Exception:
                pass
        return 0

    def _load_board_index(self) -> int:
        """读取板块成分股索引"""
        path = self.cache_dir / "board_index.json"
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    self._board_index = data
                    return len(data)
            except Exception:
                pass
        return 0

    def _load_hot_sector_constituents(self, trade_date: str) -> int:
        """读取热点板块成分股（东方财富命名体系，与 sector_ranking 同源）。

        由 build_cache.py 的 _cache_hot_sector_constituents 生成。
        """
        path = self.cache_dir / f"hot_sector_constituents_{trade_date}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    self._hot_sector_constituents = data
                    total = sum(len(v) for v in data.values())
                    logger.debug("Hot sector constituents loaded: %d sectors, %d tickers", len(data), total)
                    return len(data)
            except Exception:
                pass
        return 0

    def _load_limit_up(self, trade_date: str) -> int:
        """读取涨停数据"""
        path = self.cache_dir / f"limit_up_{trade_date}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    self._limit_up_data = data
                    return len(data.get("stocks", []))
            except Exception:
                pass
        return 0

    # ---------------------------------------------------------------
    # Compute: 计算所有 ticker 的所有信号
    # ---------------------------------------------------------------

    def compute_all(self, trade_date: str | None = None) -> dict[str, SignalReport]:
        """对所有已缓存的 ticker 计算短线信号。

        返回: {ticker: SignalReport}，按综合评分降序排列。
        """
        td = trade_date or date.today().isoformat()

        if not self._data_loaded:
            self.preload(td)

        if not self._daily_cache:
            logger.warning("No daily data loaded, cannot compute signals")
            return {}

        # 建立板块→ticker 的快速查找
        sector_of_ticker = self._build_sector_map()
        # 涨停股票集合
        limit_up_stocks = self._build_limit_up_set()

        results: dict[str, SignalReport] = {}
        for ticker, df in self._daily_cache.items():
            if df.empty or len(df) < 5:
                continue

            signals: dict[str, ShortTermSignal] = {}

            # 1. 均线多头排列
            sig = self._ma_trend(ticker, df)
            if sig is not None:
                signals["ma_trend"] = sig

            # 2. 量价突破
            sig = self._volume_breakout(ticker, df)
            if sig is not None:
                signals["volume_breakout"] = sig

            # 3. 动量加速度
            sig = self._momentum_accel(ticker, df)
            if sig is not None:
                signals["momentum_accel"] = sig

            # 4. 涨停溢价
            sig = self._limit_up_premium(ticker, df, limit_up_stocks)
            if sig is not None:
                signals["limit_up_premium"] = sig

            # 5. 板块共振
            sectors = sector_of_ticker.get(ticker, [])
            sig = self._sector_resonance(ticker, sectors)
            if sig is not None:
                signals["sector_resonance"] = sig

            # Composite
            composite = self._compute_composite(signals)

            results[ticker] = SignalReport(
                ticker=ticker,
                trade_date=td,
                signals=signals,
                composite=composite,
                n_signals=len(signals),
            )

        # 按综合评分降序排列
        results = dict(sorted(results.items(), key=lambda x: x[1].composite, reverse=True))
        logger.info("Computed short-term signals for %d tickers", len(results))
        return results

    # ---------------------------------------------------------------
    # 信号: 均线多头排列 (-100~100)
    # ---------------------------------------------------------------

    @staticmethod
    def _ma_trend(ticker: str, df: pd.DataFrame) -> ShortTermSignal | None:
        """均线趋势强度。

        判断 MA5, MA10, MA20 的多头/空头排列以及价格相对位置。
        """
        close = _col(df, "close")
        if close is None or len(df) < 20:
            return None

        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()

        last = df.iloc[-1]
        c, m5, m10, m20 = last[close.name], ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1]
        if pd.isna(m5) or pd.isna(m10) or pd.isna(m20):
            return None

        # 均线斜率（用最近 N 天的变化率衡量）
        slope5 = _compute_slope(ma5, 3)
        slope20 = _compute_slope(ma20, 3)

        # 多头排列: MA5 > MA10 > MA20
        bull_alignment = (m5 > m10 > m20)
        # 空头排列: MA5 < MA10 < MA20
        bear_alignment = (m5 < m10 < m20)

        # 价格相对位置
        above_all = c > m5 > m10 > m20
        below_all = c < m5 < m10 < m20

        score = 0.0
        confidence = 0.6

        if bull_alignment:
            # 多头: 60-100 分
            spread = (m5 - m20) / max(abs(m20), 0.01) * 100  # % diff
            score = min(60 + abs(spread) * 2, 100)
            if above_all:
                score = min(score + 10, 100)
            if slope5 > 0 and slope20 > 0:
                score = min(score + 5, 100)
                confidence = min(confidence + 0.2, 1.0)
            direction = "bullish"
        elif bear_alignment:
            # 空头: -100 到 -60
            spread = (m20 - m5) / max(abs(m20), 0.01) * 100
            score = -min(60 + abs(spread) * 2, 100)
            if below_all:
                score = max(score - 10, -100)
            if slope5 < 0 and slope20 < 0:
                score = max(score - 5, -100)
                confidence = min(confidence + 0.2, 1.0)
            direction = "bearish"
        else:
            # 交叉/缠绕: -20 到 20
            if m5 > m10:  # 短期偏多
                score = min((m5 - m10) / max(abs(m10), 0.01) * 50, 20)
                direction = "bullish" if score > 5 else "neutral"
            elif m10 > m5:  # 短期偏空
                score = -min((m10 - m5) / max(abs(m10), 0.01) * 50, 20)
                direction = "bearish" if score < -5 else "neutral"
            else:
                score = 0.0
                direction = "neutral"
            confidence = 0.3

        return ShortTermSignal(
            name="ma_trend",
            ticker=ticker,
            score=round(score, 1),
            direction=direction,
            confidence=round(confidence, 2),
            details={
                "ma5": round(m5, 2),
                "ma10": round(m10, 2),
                "ma20": round(m20, 2),
                "close": round(c, 2),
                "slope_ma5": round(slope5, 4),
                "slope_ma20": round(slope20, 4),
                "bull_alignment": bull_alignment,
                "bear_alignment": bear_alignment,
            },
        )

    # ---------------------------------------------------------------
    # 信号: 量价突破 (0~100)
    # ---------------------------------------------------------------

    @staticmethod
    def _volume_breakout(ticker: str, df: pd.DataFrame) -> ShortTermSignal | None:
        """量价突破识别。

        核心逻辑:
        - 收盘价突破 N 日高点 (N=20)
        - 成交量放大 (> MA5_vol * 1.5)
        - 突破的"新鲜度"（越新越好）
        """
        close = _col(df, "close")
        high = _col(df, "high")
        volume = _col(df, "volume")

        if close is None or high is None or volume is None or len(df) < 20:
            return None

        lookback = 20
        last = df.iloc[-1]
        c = last[close.name]
        v = last[volume.name]

        # N 日最高价
        recent_high = df[high.name].rolling(lookback).max().iloc[-1]
        recent_low = df[close.name].rolling(lookback).min().iloc[-1]
        avg_vol_5 = df[volume.name].rolling(5).mean().iloc[-1]
        avg_vol_20 = df[volume.name].rolling(20).mean().iloc[-1]

        if pd.isna(recent_high) or pd.isna(avg_vol_5):
            return None

        # 突破幅度
        breakout_pct = (c - recent_high) / max(recent_high, 0.01) * 100
        # 成交量比
        vol_ratio_5 = v / max(avg_vol_5, 0.01)
        vol_ratio_20 = v / max(avg_vol_20, 0.01)

        # 分项评分
        score = 0.0

        # 1. 突破幅度评分 (0-50)
        if breakout_pct > 0:
            score += min(breakout_pct * 10, 50)  # +1% → 10分, +5% → 50分
        else:
            # 未突破，但接近: 0-20
            distance = (recent_high - c) / max(recent_high, 0.01) * 100
            score += max(20 - distance * 5, 0)

        # 2. 成交量评分 (0-30)
        if vol_ratio_5 > 1.5:
            score += min((vol_ratio_5 - 1.5) * 15, 30)  # 2x → 7.5, 3x → 22.5, 3.5x → 30
        elif vol_ratio_5 > 1.0:
            score += min((vol_ratio_5 - 1.0) * 20, 15)  # 1.25x → 5, 1.5x → 10
        else:
            score += max((vol_ratio_5 - 1.0) * 20, -10)  # 缩量: 扣分

        # 3. 价格范围位置 (0-20)
        price_range = recent_high - recent_low
        if price_range > 0:
            pos = (c - recent_low) / price_range  # 0~1
            score += pos * 20

        # 计算置信度
        confidence = 0.4
        if breakout_pct > 0 and vol_ratio_5 > 1.5:
            confidence = min(0.5 + (breakout_pct * 5) + (vol_ratio_5 - 1.5) * 0.1, 1.0)
        elif breakout_pct > 0:
            confidence = 0.4
        elif vol_ratio_5 > 2.0:
            confidence = 0.3  # 放量但未突破，可能是出货

        final_score = max(0, min(100, round(score, 1)))
        direction = "bullish" if final_score > 40 else "neutral"

        return ShortTermSignal(
            name="volume_breakout",
            ticker=ticker,
            score=final_score,
            direction=direction,
            confidence=round(confidence, 2),
            details={
                "breakout_pct": round(breakout_pct, 2),
                "vol_ratio_5d": round(vol_ratio_5, 2),
                "vol_ratio_20d": round(vol_ratio_20, 2),
                "recent_high": round(recent_high, 2),
                "recent_low": round(recent_low, 2),
                "close": round(c, 2),
            },
        )

    # ---------------------------------------------------------------
    # 信号: 动量加速度 (-100~100)
    # ---------------------------------------------------------------

    @staticmethod
    def _momentum_accel(ticker: str, df: pd.DataFrame) -> ShortTermSignal | None:
        """动量加速度。

        比较短期/中期/长期动量的相对关系:
        - 动量_5 > 动量_10 > 动量_20 → 加速上涨 (bullish)
        - 动量_5 < 动量_10 < 动量_20 → 加速下跌 (bearish)
        - 配合 RSI 过滤超买超卖。
        """
        close = _col(df, "close")
        if close is None or len(df) < 25:
            return None

        c = close
        mom5 = c.pct_change(5).iloc[-1]
        mom10 = c.pct_change(10).iloc[-1]
        mom20 = c.pct_change(20).iloc[-1]

        if any(pd.isna(x) for x in [mom5, mom10, mom20]):
            return None

        # RSI(6)
        rsi = _compute_rsi(c, 6)

        # 判断加速度
        score = 0.0
        confidence = 0.5

        if mom5 > mom10 > mom20:
            # 加速上涨
            accel = (mom5 - mom20) * 100  # 加速度幅度
            score = min(30 + abs(accel) * 5, 100)
            confidence += 0.2
            direction = "bullish"
            # RSI 超买过滤
            if rsi is not None and rsi > 80:
                score = score * 0.6  # 超买区域，信号打折
                confidence -= 0.15
        elif mom5 < mom10 < mom20:
            # 加速下跌
            accel = (mom20 - mom5) * 100
            score = -min(30 + abs(accel) * 5, 100)
            confidence += 0.15
            direction = "bearish"
            # RSI 超卖过滤
            if rsi is not None and rsi < 20:
                score = score * 0.6
                confidence -= 0.1
        else:
            # 动量不一致
            if mom5 > 0 and mom10 > 0:
                score = min(mom5 * 100, 30)
                direction = "bullish"
            elif mom5 < 0 and mom10 < 0:
                score = -min(abs(mom5) * 100, 30)
                direction = "bearish"
            else:
                score = 0
                direction = "neutral"
            confidence = 0.3

        return ShortTermSignal(
            name="momentum_accel",
            ticker=ticker,
            score=round(score, 1),
            direction=direction,
            confidence=round(confidence, 2),
            details={
                "momentum_5d": round(mom5 * 100, 2),
                "momentum_10d": round(mom10 * 100, 2),
                "momentum_20d": round(mom20 * 100, 2),
                "rsi_6": round(rsi, 1) if rsi is not None else None,
                "acceleration": "accelerating" if mom5 > mom10 > mom20 else ("decelerating" if mom5 < mom10 < mom20 else "mixed"),
            },
        )

    # ---------------------------------------------------------------
    # 信号: 涨停溢价 (0~100)
    # ---------------------------------------------------------------

    @staticmethod
    def _limit_up_premium(
        ticker: str,
        df: pd.DataFrame,
        limit_up_stocks: dict[str, dict[str, Any]],
    ) -> ShortTermSignal | None:
        """涨停溢价评分。

        输入:
        - limit_up_stocks: {ticker: {board_count, name, ...}}

        评分维度:
        - 连板数 (board count)
        - 成交量比（越小说明封得越死）
        - 是否一字板（开盘即涨停）
        """
        if ticker not in limit_up_stocks:
            return None

        info = limit_up_stocks[ticker]
        board = int(info.get("board_count", 1))
        close = _col(df, "close")
        volume = _col(df, "volume")
        open_col = _col(df, "open")

        if close is None or volume is None:
            return None

        last = df.iloc[-1]
        v = last[volume.name]

        avg_vol_5 = df[volume.name].rolling(5).mean().iloc[-1] if len(df) >= 5 else v
        vol_ratio = v / max(avg_vol_5, 0.01)

        # 基础分: 连板
        base = min(board * 20, 60)  # 首板20, 二板40, 三板60

        # 成交量分: 缩量涨停加分，放量涨停减分
        vol_score = 0
        if vol_ratio < 0.5:
            vol_score = 30  # 极度缩量，封死
        elif vol_ratio < 0.8:
            vol_score = 20
        elif vol_ratio < 1.2:
            vol_score = 10
        elif vol_ratio > 3:
            vol_score = -10  # 巨量涨停，分歧大

        # 开盘位置: 开盘即涨停
        open_score = 0
        if open_col is not None:
            preclose = _col(df, "preclose")
            if preclose is not None:
                p = last[preclose.name]
                o = last[open_col.name]
                if p > 0:
                    open_pct = (o - p) / p * 100
                    if open_pct > 9.5:
                        open_score = 15  # 一字板/开盘涨停
                    elif open_pct > 5:
                        open_score = 10  # 高开

        final_score = max(0, min(100, base + vol_score + open_score))

        confidence = 0.5 + min(board * 0.1, 0.3)
        if vol_ratio < 0.8:
            confidence += 0.1

        return ShortTermSignal(
            name="limit_up_premium",
            ticker=ticker,
            score=float(final_score),
            direction="bullish",
            confidence=round(min(confidence, 1.0), 2),
            details={
                "board_count": board,
                "vol_ratio": round(vol_ratio, 2),
                "limit_up_type": "一字板" if open_score >= 15 else "换手板",
            },
        )

    # ---------------------------------------------------------------
    # 信号: 板块共振 (0~100)
    # ---------------------------------------------------------------

    def _sector_resonance(
        self,
        ticker: str,
        sectors: list[str],
    ) -> ShortTermSignal | None:
        """板块共振评分。

        维度:
        - 股票所在板块是否位列热点板块
        - 板块自身强度（strength_score）
        - 股票是否跨多个热点板块
        """
        if not sectors or not self._sector_ranking:
            return None

        # 构建板块名称 → 强度得分的快速查找
        sector_strength: dict[str, float] = {}
        for s in self._sector_ranking:
            name = s.get("sector_name", "")
            if name:
                score = float(s.get("strength_score", 0) or 0)
                sector_strength[name] = score

        # 取板块排名的 top N 作为热点
        top_sectors = {s.get("sector_name", "") for s in self._sector_ranking[:10] if s.get("sector_name")}

        matched = []
        total_strength = 0.0
        for sec in sectors:
            # 精确匹配
            if sec in sector_strength:
                matched.append(sec)
                total_strength += sector_strength[sec]
            # 模糊匹配
            else:
                for key in sector_strength:
                    if sec in key or key in sec:
                        matched.append(key)
                        total_strength += sector_strength[key]
                        break

        if not matched:
            # 无任何匹配 → 返回基础中性信号（确保 composite 权重不飘移）
            return ShortTermSignal(
                name="sector_resonance",
                ticker=ticker,
                score=5.0,
                direction="neutral",
                confidence=0.15,
                details={
                    "matched_sectors": [],
                    "hot_sectors": [],
                    "n_hot": 0,
                    "total_strength": 0.0,
                    "note": "no sector match in ranking",
                },
            )

        # 在热点板块中的数量
        hot_matches = [m for m in matched if m in top_sectors]
        n_hot = len(hot_matches)

        # 评分
        score = 0.0
        if n_hot >= 2:
            score = 70 + min(total_strength * 2, 30)  # 跨多个热点
        elif n_hot == 1:
            score = 40 + min(abs(total_strength) * 0.5, 30)  # 单热点
        else:
            # 不在热点中，但有板块归属
            score = max(10, min(abs(total_strength) * 0.3, 30))

        confidence = min(0.3 + n_hot * 0.15, 0.9)

        return ShortTermSignal(
            name="sector_resonance",
            ticker=ticker,
            score=round(min(score, 100), 1),
            direction="bullish" if n_hot > 0 else "neutral",
            confidence=round(confidence, 2),
            details={
                "matched_sectors": matched[:5],
                "hot_sectors": hot_matches[:5],
                "n_hot": n_hot,
                "total_strength": round(total_strength, 2),
            },
        )

    # ---------------------------------------------------------------
    # Composite
    # ---------------------------------------------------------------

    def _compute_composite(self, signals: dict[str, ShortTermSignal]) -> float:
        """综合评分 (0~100): 加权平均所有有效信号。"""
        if not signals:
            return 50.0

        total_weight = 0.0
        weighted_score = 0.0

        for name, sig in signals.items():
            weight = _SIGNAL_WEIGHTS.get(name, 0.10)
            if sig.confidence < 0.2:
                continue

            # 信号归一化到 -1~1
            if sig.name in ("ma_trend", "momentum_accel"):
                # 已经是 -100~100
                norm = sig.score / 100.0
            else:
                # 0~100 → 0~1，再映射到 -1~1
                # 0-30 → bearish, 30-50 → weak, 50-80 → moderate, 80-100 → strong
                if sig.score < 30:
                    norm = (sig.score - 30) / 70  # 30→0, 0→-0.43
                else:
                    norm = (sig.score - 30) / 70  # 30→0, 65→0.5, 100→1.0
                norm = max(-1.0, min(1.0, norm))

            w = weight * sig.confidence
            weighted_score += norm * w
            total_weight += w

        if total_weight == 0:
            return 50.0

        # 映射到 0-100: (-1~1) → (0~100)
        composite = (weighted_score / total_weight + 1) / 2 * 100
        return round(max(0, min(100, composite)), 1)

    # ---------------------------------------------------------------
    # 工具方法
    # ---------------------------------------------------------------

    def _build_sector_map(self) -> dict[str, list[str]]:
        """构建 ticker → [板块名] 的映射

        优先级:
        1. hot_sector_constituents — 东方财富热点板块成分股（与 sector_ranking 同命名体系）
        2. board_index — 东方财富概念板块（89只 probe 股票）
        3. industry_map — 证监会行业分类（覆盖全部 A 股，但命名体系不同，用作 fallback）
        """
        mapping: dict[str, list[str]] = {}

        # 1. Hot sector constituents: 与 sector_ranking 同源，直接匹配
        for board_name, tickers in self._hot_sector_constituents.items():
            for ticker in tickers:
                if ticker not in mapping:
                    mapping[ticker] = []
                mapping[ticker].append(board_name)

        # 2. board_index: 补充非热点板块数据
        for board_name, stocks in self._board_index.items():
            for s in stocks:
                code = str(s.get("code", ""))
                if not code:
                    continue
                ticker = self._normalize_ticker(code)
                # 去重
                if ticker not in mapping:
                    mapping[ticker] = []
                if board_name not in mapping[ticker]:
                    mapping[ticker].append(board_name)

        # 3. industry_map fallback: 覆盖剩余 A 股（命名体系不同，用模糊匹配）
        ind_path = self.cache_dir / "industry_map.json"
        if ind_path.exists():
            try:
                import json as _json
                import re as _re
                ind_data = _json.loads(ind_path.read_text("utf-8"))
                _prefix_re = _re.compile(r"^[A-Z]\d{2,3}")
                for ticker, industry in ind_data.items():
                    if ticker in mapping:
                        continue  # 已有精确匹配，跳过
                    mapping[ticker] = []
                    # 去掉前缀码 "J66货币金融服务" → "货币金融服务"
                    cleaned = _prefix_re.sub("", industry).strip()
                    if cleaned:
                        mapping[ticker].append(cleaned)
            except Exception as exc:
                logger.debug("Failed to load industry_map fallback: %s", exc)

        return mapping

    def _build_limit_up_set(self) -> dict[str, dict[str, Any]]:
        """构建涨停股票集合 {ticker: info}"""
        stocks = self._limit_up_data.get("stocks", [])
        result: dict[str, dict[str, Any]] = {}
        for s in stocks:
            code = str(s.get("code", ""))
            if not code:
                continue
            ticker = self._normalize_ticker(code)
            result[ticker] = {
                "board_count": int(s.get("board_count", 1)),
                "name": str(s.get("name", "")),
            }
        return result

    @staticmethod
    def _normalize_ticker(code: str) -> str:
        """标准化股票代码为 000001.SZ 格式"""
        code = code.strip().upper()
        if "." in code:
            return code
        if code.startswith("6"):
            return f"{code}.SH"
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
        if code.startswith(("8", "4")):
            return f"{code}.BJ"
        return code

    @staticmethod
    def _valid_ticker(ticker: str) -> bool:
        parts = ticker.split(".")
        if len(parts) != 2:
            return False
        digits, suffix = parts
        return len(digits) == 6 and suffix in ("SH", "SZ", "BJ")

    @staticmethod
    def _normalize_daily(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """标准化 daily parquet 的列名"""
        rename_map = {
            "date": "trade_date",
            "datetime": "trade_date",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # 确保数值列类型正确
        numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount",
                        "pctChg", "turn", "pct_chg", "turnover"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 按日期排序
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df = df.sort_values("trade_date").reset_index(drop=True)

        return df

    # ---------------------------------------------------------------
    # 序列化保存
    # ---------------------------------------------------------------

    def save_results(
        self,
        results: dict[str, SignalReport],
        output_dir: str | Path | None = None,
    ) -> Path:
        """将信号结果保存为 JSON 到缓存目录。"""
        save_dir = Path(output_dir or self.cache_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        td = self._trade_date or date.today().isoformat()
        path = save_dir / f"short_term_signals_{td}.json"

        payload = {
            "trade_date": td,
            "generated_at": pd.Timestamp.now().isoformat(),
            "total_tickers": len(results),
            "signals": {t: r.to_dict() for t, r in results.items()},
            "ranked_tickers": list(results.keys()),
        }

        from ..core.atomic_write import atomic_write_text
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("Signal results saved: %s (%d tickers)", path, len(results))
        return path


# ====================================================================
# 便捷函数 — 供外部模块直接调用
# ====================================================================

def compute_short_term_signals(
    trade_date: str | None = None,
    cache_dir: str | Path | None = None,
    save: bool = True,
) -> dict[str, SignalReport]:
    """一键计算所有短线信号。

    Args:
        trade_date: 交易日期 (默认今天)
        cache_dir: 缓存目录 (默认 results_dir/local_cache)
        save: 是否保存结果到文件

    Returns:
        {ticker: SignalReport}，按综合评分降序
    """
    engine = ShortTermSignalEngine(cache_dir=cache_dir)
    if not engine.preload(trade_date):
        logger.warning("No cache data available for %s", trade_date or "today")
        return {}

    results = engine.compute_all(trade_date)
    if results and save:
        engine.save_results(results, output_dir=cache_dir)

    return results


def format_signal_results(
    results: dict[str, SignalReport],
    top_n: int = 30,
) -> str:
    """将信号结果格式化为可读的 Markdown 表格。"""
    if not results:
        return "No signal data available."

    tickers = list(results.keys())[:top_n]

    lines = [
        "## 短线信号评分",
        "",
        f"共 {len(results)} 只股票，展示 Top {len(tickers)}",
        "",
        "| # | 代码 | 综合 | MA | 突破 | 动量 | 涨停 | 板块 | 方向 |",
        "|---|------|------|----|------|------|------|------|------|",
    ]

    for i, t in enumerate(tickers, 1):
        r = results[t]
        sig = r.signals

        def _fmt(name: str) -> str:
            s = sig.get(name)
            if s is None:
                return "-"
            return f"{s.score:.0f}"

        def _score_bar(val: float) -> str:
            if val >= 70:
                return "[^]"
            if val >= 55:
                return "[/]"
            if val >= 45:
                return "[~]"
            if val >= 30:
                return "[v]"
            return "[!]"

        direction = "^" if r.composite >= 55 else ("v" if r.composite <= 45 else "-")

        lines.append(
            f"| {i} | {t} | {_score_bar(r.composite)}{r.composite:.0f} "
            f"| {_fmt('ma_trend')} | {_fmt('volume_breakout')} "
            f"| {_fmt('momentum_accel')} | {_fmt('limit_up_premium')} "
            f"| {_fmt('sector_resonance')} | {direction} |"
        )

    return "\n".join(lines)


# ====================================================================
# 内部工具函数
# ====================================================================

def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    """模糊查找列名 (支持 pct_chg / pctChg 等变体)"""
    candidates = {
        "close": ["close", "Close"],
        "open": ["open", "Open"],
        "high": ["high", "High"],
        "low": ["low", "Low"],
        "volume": ["volume", "Volume", "vol"],
        "amount": ["amount", "Amount"],
        "pct_chg": ["pct_chg", "pctChg", "change_pct", "pctChg"],
        "preclose": ["preclose", "pre_close", "preClose"],
        "turn": ["turn", "turnover", "turnover_rate", "Turn"],
    }
    for col_name in candidates.get(name, [name]):
        if col_name in df.columns:
            return df[col_name]
    return None


def _compute_slope(series: pd.Series, periods: int = 3) -> float:
    """计算序列最近 N 期的斜率（变化率）。"""
    if len(series) < periods + 1:
        return 0.0
    recent = series.iloc[-periods:].dropna()
    if len(recent) < 2:
        return 0.0
    return (recent.iloc[-1] - recent.iloc[0]) / max(abs(recent.iloc[0]), 0.01)


def _compute_rsi(close: pd.Series, periods: int = 6) -> float | None:
    """计算 RSI (相对强弱指标)。"""
    if len(close) < periods + 1:
        return None
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(periods).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    if pd.isna(val) or np.isinf(val):
        return None
    return float(val)
