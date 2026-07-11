"""
Tier 1 / Tier 2 缓存管理 — 分层加载 + 省 token

核心逻辑:
- Tier 1: 每次默认加载的轻量摘要 (~300 token)
  → 大盘状态, 情绪档位, 板块 Top 10, 资金概览, 风险状态

- Tier 2: 按需加载的详细数据 (每次 ≤ 3 个板块, ~2000 token)
  → 个股因子明细, 结构化事件, 回测样本

- Winter 模式: 市场冰点时只加载 Tier 1, 不进深度分析 (省全部 Tier 2 token)

缓存策略:
- 个股因子 → 缓存到次日开盘 (当天不变)
- 事件 → 缓存 1 小时
- 回测样本 → 缓存到板块成分变化
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import config

logger = logging.getLogger(__name__)


_SAFE_CACHE_KEY = re.compile(r"[^A-Za-z0-9_.-]+")


# 估计 token 数 (中文字符 ~2 token/字, 英文 ~0.5 token/字符)
def estimate_tokens(text: str) -> int:
    """粗略估算 token 数"""
    cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
    en_chars = len(text) - cn_chars
    return cn_chars * 2 + en_chars // 2


@dataclass
class Tier1Data:
    """Tier 1 轻量摘要 — 始终加载"""
    market: dict[str, Any] = field(default_factory=dict)       # 大盘状态
    sentiment: dict[str, Any] = field(default_factory=dict)    # 情绪档位
    sectors: list[dict[str, Any]] = field(default_factory=list) # 板块 Top 10
    capital: dict[str, Any] = field(default_factory=dict)      # 资金概览
    risk: dict[str, Any] = field(default_factory=dict)         # 风险状态
    winter_mode: bool = False                                   # 冰点模式

    def estimated_tokens(self) -> int:
        text = json.dumps({
            "m": self.market, "s": self.sentiment,
            "sc": self.sectors[:5], "c": self.capital, "r": self.risk,
        }, ensure_ascii=False)
        return estimate_tokens(text)

    def to_prompt_block(self) -> str:
        """格式化为 Agent prompt 块"""
        if self.winter_mode:
            return (
                f"[Tier 1 - 市场冰点模式]\n"
                f"市场情绪: 冰点 (得分: {self.sentiment.get('sentiment_score', 'N/A')})\n"
                f"建议仓位上限: 20%\n"
                f"说明: 市场处于冰点, 不进深度分析\n"
            )

        lines = [
            f"[Tier 1 - 市场摘要]",
            f"大盘: {self.market.get('index_close', 'N/A')} "
            f"({self.market.get('index_change_pct', 'N/A')}%)",
            f"涨跌: {self.market.get('advance_count', 'N/A')}/"
            f"{self.market.get('decline_count', 'N/A')}",
            f"涨停/跌停: {self.market.get('limit_up_count', 'N/A')}/"
            f"{self.market.get('limit_down_count', 'N/A')}",
            f"情绪: {self.sentiment.get('sentiment', 'N/A')} "
            f"(得分: {self.sentiment.get('sentiment_score', 'N/A')})",
            f"资金: {self.capital.get('confirmation', 'N/A')}",
        ]
        if self.sectors:
            lines.append(f"板块 Top 5: {', '.join(s.get('sector_name', '') for s in self.sectors[:5])}")
        return "\n".join(lines)


@dataclass
class Tier2Data:
    """Tier 2 详细数据 — 按需加载"""
    factors: list[dict[str, Any]] = field(default_factory=list)       # 个股因子 (≤ 20只)
    events: list[dict[str, Any]] = field(default_factory=list)        # 事件 (≤ 20条)
    backtest_samples: list[dict[str, Any]] = field(default_factory=list)  # 回测样本
    target_sectors: list[str] = field(default_factory=list)           # 当前关注的板块

    def estimated_tokens(self) -> int:
        text = json.dumps({
            "f": self.factors[:5], "e": self.events[:5],
            "b": self.backtest_samples[:3], "s": self.target_sectors,
        }, ensure_ascii=False)
        return estimate_tokens(text)

    def to_prompt_block(self, max_events: int = 10,
                         max_factors: int = 10) -> str:
        """格式化为 Agent prompt 块 (可限制数量省 token)"""
        lines = ["[Tier 2 - 详细数据]"]
        if self.target_sectors:
            lines.append(f"关注板块: {', '.join(self.target_sectors)}")
        if self.factors:
            lines.append(f"因子数据 ({len(self.factors[:max_factors])}只):")
            for f in self.factors[:max_factors]:
                score = f.get("composite_score", "N/A")
                warnings = f.get("factor_warning", "")
                warn_str = f" ⚠{warnings}" if warnings else ""
                lines.append(f"  {f.get('name', f.get('code', ''))}: "
                            f"综合 {score}{warn_str}")
        if self.events:
            lines.append(f"事件 ({len(self.events[:max_events])}条):")
            for e in self.events[:max_events]:
                lines.append(f"  [{e.get('event_type', '?')}] "
                            f"{e.get('summary', '')[:80]}")
        if self.backtest_samples:
            s = self.backtest_samples[0]
            lines.append(f"回测: {s.get('sample_size', 0)}样本, "
                        f"胜率 {s.get('win_rate', 0):.0%}, "
                        f"超额 {s.get('avg_excess_return', 0):+.2%}")
        return "\n".join(lines)


class CacheManager:
    """缓存管理器 — Tier 1/Tier 2 生命周期管理

    Cache TTL:
    - market_data: 到次日 09:00 (盘后不变)
    - factors: 到次日 09:00
    - events: 1 小时
    - backtest: 24 小时
    """

    _CACHE_TTL = {
        "market_data": 3600 * 15,   # 15 小时 (15:00 → 次日 06:00)
        "factors": 3600 * 15,
        "events": 3600,              # 1 小时
        "backtest": 3600 * 24,       # 24 小时
        "news": 1800,                # 30 分钟
    }

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(
            cache_dir or config.get("data_cache_dir", "data/cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        safe_key = _SAFE_CACHE_KEY.sub("_", key).strip("._")
        if not safe_key:
            safe_key = "cache"
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Any | None:
        """获取缓存 (如果未过期)"""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = data.get("_cached_at", 0)
            category = key.split(":")[0]
            ttl = self._CACHE_TTL.get(category, 3600)
            if time.time() - cached_at > ttl:
                path.unlink(missing_ok=True)
                return None
            return data.get("data")
        except (json.JSONDecodeError, KeyError):
            return None

    def set(self, key: str, data: Any) -> None:
        """写入缓存"""
        path = self._cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"_cached_at": time.time(), "data": data},
                          f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", key, e)

    def clear_all(self) -> int:
        """清理全部缓存, 返回清理数"""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


# ============================================================
# Tier Manager — 分层加载决策
# ============================================================

# 情绪 → Winter 模式映射
WINTER_SENTIMENTS = {"冰点", "低迷"}


def decide_tier2_loading(
    tier1: Tier1Data,
    target_sectors: list[str] | None = None,
) -> dict[str, Any]:
    """决定 Tier 2 加载范围

    Returns:
        dict with:
        - load_tier2: bool (是否加载 Tier 2)
        - max_sectors: int (最多加载几个板块)
        - max_events: int (最多加载几个事件)
        - max_factors: int (最多加载几只因子)
        - reason: str (决策理由)
    """
    sentiment = tier1.sentiment.get("sentiment", "正常")

    # Winter 模式: 市场冰点 → 不加载 Tier 2, 省全部 token
    if sentiment in WINTER_SENTIMENTS:
        return {
            "load_tier2": False,
            "max_sectors": 0,
            "max_events": 0,
            "max_factors": 0,
            "reason": f"市场{sentiment}, 不进深度分析",
        }

    # 正常/温热/高潮: 加载 Tier 2, 但限制数量
    if target_sectors:
        max_sectors = min(len(target_sectors), 3)
    else:
        max_sectors = 3

    return {
        "load_tier2": True,
        "max_sectors": max_sectors,
        "max_events": 15 if sentiment == "高潮" else 10,
        "max_factors": 10,
        "reason": f"市场{sentiment}, 加载 Tier 2 深度分析",
    }
