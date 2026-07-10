"""Cache Manager 测试"""
import json
import time
import pytest
from datetime import date
from pathlib import Path
from advanced_trading_agent.core.cache_manager import (
    CacheManager, Tier1Data, Tier2Data,
    estimate_tokens, decide_tier2_loading,
    WINTER_SENTIMENTS,
)


class TestEstimateTokens:
    """Token 估算测试"""

    def test_english(self):
        n = estimate_tokens("hello world")
        assert n > 0
        assert n == 5  # 10 chars / 2 = 5

    def test_chinese(self):
        n = estimate_tokens("你好世界")
        assert n == 8  # 4 chars * 2 = 8

    def test_mixed(self):
        n = estimate_tokens("你好hello")
        assert n > 0


class TestCacheManager:
    """缓存管理器测试"""

    def test_set_and_get(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        cache.set("test_key", {"value": 123})
        result = cache.get("test_key")
        assert result == {"value": 123}

    def test_get_missing(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        assert cache.get("nonexistent") is None

    def test_get_expired(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        cache.set("test_expire", {"value": 1})
        # 手动篡改缓存时间使其过期
        path = cache._cache_path("test_expire")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_cached_at"] = 0  # epoch = 非常久以前
        path.write_text(json.dumps(data), encoding="utf-8")
        assert cache.get("test_expire") is None

    def test_clear_all(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        cache.set("k1", 1)
        cache.set("k2", 2)
        count = cache.clear_all()
        assert count == 2
        assert cache.get("k1") is None

    def test_cache_file_isolation(self, tmp_path):
        """不同 key 的缓存文件不互相影响"""
        cache = CacheManager(cache_dir=str(tmp_path))
        cache.set("key_a", "value_a")
        cache.set("key_b", "value_b")
        assert cache.get("key_a") == "value_a"
        assert cache.get("key_b") == "value_b"

    def test_overwrite(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        cache.set("key", "old")
        cache.set("key", "new")
        assert cache.get("key") == "new"


class TestTier1Data:
    """Tier 1 数据模型测试"""

    def test_to_prompt_block_winter(self):
        t = Tier1Data(
            market={"index_close": 3000},
            sentiment={"sentiment": "冰点", "sentiment_score": 15},
            winter_mode=True,
        )
        block = t.to_prompt_block()
        assert "冰点" in block
        assert "不进深度分析" in block

    def test_to_prompt_block_normal(self):
        t = Tier1Data(
            market={"index_close": 3000, "index_change_pct": 0.5},
            sentiment={"sentiment": "正常", "sentiment_score": 55},
            capital={"confirmation": "资金确认"},
        )
        block = t.to_prompt_block()
        assert "3000" in block
        assert "0.5" in block

    def test_estimated_tokens(self):
        t = Tier1Data(
            market={"index_close": 3000},
            sentiment={"sentiment": "正常"},
        )
        assert t.estimated_tokens() > 0


class TestDecideTier2Loading:
    """Tier 2 加载决策测试"""

    def test_winter_mode_skips_tier2(self):
        for sentiment in WINTER_SENTIMENTS:
            t = Tier1Data(sentiment={"sentiment": sentiment})
            decision = decide_tier2_loading(t)
            assert decision["load_tier2"] is False
            assert decision["max_sectors"] == 0

    def test_normal_mode_loads_tier2(self):
        t = Tier1Data(sentiment={"sentiment": "正常"})
        decision = decide_tier2_loading(t)
        assert decision["load_tier2"] is True
        assert decision["max_sectors"] > 0

    def test_high_sentiment_reduces_events(self):
        t = Tier1Data(sentiment={"sentiment": "高潮"})
        decision = decide_tier2_loading(t)
        assert decision["load_tier2"] is True
