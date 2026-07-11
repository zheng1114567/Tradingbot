"""
全局配置管理 — 借鉴 TradingAgents' default_config.py + config.py

采用单例模式 + env 覆盖:
1. 代码设定默认值
2. .env 文件覆盖
3. TRADINGAGENTS_* / ATA_* 环境变量覆盖
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

# ============================================================
# 配置加载
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ATA_HOME = Path(os.getenv("ATA_HOME", os.path.join(Path.home(), ".advanced_trading_agent")))

# TRADINGAGENTS_* 保持向后兼容, ATA_* 是新的前缀
_ENV_OVERRIDES = {
    "ATA_LLM_PROVIDER": "llm_provider",
    "ATA_DEEP_THINK_LLM": "deep_think_llm",
    "ATA_QUICK_THINK_LLM": "quick_think_llm",
    "ATA_TEMPERATURE": "temperature",
    "ATA_MAX_DEBATE_ROUNDS": "max_debate_rounds",
    "ATA_OUTPUT_LANGUAGE": "output_language",
    "ATA_DATA_CACHE_DIR": "data_cache_dir",
    "ATA_RESULTS_DIR": "results_dir",
    "ATA_MEMORY_LOG_PATH": "memory_log_path",
    "ATA_MEMORY_INDEX_PATH": "memory_index_path",
    # 兼容旧前缀
    "TRADINGAGENTS_LLM_PROVIDER": "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM": "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM": "quick_think_llm",
    "TRADINGAGENTS_TEMPERATURE": "temperature",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "max_debate_rounds",
    "TRADINGAGENTS_OUTPUT_LANGUAGE": "output_language",
}

_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _load_dotenv() -> None:
    """加载 .env 文件 (如果存在)"""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key not in os.environ:  # 环境变量优先
                os.environ[key] = value


def _coerce_env(value: str, reference) -> Any:
    """将环境变量字符串转换为目标类型"""
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(f"Invalid boolean: {value}")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


class Config:
    """全局配置 — 单例"""

    _instance: "Config | None" = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        _load_dotenv()
        self._config = self._build_defaults()
        self._apply_env_overrides()

    # ----------------------------------------------------------
    # 默认值
    # ----------------------------------------------------------
    @staticmethod
    def _build_defaults() -> dict[str, Any]:
        return {
            # 路径
            "project_dir": str(_PROJECT_ROOT),
            "results_dir": os.getenv("ATA_RESULTS_DIR", str(_ATA_HOME / "results")),
            "data_cache_dir": os.getenv("ATA_DATA_CACHE_DIR", str(_ATA_HOME / "cache")),
            "memory_log_path": os.getenv("ATA_MEMORY_LOG_PATH", str(_ATA_HOME / "memory" / "trading_memory.md")),
            "memory_index_path": os.getenv("ATA_MEMORY_INDEX_PATH", str(_ATA_HOME / "memory" / "trading_memory.jsonl")),
            "strategy_audit_queue_path": os.getenv(
                "ATA_STRATEGY_AUDIT_QUEUE_PATH",
                str(_ATA_HOME / "memory" / "strategy_audit_queue.jsonl"),
            ),
            # LLM
            "llm_provider": "deepseek",
            "deep_think_llm": "deepseek-chat",
            "quick_think_llm": "deepseek-chat",
            "temperature": 0.1,
            # 运行时
            "max_debate_rounds": 2,
            "max_risk_discuss_rounds": 2,
            "output_language": "Chinese",
            # 数据
            "data_vendors": {
                "market_data": "akshare,tushare",   # 行情: 免费 akshare 优先, tushare 可选增强
                "fundamental_data": "akshare,tushare",
                "news_data": "akshare",
                "capital_flow": "akshare,tushare",
                "a_share_specific": "akshare,tushare",
                "analysis": "akshare,tushare",
            },
            # 风控
            "risk_config": {
                "max_single_position_pct": 0.10,     # 单票 ≤ 10%
                "max_sector_pct": 0.30,             # 单板块 ≤ 30%
                "max_total_pct": 0.60,              # 总仓位 ≤ 60%
                "min_daily_volume_cny": 10_000_000, # 日成交额 ≥ 1000万
                "impact_cost_threshold": 0.30,      # 冲击成本 > 预期收益 30%  veto
                "stop_loss_pct": -0.07,             # 止损 -7%
                "take_profit_pct": 0.15,            # 止盈 15%
                "max_holding_days": 20,              # 最长持有天数
            },
            # 回测
            "backtest_config": {
                "default_holding_days": [1, 3, 5, 10, 20],
                "primary_holding_days": 5,
                "benchmark": "000300.SH",  # 沪深300
                "slippage_bps": 3,         # 滑点 3bp
                "stamp_tax_bps": 10,       # 印花税 10bp (卖出)
                "commission_bps": 3,       # 佣金 3bp
                "min_sample_size": 30,     # 最小样本量
            },
            # 复盘与策略校准
            "review_config": {
                "review_horizons": [1, 3, 5, 10, 20],
                "primary_horizon_days": 5,
                "min_samples_to_adjust": 30,
                "min_hit_rate": 0.45,
                "min_avg_excess_return": 0.0,
                "pause_hit_rate": 0.35,
                "pause_avg_excess_return": -0.02,
            },
            # 策略规则版本: 所有确定性阈值进入同一 rulebook snapshot
            "strategy_rules": {
                "version": os.getenv("ATA_STRATEGY_RULE_VERSION", "rules-v1"),
                "rubric_thresholds": {
                    "recommend_min_total": 9,
                    "watch_below_total": 9,
                    "reject_on_risk_score": 0,
                },
                "memory_policy": {
                    "primary_store": "markdown",
                    "index_store": "jsonl",
                    "deferred_reflection": True,
                },
            },
        }

    # ----------------------------------------------------------
    # 环境变量覆盖
    # ----------------------------------------------------------
    def _apply_env_overrides(self) -> None:
        for env_var, config_key in _ENV_OVERRIDES.items():
            raw = os.environ.get(env_var)
            if raw is None or raw == "":
                continue
            try:
                self._config[config_key] = _coerce_env(raw, self._config.get(config_key))
            except ValueError as e:
                raise ValueError(f"Invalid value for {env_var}: {e}") from e

    # ----------------------------------------------------------
    # 公开 API
    # ----------------------------------------------------------
    def get(self, key: str, default=None) -> Any:
        return self._config.get(key, default)

    def get_all(self) -> dict[str, Any]:
        return deepcopy(self._config)

    def update(self, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(self._config.get(key), dict):
                self._config[key].update(value)
            else:
                self._config[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config


# 全局实例
config = Config()


# 便捷函数 (与 TradingAgents 风格兼容)
def get_config() -> dict[str, Any]:
    return config.get_all()


def set_config(updates: dict[str, Any]) -> None:
    config.update(updates)
