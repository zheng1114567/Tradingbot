"""配置测试"""
import pytest
from advanced_trading_agent.config import Config, config


def test_config_singleton():
    c1 = Config()
    c2 = Config()
    assert c1 is c2


def test_config_defaults():
    assert config.get("llm_provider") is not None
    assert config.get("deep_think_llm") is not None
    assert config.get("risk_config") is not None
    assert config.get("backtest_config") is not None


def test_config_update():
    config.update({"llm_provider": "test"})
    assert config.get("llm_provider") == "test"
    # 恢复
    config.update({"llm_provider": "deepseek"})


def test_risk_config_values():
    rc = config.get("risk_config", {})
    assert rc.get("max_single_position_pct") == 0.10
    assert rc.get("min_daily_volume_cny") >= 1_000_000


def test_backtest_config_values():
    bc = config.get("backtest_config", {})
    assert bc.get("primary_holding_days") == 5
    assert bc.get("benchmark") is not None
