"""配置测试"""
from advanced_trading_agent.config import Config, config


def test_config_singleton():
    c1 = Config()
    c2 = Config()
    assert c1 is c2


def test_config_defaults():
    assert config.get("llm_provider") is not None
    assert config.get("llm_model") == "qwen3.6-flash"
    assert config.get("deep_think_llm") is not None
    assert config.get("risk_config") is not None
    assert config.get("backtest_config") is not None


def test_config_update():
    previous_provider = config.get("llm_provider")
    config.update({"llm_provider": "test"})
    assert config.get("llm_provider") == "test"
    config.update({"llm_provider": previous_provider})


def test_risk_config_values():
    rc = config.get("risk_config", {})
    assert rc.get("max_single_position_pct") == 0.10
    assert rc.get("min_daily_volume_cny") >= 1_000_000


def test_backtest_config_values():
    bc = config.get("backtest_config", {})
    assert bc.get("primary_holding_days") == 5
    assert bc.get("benchmark") is not None


def test_strategy_rule_config_values():
    rules = config.get("strategy_rules", {})
    assert rules.get("version")
    assert rules.get("rubric_thresholds", {}).get("recommend_min_total") == 9
    assert config.get("strategy_audit_queue_path")
