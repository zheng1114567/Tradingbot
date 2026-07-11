"""Profit-evaluation framework tests."""
from __future__ import annotations

import pandas as pd

from advanced_trading_agent.backtest.alpha import AlphaAttributionAnalyzer
from advanced_trading_agent.backtest.data_qa import DataQualityGate
from advanced_trading_agent.backtest.experiment import ExperimentRegistry
from advanced_trading_agent.backtest.paper import PaperTradingLedger


def _signals() -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": ["2026-07-10"],
        "available_at": ["2026-07-10 15:30:00"],
        "code": ["000001.SZ"],
        "decision": ["推荐"],
        "score": [9.0],
        "alpha_source": ["event/factor"],
    })


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2026-07-10", periods=8, freq="B")
    return pd.DataFrame({
        "trade_date": list(dates),
        "code": ["000001.SZ"] * len(dates),
        "open": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5],
        "close": [10.2, 10.7, 11.2, 11.7, 12.2, 12.7, 13.2, 13.7],
    })


def test_data_quality_gate_passes_valid_point_in_time_data():
    report = DataQualityGate().validate(
        signals=_signals(),
        prices=_prices(),
        run_time="2026-07-10 16:00:00",
    )

    assert report.passed
    assert report.summary["signals_count"] == 1
    assert report.summary["unique_codes"] == 1


def test_data_quality_gate_blocks_future_available_at():
    signals = _signals()
    signals.loc[0, "available_at"] = "2026-07-11 09:30:00"

    report = DataQualityGate().validate(
        signals=signals,
        prices=_prices(),
        run_time="2026-07-10 16:00:00",
    )

    assert not report.passed
    assert any(issue.field == "signals.available_at" for issue in report.issues)


def test_experiment_registry_persists_input_fingerprints(tmp_path):
    registry = ExperimentRegistry(path=str(tmp_path / "experiments.jsonl"))

    record = registry.register(
        name="walk-forward-smoke",
        signals=_signals(),
        prices=_prices(),
        metrics={"total_return": 0.03},
        notes=["smoke"],
    )

    loaded = registry.load()
    assert len(loaded) == 1
    assert loaded[0].run_id == record.run_id
    assert loaded[0].inputs["signals_count"] == 1
    assert loaded[0].metrics["total_return"] == 0.03


def test_alpha_attribution_splits_multi_source_pnl():
    trades = pd.DataFrame({
        "alpha_source": ["event/factor", "factor"],
        "return": [0.02, -0.01],
        "pnl": [1000.0, -300.0],
    })

    summary = AlphaAttributionAnalyzer().summarize(trades)
    by_source = {item.alpha_source: item for item in summary}

    assert by_source["event"].total_pnl == 500.0
    assert by_source["factor"].sample_size == 2
    assert by_source["factor"].total_pnl == 200.0


def test_paper_trading_ledger_records_and_resolves(tmp_path):
    ledger = PaperTradingLedger(path=str(tmp_path / "paper.jsonl"))

    result = ledger.record_and_resolve(
        signals=_signals(),
        prices=_prices(),
        holding_days=2,
    )
    rows = ledger.load()

    assert result.recorded_count == 1
    assert result.resolved_count == 1
    assert rows[0]["status"] == "resolved"
    assert rows[0]["trade"]["entry_date"] == "2026-07-13"
    assert result.summary["trade_count"] == 1
