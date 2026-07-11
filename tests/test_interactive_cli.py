"""Tests for the interactive slash-command CLI."""
from __future__ import annotations

from io import StringIO

from advanced_trading_agent import interactive_cli


def test_help_command_lists_core_slash_commands():
    result = interactive_cli.execute_command("/help")

    assert result.should_exit is False
    assert "/analyze <ticker>" in result.output
    assert "/data <ticker>" in result.output
    assert "/status" in result.output
    assert "/exit" in result.output


def test_bom_prefixed_slash_command_is_not_treated_as_ticker():
    calls = []

    def fake_analyze(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "# should not run"

    result = interactive_cli.execute_command(
        "\ufeff/help",
        analyze_runner=fake_analyze,
    )

    assert "Advanced Trading Agent CLI" in result.output
    assert calls == []


def test_bare_ticker_runs_analyze_with_skip_backtest_default():
    calls = []

    def fake_analyze(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "# fake report"

    result = interactive_cli.execute_command(
        "000001.SZ",
        analyze_runner=fake_analyze,
    )

    assert result.output == "# fake report"
    assert calls == [{
        "args": ("000001.SZ", interactive_cli.date.today().isoformat()),
        "kwargs": {"debug": False, "skip_backtest": True},
    }]


def test_analyze_command_parses_date_debug_and_skip_backtest():
    calls = []

    def fake_analyze(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "analysis ok"

    result = interactive_cli.execute_command(
        "/analyze 000001.SZ --date 2026-07-10 --skip-backtest --debug",
        analyze_runner=fake_analyze,
    )

    assert result.output == "analysis ok"
    assert calls == [{
        "args": ("000001.SZ", "2026-07-10"),
        "kwargs": {"debug": True, "skip_backtest": True},
    }]


def test_data_command_parses_dataagent_options():
    calls = []

    def fake_data(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return '{"run_id": "run-1"}'

    result = interactive_cli.execute_command(
        "/data 000001.SZ --date 2026-07-10 --start-date 20260701 "
        "--end-date 20260710 --output-dir out --react-planner "
        "--news-keyword 银行 --no-llm-news-filter --no-news-full-text",
        data_runner=fake_data,
    )

    assert result.output == '{"run_id": "run-1"}'
    assert calls == [{
        "args": ("000001.SZ",),
        "kwargs": {
            "trade_date": "2026-07-10",
            "start_date": "20260701",
            "end_date": "20260710",
            "output_dir": "out",
            "use_react_planner": True,
            "news_keyword": "银行",
            "use_llm_news_filter": False,
            "fetch_news_full_text": False,
        },
    }]


def test_unknown_and_exit_commands_do_not_raise():
    unknown = interactive_cli.execute_command("/missing")
    exiting = interactive_cli.execute_command("/exit")

    assert "Unknown command" in unknown.output
    assert exiting.should_exit is True
    assert exiting.output == "bye"


def test_repl_reads_multiple_commands_until_exit():
    output = StringIO()
    input_stream = StringIO("/help\n/exit\n")

    exit_code = interactive_cli.repl(
        stdin=input_stream,
        stdout=output,
        show_banner=False,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert rendered.count("ata> ") == 2
    assert "Advanced Trading Agent CLI" in rendered
    assert "bye" in rendered
