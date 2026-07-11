"""Tests for the interactive slash-command CLI."""
from __future__ import annotations

import json
from io import StringIO

from advanced_trading_agent import interactive_cli


def test_help_command_lists_core_slash_commands():
    result = interactive_cli.execute_command("/help")

    assert result.should_exit is False
    assert "/a <ticker>" in result.output
    assert "/data [ticker]" in result.output
    assert "/datas [ticker]" in result.output
    assert "/data <natural language request>" in result.output
    assert "/date [ticker]" in result.output
    assert "/dates [ticker]" in result.output
    assert "/run [ticker]" in result.output
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


def test_bare_ticker_accepts_positional_date():
    calls = []

    def fake_analyze(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "# fake report"

    result = interactive_cli.execute_command(
        "000001.SZ 2026-07-10",
        analyze_runner=fake_analyze,
    )

    assert result.output == "# fake report"
    assert calls == [{
        "args": ("000001.SZ", "2026-07-10"),
        "kwargs": {"debug": False, "skip_backtest": True},
    }]


def test_analyze_alias_parses_positional_date_and_debug():
    calls = []

    def fake_analyze(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "analysis ok"

    result = interactive_cli.execute_command(
        "/a 000001.SZ 2026-07-10 --debug",
        analyze_runner=fake_analyze,
    )

    assert result.output == "analysis ok"
    assert calls == [{
        "args": ("000001.SZ", "2026-07-10"),
        "kwargs": {"debug": True, "skip_backtest": True},
    }]


def test_analyze_can_enable_backtest_explicitly():
    calls = []

    def fake_analyze(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "analysis ok"

    result = interactive_cli.execute_command(
        "/analyze 000001.SZ 2026-07-10 --with-backtest",
        analyze_runner=fake_analyze,
    )

    assert result.output == "analysis ok"
    assert calls == [{
        "args": ("000001.SZ", "2026-07-10"),
        "kwargs": {"debug": False, "skip_backtest": False},
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


def test_data_alias_accepts_positional_date():
    calls = []

    def fake_data(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return '{"run_id": "run-1"}'

    result = interactive_cli.execute_command(
        "/d 000001.SZ 2026-07-10",
        data_runner=fake_data,
    )

    assert result.output == '{"run_id": "run-1"}'
    assert calls[0]["args"] == ("000001.SZ",)
    assert calls[0]["kwargs"]["trade_date"] == "2026-07-10"


def test_data_sets_current_context_for_parameterless_analyze():
    context = interactive_cli.SessionContext()
    data_calls = []
    analyze_calls = []

    def fake_data(*args, **kwargs):
        data_calls.append({"args": args, "kwargs": kwargs})
        return "data ok"

    def fake_analyze(*args, **kwargs):
        analyze_calls.append({"args": args, "kwargs": kwargs})
        return "analysis ok"

    data_result = interactive_cli.execute_command(
        "/data 000001.SZ 2026-07-10",
        context=context,
        data_runner=fake_data,
    )
    analyze_result = interactive_cli.execute_command(
        "/analyze",
        context=context,
        analyze_runner=fake_analyze,
    )

    assert data_result.output == "data ok"
    assert analyze_result.output == "analysis ok"
    assert context.ticker == "000001.SZ"
    assert context.trade_date == "2026-07-10"
    assert data_calls[0]["args"] == ("000001.SZ",)
    assert data_calls[0]["kwargs"]["trade_date"] == "2026-07-10"
    assert analyze_calls == [{
        "args": ("000001.SZ", "2026-07-10"),
        "kwargs": {"debug": False, "skip_backtest": True},
    }]


def test_parameterless_data_reuses_current_context():
    context = interactive_cli.SessionContext(ticker="000001.SZ", trade_date="2026-07-10")
    calls = []

    def fake_data(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "data ok"

    result = interactive_cli.execute_command(
        "/data",
        context=context,
        data_runner=fake_data,
    )

    assert result.output == "data ok"
    assert calls[0]["args"] == ("000001.SZ",)
    assert calls[0]["kwargs"]["trade_date"] == "2026-07-10"


def test_data_natural_language_today_collects_current_close_data():
    calls = []

    def fake_data(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return '{"run_id": "run-today"}'

    result = interactive_cli.execute_command(
        "/data 分析今天收盘后的数据 000001.SZ",
        data_runner=fake_data,
    )

    today = interactive_cli.date.today()
    assert "# DataAgent Intent" in result.output
    assert "run-today" in result.output
    assert calls == [{
        "args": ("000001.SZ",),
        "kwargs": {
            "trade_date": today.isoformat(),
            "start_date": today.strftime("%Y%m%d"),
            "end_date": today.strftime("%Y%m%d"),
            "output_dir": None,
            "use_react_planner": True,
            "news_keyword": None,
            "use_llm_news_filter": True,
            "fetch_news_full_text": True,
        },
    }]


def test_data_natural_language_year_range_uses_current_year_window():
    calls = []

    def fake_data(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return '{"run_id": "run-year"}'

    result = interactive_cli.execute_command(
        "/data 拉一下今年的数据 000001.SZ",
        data_runner=fake_data,
    )

    today = interactive_cli.date.today()
    assert "# DataAgent Intent" in result.output
    assert calls[0]["args"] == ("000001.SZ",)
    assert calls[0]["kwargs"]["trade_date"] == today.isoformat()
    assert calls[0]["kwargs"]["start_date"] == f"{today.year}0101"
    assert calls[0]["kwargs"]["end_date"] == today.strftime("%Y%m%d")
    assert calls[0]["kwargs"]["use_react_planner"] is True


def test_data_natural_language_can_use_injected_llm_intent_parser():
    context = interactive_cli.SessionContext(ticker="000001.SZ", trade_date="2026-07-10")
    parser_calls = []
    data_calls = []

    def fake_intent_parser(text, session_context):
        parser_calls.append((text, session_context.ticker, session_context.trade_date))
        return interactive_cli.DataIntent(
            action="collect",
            ticker="000001.SZ",
            trade_date="2026-07-10",
            start_date="20260701",
            end_date="20260710",
            query=text,
            reasoning="fake llm intent",
        )

    def fake_data(*args, **kwargs):
        data_calls.append({"args": args, "kwargs": kwargs})
        return '{"run_id": "run-window"}'

    result = interactive_cli.execute_command(
        "/data 帮我拉一段时间的数据",
        context=context,
        data_runner=fake_data,
        intent_parser=fake_intent_parser,
    )

    assert parser_calls == [("帮我拉一段时间的数据", "000001.SZ", "2026-07-10")]
    assert "fake llm intent" in result.output
    assert data_calls[0]["args"] == ("000001.SZ",)
    assert data_calls[0]["kwargs"]["start_date"] == "20260701"
    assert data_calls[0]["kwargs"]["end_date"] == "20260710"
    assert data_calls[0]["kwargs"]["use_react_planner"] is True


def test_datas_alias_supports_natural_language_requests():
    calls = []

    def fake_data(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return '{"run_id": "run-alias"}'

    result = interactive_cli.execute_command(
        "/datas 分析今天的数据 000001.SZ",
        data_runner=fake_data,
    )

    assert "# DataAgent Intent" in result.output
    assert calls[0]["args"] == ("000001.SZ",)
    assert calls[0]["kwargs"]["use_react_planner"] is True


def test_data_natural_language_asks_for_ticker_before_collecting():
    calls = []

    def fake_data(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return '{"run_id": "should-not-run"}'

    result = interactive_cli.execute_command(
        "/data 分析今天收盘后的数据",
        data_runner=fake_data,
    )

    assert "# DataAgent Needs Clarification" in result.output
    assert "你想分析哪只股票" in result.output
    assert calls == []


def test_data_natural_language_screen_uses_local_candidate_summary():
    data_calls = []

    def fake_data(*args, **kwargs):
        data_calls.append({"args": args, "kwargs": kwargs})
        return "should not run"

    def fake_intent_parser(text, session_context):
        return interactive_cli.DataIntent(
            action="screen",
            top_n=3,
            query=text,
            reasoning="fake screen intent",
        )

    result = interactive_cli.execute_command(
        "/data 最近表现比较好的股票",
        data_runner=fake_data,
        intent_parser=fake_intent_parser,
    )

    assert "Candidate screen: top 3" in result.output
    assert "最近表现比较好的股票" in result.output
    assert data_calls == []


def test_parameterless_analyze_requires_current_context():
    result = interactive_cli.execute_command("/analyze", context=interactive_cli.SessionContext())

    assert "No current ticker" in result.output


def test_dates_command_lists_local_data_dates(tmp_path):
    run_dir = tmp_path / "data_agent_runs" / "2026-07-10_000001_SZ_20260712_000637"
    manifest_dir = run_dir / "06_final" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest_000001_SZ_2026-07-10.json").write_text(
        json.dumps({"ticker": "000001.SZ", "trade_date": "2026-07-10"}),
        encoding="utf-8",
    )

    result = interactive_cli.execute_command(
        f"/dates 000001.SZ --results-dir {tmp_path}",
    )

    assert "Data dates for 000001.SZ: 1" in result.output
    assert "2026-07-10" in result.output
    assert "000001_SZ" in result.output


def test_dates_command_reuses_current_context(tmp_path):
    context = interactive_cli.SessionContext(ticker="000001.SZ", trade_date="2026-07-10")
    run_dir = tmp_path / "data_agent_runs" / "2026-07-10_000001_SZ_20260712_000637"
    run_dir.mkdir(parents=True)

    result = interactive_cli.execute_command(
        f"/dates --results-dir {tmp_path}",
        context=context,
    )

    assert "Data dates for 000001.SZ: 1" in result.output
    assert "2026-07-10" in result.output


def test_date_alias_lists_local_data_dates(tmp_path):
    run_dir = tmp_path / "data_agent_runs" / "2026-07-10_000001_SZ_20260712_000637"
    run_dir.mkdir(parents=True)

    result = interactive_cli.execute_command(
        f"/date 000001.SZ --results-dir {tmp_path}",
    )

    assert "Data dates for 000001.SZ: 1" in result.output
    assert "2026-07-10" in result.output


def test_run_command_runs_data_then_analyze_with_backtest_by_default():
    context = interactive_cli.SessionContext()
    calls = []

    def fake_data(*args, **kwargs):
        calls.append(("data", args, kwargs))
        return "data ok"

    def fake_analyze(*args, **kwargs):
        calls.append(("analyze", args, kwargs))
        return "analysis ok"

    result = interactive_cli.execute_command(
        "/run 000001.SZ 2026-07-10",
        context=context,
        data_runner=fake_data,
        analyze_runner=fake_analyze,
    )

    assert "# Full Run Complete" in result.output
    assert "data ok" in result.output
    assert "analysis ok" in result.output
    assert context.ticker == "000001.SZ"
    assert context.trade_date == "2026-07-10"
    assert calls[0][0] == "data"
    assert calls[0][1] == ("000001.SZ",)
    assert calls[0][2]["trade_date"] == "2026-07-10"
    assert calls[0][2]["start_date"] == "20260710"
    assert calls[0][2]["end_date"] == "20260710"
    assert calls[1] == (
        "analyze",
        ("000001.SZ", "2026-07-10"),
        {"debug": False, "skip_backtest": False},
    )


def test_run_command_can_skip_backtest_explicitly():
    calls = []

    def fake_data(*args, **kwargs):
        calls.append(("data", args, kwargs))
        return "data ok"

    def fake_analyze(*args, **kwargs):
        calls.append(("analyze", args, kwargs))
        return "analysis ok"

    result = interactive_cli.execute_command(
        "/run 000001.SZ 2026-07-10 --skip-backtest --debug",
        data_runner=fake_data,
        analyze_runner=fake_analyze,
    )

    assert "# Full Run Complete" in result.output
    assert calls[1] == (
        "analyze",
        ("000001.SZ", "2026-07-10"),
        {"debug": True, "skip_backtest": True},
    )


def test_parameterless_run_reuses_current_context():
    context = interactive_cli.SessionContext(ticker="000001.SZ", trade_date="2026-07-10")
    calls = []

    def fake_data(*args, **kwargs):
        calls.append(("data", args, kwargs))
        return "data ok"

    def fake_analyze(*args, **kwargs):
        calls.append(("analyze", args, kwargs))
        return "analysis ok"

    result = interactive_cli.execute_command(
        "/run",
        context=context,
        data_runner=fake_data,
        analyze_runner=fake_analyze,
    )

    assert "# Full Run Complete" in result.output
    assert calls[0][1] == ("000001.SZ",)
    assert calls[0][2]["trade_date"] == "2026-07-10"
    assert calls[1][1] == ("000001.SZ", "2026-07-10")


def test_parameterless_run_requires_current_context():
    result = interactive_cli.execute_command("/run", context=interactive_cli.SessionContext())

    assert "No current ticker" in result.output


def test_unknown_and_exit_commands_do_not_raise():
    unknown = interactive_cli.execute_command("/missing")
    exiting = interactive_cli.execute_command("/q")

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
