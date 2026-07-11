"""Interactive slash-command CLI for the trading-agent workflow."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, TextIO

from .config import config
from .main import analyze_single, run_standalone_data_agent


HELP_TEXT = """\
Advanced Trading Agent CLI

Slash commands:
  /help
      Show this help.

  <ticker> [YYYY-MM-DD]
      Fast path. Analyze a ticker and skip backtest by default.
      Example: 000001.SZ 2026-07-10

  /a <ticker> [YYYY-MM-DD]
  /analyze [ticker] [YYYY-MM-DD] [--with-backtest] [--debug]
      Run the full multi-agent analysis workflow.
      Example: /a 000001.SZ 2026-07-10
      If ticker is omitted, analyze the current ticker set by /data or a previous analysis.

  /d <ticker> [YYYY-MM-DD]
  /data [ticker] [--date YYYY-MM-DD] [--start-date YYYYMMDD] [--end-date YYYYMMDD]
      Run DataAgent only and print its persisted artifact summary.
      Example: /d 000001.SZ 2026-07-10
      If ticker is omitted, refresh data for the current ticker.

  /date [ticker]
  /dates [ticker]
      Show which trade dates exist in local DataAgent artifacts.
      Example: /dates 000001.SZ

  /run [ticker] [YYYY-MM-DD] [--skip-backtest] [--debug]
      Run the normal end-to-end flow: data collection, analysis, report, and memory write.
      Example: /run 000001.SZ
      If ticker is omitted, run the current ticker. Date defaults to today.
      Backtest is included by default for /run.

  /s or /status
      Show configured runtime paths and defaults.

  /config
      Show selected configuration values.

  /clear
      Print ANSI clear-screen sequence.

  /q, /exit, or /quit
      Exit the session.

Tip:
  Backtest is skipped by default in the interactive CLI. Add --with-backtest only when needed.
"""


@dataclass(frozen=True)
class CommandResult:
    """Result returned by one slash-command execution."""

    output: str
    should_exit: bool = False


@dataclass
class SessionContext:
    """Mutable context for one interactive CLI session."""

    ticker: str | None = None
    trade_date: str | None = None
    last_data_output: str | None = None
    last_report: str | None = None

    def set_current(self, ticker: str, trade_date: str | None) -> None:
        self.ticker = ticker
        self.trade_date = trade_date or str(date.today())


AnalyzeRunner = Callable[..., str]
DataRunner = Callable[..., str]


def build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-agent",
        description="Claude-style interactive CLI for Advanced Trading Agent.",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Start without printing the welcome banner.",
    )
    return parser


def build_analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="/analyze", add_help=False)
    parser.add_argument("ticker", nargs="?")
    parser.add_argument("date_arg", nargs="?")
    parser.add_argument("--date", "-d", dest="trade_date")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--with-backtest", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def build_data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="/data", add_help=False)
    parser.add_argument("ticker", nargs="?")
    parser.add_argument("date_arg", nargs="?")
    parser.add_argument("--date", "-d", dest="trade_date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output-dir")
    parser.add_argument("--react-planner", action="store_true")
    parser.add_argument("--news-keyword")
    parser.add_argument("--no-llm-news-filter", action="store_true")
    parser.add_argument("--no-news-full-text", action="store_true")
    return parser


def build_dates_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="/dates", add_help=False)
    parser.add_argument("ticker", nargs="?")
    parser.add_argument("--results-dir")
    return parser


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="/run", add_help=False)
    parser.add_argument("ticker", nargs="?")
    parser.add_argument("date_arg", nargs="?")
    parser.add_argument("--date", "-d", dest="trade_date")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--react-planner", action="store_true")
    parser.add_argument("--news-keyword")
    parser.add_argument("--no-llm-news-filter", action="store_true")
    parser.add_argument("--no-news-full-text", action="store_true")
    return parser


class _ParserError(Exception):
    pass


class _NonExitingParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ParserError(message)


def _parse_args(parser: argparse.ArgumentParser, args: list[str]) -> argparse.Namespace:
    non_exiting = _NonExitingParser(prog=parser.prog, add_help=False)
    for action in parser._actions:
        if action.dest == "help":
            continue
        option_strings = list(action.option_strings)
        kwargs = {
            "dest": action.dest,
            "default": action.default,
            "nargs": action.nargs,
            "const": action.const,
            "choices": action.choices,
            "type": action.type,
            "required": action.required,
            "help": action.help,
            "metavar": action.metavar,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if isinstance(action, argparse._StoreTrueAction):
            kwargs.pop("nargs", None)
            kwargs.pop("const", None)
            kwargs.pop("type", None)
            non_exiting.add_argument(*option_strings, action="store_true", **kwargs)
        elif option_strings:
            non_exiting.add_argument(*option_strings, **kwargs)
        else:
            kwargs.pop("dest", None)
            kwargs.pop("required", None)
            non_exiting.add_argument(action.dest, **kwargs)
    try:
        return non_exiting.parse_args(args)
    except _ParserError as exc:
        raise ValueError(f"{parser.prog}: {exc}") from exc
    except SystemExit as exc:
        raise ValueError(f"{parser.prog}: invalid arguments") from exc


def normalize_line(line: str) -> str:
    stripped = line.strip().lstrip("\ufeff")
    if not stripped:
        return ""
    if stripped.startswith("/"):
        return stripped
    return f"/analyze {stripped}"


def status_text(context: SessionContext | None = None) -> str:
    results_dir = config.get("results_dir", "data/results")
    memory_dir = config.get("memory_dir", "data/memory")
    lines = [
        "Runtime status:",
        f"  results_dir: {results_dir}",
        f"  memory_dir: {memory_dir}",
        "  fast path: 000001.SZ 2026-07-10",
        "  default: /a skips backtest unless --with-backtest is set",
        "  default: /run includes backtest unless --skip-backtest is set",
    ]
    if context is not None and context.ticker:
        lines.append(f"  current: {context.ticker} {context.trade_date or str(date.today())}")
    else:
        lines.append("  current: <none>")
    return "\n".join(lines)


def config_text() -> str:
    keys = [
        "results_dir",
        "memory_dir",
        "llm_provider",
        "model",
        "temperature",
        "max_tokens",
    ]
    lines = ["Selected config:"]
    for key in keys:
        lines.append(f"  {key}: {config.get(key, '<unset>')}")
    return "\n".join(lines)


def _configured_results_dir(results_dir: str | None = None) -> Path:
    configured = results_dir or config.get("results_dir", "data/results")
    return Path(str(configured)).expanduser()


def _normalize_ticker_for_path(ticker: str | None) -> str | None:
    if not ticker:
        return None
    return ticker.replace(".", "_")


def list_data_dates(
    ticker: str | None = None,
    *,
    results_dir: str | None = None,
) -> dict[str, object]:
    """Return available trade dates discovered from local DataAgent artifacts."""

    root = _configured_results_dir(results_dir)
    runs_dir = root / "data_agent_runs"
    wanted_ticker = _normalize_ticker_for_path(ticker)
    dates: dict[str, dict[str, object]] = {}

    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            parts = run_dir.name.split("_")
            if len(parts) < 3:
                continue
            trade_date = parts[0]
            run_ticker = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
            if wanted_ticker and run_ticker != wanted_ticker:
                continue
            entry = dates.setdefault(
                trade_date,
                {"date": trade_date, "tickers": set(), "run_count": 0},
            )
            entry["tickers"].add(run_ticker)
            entry["run_count"] = int(entry["run_count"]) + 1

    if root.exists():
        for manifest_path in root.rglob("manifest_*.json"):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            trade_date = str(payload.get("trade_date") or "")
            run_ticker = _normalize_ticker_for_path(str(payload.get("ticker") or ""))
            if not trade_date:
                continue
            if wanted_ticker and run_ticker != wanted_ticker:
                continue
            entry = dates.setdefault(
                trade_date,
                {"date": trade_date, "tickers": set(), "run_count": 0},
            )
            if run_ticker:
                entry["tickers"].add(run_ticker)

    normalized = []
    for trade_date, entry in sorted(dates.items()):
        normalized.append({
            "date": trade_date,
            "tickers": sorted(entry["tickers"]),
            "run_count": entry["run_count"],
        })

    return {
        "results_dir": str(root),
        "ticker": ticker,
        "date_count": len(normalized),
        "dates": normalized,
    }


def format_dates_summary(payload: dict[str, object]) -> str:
    dates = payload.get("dates", [])
    ticker = payload.get("ticker") or "all"
    lines = [
        f"Data dates for {ticker}: {payload.get('date_count', 0)}",
        f"results_dir: {payload.get('results_dir')}",
    ]
    if not dates:
        lines.append("  <none>")
        return "\n".join(lines)
    for item in dates:
        tickers = ", ".join(item.get("tickers", [])) if isinstance(item, dict) else ""
        run_count = item.get("run_count", 0) if isinstance(item, dict) else 0
        trade_date = item.get("date", "") if isinstance(item, dict) else ""
        suffix = f" ({tickers})" if tickers else ""
        lines.append(f"  {trade_date}: {run_count} run(s){suffix}")
    return "\n".join(lines)


def execute_command(
    line: str,
    *,
    context: SessionContext | None = None,
    analyze_runner: AnalyzeRunner = analyze_single,
    data_runner: DataRunner = run_standalone_data_agent,
) -> CommandResult:
    context = context or SessionContext()
    normalized = normalize_line(line)
    if not normalized:
        return CommandResult("")

    try:
        parts = shlex.split(normalized, posix=sys.platform != "win32")
    except ValueError as exc:
        return CommandResult(f"Parse error: {exc}")

    command = parts[0].lower()
    args = parts[1:]

    if command in {"/help", "/?", "/h"}:
        return CommandResult(HELP_TEXT)
    if command in {"/exit", "/quit", "/q"}:
        return CommandResult("bye", should_exit=True)
    if command == "/clear":
        return CommandResult("\033[2J\033[H")
    if command in {"/status", "/s"}:
        return CommandResult(status_text(context))
    if command == "/config":
        return CommandResult(config_text())
    if command in {"/date", "/dates"}:
        try:
            parsed = _parse_args(build_dates_parser(), args)
        except ValueError as exc:
            return CommandResult(str(exc))
        ticker = parsed.ticker or context.ticker
        return CommandResult(format_dates_summary(
            list_data_dates(ticker, results_dir=parsed.results_dir)
        ))
    if command in {"/analyze", "/a"}:
        try:
            parsed = _parse_args(build_analyze_parser(), args)
        except ValueError as exc:
            return CommandResult(str(exc))
        ticker = parsed.ticker or context.ticker
        if not ticker:
            return CommandResult("No current ticker. Run /data <ticker> first or use /a <ticker>.")
        trade_date = (
            parsed.trade_date
            or parsed.date_arg
            or context.trade_date
            or str(date.today())
        )
        report = analyze_runner(
            ticker,
            trade_date,
            debug=parsed.debug,
            skip_backtest=not parsed.with_backtest,
        )
        context.set_current(ticker, trade_date)
        context.last_report = report
        return CommandResult(report)
    if command in {"/data", "/d"}:
        try:
            parsed = _parse_args(build_data_parser(), args)
        except ValueError as exc:
            return CommandResult(str(exc))
        ticker = parsed.ticker or context.ticker
        if not ticker:
            return CommandResult("No current ticker. Use /data <ticker> first.")
        trade_date = parsed.trade_date or parsed.date_arg or context.trade_date
        output = data_runner(
            ticker,
            trade_date=trade_date,
            start_date=parsed.start_date,
            end_date=parsed.end_date,
            output_dir=parsed.output_dir,
            use_react_planner=parsed.react_planner,
            news_keyword=parsed.news_keyword,
            use_llm_news_filter=not parsed.no_llm_news_filter,
            fetch_news_full_text=not parsed.no_news_full_text,
        )
        context.set_current(ticker, trade_date)
        context.last_data_output = output
        return CommandResult(output)
    if command == "/run":
        try:
            parsed = _parse_args(build_run_parser(), args)
        except ValueError as exc:
            return CommandResult(str(exc))
        ticker = parsed.ticker or context.ticker
        if not ticker:
            return CommandResult("No current ticker. Use /run <ticker> first.")
        trade_date = (
            parsed.trade_date
            or parsed.date_arg
            or context.trade_date
            or str(date.today())
        )
        data_output = data_runner(
            ticker,
            trade_date=trade_date,
            start_date=trade_date.replace("-", ""),
            end_date=trade_date.replace("-", ""),
            output_dir=parsed.output_dir,
            use_react_planner=parsed.react_planner,
            news_keyword=parsed.news_keyword,
            use_llm_news_filter=not parsed.no_llm_news_filter,
            fetch_news_full_text=not parsed.no_news_full_text,
        )
        report = analyze_runner(
            ticker,
            trade_date,
            debug=parsed.debug,
            skip_backtest=parsed.skip_backtest,
        )
        context.set_current(ticker, trade_date)
        context.last_data_output = data_output
        context.last_report = report
        return CommandResult("\n\n".join([
            "# Full Run Complete",
            "## Data",
            data_output,
            "## Analysis",
            report,
        ]))

    return CommandResult(f"Unknown command: {command}\nRun /help to see available commands.")


def banner() -> str:
    return "\n".join([
        "Advanced Trading Agent",
        "Type /help for commands. Type /exit to quit.",
    ])


def repl(
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    analyze_runner: AnalyzeRunner = analyze_single,
    data_runner: DataRunner = run_standalone_data_agent,
    show_banner: bool = True,
) -> int:
    context = SessionContext()
    if show_banner:
        print(banner(), file=stdout)

    while True:
        print("ata> ", end="", file=stdout, flush=True)
        line = stdin.readline()
        if line == "":
            print("", file=stdout)
            return 0
        result = execute_command(
            line,
            context=context,
            analyze_runner=analyze_runner,
            data_runner=data_runner,
        )
        if result.output:
            print(result.output, file=stdout)
        if result.should_exit:
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_root_parser()
    args = parser.parse_args(argv)
    return repl(show_banner=not args.no_banner)


if __name__ == "__main__":
    raise SystemExit(main())
