"""Interactive slash-command CLI for the trading-agent workflow."""
from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from datetime import date
from typing import Callable, TextIO

from .config import config
from .main import analyze_single, run_standalone_data_agent


HELP_TEXT = """\
Advanced Trading Agent CLI

Slash commands:
  /help
      Show this help.

  /analyze <ticker> [--date YYYY-MM-DD] [--skip-backtest] [--debug]
      Run the full multi-agent analysis workflow.
      Example: /analyze 000001.SZ --date 2026-07-10 --skip-backtest

  /data <ticker> [--date YYYY-MM-DD] [--start-date YYYYMMDD] [--end-date YYYYMMDD]
      Run DataAgent only and print its persisted artifact summary.
      Example: /data 000001.SZ --date 2026-07-10 --start-date 20260701 --end-date 20260710

  /status
      Show configured runtime paths and defaults.

  /config
      Show selected configuration values.

  /clear
      Print ANSI clear-screen sequence.

  /exit or /quit
      Exit the session.

Tip:
  A bare ticker, for example 000001.SZ, is treated as /analyze 000001.SZ --skip-backtest.
"""


@dataclass(frozen=True)
class CommandResult:
    """Result returned by one slash-command execution."""

    output: str
    should_exit: bool = False


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
    parser.add_argument("ticker")
    parser.add_argument("--date", "-d", dest="trade_date")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def build_data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="/data", add_help=False)
    parser.add_argument("ticker")
    parser.add_argument("--date", "-d", dest="trade_date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
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
    return f"/analyze {stripped} --skip-backtest"


def status_text() -> str:
    results_dir = config.get("results_dir", "data/results")
    memory_dir = config.get("memory_dir", "data/memory")
    return "\n".join([
        "Runtime status:",
        f"  results_dir: {results_dir}",
        f"  memory_dir: {memory_dir}",
        "  default bare ticker behavior: /analyze <ticker> --skip-backtest",
    ])


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


def execute_command(
    line: str,
    *,
    analyze_runner: AnalyzeRunner = analyze_single,
    data_runner: DataRunner = run_standalone_data_agent,
) -> CommandResult:
    normalized = normalize_line(line)
    if not normalized:
        return CommandResult("")

    try:
        parts = shlex.split(normalized)
    except ValueError as exc:
        return CommandResult(f"Parse error: {exc}")

    command = parts[0].lower()
    args = parts[1:]

    if command in {"/help", "/?"}:
        return CommandResult(HELP_TEXT)
    if command in {"/exit", "/quit"}:
        return CommandResult("bye", should_exit=True)
    if command == "/clear":
        return CommandResult("\033[2J\033[H")
    if command == "/status":
        return CommandResult(status_text())
    if command == "/config":
        return CommandResult(config_text())
    if command == "/analyze":
        if not args:
            return CommandResult("Usage: /analyze <ticker> [--date YYYY-MM-DD] [--skip-backtest] [--debug]")
        try:
            parsed = _parse_args(build_analyze_parser(), args)
        except ValueError as exc:
            return CommandResult(str(exc))
        trade_date = parsed.trade_date or str(date.today())
        report = analyze_runner(
            parsed.ticker,
            trade_date,
            debug=parsed.debug,
            skip_backtest=parsed.skip_backtest,
        )
        return CommandResult(report)
    if command == "/data":
        if not args:
            return CommandResult("Usage: /data <ticker> [--date YYYY-MM-DD] [--start-date YYYYMMDD] [--end-date YYYYMMDD]")
        try:
            parsed = _parse_args(build_data_parser(), args)
        except ValueError as exc:
            return CommandResult(str(exc))
        output = data_runner(
            parsed.ticker,
            trade_date=parsed.trade_date,
            start_date=parsed.start_date,
            end_date=parsed.end_date,
            output_dir=parsed.output_dir,
            use_react_planner=parsed.react_planner,
            news_keyword=parsed.news_keyword,
            use_llm_news_filter=not parsed.no_llm_news_filter,
            fetch_news_full_text=not parsed.no_news_full_text,
        )
        return CommandResult(output)

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
