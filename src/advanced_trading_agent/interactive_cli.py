"""Interactive slash-command CLI for the trading-agent workflow."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Literal, TextIO

from pydantic import BaseModel, Field

from .config import config
from .data_agent.data_agent import DataAgentRequest
from .data_agent.planner import DataAgentPlanner
from .main import analyze_single, run_standalone_data_agent


HELP_TEXT = "\n".join([
    "",
    "  ╔══════════════════════════════════════════════════════════════╗",
    "  ║                    TRADESIGHT  命令参考                       ║",
    "  ╠══════════════════════════════════════════════════════════════╣",
    "  ║                                                              ║",
    "  ║  市场发现                                                    ║",
    "  ║    /scan          扫描市场热点板块和强势股                    ║",
    "  ║    /market        同 /scan                                   ║",
    "  ║                                                              ║",
    "  ║  分析                                                        ║",
    "  ║    /a <代码>      分析指定股票 (简写)                         ║",
    "  ║    /run <代码>    完整流水线: 数据 + 分析 + 报告              ║",
    "  ║                                                              ║",
    "  ║  数据                                                        ║",
    "  ║    /d <代码>      采集并持久化数据 (简写)                     ║",
    "  ║    /dates <代码>  列出本地已有的交易日                        ║",
    "  ║                                                              ║",
    "  ║  信息                                                        ║",
    "  ║    /status        查看运行时配置和路径                        ║",
    "  ║    /config        查看选定的配置项                            ║",
    "  ║    /help          显示此帮助                                  ║",
    "  ║                                                              ║",
    "  ║  会话                                                        ║",
    "  ║    /clear         清屏                                        ║",
    "  ║    /q, /exit      退出                                        ║",
    "  ║                                                              ║",
    "  ╚══════════════════════════════════════════════════════════════╝",
    "",
    "  示例:",
    "    /scan                     发现市场热点",
    "    /a 000001.SZ              快速分析平安银行",
    "    /run 000001.SZ            完整流水线 (含回测)",
    "    /d 000001.SZ --start 20260101 --end 20260712",
    "",
])


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


class DataIntent(BaseModel):
    """Structured data task parsed from a natural-language request."""

    action: Literal["collect", "dates", "screen"] = Field(
        description="collect=run DataAgent, dates=list stored dates, screen=find strong candidates"
    )
    ticker: str | None = Field(default=None, description="A-share ticker, e.g. 000001.SZ")
    trade_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD")
    start_date: str | None = Field(default=None, description="Data start date YYYYMMDD")
    end_date: str | None = Field(default=None, description="Data end date YYYYMMDD")
    top_n: int = Field(default=10, ge=1, le=100, description="Candidate count for screen tasks")
    query: str = Field(default="", description="Original user request")
    reasoning: str = Field(default="", description="Brief rationale for parsed intent")


IntentParser = Callable[[str, SessionContext], DataIntent]


_TICKER_RE = re.compile(r"\b\d{6}\.(?:SH|SZ|BJ)\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_COMPACT_DATE_RE = re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b")
_TOP_N_RE = re.compile(r"(?:top|前|最近|最好|强势)\s*(\d{1,3})", re.IGNORECASE)


def _today() -> date:
    return date.today()


def _compact_date(value: str | None) -> str | None:
    if not value:
        return None
    return value.replace("-", "")


def _extract_ticker(text: str) -> str | None:
    match = _TICKER_RE.search(text)
    return match.group(0).upper() if match else None


def _extract_date(text: str) -> str | None:
    match = _ISO_DATE_RE.search(text)
    if match:
        return "-".join(match.groups())
    match = _COMPACT_DATE_RE.search(text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return None


def _extract_top_n(text: str, default: int = 10) -> int:
    match = _TOP_N_RE.search(text)
    if not match:
        return default
    try:
        return max(1, min(100, int(match.group(1))))
    except ValueError:
        return default


def _fallback_data_intent(text: str, context: SessionContext | None = None) -> DataIntent:
    context = context or SessionContext()
    normalized = text.strip()
    today = _today()
    ticker = _extract_ticker(normalized) or context.ticker
    explicit_date = _extract_date(normalized)
    trade_date = explicit_date or context.trade_date
    lowered = normalized.lower()

    wants_dates = any(token in normalized for token in ("有哪些日期", "多少 date", "多少日期", "已有日期", "数据日期"))
    wants_screen = any(token in normalized for token in ("比较好", "表现好", "强势", "候选", "筛", "排名", "top", "Top"))
    wants_year = "今年" in normalized or "year" in lowered
    wants_recent = any(token in normalized for token in ("最近", "这一段", "一段时间", "近一段", "近段"))
    wants_today = "今天" in normalized or "当天" in normalized or "收盘" in normalized or "today" in lowered

    if wants_dates:
        return DataIntent(
            action="dates",
            ticker=ticker,
            trade_date=trade_date,
            query=normalized,
            reasoning="规则解析: 用户询问本地已落盘数据日期",
        )

    if wants_screen and not ticker:
        return DataIntent(
            action="screen",
            trade_date=explicit_date or context.trade_date or today.isoformat(),
            start_date=(today - timedelta(days=30)).strftime("%Y%m%d"),
            end_date=today.strftime("%Y%m%d"),
            top_n=_extract_top_n(normalized),
            query=normalized,
            reasoning="规则解析: 用户询问最近表现较好的股票",
        )

    if wants_year:
        start_date = f"{today.year}0101"
        end_date = today.strftime("%Y%m%d")
        return DataIntent(
            action="collect",
            ticker=ticker,
            trade_date=explicit_date or today.isoformat(),
            start_date=start_date,
            end_date=end_date,
            query=normalized,
            reasoning="规则解析: 用户请求今年数据",
        )

    if wants_recent:
        return DataIntent(
            action="collect",
            ticker=ticker,
            trade_date=explicit_date or today.isoformat(),
            start_date=(today - timedelta(days=30)).strftime("%Y%m%d"),
            end_date=today.strftime("%Y%m%d"),
            query=normalized,
            reasoning="规则解析: 用户请求最近一段时间数据",
        )

    if wants_today or not trade_date:
        trade_date = explicit_date or today.isoformat()

    return DataIntent(
        action="collect",
        ticker=ticker,
        trade_date=trade_date,
        start_date=_compact_date(trade_date),
        end_date=_compact_date(trade_date),
        query=normalized,
        reasoning="规则解析: 默认采集指定或当前交易日数据",
    )


def parse_data_intent(
    text: str,
    context: SessionContext | None = None,
    *,
    llm: object | None = None,
) -> DataIntent:
    """Parse a natural-language data request into an auditable DataIntent."""

    context = context or SessionContext()
    fallback = _fallback_data_intent(text, context)
    if llm is None:
        try:
            from .llm.client import create_llm

            llm = create_llm()
        except Exception:
            return fallback

    try:
        intent = llm.chat(
            messages=[
                (
                    "system",
                    "你是 DataAgent 的意图解析器。只把用户自然语言转换成结构化数据任务，"
                    "不要编造行情数据。action 只能是 collect/dates/screen。"
                    "今天/当天/收盘默认指当前日期；今年要转为年初到今天；"
                    "最近一段时间默认近 30 天；如果用户没有给股票代码但上下文有当前股票，可复用上下文。",
                ),
                (
                    "user",
                    json.dumps(
                        {
                            "query": text,
                            "current_ticker": context.ticker,
                            "current_trade_date": context.trade_date,
                            "today": _today().isoformat(),
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
            response_format=DataIntent,
            temperature=0,
        )
    except Exception:
        return fallback

    if not isinstance(intent, DataIntent):
        return fallback
    if intent.ticker is None:
        intent.ticker = fallback.ticker
    if intent.trade_date is None:
        intent.trade_date = fallback.trade_date
    if intent.start_date is None:
        intent.start_date = fallback.start_date
    if intent.end_date is None:
        intent.end_date = fallback.end_date
    if intent.query == "":
        intent.query = text
    return intent


def _looks_like_natural_data_request(args: list[str]) -> bool:
    if not args:
        return False
    option_like = any(part.startswith("-") for part in args)
    if option_like:
        return False
    joined = " ".join(args)
    if re.search(r"[\u4e00-\u9fff]", joined):
        return True
    if len(args) <= 2 and _extract_ticker(" ".join(args)):
        return False
    return len(args) > 2


def _safe_json_summary(text: str, limit: int = 1200) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "\n... <truncated>"


def _format_clarification_questions(intent: DataIntent, questions: list[dict[str, object]]) -> str:
    lines = [
        "# DataAgent Needs Clarification",
        f"query: {intent.query or '<empty>'}",
        f"action: {intent.action}",
        f"reasoning: {intent.reasoning or '<none>'}",
        "",
        "请补充以下信息后再继续：",
    ]
    for idx, item in enumerate(questions, start=1):
        default = item.get("default") if isinstance(item, dict) else None
        suffix = f" 默认: {default}" if default not in (None, "") else ""
        question = item.get("question", "") if isinstance(item, dict) else str(item)
        lines.append(f"{idx}. {question}{suffix}")
    return "\n".join(lines)


def _clarification_questions_for_intent(intent: DataIntent, context: SessionContext) -> list[dict[str, object]]:
    request = DataAgentRequest(
        ticker=intent.ticker or context.ticker or "",
        trade_date=intent.trade_date or context.trade_date,
        start_date=intent.start_date,
        end_date=intent.end_date,
        news_keyword=None,
    )
    questions = DataAgentPlanner.clarification_questions(request)
    return [item for item in questions if item.get("required")]


def _load_recent_candidate_rows(results_dir: str | None = None) -> list[dict[str, object]]:
    root = _configured_results_dir(results_dir)
    rows: list[dict[str, object]] = []
    for response_path in root.rglob("response.json"):
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        request = payload.get("request", {})
        final_data = payload.get("final_data", {})
        cleaned = final_data.get("cleaned", {})
        analysis = final_data.get("analysis", {})
        daily = cleaned.get("daily", {})
        latest = daily.get("latest", {}) if isinstance(daily, dict) else {}
        ticker = request.get("ticker") or latest.get("code")
        trade_date = request.get("trade_date") or latest.get("trade_date")
        if not ticker:
            continue
        score = 0.0
        pct_chg = latest.get("pct_chg") or daily.get("price_change_pct") if isinstance(daily, dict) else None
        try:
            score += float(pct_chg or 0)
        except (TypeError, ValueError):
            pass
        factors = analysis.get("factors") if isinstance(analysis, dict) else None
        if isinstance(factors, dict):
            try:
                score += float(factors.get("composite_score") or 0)
            except (TypeError, ValueError):
                pass
        rows.append({
            "ticker": str(ticker),
            "trade_date": str(trade_date or ""),
            "score": score,
            "pct_chg": pct_chg,
            "response_path": str(response_path),
        })
    return sorted(rows, key=lambda item: float(item.get("score") or 0), reverse=True)


def format_screen_summary(intent: DataIntent, *, results_dir: str | None = None) -> str:
    rows = _load_recent_candidate_rows(results_dir)[: intent.top_n]
    lines = [
        f"Candidate screen: top {intent.top_n}",
        f"query: {intent.query}",
        f"window: {intent.start_date or '<auto>'} -> {intent.end_date or '<auto>'}",
    ]
    if not rows:
        lines.append("  <no local candidate data found; run /data or /run for more tickers first>")
        return "\n".join(lines)
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"  {idx}. {row['ticker']} {row['trade_date']} score={row['score']:.2f} pct_chg={row.get('pct_chg')}"
        )
    return "\n".join(lines)


def execute_data_intent(
    intent: DataIntent,
    *,
    context: SessionContext,
    data_runner: DataRunner,
) -> CommandResult:
    if intent.action == "dates":
        output = format_dates_summary(list_data_dates(intent.ticker))
        return CommandResult(output)

    if intent.action == "screen":
        return CommandResult(format_screen_summary(intent))

    ticker = intent.ticker or context.ticker
    clarification_questions = _clarification_questions_for_intent(intent, context)
    if clarification_questions:
        return CommandResult(_format_clarification_questions(intent, clarification_questions))
    if not ticker:
        return CommandResult("No ticker resolved from data request. Include a ticker, e.g. /data 分析今天收盘后的数据 000001.SZ")

    trade_date = intent.trade_date or context.trade_date or str(date.today())
    output = data_runner(
        ticker,
        trade_date=trade_date,
        start_date=intent.start_date,
        end_date=intent.end_date or _compact_date(trade_date),
        output_dir=None,
        use_react_planner=True,
        news_keyword=None,
        use_llm_news_filter=True,
        fetch_news_full_text=True,
    )
    context.set_current(ticker, trade_date)
    context.last_data_output = output
    return CommandResult("\n".join([
        "# DataAgent Intent",
        f"action: {intent.action}",
        f"ticker: {ticker}",
        f"trade_date: {trade_date}",
        f"start_date: {intent.start_date or '<auto>'}",
        f"end_date: {intent.end_date or _compact_date(trade_date) or '<auto>'}",
        f"reasoning: {intent.reasoning or '<none>'}",
        "",
        _safe_json_summary(output),
    ]))


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


def _handle_scan(
    args: list[str],
    *,
    context: SessionContext,
    analyze_runner: AnalyzeRunner,
) -> CommandResult:
    """Handle /scan — discover hot stocks.

    /scan               Lightweight scan, show ranked candidates.
    /scan --full         Full pipeline: scan + collect + LLM summary + analyze top-N.
    /scan --force        Force re-scan even if today's report is cached.
    """
    from .data_agent.scanner import MarketScanner

    force = "--force" in args or "-f" in args
    full = "--full" in args

    if full or force:
        from .main import scan_and_analyze
        top_n = 10
        for i, a in enumerate(args):
            if a in ("--top", "-n") and i + 1 < len(args):
                try:
                    top_n = int(args[i + 1])
                except ValueError:
                    pass
        report = scan_and_analyze(top_n=top_n, force=force)
        return CommandResult(report)

    scanner = MarketScanner(top_sectors=5, top_n=15)
    results = scanner.scan()

    if not results:
        return CommandResult("No hot stocks found. The market may be quiet today.")

    output = scanner.format_results(results)
    output += "\n\nUse /a <ticker> to analyze any of these candidates."
    output += "\nUse /scan --full for deep analysis with LLM summary."

    return CommandResult(output)


def execute_command(
    line: str,
    *,
    context: SessionContext | None = None,
    analyze_runner: AnalyzeRunner = analyze_single,
    data_runner: DataRunner = run_standalone_data_agent,
    intent_parser: IntentParser = parse_data_intent,
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
    if command in {"/data", "/d", "/datas"}:
        if _looks_like_natural_data_request(args):
            intent = intent_parser(" ".join(args), context)
            return execute_data_intent(intent, context=context, data_runner=data_runner)
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

    if command in {"/scan", "/scanner", "/market"}:
        return _handle_scan(args, context=context, analyze_runner=analyze_runner)

    return CommandResult(f"Unknown command: {command}\nRun /help to see available commands.")


def banner() -> str:
    return "\n".join([
        "",
        "  ╔══════════════════════════════════════════════════════════════╗",
        "  ║                                                              ║",
        "  ║    ████████╗██████╗  █████╗ ██████╗ ███████╗███████╗██╗ ██████╗ ██╗  ██╗████████╗",
        "  ║    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝",
        "  ║       ██║   ██████╔╝███████║██║  ██║█████╗  ███████╗██║██║  ███╗███████║   ██║",
        "  ║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ╚════██║██║██║   ██║██╔══██║   ██║",
        "  ║       ██║   ██║  ██║██║  ██║██████╔╝███████╗███████║██║╚██████╔╝██║  ██║   ██║",
        "  ║       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝",
        "  ║",
        "  ║            A股多智能体量化分析系统  |  Multi-Agent A-Share Analysis",
        "  ║",
        "  ╚══════════════════════════════════════════════════════════════╝",
        "",
        "  /help 查看命令  |  /scan 发现热点  |  /exit 退出",
        "",
    ])


def repl(
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    analyze_runner: AnalyzeRunner = analyze_single,
    data_runner: DataRunner = run_standalone_data_agent,
    intent_parser: IntentParser = parse_data_intent,
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
            intent_parser=intent_parser,
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
