"""Roundtable harness for data-aware multi-agent debate.

This module is the seam between collected DataAgent payloads and the debate
adapters. Callers provide workflow state and contradictions; the harness
returns scoped agent contexts plus a shared task prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..agents.specs import build_roundtable_system_message
from ..tool_nodes.registry import get_allowed_tool_names
from .schemas import EvidenceItem


# Roundtable participant ordering.
# Default participants are always present; specialists join conditionally
# based on A-share signal triggers.
_ALL_PARTICIPANTS_ORDERED: tuple[str, ...] = (
    "Market", "Event", "Policy", "HotMoney", "Analysis", "Unlock", "Backtest",
)
_DEFAULT_PARTICIPANTS: tuple[str, ...] = ("Market", "Event", "Analysis", "Backtest")

# Trigger rules for conditional specialist participants.
# Keys match tier2_data.a_share_signals dict keys (lowercase with underscores).
_TRIGGER_RULES: dict[str, dict[str, Any]] = {
    "policy": {
        "min_strength": 0.6,
        "required_signals": ["positive", "negative"],
        "require_data_status": ["available", "partial"],
    },
    "hot_money": {
        "required_signals": ["confirmed", "speculative", "overheated"],
        "require_data_status": ["available", "partial"],
    },
    "unlock": {
        "required_risk_levels": ["high", "medium"],
        "require_data_status": ["available"],
    },
}

# Map signal dict keys → participant display names used in _ALL_PARTICIPANTS_ORDERED.
_SIGNAL_TO_PARTICIPANT: dict[str, str] = {
    "policy": "Policy",
    "hot_money": "HotMoney",
    "unlock": "Unlock",
}


@dataclass(frozen=True)
class DataAgentBrief:
    """Compact summary of the DataAgent payload entering Round 2."""

    ticker: str
    trade_date: str
    manifest_text: str
    quality_text: str
    risk_text: str
    shared_text: str


@dataclass(frozen=True)
class RoundtableAgentContext:
    """Evidence and instructions for one debate participant."""

    name: str
    report: str
    evidence_text: str
    system_message: str
    allowed_tool_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoundtableContext:
    """Complete public output of the roundtable harness."""

    data_brief: DataAgentBrief
    agent_contexts: dict[str, RoundtableAgentContext] = field(default_factory=dict)
    shared_evidence_text: str = ""
    evidence_board: list[EvidenceItem] = field(default_factory=list)
    task: str = ""
    agent_order: tuple[str, ...] = ()


class RoundtableHarness:
    """Build scoped debate inputs from workflow state.

    Interface contract:
    - Inputs are plain workflow state dictionaries and contradiction strings.
    - Outputs are immutable context objects that can feed AutoGen or the local
      fallback without either adapter knowing DataAgent internals.
    - Agent prompts are scoped: each participant sees shared metadata plus its
      own report and evidence pack, and must not invent unavailable data.
    """

    _REPORT_KEYS: dict[str, str] = {
        "Market": "market_report",
        "Event": "event_report",
        "Analysis": "analysis_report",
        "Backtest": "backtest_report",
        "HotMoney": "hot_money_report",
        "Policy": "policy_report",
        "Unlock": "unlock_report",
    }

    def __init__(self, *, char_limit: int = 1800) -> None:
        self.char_limit = char_limit

    @staticmethod
    def _signal_meets_criteria(
        signal_data: dict[str, Any],
        rules: dict[str, Any],
    ) -> bool:
        """Check if a signal dict meets its trigger rules.

        Returns True when all conditions in *rules* are satisfied.
        """
        # data_status gate
        required_statuses = rules.get("require_data_status", [])
        if required_statuses:
            if signal_data.get("data_status") not in required_statuses:
                return False

        # signal value match (at least one must match)
        if signal_data.get("signal") in rules.get("required_signals", []):
            return True

        # min_strength threshold
        min_strength = rules.get("min_strength")
        if min_strength is not None:
            try:
                if float(signal_data.get("strength", 0)) >= min_strength:
                    return True
            except (TypeError, ValueError):
                pass

        # risk_level match (for Unlock)
        if signal_data.get("risk_level") in rules.get("required_risk_levels", []):
            return True

        return False

    @staticmethod
    def _resolve_participants(
        tier2: dict[str, Any],
    ) -> tuple[str, ...]:
        """Resolve the participant list from A-share signal triggers.

        Always includes default participants; conditionally activates
        specialists whose signals pass the trigger rules.
        """
        a_share = tier2.get("a_share_signals", {}) or {}
        active_specialists: set[str] = set()

        for signal_key, rules in _TRIGGER_RULES.items():
            signal_data = a_share.get(signal_key, {}) or {}
            if RoundtableHarness._signal_meets_criteria(signal_data, rules):
                participant = _SIGNAL_TO_PARTICIPANT.get(signal_key)
                if participant:
                    active_specialists.add(participant)

        return tuple(
            p for p in _ALL_PARTICIPANTS_ORDERED
            if p in _DEFAULT_PARTICIPANTS or p in active_specialists
        )

    def build_context(
        self,
        state: dict[str, Any],
        contradictions: list[str],
    ) -> RoundtableContext:
        tier1 = state.get("tier1_data", {}) or {}
        tier2 = state.get("tier2_data", {}) or {}
        brief = self._build_brief(state, tier1=tier1, tier2=tier2)
        shared = self._build_shared_evidence(brief, contradictions)
        evidence_board = self.build_evidence_board(state)
        participants = self._resolve_participants(tier2)
        agent_contexts = {
            agent: self._build_agent_context(
                agent,
                state=state,
                tier1=tier1,
                tier2=tier2,
                shared_evidence=shared,
                evidence_board=evidence_board,
            )
            for agent in participants
        }
        task = self._build_task(
            ticker=brief.ticker,
            trade_date=brief.trade_date,
            contradictions=contradictions,
            shared_evidence=shared,
            evidence_board=evidence_board,
        )
        return RoundtableContext(
            data_brief=brief,
            agent_contexts=agent_contexts,
            shared_evidence_text=shared,
            evidence_board=evidence_board,
            task=task,
            agent_order=participants,
        )

    def _build_brief(
        self,
        state: dict[str, Any],
        *,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
    ) -> DataAgentBrief:
        manifest = state.get("pit_manifest") or tier1.get("_data_manifest") or {}
        quality = tier2.get("data_quality") or state.get("data_quality_report") or {}
        risk = tier1.get("risk") or {}
        shared_payload = {
            "market": tier1.get("market", {}),
            "sentiment": tier1.get("sentiment", {}),
            "capital": tier1.get("capital", {}),
            "sector": tier1.get("sector", {}),
            "data_summary": tier2.get("data_summary", {}),
        }
        return DataAgentBrief(
            ticker=str(state.get("company_of_interest", "")),
            trade_date=str(state.get("trade_date", "")),
            manifest_text=self._json_excerpt(manifest),
            quality_text=self._json_excerpt(quality),
            risk_text=self._json_excerpt(risk),
            shared_text=self._json_excerpt(shared_payload),
        )

    def _build_shared_evidence(
        self,
        brief: DataAgentBrief,
        contradictions: list[str],
    ) -> str:
        lines = [
            "DATA_AGENT_BRIEF",
            f"ticker={brief.ticker}",
            f"trade_date={brief.trade_date}",
            "contradictions=" + self._json_excerpt(contradictions, limit=800),
            "manifest=" + brief.manifest_text,
            "data_quality=" + brief.quality_text,
            "risk=" + brief.risk_text,
            "shared_market_snapshot=" + brief.shared_text,
        ]
        return "\n".join(lines)

    def build_evidence_board(
        self,
        state: dict[str, Any],
    ) -> list[EvidenceItem]:
        """Build a shared, traceable evidence board from workflow state.

        Extracts key data points from tier1/tier2 and agent reports,
        assigning each a unique ID for cross-referencing in debate.
        """
        evidence: list[EvidenceItem] = []
        tier1 = state.get("tier1_data", {}) or {}
        tier2 = state.get("tier2_data", {}) or {}
        idx = 0

        def _add(
            agent: str,
            field: str,
            value: Any,
            *,
            tag: str | None = None,
        ) -> str:
            nonlocal idx
            idx += 1
            eid = f"ev_{agent.lower()}_{idx:03d}"
            evidence.append(
                EvidenceItem(
                    id=eid,
                    agent=agent,
                    field_path=field,
                    value=str(value)[:200],
                    contradiction_tag=tag,
                )
            )
            return eid

        # Market evidence from tier1
        capital = tier1.get("capital", {}) or {}
        if capital.get("confirmation"):
            _add("Market", "tier1_data.capital.confirmation", capital["confirmation"])
        sentiment = tier1.get("sentiment", {}) or {}
        if sentiment.get("sentiment"):
            _add("Market", "tier1_data.sentiment.sentiment", sentiment["sentiment"])
        market = tier1.get("market", {}) or {}
        if market.get("index_change_pct"):
            _add(
                "Market",
                "tier1_data.market.index_change_pct",
                market["index_change_pct"],
            )
        sector = tier1.get("sector", {}) or {}
        sector_name = sector.get("matched_sector") or sector.get("leading_sector")
        if sector_name:
            field = "tier1_data.sector.matched_sector" if sector.get("matched_sector") else "tier1_data.sector.leading_sector"
            _add("Market", field, sector_name)

        # Event evidence from tier2
        events = tier2.get("events", []) or []
        for i, ev in enumerate(events[:3]):
            _add("Event", f"tier2_data.events[{i}].title", ev.get("title", ""))
            if ev.get("direction"):
                _add("Event", f"tier2_data.events[{i}].direction", ev["direction"])

        # Analysis evidence from tier2
        factors = tier2.get("factors", []) or []
        for i, fct in enumerate(factors[:5]):
            _add("Analysis", f"tier2_data.factors[{i}].name", fct.get("name", ""))
            score = fct.get("composite_score", fct.get("score"))
            if score is not None:
                field = "composite_score" if fct.get("composite_score") is not None else "score"
                _add("Analysis", f"tier2_data.factors[{i}].{field}", score)

        # Backtest evidence from tier2
        samples = tier2.get("backtest_samples", []) or []
        _add("Backtest", "tier2_data.backtest_samples.count", len(samples))

        # HotMoney evidence from tier2_data.a_share_signals
        a_share = tier2.get("a_share_signals", {}) or {}
        hm = a_share.get("hot_money", {}) or {}
        if hm.get("signal"):
            _add("HotMoney", "a_share_signals.hot_money.signal", hm["signal"])
        if hm.get("score"):
            _add("HotMoney", "a_share_signals.hot_money.score", hm["score"])
        if hm.get("board_count"):
            _add("HotMoney", "a_share_signals.hot_money.board_count", hm["board_count"])
        if hm.get("warnings"):
            for w in hm["warnings"][:2]:
                _add("HotMoney", "a_share_signals.hot_money.warning", w)

        # Agent report summaries
        for agent_name, key in [
            ("Market", "market_report_obj"),
            ("Event", "event_report_obj"),
            ("Analysis", "analysis_report_obj"),
            ("Backtest", "backtest_report_obj"),
            ("HotMoney", "hot_money_report_obj"),
        ]:
            rpt = state.get(key)
            if rpt:
                _add(
                    agent_name,
                    f"{key}.decision",
                    getattr(rpt, "decision", None)
                    if hasattr(rpt, "decision")
                    else "N/A",
                )

        return evidence

    def _build_agent_context(
        self,
        agent: str,
        *,
        state: dict[str, Any],
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        shared_evidence: str,
        evidence_board: list | None = None,
    ) -> RoundtableAgentContext:
        report = self._agent_report(state, agent)
        evidence = self._agent_evidence(agent, tier1=tier1, tier2=tier2)
        allowed_tool_names = get_allowed_tool_names(agent.lower())
        system_message = self._system_message(
            agent=agent,
            report=report,
            evidence=evidence,
            shared_evidence=shared_evidence,
            evidence_board=evidence_board or [],
        )
        return RoundtableAgentContext(
            name=agent,
            report=report,
            evidence_text=evidence,
            system_message=system_message,
            allowed_tool_names=allowed_tool_names,
        )

    def _agent_evidence(
        self,
        agent: str,
        *,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
    ) -> str:
        if agent == "Market":
            payload = {
                "market": tier1.get("market", {}),
                "sentiment": tier1.get("sentiment", {}),
                "capital": tier1.get("capital", {}),
                "sector": tier1.get("sector", {}),
                "risk": tier1.get("risk", {}),
            }
        elif agent == "Event":
            payload = {
                "events": self._limit_list(tier2.get("events", []), 5),
                "sector_context": tier2.get("sector_context", {}),
                "data_quality": tier2.get("data_quality", {}),
            }
        elif agent == "Analysis":
            payload = {
                "factors": self._limit_list(tier2.get("factors", []), 8),
                "sector_context": tier2.get("sector_context", {}),
                "price_data_tail": self._limit_list(tier2.get("price_data", []), 5),
                "data_quality": tier2.get("data_quality", {}),
            }
        elif agent == "Backtest":
            payload = {
                "backtest_samples": self._limit_list(
                    tier2.get("backtest_samples", []), 8
                ),
                "price_data_tail": self._limit_list(tier2.get("price_data", []), 10),
                "data_quality": tier2.get("data_quality", {}),
            }
        elif agent == "HotMoney":
            a_share = tier2.get("a_share_signals", {}) or {}
            payload = {
                "hot_money": a_share.get("hot_money", {}),
                "limit_up_summary": tier2.get("limit_up_summary", {}),
                "dragon_tiger": self._limit_list(
                    tier2.get("dragon_tiger", []), 10
                ),
            }
        else:
            payload = {}
        return self._json_excerpt(payload)

    def _system_message(
        self,
        *,
        agent: str,
        report: str,
        evidence: str,
        shared_evidence: str,
        evidence_board: list | None = None,
    ) -> str:
        return build_roundtable_system_message(
            agent=agent,
            report=report,
            evidence=evidence,
            shared_evidence=shared_evidence,
            evidence_board=evidence_board or [],
            char_limit=self.char_limit,
        )

    def _build_task(
        self,
        *,
        ticker: str,
        trade_date: str,
        contradictions: list[str],
        shared_evidence: str,
        evidence_board: list | None = None,
    ) -> str:
        board_section = ""
        if evidence_board:
            formatted = "\n".join(
                f"  [{e.id}] {e.agent}: {e.field_path} = {e.value[:80]}"
                for e in evidence_board
            )
            board_section = (
                f"\n共享证据板 (引用时使用 [ev_xxxxx]):\n{formatted[:1000]}\n"
            )
        return f"""请进行 Round 2 圆桌会议。

DATA_AGENT_BRIEF:
{shared_evidence[: self.char_limit]}
{board_section}
标的: {ticker}
交易日: {trade_date}

矛盾点:
{chr(10).join(f"- {c}" for c in contradictions)}

会议规则:
1. 每个 Agent 必须基于自己的 AgentContext 发言，不能复述或发明其他 Agent 的私有数据。
2. 每次发言都要指出引用的 DataAgent 字段或报告片段，使用 [ev_xxxxx] ID 引用证据。
3. Moderator 最后输出 unresolved_conflicts、final_pressure(upgrade/neutral/downgrade) 和风控关注点。
4. 如果证据链不足，final_pressure 应偏 neutral 或 downgrade，不得强行 upgrade。
"""

    def _agent_report(self, state: dict[str, Any], agent: str) -> str:
        report = str(state.get(self._REPORT_KEYS[agent], "")).strip()
        return report[: self.char_limit] if report else "暂无该 Agent 报告"

    def _json_excerpt(self, value: Any, *, limit: int | None = None) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
        max_len = limit or self.char_limit
        return text[:max_len]

    @staticmethod
    def _limit_list(value: Any, limit: int) -> list[Any]:
        if not isinstance(value, list):
            return []
        return value[-limit:]
