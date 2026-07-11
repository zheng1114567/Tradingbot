"""Roundtable harness for data-aware multi-agent debate.

This module is the seam between collected DataAgent payloads and the debate
adapters. Callers provide workflow state and contradictions; the harness
returns scoped agent contexts plus a shared task prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


_AGENT_ORDER = ("Market", "Event", "Analysis", "Backtest")


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


@dataclass(frozen=True)
class RoundtableContext:
    """Complete public output of the roundtable harness."""

    data_brief: DataAgentBrief
    agent_contexts: dict[str, RoundtableAgentContext] = field(default_factory=dict)
    shared_evidence_text: str = ""
    task: str = ""


class RoundtableHarness:
    """Build scoped debate inputs from workflow state.

    Interface contract:
    - Inputs are plain workflow state dictionaries and contradiction strings.
    - Outputs are immutable context objects that can feed AutoGen or the local
      fallback without either adapter knowing DataAgent internals.
    - Agent prompts are scoped: each participant sees shared metadata plus its
      own report and evidence pack, and must not invent unavailable data.
    """

    _REPORT_KEYS = {
        "Market": "market_report",
        "Event": "event_report",
        "Analysis": "analysis_report",
        "Backtest": "backtest_report",
    }

    def __init__(self, *, char_limit: int = 1800) -> None:
        self.char_limit = char_limit

    def build_context(
        self,
        state: dict[str, Any],
        contradictions: list[str],
    ) -> RoundtableContext:
        tier1 = state.get("tier1_data", {}) or {}
        tier2 = state.get("tier2_data", {}) or {}
        brief = self._build_brief(state, tier1=tier1, tier2=tier2)
        shared = self._build_shared_evidence(brief, contradictions)
        agent_contexts = {
            agent: self._build_agent_context(
                agent,
                state=state,
                tier1=tier1,
                tier2=tier2,
                shared_evidence=shared,
            )
            for agent in _AGENT_ORDER
        }
        task = self._build_task(
            ticker=brief.ticker,
            trade_date=brief.trade_date,
            contradictions=contradictions,
            shared_evidence=shared,
        )
        return RoundtableContext(
            data_brief=brief,
            agent_contexts=agent_contexts,
            shared_evidence_text=shared,
            task=task,
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

    def _build_agent_context(
        self,
        agent: str,
        *,
        state: dict[str, Any],
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        shared_evidence: str,
    ) -> RoundtableAgentContext:
        report = self._agent_report(state, agent)
        evidence = self._agent_evidence(agent, tier1=tier1, tier2=tier2)
        system_message = self._system_message(
            agent=agent,
            report=report,
            evidence=evidence,
            shared_evidence=shared_evidence,
        )
        return RoundtableAgentContext(
            name=agent,
            report=report,
            evidence_text=evidence,
            system_message=system_message,
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
                "backtest_samples": self._limit_list(tier2.get("backtest_samples", []), 8),
                "price_data_tail": self._limit_list(tier2.get("price_data", []), 10),
                "data_quality": tier2.get("data_quality", {}),
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
    ) -> str:
        focus = {
            "Market": "市场温度、资金确认、仓位约束、行业环境",
            "Event": "事件传导、证据等级、定价状态、证伪条件",
            "Analysis": "因子排序、拥挤风险、择时过滤、数据质量",
            "Backtest": "样本量、胜率、超额收益、统计可靠性",
        }[agent]
        return f"""你是 {agent} Agent，参加 Round 2 圆桌会议。

边界规则:
1. 只能引用自己的 AgentContext、DATA_AGENT_BRIEF 和你已有的报告。
2. 只能引用自己的 AgentContext 中的数据字段；不要替其他 Agent 解释其私有证据。
3. 如果证据缺失，必须明确说“数据不足”，不得补造外部数据。
4. 必须回应矛盾点，并说明对最终裁定的影响: upgrade/neutral/downgrade。
5. 输出包含: 立场、引用证据、对矛盾的解释、对最终裁定的压力。

关注范围: {focus}

DATA_AGENT_BRIEF:
{shared_evidence[: self.char_limit]}

AgentContext:
{evidence[: self.char_limit]}

既有报告:
{report[: self.char_limit]}
"""

    def _build_task(
        self,
        *,
        ticker: str,
        trade_date: str,
        contradictions: list[str],
        shared_evidence: str,
    ) -> str:
        return f"""请进行 Round 2 圆桌会议。

DATA_AGENT_BRIEF:
{shared_evidence[: self.char_limit]}

标的: {ticker}
交易日: {trade_date}

矛盾点:
{chr(10).join(f"- {c}" for c in contradictions)}

会议规则:
1. 每个 Agent 必须基于自己的 AgentContext 发言，不能复述或发明其他 Agent 的私有数据。
2. 每次发言都要指出引用的 DataAgent 字段或报告片段。
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
