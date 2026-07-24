"""AutoGen-backed roundtable for the batch sector ETF watchlist."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import config
from ..data_agent.etf_watchlist import (
    RoundtableAgentOutput,
    RoundtableDialogueTurn,
    SectorCandidatePayload,
)
from ..llm.client import openai_compatible_model_info, resolve_openai_compatible_settings

logger = logging.getLogger(__name__)

_AGENT_SPEAKERS = {
    "Market_Agent": "Market",
    "Event_Agent": "Event",
    "Analysis_Agent": "Analysis",
    "Risk_Agent": "Risk",
    "System_Moderator": "Moderator",
}


@dataclass
class ETFWatchlistAutoGenResult:
    """Structured AutoGen output for the batch ETF watchlist report."""

    provider: str = "autogen"
    mode: str = "autogen_batch_roundtable"
    summary: str = ""
    agent_outputs: list[RoundtableAgentOutput] = field(default_factory=list)
    dialogue_records: list[RoundtableDialogueTurn] = field(default_factory=list)
    round_history: list[dict[str, Any]] = field(default_factory=list)
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    final_decisions: list[dict[str, Any]] = field(default_factory=list)
    excluded_by_roundtable: list[dict[str, Any]] = field(default_factory=list)
    llm_provider: str = ""
    model: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "mode": self.mode,
            "backtest_used": False,
            "agent_outputs": [item.model_dump(mode="json") for item in self.agent_outputs],
            "dialogue_records": [item.model_dump(mode="json") for item in self.dialogue_records],
            "round_history": self.round_history,
            "summary": self.summary,
            "raw_messages": self.raw_messages,
            "final_decisions": self.final_decisions,
            "excluded_by_roundtable": self.excluded_by_roundtable,
            "llm_provider": self.llm_provider,
            "model": self.model,
            "note": "AutoGen batch roundtable; no backtest is used for ETF watchlist decisions.",
        }


class ETFWatchlistAutoGenRoundtable:
    """Run a bounded AutoGen roundtable for all ETF sector candidates."""

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        timeout_seconds: float | None = None,
        max_turns: int | None = None,
    ) -> None:
        cfg = config.get_all()
        self.provider = provider or cfg.get("llm_provider", "deepseek")
        self.model = model or cfg.get("deep_think_llm", "deepseek-chat")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else cfg.get("autogen_timeout", 120.0)
        )
        self.max_turns = max_turns or cfg.get("autogen_max_turns", 20)

    def run(
        self,
        *,
        trade_date: str,
        candidates: list[SectorCandidatePayload],
        max_final_decisions: int = 3,
    ) -> ETFWatchlistAutoGenResult:
        """Run AutoGen roundtable synchronously (wraps async)."""
        if not candidates:
            return ETFWatchlistAutoGenResult(summary="没有可进入圆桌的 ETF 板块候选。")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("ETF AutoGen roundtable cannot run inside an active event loop")

        return asyncio.run(
            self._run_autogen_roundtable(
                trade_date=trade_date,
                candidates=candidates,
                max_final_decisions=max_final_decisions,
            )
        )

    async def _run_autogen_roundtable(
        self,
        *,
        trade_date: str,
        candidates: list[SectorCandidatePayload],
        max_final_decisions: int = 3,
    ) -> ETFWatchlistAutoGenResult:
        """Async core: create agents, run roundtable, parse output."""
        if not candidates:
            return ETFWatchlistAutoGenResult(summary="没有可进入圆桌的 ETF 板块候选。")

        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        logging.getLogger("autogen_core").setLevel(logging.WARNING)
        logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
        logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)

        model_client = self._build_model_client()
        agents = self._build_agents(model_client, trade_date, candidates)
        team = RoundRobinGroupChat(agents, max_turns=self.max_turns)

        raw_messages: list[dict[str, Any]] = []
        try:
            async for event in team.run_stream():
                raw_messages.append(self._serialize_event(event))
        except Exception as exc:
            logger.error("AutoGen roundtable stream failed: %s", exc)
            return ETFWatchlistAutoGenResult(
                summary=f"AutoGen roundtable failed: {exc}",
                raw_messages=raw_messages,
            )

        return self._parse_output(raw_messages, candidates, max_final_decisions)

    def _build_model_client(self) -> OpenAIChatCompletionClient:
        """Create an OpenAI-compatible model client for AutoGen."""
        settings = resolve_openai_compatible_settings(self.provider)
        model_info = openai_compatible_model_info(settings["base_url"])
        return OpenAIChatCompletionClient(
            model=self.model,
            base_url=settings["base_url"],
            api_key=settings.get("api_key") or os.environ.get(settings.get("api_key_env", "")),
            model_info=model_info,
        )

    def _build_agents(
        self,
        model_client: OpenAIChatCompletionClient,
        trade_date: str,
        candidates: list[SectorCandidatePayload],
    ) -> list[AssistantAgent]:
        """Build the four specialist agents plus moderator."""
        candidate_summary = "\n\n".join(
            f"## 候选 {i + 1}: {c.sector}\n"
            f"  热度排名: {c.hot_rank}\n"
            f"  板块涨跌幅: {c.sector_change_pct:+.2f}%\n"
            f"  资金流向: {c.capital_flow_summary or 'N/A'}\n"
            f"  主要ETF: {', '.join(f'{e.code} {e.name}' for e in c.etf_candidates[:2])}\n"
            for i, c in enumerate(candidates)
        )

        system_context = (
            f"当前交易日: {trade_date}\n"
            f"最大最终决策数: {max_final_decisions}\n\n"
            f"## 候选板块一览\n{candidate_summary}"
        )

        market_agent = AssistantAgent(
            name="Market_Agent",
            model_client=model_client,
            system_message=(
                "你是 Market Agent，负责从市场情绪和资金面角度评估板块。\n"
                f"{system_context}\n"
                "评估标准：板块热度、资金流入强度、市场情绪阶段。\n"
                "输出建议保留/排除的板块及理由。"
            ),
        )

        event_agent = AssistantAgent(
            name="Event_Agent",
            model_client=model_client,
            system_message=(
                "你是 Event Agent，负责从事件驱动角度评估板块。\n"
                f"{system_context}\n"
                "评估标准：近期政策催化、行业事件、新闻情绪。\n"
                "输出建议保留/排除的板块及理由。"
            ),
        )

        analysis_agent = AssistantAgent(
            name="Analysis_Agent",
            model_client=model_client,
            system_message=(
                "你是 Analysis Agent，负责从因子和技术面角度评估板块。\n"
                f"{system_context}\n"
                "评估标准：动量因子、拥挤度、估值分位、技术形态。\n"
                "输出建议保留/排除的板块及理由。"
            ),
        )

        risk_agent = AssistantAgent(
            name="Risk_Agent",
            model_client=model_client,
            system_message=(
                "你是 Risk Agent，负责从风控角度评估板块。\n"
                f"{system_context}\n"
                "评估标准：ETF流动性、溢折价、停牌风险、板块集中度。\n"
                "你有否决权：如果某板块的ETF不可交易或风险过高，应排除。"
            ),
        )

        moderator = AssistantAgent(
            name="System_Moderator",
            model_client=model_client,
            system_message=(
                "你是 Moderator，负责综合 Market、Event、Analysis、Risk 四方的意见，"
                "做出最终决策。\n"
                f"{system_context}\n"
                "输出最终决策时请按优先级列出选中的板块及每个板块对应的首选ETF。"
            ),
        )

        return [market_agent, event_agent, analysis_agent, risk_agent, moderator]

    def _parse_output(
        self,
        raw_messages: list[dict[str, Any]],
        candidates: list[SectorCandidatePayload],
        max_final_decisions: int,
    ) -> ETFWatchlistAutoGenResult:
        """Parse AutoGen raw messages into structured result."""
        summary = ""
        for msg in raw_messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > len(summary):
                summary = content

        # Extract JSON decision blocks from moderator output
        final_decisions: list[dict[str, Any]] = []
        excluded_by_roundtable: list[dict[str, Any]] = []
        moderator_messages = [
            m for m in raw_messages
            if m.get("source") == "System_Moderator" and isinstance(m.get("content"), str)
        ]
        for msg in moderator_messages[-3:]:
            content: str = msg.get("content", "")
            for candidate in self._json_object_candidates(content):
                sector_name = candidate.get("sector", "").strip()
                action = candidate.get("action", "").strip().lower()
                matched = [c for c in candidates if c.sector == sector_name]
                entry = {
                    "sector": sector_name,
                    "action": action,
                    "primary_etf": candidate.get("primary_etf", ""),
                    "reason": candidate.get("reason", ""),
                }
                if action in ("exclude", "排除"):
                    excluded_by_roundtable.append(entry)
                else:
                    final_decisions.append(entry)

        agent_outputs = self._coerce_agent_outputs(raw_messages)
        dialogue = self._coerce_dialogue_records(raw_messages)

        structured = self._extract_structured_summary(summary)
        return ETFWatchlistAutoGenResult(
            provider="autogen",
            mode="autogen_batch_roundtable",
            summary=structured.get("summary", summary[:2000]) if summary else "",
            agent_outputs=agent_outputs,
            dialogue_records=dialogue,
            raw_messages=raw_messages,
            final_decisions=final_decisions[:max_final_decisions],
            excluded_by_roundtable=excluded_by_roundtable,
            llm_provider=self.provider,
            model=self.model,
        )

    @staticmethod
    def _serialize_event(event: Any) -> dict[str, Any]:
        content = ""
        if hasattr(event, "content"):
            raw = event.content
            if isinstance(raw, list):
                content = " ".join(
                    str(getattr(item, "text", str(item))) for item in raw
                )
            elif isinstance(raw, str):
                content = raw
            else:
                content = str(raw)
        source = ""
        if hasattr(event, "source"):
            source = event.source
        return {
            "type": type(event).__name__,
            "source": source,
            "content": content,
        }

    @staticmethod
    def _json_object_candidates(content: str) -> list[dict[str, Any]]:
        """Extract JSON-like sector decision objects from text."""
        results = []
        for match in re.finditer(
            r'\{\s*"sector"\s*:\s*"[^"]*"\s*,\s*"action"\s*:\s*"[^"]*"\s*[^}]*\}',
            content,
        ):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict) and obj.get("sector") and obj.get("action"):
                    results.append(obj)
            except json.JSONDecodeError:
                continue
        return results

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(item) for item in content)
        return str(content)

    @staticmethod
    def _infer_stance(content: str) -> str:
        lowered = content.lower()
        pos_words = ["保留", "推荐", "看好", "positive", "保留/推荐", "建议保留"]
        neg_words = ["排除", "不推荐", "不看好", "negative", "排除/不推荐", "建议排除"]
        pos_score = sum(1 for w in pos_words if w in lowered)
        neg_score = sum(1 for w in neg_words if w in lowered)
        if pos_score > neg_score:
            return "保留"
        if neg_score > pos_score:
            return "排除"
        return "中性"

    @staticmethod
    def _extract_objections(content: str) -> list[str]:
        objections = []
        for marker in ("风险", "担忧", "不足", "问题", "注意"):
            if marker in content:
                for sentence in re.split(r'[。！？\n]', content):
                    if marker in sentence:
                        s = sentence.strip()
                        if s and len(s) > 4:
                            objections.append(s)
        return objections[:5]

    @staticmethod
    def _extract_structured_summary(summary: str) -> dict[str, Any]:
        """Extract structured fields from moderator summary text."""
        result: dict[str, Any] = {"summary": summary[:2000] if summary else ""}
        patterns = {
            "market_verdict": r"市场[：:](.+?)(?=\s|$)",
            "final_action": r"(最终)?决策[：:](.+?)(?=\s|$)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, summary)
            if match:
                result[key] = match.group(1).strip()
        return result

    @staticmethod
    def _coerce_agent_outputs(
        raw_messages: list[dict[str, Any]],
    ) -> list[RoundtableAgentOutput]:
        """Coerce raw messages into structured agent outputs."""
        outputs: list[RoundtableAgentOutput] = []
        seen = set()
        for msg in raw_messages:
            source = msg.get("source", "")
            display = _AGENT_SPEAKERS.get(source, source)
            if display in seen:
                continue
            content = ETFWatchlistAutoGenRoundtable._stringify_content(msg.get("content", ""))
            if not content:
                continue
            stance = ETFWatchlistAutoGenRoundtable._infer_stance(content)
            outputs.append(RoundtableAgentOutput(
                agent=display,
                stance=stance,
                summary=content[:300],
                objections=ETFWatchlistAutoGenRoundtable._extract_objections(content),
                evidence_keys=[],
            ))
            seen.add(display)
        return outputs

    @staticmethod
    def _coerce_dialogue_records(
        raw_messages: list[dict[str, Any]],
    ) -> list[RoundtableDialogueTurn]:
        """Coerce raw messages into chronological dialogue turns."""
        records: list[RoundtableDialogueTurn] = []
        for idx, msg in enumerate(raw_messages, start=1):
            source = msg.get("source", "")
            content = ETFWatchlistAutoGenRoundtable._stringify_content(msg.get("content", ""))
            if not content:
                continue
            records.append(RoundtableDialogueTurn(
                turn=idx,
                agent=_AGENT_SPEAKERS.get(source, source),
                content=content[:500],
            ))
        return records
