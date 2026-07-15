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
            "note": "AutoGen batch roundtable; no backtest is used for ETF watchlist decisions.",
        }


class ETFWatchlistAutoGenRoundtable:
    """Run a bounded AutoGen roundtable for all ETF sector candidates."""

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        timeout_seconds: float = 90.0,
        max_turns: int = 5,
    ) -> None:
        cfg = config.get_all()
        self.provider = provider or cfg.get("llm_provider", "deepseek")
        self.model = model or cfg.get("deep_think_llm", "deepseek-chat")
        self.timeout_seconds = timeout_seconds
        self.max_turns = max_turns

    def run(
        self,
        *,
        trade_date: str,
        candidates: list[SectorCandidatePayload],
        max_final_decisions: int,
    ) -> ETFWatchlistAutoGenResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(asyncio.wait_for(
                self._run_async(
                    trade_date=trade_date,
                    candidates=candidates,
                    max_final_decisions=max_final_decisions,
                ),
                timeout=self.timeout_seconds,
            ))
        raise RuntimeError("ETF AutoGen roundtable cannot run inside an active event loop")

    async def _run_async(
        self,
        *,
        trade_date: str,
        candidates: list[SectorCandidatePayload],
        max_final_decisions: int,
    ) -> ETFWatchlistAutoGenResult:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        logging.getLogger("autogen_core").setLevel(logging.WARNING)
        logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
        logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)

        if not candidates:
            return ETFWatchlistAutoGenResult(summary="没有可进入圆桌的 ETF 板块候选。")

        model_client = self._create_model_client(OpenAIChatCompletionClient)
        try:
            agents = [
                AssistantAgent(
                    "Market_Agent",
                    model_client=model_client,
                    system_message=(
                        "你是 Market Agent。只讨论板块强度、动量、宽度和市场状态。"
                        "逐板块指出支持/谨慎/反对，必须引用输入里的分数字段。"
                    ),
                ),
                AssistantAgent(
                    "Event_Agent",
                    model_client=model_client,
                    system_message=(
                        "你是 Event Agent。只讨论新闻、事件催化和证伪条件。"
                        "不能编造新闻；缺少事件时必须明确降级理由。"
                    ),
                ),
                AssistantAgent(
                    "Analysis_Agent",
                    model_client=model_client,
                    system_message=(
                        "你是 Analysis Agent。只讨论 ETF 候选、首选 ETF、备选 ETF、"
                        "匹配度、流动性和跟踪纯度。每个板块必须落到首选 ETF。"
                    ),
                ),
                AssistantAgent(
                    "Risk_Agent",
                    model_client=model_client,
                    system_message=(
                        "你是 Risk Agent。只讨论流动性、停牌/涨跌停、映射不确定、组合上限和人工审批。"
                        "不要使用回测，不要给出自动交易许可。"
                    ),
                ),
                AssistantAgent(
                    "System_Moderator",
                    model_client=model_client,
                    system_message=(
                        "你是 System Moderator。你要综合四个 Agent 的发言，"
                        "给出最终保留的 Top ETF 板块排序、每个板块的首选 ETF、主要理由和反对意见。"
                        "你最后必须输出一个 JSON 对象，字段为 final_decisions 和 excluded_by_roundtable，"
                        "final_decisions 每项必须包含 sector/status/primary_etf_code/support_reasons/objections/confidence。"
                        "JSON 之后结尾必须写 TERMINATE。"
                    ),
                ),
            ]
            team = RoundRobinGroupChat(agents, max_turns=self.max_turns)
            result = await team.run(task=self._build_task(
                trade_date=trade_date,
                candidates=candidates,
                max_final_decisions=max_final_decisions,
            ))
            messages = self._extract_messages(result)
            return self._to_result(messages, candidates=candidates)
        finally:
            await model_client.close()

    def _create_model_client(self, client_cls: type) -> Any:
        if self.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is not set")
            return client_cls(
                model=self.model,
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": False,
                    "family": "unknown",
                    "structured_output": False,
                },
            )
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            return client_cls(model=self.model, api_key=api_key)

        api_key = os.environ.get(f"{self.provider.upper()}_API_KEY")
        base_url = os.environ.get(f"{self.provider.upper()}_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError(f"{self.provider} API key/base URL is not configured")
        return client_cls(
            model=self.model,
            api_key=api_key,
            base_url=base_url,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            },
        )

    @staticmethod
    def _build_task(
        *,
        trade_date: str,
        candidates: list[SectorCandidatePayload],
        max_final_decisions: int,
    ) -> str:
        payload = []
        for candidate in candidates:
            payload.append({
                "sector_name": candidate.sector_name,
                "pre_score": candidate.pre_score,
                "momentum_score": candidate.momentum_score,
                "breadth_score": candidate.breadth_score,
                "event_score": candidate.event_score,
                "support_evidence": candidate.support_evidence[:6],
                "risk_flags": candidate.risk_flags,
                "etf_candidates": [
                    {
                        "code": etf.code,
                        "name": etf.name,
                        "match_score": etf.match_score,
                        "liquidity_score": etf.liquidity_score,
                        "tracking_purity_score": etf.tracking_purity_score,
                        "total_score": etf.total_score,
                        "reason": etf.reason,
                        "blocked_reasons": etf.blocked_reasons,
                    }
                    for etf in candidate.raw_etf_candidates[:3]
                ],
            })
        return (
            f"请召开 A 股板块 ETF 批量圆桌会议。交易日: {trade_date}\n"
            f"最终最多保留 {max_final_decisions} 个板块，每个板块必须有首选 ETF。\n"
            "不要使用回测，不要允许自动执行交易。请逐个 Agent 发言，最后 Moderator 总结。\n"
            "Moderator 最后必须输出严格 JSON，格式如下:\n"
            "{\n"
            '  "final_decisions": [\n'
            "    {\n"
            '      "sector": "板块名",\n'
            '      "status": "active|monitor",\n'
            '      "primary_etf_code": "ETF代码",\n'
            '      "support_reasons": ["理由1", "理由2"],\n'
            '      "objections": ["反对意见或风险"],\n'
            '      "confidence": "high|medium|low"\n'
            "    }\n"
            "  ],\n"
            '  "excluded_by_roundtable": [\n'
            '    {"sector": "板块名", "reason": "圆桌否决理由"}\n'
            "  ]\n"
            "}\n"
            "候选 JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _extract_messages(result: Any) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in getattr(result, "messages", []):
            source = getattr(message, "source", "") or getattr(message, "name", "")
            content = getattr(message, "content", "")
            if not source and not content:
                continue
            messages.append({
                "source": str(source),
                "type": type(message).__name__,
                "content": ETFWatchlistAutoGenRoundtable._stringify_content(content),
            })
        return messages

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(getattr(item, "content", item)) for item in content)
        return str(content)

    @staticmethod
    def _to_result(
        messages: list[dict[str, Any]],
        *,
        candidates: list[SectorCandidatePayload],
    ) -> ETFWatchlistAutoGenResult:
        dialogue: list[RoundtableDialogueTurn] = []
        agent_outputs: list[RoundtableAgentOutput] = []
        candidate_names = [candidate.sector_name for candidate in candidates]
        for idx, message in enumerate(messages, start=1):
            speaker = _AGENT_SPEAKERS.get(str(message.get("source", "")))
            if speaker is None:
                continue
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            dialogue.append(
                RoundtableDialogueTurn(
                    round=1,
                    sector="batch",
                    speaker=speaker,  # type: ignore[arg-type]
                    message=content[:2000],
                    references=candidate_names,
                )
            )
            if speaker != "Moderator":
                agent_outputs.append(
                    RoundtableAgentOutput(
                        agent=speaker,  # type: ignore[arg-type]
                        sector="batch",
                        stance=ETFWatchlistAutoGenRoundtable._infer_stance(content),
                        summary=content[:500],
                        evidence=candidate_names,
                        objections=ETFWatchlistAutoGenRoundtable._extract_objections(content),
                    )
                )
        summary = ""
        for message in reversed(messages):
            if message.get("source") == "System_Moderator":
                summary = str(message.get("content", "")).strip()
                break
        if not summary:
            summary = "\n".join(turn.message for turn in dialogue[-3:])
        structured = ETFWatchlistAutoGenRoundtable._extract_structured_summary(summary)
        return ETFWatchlistAutoGenResult(
            summary=summary[:3000],
            agent_outputs=agent_outputs,
            dialogue_records=dialogue,
            round_history=[{
                "round": 1,
                "sector": "batch",
                "turn_count": len(dialogue),
                "turns": [turn.model_dump(mode="json") for turn in dialogue],
            }],
            raw_messages=messages,
            final_decisions=structured.get("final_decisions", []),
            excluded_by_roundtable=structured.get("excluded_by_roundtable", []),
        )

    @staticmethod
    def _extract_structured_summary(content: str) -> dict[str, list[dict[str, Any]]]:
        """Extract the Moderator's final JSON object from free-form text."""
        for candidate in ETFWatchlistAutoGenRoundtable._json_object_candidates(content):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            final_decisions = parsed.get("final_decisions")
            excluded = parsed.get("excluded_by_roundtable")
            if isinstance(final_decisions, list):
                return {
                    "final_decisions": [item for item in final_decisions if isinstance(item, dict)],
                    "excluded_by_roundtable": [item for item in excluded if isinstance(item, dict)] if isinstance(excluded, list) else [],
                }
        return {"final_decisions": [], "excluded_by_roundtable": []}

    @staticmethod
    def _json_object_candidates(content: str) -> list[str]:
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        candidates = list(fenced)
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(content[start : end + 1])
        return candidates

    @staticmethod
    def _infer_stance(content: str) -> str:
        if any(token in content for token in ("反对", "剔除", "不建议", "block", "veto")):
            return "block"
        if any(token in content for token in ("谨慎", "观察", "风险", "caution", "monitor")):
            return "caution"
        return "support"

    @staticmethod
    def _extract_objections(content: str) -> list[str]:
        objections = []
        for line in content.splitlines():
            if any(token in line for token in ("风险", "反对", "不足", "剔除", "不建议")):
                objections.append(line.strip("- 	")[:300])
            if len(objections) >= 5:
                break
        return objections
