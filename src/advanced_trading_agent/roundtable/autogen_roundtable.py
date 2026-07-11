"""AutoGen-backed Round 2 roundtable.

The public interface is intentionally small so the workflow can fall back to
the deterministic in-process roundtable if AutoGen or the model endpoint fails.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from ..config import config
from .harness import RoundtableHarness

logger = logging.getLogger(__name__)


@dataclass
class RoundtableResult:
    """Structured result returned to the LangGraph workflow."""

    questions: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    unresolved_conflicts: list[str] = field(default_factory=list)
    final_pressure: str = "neutral"
    provider: str = "autogen"
    fallback_reason: str = ""


class AutoGenRoundtable:
    """Run a fixed-order AutoGen roundtable for Round 2 conflicts."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        harness: RoundtableHarness | None = None,
    ):
        cfg = config.get_all()
        self.provider = provider or cfg.get("llm_provider", "deepseek")
        self.model = model or cfg.get("deep_think_llm", "deepseek-chat")
        self.harness = harness or RoundtableHarness()

    def run(self, state: dict[str, Any]) -> RoundtableResult:
        """Run AutoGen synchronously from the workflow node."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run_async(state))
        raise RuntimeError("AutoGen roundtable cannot run inside an active event loop")

    async def _run_async(self, state: dict[str, Any]) -> RoundtableResult:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        round2 = state.get("round2_state", {})
        contradictions = list(round2.get("contradictions", []))
        if not contradictions:
            return RoundtableResult(summary="", unresolved_conflicts=[])

        context = self.harness.build_context(state, contradictions)
        model_client = self._create_model_client(OpenAIChatCompletionClient)
        agent_contexts = context.agent_contexts
        agents = [
            AssistantAgent(
                "Market_Agent",
                model_client=model_client,
                system_message=agent_contexts["Market"].system_message,
            ),
            AssistantAgent(
                "Event_Agent",
                model_client=model_client,
                system_message=agent_contexts["Event"].system_message,
            ),
            AssistantAgent(
                "Analysis_Agent",
                model_client=model_client,
                system_message=agent_contexts["Analysis"].system_message,
            ),
            AssistantAgent(
                "Backtest_Agent",
                model_client=model_client,
                system_message=agent_contexts["Backtest"].system_message,
            ),
            AssistantAgent(
                "System_Moderator",
                model_client=model_client,
                system_message=(
                    "你是 System Moderator。你只能基于 DATA_AGENT_BRIEF、各 Agent 发言和"
                    "Round 2 矛盾点裁定。最后必须输出 unresolved_conflicts、final_pressure"
                    "(upgrade/neutral/downgrade) 和风控关注点。回复结尾写 TERMINATE。"
                ),
            ),
        ]
        team = RoundRobinGroupChat(agents, max_turns=len(agents))
        task = context.task
        result = await team.run(task=task)
        messages = self._extract_messages(result)
        await model_client.close()
        return self._to_result(contradictions, messages, context=context)

    def _create_model_client(self, client_cls):
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
                    "function_calling": False,
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
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            },
        )

    @staticmethod
    def _agent_report(state: dict[str, Any], key: str) -> str:
        report = str(state.get(key, "")).strip()
        return report[:1200] if report else "暂无该 Agent 报告"

    @staticmethod
    def _build_task(state: dict[str, Any], contradictions: list[str]) -> str:
        return RoundtableHarness().build_context(state, contradictions).task

    @staticmethod
    def _extract_messages(result: Any) -> list[dict[str, str]]:
        messages = []
        for message in getattr(result, "messages", []):
            source = getattr(message, "source", "") or getattr(message, "name", "")
            content = getattr(message, "content", "")
            if content:
                messages.append({"source": str(source), "content": str(content)})
        return messages

    @staticmethod
    def _to_result(
        contradictions: list[str],
        messages: list[dict[str, str]],
        *,
        context: Any | None = None,
    ) -> RoundtableResult:
        evidence_by_source = {}
        if context is not None:
            evidence_by_source = {
                "Market_Agent": context.agent_contexts["Market"].evidence_text,
                "Event_Agent": context.agent_contexts["Event"].evidence_text,
                "Analysis_Agent": context.agent_contexts["Analysis"].evidence_text,
                "Backtest_Agent": context.agent_contexts["Backtest"].evidence_text,
            }
        questions = [{
            "source_agent": "System",
            "target_agent": ",".join(m["source"] for m in messages if m["source"]),
            "question": "请围绕 Round 2 矛盾点进行圆桌发言。",
            "answer": "\n".join(f"{m['source']}: {m['content']}" for m in messages),
            "answers": [
                {
                    "target_agent": m["source"],
                    "answer": m["content"],
                    "evidence": evidence_by_source.get(m["source"], ""),
                }
                for m in messages
            ],
            "data_source": "; ".join(contradictions),
        }]
        summary = "\n".join(f"{m['source']}: {m['content']}" for m in messages[-5:])
        lowered = summary.lower()
        if "downgrade" in lowered or "降级" in summary or "拒绝" in summary:
            pressure = "downgrade"
        elif "upgrade" in lowered or "推荐" in summary:
            pressure = "upgrade"
        else:
            pressure = "neutral"
        return RoundtableResult(
            questions=questions,
            summary=summary,
            unresolved_conflicts=contradictions,
            final_pressure=pressure,
            provider="autogen",
        )
