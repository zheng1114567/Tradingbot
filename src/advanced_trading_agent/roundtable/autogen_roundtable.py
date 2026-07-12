"""AutoGen-backed Round 2 roundtable.

The workflow uses this adapter as the primary Round 2 executor. If AutoGen or
the model endpoint fails, the workflow falls back to the local DebateEngine.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import config
from ..tool_nodes.registry import get_agent_tools
from .harness import RoundtableHarness

logger = logging.getLogger(__name__)

_AGENT_ORDER = ("Market", "Event", "Analysis", "Backtest")
_AGENT_TO_KEY = {
    "Market": "market",
    "Event": "event",
    "Analysis": "analysis",
    "Backtest": "backtest",
}


@dataclass
class RoundtableResult:
    """Structured result returned to the LangGraph workflow."""

    questions: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    unresolved_conflicts: list[str] = field(default_factory=list)
    final_pressure: str = "neutral"
    provider: str = "autogen"
    fallback_reason: str = ""
    evidence_board: list[dict[str, Any]] = field(default_factory=list)
    round_history: list[dict[str, Any]] = field(default_factory=list)
    moderator_output: dict[str, Any] | None = None
    round_count: int = 0
    tool_calls_by_agent: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


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
        from autogen_core.tools import FunctionTool
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        round2 = state.get("round2_state", {})
        contradiction_records = list(round2.get("contradiction_records", []))
        contradictions = [
            c.get("description", str(c)) if isinstance(c, dict) else str(c)
            for c in contradiction_records
        ]
        if not contradictions:
            return RoundtableResult(summary="", unresolved_conflicts=[])

        context = self.harness.build_context(state, contradictions)
        model_client = self._create_model_client(OpenAIChatCompletionClient)
        try:
            agents = []
            for agent_name in _AGENT_ORDER:
                agent_context = context.agent_contexts[agent_name]
                tools = self._build_autogen_tools(
                    agent_key=_AGENT_TO_KEY[agent_name],
                    function_tool_cls=FunctionTool,
                )
                agents.append(
                    AssistantAgent(
                        f"{agent_name}_Agent",
                        model_client=model_client,
                        system_message=agent_context.system_message,
                        tools=tools,
                    )
                )
            agents.append(
                AssistantAgent(
                    "System_Moderator",
                    model_client=model_client,
                    system_message=(
                        "你是 System Moderator。你只能基于 DATA_AGENT_BRIEF、各 Agent 发言、"
                        "已注册工具返回结果和 Round 2 矛盾点裁定。"
                        "最后必须明确 unresolved_conflicts、final_pressure"
                        "(upgrade/neutral/downgrade) 和风控关注点。回复结尾写 TERMINATE。"
                    ),
                )
            )
            team = RoundRobinGroupChat(agents, max_turns=len(agents))
            result = await team.run(task=context.task)
            messages = self._extract_messages(result)
            return self._to_result(contradictions, messages, context=context)
        finally:
            await model_client.close()

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
    def _build_autogen_tools(
        *,
        agent_key: str,
        function_tool_cls: type,
    ) -> list[Any]:
        tools = []
        for tool in get_agent_tools(agent_key):
            func = getattr(tool, "func", None)
            name = getattr(tool, "name", None) or getattr(func, "__name__", "tool")
            description = getattr(tool, "description", None) or getattr(func, "__doc__", "") or name
            if not callable(func):
                func = AutoGenRoundtable._wrap_langchain_tool(tool, name=name, description=description)
            tools.append(
                function_tool_cls(
                    func,
                    description=description,
                    name=name,
                    strict=False,
                )
            )
        return tools

    @staticmethod
    def _wrap_langchain_tool(tool: Any, *, name: str, description: str) -> Callable[..., Any]:
        def invoke_tool(**kwargs: Any) -> Any:
            return tool.invoke(kwargs)

        invoke_tool.__name__ = name
        invoke_tool.__doc__ = description
        return invoke_tool

    @staticmethod
    def _extract_messages(result: Any) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in getattr(result, "messages", []):
            source = getattr(message, "source", "") or getattr(message, "name", "")
            message_type = type(message).__name__
            content = getattr(message, "content", "")
            if not source and not content:
                continue
            messages.append(
                {
                    "source": str(source),
                    "type": message_type,
                    "content": AutoGenRoundtable._stringify_content(content),
                }
            )
        return messages

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(str(item))
                else:
                    chunks.append(str(getattr(item, "content", item)))
            return "\n".join(chunks)
        return str(content)

    @staticmethod
    def _collect_tool_calls(messages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        tool_calls: dict[str, list[dict[str, Any]]] = {}
        for message in messages:
            message_type = str(message.get("type", ""))
            if "ToolCall" not in message_type and "Tool" not in message_type:
                continue
            source = str(message.get("source", "") or "unknown")
            tool_calls.setdefault(source, []).append(
                {
                    "tool": source if "tool" in source.lower() else "",
                    "event_type": message_type,
                    "content": str(message.get("content", ""))[:1000],
                }
            )
        return tool_calls

    @staticmethod
    def _to_result(
        contradictions: list[str],
        messages: list[dict[str, Any]],
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

        speaker_messages = [
            m for m in messages
            if m.get("source") in {
                "Market_Agent", "Event_Agent", "Analysis_Agent", "Backtest_Agent", "System_Moderator"
            }
        ]
        questions = [{
            "source_agent": "System",
            "target_agent": ",".join(m["source"] for m in speaker_messages if m["source"]),
            "question": "请围绕 Round 2 矛盾点进行圆桌发言。",
            "answer": "\n".join(f"{m['source']}: {m['content']}" for m in speaker_messages),
            "answers": [
                {
                    "target_agent": m["source"],
                    "answer": m["content"],
                    "evidence": evidence_by_source.get(m["source"], ""),
                }
                for m in speaker_messages
            ],
            "data_source": "; ".join(contradictions),
        }]

        summary = "\n".join(f"{m['source']}: {m['content']}" for m in speaker_messages[-5:])
        lowered = summary.lower()
        if "downgrade" in lowered or "降级" in summary or "拒绝" in summary:
            pressure = "downgrade"
        elif "upgrade" in lowered or "推荐" in summary:
            pressure = "upgrade"
        else:
            pressure = "neutral"

        moderator_message = next(
            (m for m in reversed(speaker_messages) if m.get("source") == "System_Moderator"),
            None,
        )
        moderator_output = None
        if moderator_message is not None:
            moderator_output = {
                "round_number": 0,
                "final_pressure": pressure,
                "unresolved_contradiction_ids": [],
                "consensus_items": [],
                "dissent_items": contradictions,
                "converged": True,
                "reasoning": moderator_message.get("content", "")[:2000],
                "risk_focus": [],
            }

        return RoundtableResult(
            questions=questions,
            summary=summary,
            unresolved_conflicts=contradictions,
            final_pressure=pressure,
            provider="autogen",
            evidence_board=[
                item.model_dump(mode="json") for item in getattr(context, "evidence_board", [])
            ] if context is not None else [],
            round_history=[{
                "round_number": 0,
                "provider": "autogen",
                "messages": speaker_messages,
            }],
            moderator_output=moderator_output,
            round_count=1 if speaker_messages else 0,
            tool_calls_by_agent=AutoGenRoundtable._collect_tool_calls(messages),
        )
