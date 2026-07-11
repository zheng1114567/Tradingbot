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

    def __init__(self, model: str | None = None, provider: str | None = None):
        cfg = config.get_all()
        self.provider = provider or cfg.get("llm_provider", "deepseek")
        self.model = model or cfg.get("deep_think_llm", "deepseek-chat")

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

        model_client = self._create_model_client(OpenAIChatCompletionClient)
        agents = [
            AssistantAgent(
                "Market_Agent",
                model_client=model_client,
                system_message=(
                    "你是 Market Agent。只基于给定 Market 报告讨论市场温度、资金状态、仓位约束。"
                    "不要编造外部数据。\n\n"
                    f"Market 报告:\n{self._agent_report(state, 'market_report')}"
                ),
            ),
            AssistantAgent(
                "Event_Agent",
                model_client=model_client,
                system_message=(
                    "你是 Event Agent。只基于给定 Event 报告讨论事件传导、证据等级和定价状态。"
                    "不要编造外部数据。\n\n"
                    f"Event 报告:\n{self._agent_report(state, 'event_report')}"
                ),
            ),
            AssistantAgent(
                "Analysis_Agent",
                model_client=model_client,
                system_message=(
                    "你是 Analysis Agent。只基于给定 Analysis 报告讨论因子排序、拥挤和择时。"
                    "不要编造外部数据。\n\n"
                    f"Analysis 报告:\n{self._agent_report(state, 'analysis_report')}"
                ),
            ),
            AssistantAgent(
                "Backtest_Agent",
                model_client=model_client,
                system_message=(
                    "你是 Backtest Agent。只基于给定 Backtest 报告讨论样本量、胜率和统计可靠性。"
                    "不要编造外部数据。\n\n"
                    f"Backtest 报告:\n{self._agent_report(state, 'backtest_report')}"
                ),
            ),
            AssistantAgent(
                "System_Moderator",
                model_client=model_client,
                system_message=(
                    "你是 System Moderator。最后发言必须总结未解决分歧、裁定压力"
                    "(upgrade/neutral/downgrade)和风控关注点。回复结尾写 TERMINATE。"
                ),
            ),
        ]
        team = RoundRobinGroupChat(agents, max_turns=len(agents))
        task = self._build_task(state, contradictions)
        result = await team.run(task=task)
        messages = self._extract_messages(result)
        await model_client.close()
        return self._to_result(contradictions, messages)

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
        return f"""请进行 Round 2 圆桌会议。

矛盾点:
{chr(10).join(f"- {c}" for c in contradictions)}

规则:
1. 每个 Agent 只能基于自身 system message 中的专属报告发言。
2. 必须回应矛盾点，不允许补造数据。
3. Moderator 最后输出未解决分歧、final_pressure 和风控关注点。
"""

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
    def _to_result(contradictions: list[str], messages: list[dict[str, str]]) -> RoundtableResult:
        questions = [{
            "source_agent": "System",
            "target_agent": ",".join(m["source"] for m in messages if m["source"]),
            "question": "请围绕 Round 2 矛盾点进行圆桌发言。",
            "answer": "\n".join(f"{m['source']}: {m['content']}" for m in messages),
            "answers": [
                {
                    "target_agent": m["source"],
                    "answer": m["content"],
                    "evidence": "",
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
