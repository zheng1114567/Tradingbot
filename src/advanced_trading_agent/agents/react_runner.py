"""LangChain/LangGraph ReAct bridge for project agents."""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


def run_prebuilt_react(
    *,
    llm: Any,
    tools: list[Any],
    prompt: str,
    user_content: str,
    response_format: type,
    recursion_limit: int = 8,
) -> tuple[Any | None, list[dict[str, Any]]]:
    """Run the official ReAct-style agent if the LLM supports LangChain."""
    if not hasattr(llm, "as_langchain_chat_model"):
        return None, []

    try:
        graph = create_agent(
            model=llm.as_langchain_chat_model(),
            tools=tools,
            system_prompt=prompt,
            response_format=response_format,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_content)]},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as e:
        logger.warning("ReAct agent failed, falling back: %s", e)
        return None, []

    trace = _extract_trace(result.get("messages", []))
    return result.get("structured_response"), trace


def _extract_trace(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            trace.append({
                "tool": call.get("name", ""),
                "args": call.get("args", {}),
                "id": call.get("id", ""),
            })
        if getattr(message, "type", "") == "tool":
            trace.append({
                "tool": getattr(message, "name", ""),
                "observation": str(getattr(message, "content", ""))[:1000],
                "id": getattr(message, "tool_call_id", ""),
            })
    return trace
