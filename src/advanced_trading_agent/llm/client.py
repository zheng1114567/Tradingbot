"""
LLM 客户端 — 多供应商支持 (DeepSeek, OpenAI, Anthropic)

借鉴 TradingAgents 的 llm_clients/*.py 模式,
但简化了配置, 默认使用 DeepSeek。
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from ..config import config

logger = logging.getLogger(__name__)


_ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
}


class LLMClient:
    """LLM 客户端 — DeepSeek 优先, 支持降级"""

    def __init__(self, provider: str | None = None,
                 model: str | None = None,
                 temperature: float | None = None):
        cfg = config.get_all()
        self.provider = provider or cfg.get("llm_provider", "deepseek")
        self.model = model or cfg.get("deep_think_llm", "deepseek-chat")
        self.temperature = temperature if temperature is not None else cfg.get("temperature", 0.1)
        self._client = None

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI 兼容客户端"""
        if self._client is not None:
            return self._client

        if self.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
            )
        elif self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            self._client = OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            # Anthropic 使用独立的 anthropic SDK (非 OpenAI 兼容)
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic provider requires `pip install anthropic`"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            self._client = Anthropic(api_key=api_key)
            return self._client
        else:
            # 自定义 OpenAI 兼容端点
            api_key = os.environ.get(f"{self.provider.upper()}_API_KEY", "")
            base_url = os.environ.get(f"{self.provider.upper()}_BASE_URL", "")
            self._client = OpenAI(api_key=api_key, base_url=base_url)

        return self._client

    @staticmethod
    def _normalize_messages(messages: Sequence[Any]) -> list[dict[str, str]]:
        """Accept LangChain-style tuples and return provider-safe chat messages."""
        normalized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "user"))
                content = message.get("content", "")
            elif isinstance(message, tuple) and len(message) == 2:
                role, content = message
            else:
                raise TypeError(f"Unsupported message format: {type(message).__name__}")

            role = _ROLE_ALIASES.get(str(role), str(role))
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"Unsupported message role: {role}")
            normalized.append({"role": role, "content": str(content)})
        return normalized

    def _call_openai(self, kwargs: dict[str, Any],
                      response_format: type | None = None) -> str:
        """调用 OpenAI 兼容 API"""
        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": response_format.model_json_schema(),
                },
            }

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

        if response_format is not None:
            return self._parse_structured(content, response_format)

        return content

    def _call_anthropic(self, kwargs: dict[str, Any],
                         response_format: type | None = None) -> str:
        """调用 Anthropic API"""
        import json

        # 转换消息格式: OpenAI → Anthropic
        system_msg = None
        anthropic_messages = []
        for m in kwargs["messages"]:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                anthropic_messages.append({
                    "role": "user" if m["role"] == "user" else "assistant",
                    "content": m["content"],
                })

        if response_format is not None:
            system_msg = (
                f"{system_msg}\n\nYou must respond with valid JSON conforming to "
                f"this schema: {json.dumps(response_format.model_json_schema(), ensure_ascii=False)}"
            )

        params = {
            "model": kwargs["model"],
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": anthropic_messages,
        }
        if system_msg:
            params["system"] = system_msg

        response = self.client.messages.create(**params)
        content = response.content[0].text if response.content else ""

        if response_format is not None:
            return self._parse_structured(content, response_format)

        return content

    @staticmethod
    def _parse_structured(content: str, response_format: type) -> Any:
        """解析 JSON 到 Pydantic"""
        try:
            import json
            parsed = json.loads(content)
            return response_format.model_validate(parsed)
        except Exception as e:
            raise ValueError(f"Failed to parse structured output: {e}") from e

    def chat(self, messages: Sequence[Any],
             response_format: type | None = None,
             temperature: float | None = None,
             max_tokens: int = 4096) -> Any:
        """调用 LLM 聊天

        Args:
            messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            response_format: Pydantic 模型 (用 structured output)
            temperature: 采样温度
            max_tokens: 最大 token 数

        Returns:
            如果指定了 response_format, 返回 Pydantic 实例;
            否则返回 str.
        """
        normalized_messages = self._normalize_messages(messages)
        kwargs = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens,
        }

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(kwargs, response_format=response_format)
            return self._call_openai(kwargs, response_format=response_format)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    def invoke(self, messages: Sequence[Any],
               response_format: type | None = None) -> str | Any:
        """便捷调用方法"""
        return self.chat(messages, response_format=response_format)


def create_llm(provider: str | None = None,
               model: str | None = None,
               temperature: float | None = None) -> LLMClient:
    """工厂函数创建 LLM 客户端"""
    return LLMClient(provider=provider, model=model, temperature=temperature)
