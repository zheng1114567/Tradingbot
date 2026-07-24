"""
LLM 客户端 — 多供应商支持 (Qwen, DeepSeek, OpenAI, Anthropic)

借鉴 TradingAgents 的 llm_clients/*.py 模式,
但简化了配置, 默认使用 Qwen。
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai import BadRequestError
from openai import OpenAI

from ..config import config
from .provider_registry import first_env_value, provider_spec, resolve_base_url

logger = logging.getLogger(__name__)


_ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
}

@dataclass(frozen=True)
class OpenAICompatibleSettings:
    provider: str
    api_key: str | None
    api_key_env: str
    base_url: str | None = None


def _llm_disabled() -> bool:
    """Return True when callers explicitly request deterministic offline mode."""
    return os.environ.get("ATA_DISABLE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def _api_key_or_placeholder(value: str | None) -> str:
    """Allow lazy client construction; real auth still fails at call time."""
    return value or "missing-api-key"


def resolve_openai_compatible_settings(provider: str) -> OpenAICompatibleSettings:
    """Resolve API settings for providers served through the OpenAI SDK."""
    provider_key = str(provider or "").lower()
    spec = provider_spec(provider_key)
    api_key, api_key_env = first_env_value(spec.api_key_envs)
    return OpenAICompatibleSettings(
        provider=provider_key,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=resolve_base_url(spec),
    )


def openai_compatible_model_info() -> dict[str, Any]:
    """AutoGen model metadata for OpenAI-compatible non-OpenAI providers."""
    return {
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        "structured_output": False,
    }


def llm_api_key_configured(provider: str) -> bool:
    """Return whether the configured provider has an API key available."""
    provider_key = str(provider or "").lower()
    if provider_key == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return bool(resolve_openai_compatible_settings(provider_key).api_key)


class LLMClient:
    """LLM 客户端 — DeepSeek 优先, 支持降级"""

    def __init__(self, provider: str | None = None,
                 model: str | None = None,
                 temperature: float | None = None):
        cfg = config.get_all()
        configured_provider = str(cfg.get("llm_provider", "qwen")).lower()
        self.provider = str(provider or configured_provider).lower()
        spec = provider_spec(self.provider)
        configured_model = cfg.get("llm_model") or cfg.get("deep_think_llm")
        if model is not None:
            self.model = model
        elif provider is not None and self.provider != configured_provider:
            self.model = spec.default_model
        else:
            self.model = configured_model or spec.default_model
        self.temperature = temperature if temperature is not None else cfg.get("temperature", 0.1)
        self.timeout_seconds = float(os.environ.get("ATA_LLM_TIMEOUT_SECONDS", "60"))
        self._client = None

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI 兼容客户端"""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
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

        settings = resolve_openai_compatible_settings(self.provider)
        kwargs: dict[str, Any] = {
            "api_key": _api_key_or_placeholder(settings.api_key),
            "timeout": self.timeout_seconds,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = OpenAI(**kwargs)

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

    def _build_structured_messages(
        self,
        messages: list[dict[str, str]],
        response_format: type,
    ) -> list[dict[str, str]]:
        """Ask providers without native JSON schema support for raw JSON."""
        schema = response_format.model_json_schema()
        instruction = (
            "You must respond with valid JSON only. Do not wrap it in markdown. "
            "The JSON must conform to this schema: "
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        return [*messages, {"role": "system", "content": instruction}]

    def _call_openai(
        self,
        kwargs: dict[str, Any],
        response_format: type | None = None,
        *,
        use_native_response_format: bool = True,
    ) -> str:
        """调用 OpenAI 兼容 API"""
        if response_format is not None and use_native_response_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": response_format.model_json_schema(),
                },
            }
        elif response_format is not None:
            kwargs["messages"] = self._build_structured_messages(kwargs["messages"], response_format)

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
            text = content.strip()
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            parsed = json.loads(text)
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
        if _llm_disabled():
            raise RuntimeError("LLM calls disabled by ATA_DISABLE_LLM")

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
        except BadRequestError as e:
            message = str(e)
            if response_format is not None and "response_format" in message:
                logger.warning(
                    "Provider rejected native structured output; retrying JSON prompt mode: %s",
                    e,
                )
                fallback_kwargs = {
                    "model": self.model,
                    "messages": normalized_messages,
                    "temperature": temperature or self.temperature,
                    "max_tokens": max_tokens,
                }
                return self._call_openai(
                    fallback_kwargs,
                    response_format=response_format,
                    use_native_response_format=False,
                )
            logger.error("LLM call failed: %s", e)
            raise
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    def invoke(self, messages: Sequence[Any],
               response_format: type | None = None) -> str | Any:
        """便捷调用方法"""
        return self.chat(messages, response_format=response_format)

    def as_langchain_chat_model(self):
        """Return a LangChain chat model for LangGraph prebuilt agents."""
        if _llm_disabled():
            raise RuntimeError("LLM calls disabled by ATA_DISABLE_LLM")

        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )

        from langchain_openai import ChatOpenAI

        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=resolve_openai_compatible_settings("openai").api_key or "",
                timeout=self.timeout_seconds,
            )

        settings = resolve_openai_compatible_settings(self.provider)
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=settings.api_key or "",
            base_url=settings.base_url or "",
            timeout=self.timeout_seconds,
        )


def create_llm(provider: str | None = None,
               model: str | None = None,
               temperature: float | None = None) -> LLMClient:
    """工厂函数创建 LLM 客户端"""
    return LLMClient(provider=provider, model=model, temperature=temperature)
