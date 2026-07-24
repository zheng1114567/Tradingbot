"""Central LLM provider registry.

The rest of the project should not need to know provider-specific key names or
OpenAI-compatible endpoint URLs.  Keep those details behind this small seam.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


QWEN_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    api_key_envs: tuple[str, ...]
    default_model: str
    base_url: str | None = None
    base_url_envs: tuple[str, ...] = ()
    openai_compatible: bool = True
    pricing_hint: str = ""


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "qwen": ProviderSpec(
        provider="qwen",
        api_key_envs=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        base_url=QWEN_COMPATIBLE_BASE_URL,
        base_url_envs=("QWEN_BASE_URL", "DASHSCOPE_BASE_URL"),
        default_model="qwen3.6-flash",
        pricing_hint="Qwen 3.6 Flash: lowest-cost Qwen 3.6 text model tier.",
    ),
    "deepseek": ProviderSpec(
        provider="deepseek",
        api_key_envs=("DEEPSEEK_API_KEY",),
        base_url="https://api.deepseek.com/v1",
        base_url_envs=("DEEPSEEK_BASE_URL",),
        default_model="deepseek-chat",
    ),
    "openai": ProviderSpec(
        provider="openai",
        api_key_envs=("OPENAI_API_KEY",),
        base_url=None,
        base_url_envs=("OPENAI_BASE_URL",),
        default_model="gpt-4o-mini",
    ),
    "kimi": ProviderSpec(
        provider="kimi",
        api_key_envs=("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
        base_url_envs=("KIMI_BASE_URL", "MOONSHOT_BASE_URL"),
        default_model="moonshot-v1-8k",
    ),
    "glm": ProviderSpec(
        provider="glm",
        api_key_envs=("GLM_API_KEY", "ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
        base_url_envs=("GLM_BASE_URL", "ZHIPU_BASE_URL"),
        default_model="glm-4-flash",
    ),
    "anthropic": ProviderSpec(
        provider="anthropic",
        api_key_envs=("ANTHROPIC_API_KEY",),
        default_model="claude-3-5-haiku-latest",
        openai_compatible=False,
    ),
}


def valid_providers() -> frozenset[str]:
    """Return provider names accepted by the first-class config surface."""
    return frozenset(PROVIDER_SPECS)


def provider_spec(provider: str) -> ProviderSpec:
    """Return a first-class provider spec or a generic OpenAI-compatible one."""
    provider_key = str(provider or "").lower()
    if provider_key in PROVIDER_SPECS:
        return PROVIDER_SPECS[provider_key]
    prefix = provider_key.upper()
    return ProviderSpec(
        provider=provider_key,
        api_key_envs=(f"{prefix}_API_KEY",),
        base_url=os.environ.get(f"{prefix}_BASE_URL", ""),
        base_url_envs=(f"{prefix}_BASE_URL",),
        default_model=f"{provider_key}-model",
    )


def first_env_value(names: tuple[str, ...]) -> tuple[str | None, str]:
    """Return the first configured environment value plus its display label."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, " or ".join(names)


def resolve_base_url(spec: ProviderSpec) -> str | None:
    """Resolve provider base URL, honoring provider-specific env overrides."""
    for name in spec.base_url_envs:
        value = os.environ.get(name)
        if value:
            return value
    return spec.base_url
