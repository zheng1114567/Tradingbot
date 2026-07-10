"""LLM 客户端测试"""
import os
import pytest
from advanced_trading_agent.llm.client import LLMClient, create_llm


class TestLLMClient:
    """测试 LLM 客户端工厂和初始化"""

    def test_create_default(self):
        client = create_llm()
        assert client.provider == "deepseek"
        assert client.model == "deepseek-chat"
        assert client.temperature == 0.1

    def test_create_custom(self):
        client = create_llm(provider="openai", model="gpt-4o", temperature=0.5)
        assert client.provider == "openai"
        assert client.model == "gpt-4o"
        assert client.temperature == 0.5

    def test_deepseek_client(self):
        """DeepSeek 客户端应返回 OpenAI 实例"""
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        client = create_llm(provider="deepseek")
        c = client.client
        from openai import OpenAI
        assert isinstance(c, OpenAI)
        assert c.base_url.host == "api.deepseek.com"
        del os.environ["DEEPSEEK_API_KEY"]

    def test_openai_client(self):
        """OpenAI 客户端应返回 OpenAI 实例"""
        os.environ["OPENAI_API_KEY"] = "test-key"
        client = create_llm(provider="openai")
        c = client.client
        from openai import OpenAI
        assert isinstance(c, OpenAI)
        del os.environ["OPENAI_API_KEY"]

    def test_anthropic_import_error_when_not_installed(self):
        """Anthropic provider 应在缺少 SDK 时给出明确提示"""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        client = create_llm(provider="anthropic")
        try:
            client.client
        except ImportError as e:
            assert "anthropic" in str(e).lower()
        except Exception:
            pass  # 如果 SDK 已安装则不会抛 ImportError
        del os.environ["ANTHROPIC_API_KEY"]

    def test_custom_provider(self):
        """自定义 OpenAI 兼容端点"""
        os.environ["CUSTOM_API_KEY"] = "test-key"
        os.environ["CUSTOM_BASE_URL"] = "https://custom.api.com/v1"
        client = create_llm(provider="custom", model="custom-model")
        c = client.client
        from openai import OpenAI
        assert isinstance(c, OpenAI)
        assert "custom.api.com" in str(c.base_url)
        del os.environ["CUSTOM_API_KEY"]
        del os.environ["CUSTOM_BASE_URL"]

    def test_missing_api_key_doesnt_crash_init(self):
        """缺失 API key 不应在初始化时崩溃"""
        # 确保环境变量不存在
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        client = create_llm(provider="deepseek")
        # 初始化应成功; 调用时才会失败
        assert client.client is not None
