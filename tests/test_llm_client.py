"""LLM 客户端测试"""
import os
import pytest
from advanced_trading_agent.llm.client import LLMClient, create_llm
from advanced_trading_agent.agents.schemas import MarketReport


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

    def test_parse_structured_accepts_markdown_json_block(self):
        payload = """```json
{"market_state":"正常","position_cap":0.6,"capital_confirmation":"资金确认","sector_preference":[],"risk_warning":null,"reasoning":"ok"}
```"""

        parsed = LLMClient._parse_structured(payload, MarketReport)

        assert parsed.market_state == "正常"

    def test_openai_structured_fallback_uses_json_prompt(self):
        captured = {}

        class Message:
            content = (
                '{"market_state":"正常","position_cap":0.6,'
                '"capital_confirmation":"资金确认","sector_preference":[],'
                '"risk_warning":null,"reasoning":"fallback"}'
            )

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return Response()

        class Chat:
            completions = Completions()

        class FakeClient:
            chat = Chat()

        client = create_llm(provider="openai")
        client._client = FakeClient()

        result = client._call_openai(
            {
                "model": "fake",
                "messages": [{"role": "user", "content": "x"}],
                "temperature": 0.1,
                "max_tokens": 100,
            },
            response_format=MarketReport,
            use_native_response_format=False,
        )

        assert result.reasoning == "fallback"
        assert "response_format" not in captured
        assert "valid JSON only" in captured["messages"][-1]["content"]
