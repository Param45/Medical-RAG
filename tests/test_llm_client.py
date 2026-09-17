"""
Tests for LLM Client Wrapper (SRS §5.6, BUILD_GUIDE Task 2.1).

All tests use mocks to avoid spending live API credits or loading large models.
"""

from unittest.mock import MagicMock, patch
import pytest

import llm_client
from normalize import llm_normalize, NormalizedEntity


class TestLLMClientLocal:
    """Tests for Local LLM (MedGemma / Llama-cpp) provider dispatch."""

    @patch("llm_client._call_local")
    def test_local_dispatch_basic(self, mock_local, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.setenv("LLM_MODEL", "medgemma-1.5-4b-it-Q4_K_M.gguf")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        mock_local.return_value = "Hello from Local MedGemma"

        messages = [{"role": "user", "content": "Hello!"}]
        response = llm_client.chat(messages, system="You are a medical assistant.")

        assert response == "Hello from Local MedGemma"
        mock_local.assert_called_once_with(
            model="medgemma-1.5-4b-it-Q4_K_M.gguf",
            messages=messages,
            system="You are a medical assistant.",
        )

    def test_local_call_sdk_mock(self):
        mock_llama_instance = MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Local inference reply"}}]
        }

        with patch("llm_client._get_local_llama_client", return_value=mock_llama_instance):
            messages = [{"role": "user", "content": "Test prompt"}]
            result = llm_client._call_local(
                model="medgemma-1.5-4b-it-Q4_K_M.gguf",
                messages=messages,
                system="System prompt",
            )
            assert result == "Local inference reply"
            mock_llama_instance.create_chat_completion.assert_called_once()
            call_args = mock_llama_instance.create_chat_completion.call_args[1]
            assert call_args["messages"][0] == {"role": "system", "content": "System prompt"}
            assert call_args["messages"][1] == {"role": "user", "content": "Test prompt"}


class TestLLMClientGemini:
    """Tests for Gemini provider dispatch."""

    @patch("llm_client._call_gemini")
    def test_gemini_dispatch_basic(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_API_KEY", "test_key_123")
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")

        mock_gemini.return_value = "Hello from Gemini"

        messages = [{"role": "user", "content": "Hello!"}]
        response = llm_client.chat(messages, system="You are a helpful assistant.")

        assert response == "Hello from Gemini"
        mock_gemini.assert_called_once_with(
            api_key="test_key_123",
            model="gemini-2.5-flash",
            messages=messages,
            system="You are a helpful assistant.",
        )

    def test_gemini_call_sdk_integration(self):
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Mocked Gemini SDK response"
        mock_client.models.generate_content.return_value = mock_response

        class DummyContent:
            def __init__(self, role, parts):
                self.role = role
                self.parts = parts

        mock_types.Content = DummyContent
        mock_types.Part.from_text = lambda text: text

        mock_google = MagicMock()
        mock_google.genai = mock_genai
        mock_genai.types = mock_types

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            messages = [
                {"role": "user", "content": "First prompt"},
                {"role": "assistant", "content": "Assistant reply"},
                {"role": "user", "content": "Second prompt"},
            ]
            result = llm_client._call_gemini(
                api_key="fake_key",
                model="gemini-2.5-flash",
                messages=messages,
                system="Test system",
            )

            assert result == "Mocked Gemini SDK response"
            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert call_kwargs["model"] == "gemini-2.5-flash"
            assert len(call_kwargs["contents"]) == 3
            assert call_kwargs["contents"][0].role == "user"
            assert call_kwargs["contents"][1].role == "model"
            assert call_kwargs["contents"][2].role == "user"


class TestLLMClientAnthropic:
    """Tests for Anthropic provider dispatch."""

    @patch("llm_client._call_anthropic")
    def test_anthropic_dispatch(self, mock_anthropic, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_API_KEY", "sk-ant-test")
        monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet-20241022")

        mock_anthropic.return_value = "Hello from Claude"

        messages = [{"role": "user", "content": "Hi!"}]
        response = llm_client.chat(messages, system="System prompt")

        assert response == "Hello from Claude"
        mock_anthropic.assert_called_once_with(
            api_key="sk-ant-test",
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            system="System prompt",
        )


class TestLLMClientOpenAI:
    """Tests for OpenAI provider dispatch."""

    @patch("llm_client._call_openai")
    def test_openai_dispatch(self, mock_openai, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")

        mock_openai.return_value = "Hello from GPT-4o"

        messages = [{"role": "user", "content": "Hi!"}]
        response = llm_client.chat(messages, system="System prompt")

        assert response == "Hello from GPT-4o"
        mock_openai.assert_called_once_with(
            api_key="sk-test",
            model="gpt-4o",
            messages=messages,
            system="System prompt",
        )


class TestLLMClientErrorsAndRetry:
    """Tests for retry logic and error reporting."""

    def test_missing_api_key_raises_value_error_for_cloud(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_API_KEY", "")
        with pytest.raises(ValueError, match="LLM_API_KEY environment variable is not set"):
            llm_client.chat([{"role": "user", "content": "test"}])

    def test_unsupported_provider_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "unsupported_provider_xyz")
        monkeypatch.setenv("LLM_API_KEY", "test_key")
        with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
            llm_client.chat([{"role": "user", "content": "test"}])

    @patch("llm_client._call_gemini")
    def test_retry_on_transient_failure_succeeds(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_API_KEY", "test_key")
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")

        # First call fails, second call succeeds
        mock_gemini.side_effect = [ConnectionResetError("Transient network glitch"), "Recovered!"]

        result = llm_client.chat([{"role": "user", "content": "test"}], retry_delay_seconds=0.01)
        assert result == "Recovered!"
        assert mock_gemini.call_count == 2

    @patch("llm_client._call_gemini")
    def test_retry_exhaustion_raises_runtime_error(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_API_KEY", "test_key")
        monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")

        mock_gemini.side_effect = [ConnectionResetError("Error 1"), ConnectionResetError("Error 2")]

        with pytest.raises(RuntimeError, match="failed after 2 attempt"):
            llm_client.chat([{"role": "user", "content": "test"}], max_retries=1, retry_delay_seconds=0.01)

        assert mock_gemini.call_count == 2


class TestLLMNormalizeIntegration:
    """Tests for normalize.llm_normalize integration with llm_client."""

    @patch("llm_client.chat")
    def test_llm_normalize_success(self, mock_chat):
        mock_chat.return_value = '{"canonical": "Modified Radical Mastectomy", "type": "Procedure"}'

        result = llm_normalize("mod rad mast", context="Patient underwent mod rad mast on left breast", base_confidence=0.90)

        assert result is not None
        assert isinstance(result, NormalizedEntity)
        assert result.raw_text == "mod rad mast"
        assert result.normalized_term == "Modified Radical Mastectomy"
        assert result.entity_type == "Procedure"
        assert result.method == "llm"
        # 0.90 - 0.1 penalty = 0.80
        assert result.confidence == 0.80

    @patch("llm_client.chat")
    def test_llm_normalize_with_code_fences(self, mock_chat):
        mock_chat.return_value = '```json\n{"canonical": "Paclitaxel", "type": "Medication"}\n```'

        result = llm_normalize("pacli", context="Administer pacli 80mg", base_confidence=1.0)

        assert result is not None
        assert result.normalized_term == "Paclitaxel"
        assert result.entity_type == "Medication"
        assert result.method == "llm"
        assert result.confidence == 0.90

    @patch("llm_client.chat")
    def test_llm_normalize_null_response(self, mock_chat):
        mock_chat.return_value = "null"

        result = llm_normalize("unknown random gibberish", context="some text")
        assert result is None

    @patch("llm_client.chat")
    def test_llm_normalize_error_graceful_fallback(self, mock_chat):
        mock_chat.side_effect = RuntimeError("API unavailable")

        result = llm_normalize("some term", context="some context")
        assert result is None


class TestTaskRouting:
    """Tests for task-specific routing: build -> Gemini, inference -> Local."""

    @patch("llm_client._call_gemini")
    @patch("llm_client._call_local")
    def test_task_build_routes_to_gemini(self, mock_local, mock_gemini, monkeypatch):
        monkeypatch.setenv("LLM_BUILD_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_BUILD_MODEL", "gemini-3.1-flash-lite")
        monkeypatch.setenv("LLM_API_KEY", "test_gemini_key")
        monkeypatch.setenv("LLM_PROVIDER", "local")

        mock_gemini.return_value = "Gemini build output"

        messages = [{"role": "user", "content": "Extract triples"}]
        res = llm_client.chat(messages, task="build")

        assert res == "Gemini build output"
        mock_gemini.assert_called_once_with(
            api_key="test_gemini_key",
            model="gemini-3.1-flash-lite",
            messages=messages,
            system=None,
        )
        mock_local.assert_not_called()

    @patch("llm_client._call_gemini")
    @patch("llm_client._call_local")
    def test_task_inference_routes_to_local(self, mock_local, mock_gemini, monkeypatch):
        monkeypatch.setenv("LLM_BUILD_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_INFERENCE_PROVIDER", "local")
        monkeypatch.setenv("LLM_INFERENCE_MODEL", "medgemma-1.5-4b-it-Q4_K_M.gguf")

        mock_local.return_value = "Local inference output"

        messages = [{"role": "user", "content": "Generate Cypher"}]
        res = llm_client.chat(messages, task="inference")

        assert res == "Local inference output"
        mock_local.assert_called_once_with(
            model="medgemma-1.5-4b-it-Q4_K_M.gguf",
            messages=messages,
            system=None,
        )
        mock_gemini.assert_not_called()

    @patch("llm_client._call_gemini")
    def test_chat_build_convenience_helper(self, mock_gemini, monkeypatch):
        monkeypatch.setenv("LLM_BUILD_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_BUILD_MODEL", "gemini-3.1-flash-lite")
        monkeypatch.setenv("LLM_API_KEY", "test_key")

        mock_gemini.return_value = "Build summary"
        messages = [{"role": "user", "content": "Summarize"}]
        res = llm_client.chat_build(messages)

        assert res == "Build summary"
        mock_gemini.assert_called_once()

    @patch("llm_client._call_local")
    def test_chat_inference_convenience_helper(self, mock_local, monkeypatch):
        monkeypatch.setenv("LLM_INFERENCE_PROVIDER", "local")
        monkeypatch.setenv("LLM_INFERENCE_MODEL", "medgemma-1.5-4b-it-Q4_K_M.gguf")

        mock_local.return_value = "Inference answer"
        messages = [{"role": "user", "content": "Answer question"}]
        res = llm_client.chat_inference(messages)

        assert res == "Inference answer"
        mock_local.assert_called_once()

