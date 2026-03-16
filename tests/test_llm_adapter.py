"""Tests for LLM adapter."""
import pytest
from unittest.mock import MagicMock, patch

from plugins.chat.src.llm_adapter import LLMAdapter, LLMError


class TestLLMAdapter:
    def setup_method(self):
        self.adapter = LLMAdapter(
            api_endpoint="https://api.example.com/v1/chat/completions",
            api_key="sk-test-key",
            model="gpt-4o-mini",
            system_prompt="You are a test bot.",
            timeout=10,
        )

    def test_constructor_stores_config(self):
        assert self.adapter.api_endpoint == "https://api.example.com/v1/chat/completions"
        assert self.adapter.api_key == "sk-test-key"
        assert self.adapter.model == "gpt-4o-mini"
        assert self.adapter.system_prompt == "You are a test bot."
        assert self.adapter.timeout == 10

    @patch("plugins.chat.src.llm_adapter.requests")
    def test_chat_success_returns_content(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello there!"}}]
        }
        mock_requests.post.return_value = mock_resp

        result = self.adapter.chat([{"role": "user", "content": "Hi"}])
        assert result == "Hello there!"

    @patch("plugins.chat.src.llm_adapter.requests")
    def test_chat_sends_system_prompt(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Ok"}}]
        }
        mock_requests.post.return_value = mock_resp

        self.adapter.chat([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_requests.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are a test bot."

    @patch("plugins.chat.src.llm_adapter.requests")
    def test_chat_sends_model_in_payload(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Ok"}}]
        }
        mock_requests.post.return_value = mock_resp

        self.adapter.chat([])

        call_kwargs = mock_requests.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "gpt-4o-mini"

    @patch("plugins.chat.src.llm_adapter.requests")
    def test_chat_sends_auth_header(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Ok"}}]
        }
        mock_requests.post.return_value = mock_resp

        self.adapter.chat([])

        call_kwargs = mock_requests.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer sk-test-key"

    @patch("plugins.chat.src.llm_adapter.requests")
    def test_chat_raises_on_non_200(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_requests.post.return_value = mock_resp

        with pytest.raises(LLMError, match="500"):
            self.adapter.chat([{"role": "user", "content": "Hi"}])

    @patch("plugins.chat.src.llm_adapter.requests")
    def test_chat_raises_on_timeout(self, mock_requests):
        import requests as real_requests
        mock_requests.post.side_effect = real_requests.Timeout("timed out")
        mock_requests.Timeout = real_requests.Timeout
        mock_requests.RequestException = real_requests.RequestException

        with pytest.raises(LLMError, match="timed out"):
            self.adapter.chat([{"role": "user", "content": "Hi"}])

    @patch("plugins.chat.src.llm_adapter.requests")
    def test_chat_raises_on_invalid_json(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unexpected": "format"}
        mock_requests.post.return_value = mock_resp

        with pytest.raises(LLMError, match="Invalid LLM API response"):
            self.adapter.chat([{"role": "user", "content": "Hi"}])

    @patch("plugins.chat.src.llm_adapter.requests")
    def test_chat_includes_message_history(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Reply"}}]
        }
        mock_requests.post.return_value = mock_resp

        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        self.adapter.chat(history)

        call_kwargs = mock_requests.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        # system + 3 history messages
        assert len(payload["messages"]) == 4
        assert payload["messages"][1]["content"] == "First"

    def test_chat_raises_if_no_endpoint(self):
        adapter = LLMAdapter(
            api_endpoint="", api_key="key", model="model"
        )
        with pytest.raises(LLMError, match="not configured"):
            adapter.chat([{"role": "user", "content": "Hi"}])


class TestEndpointNormalization:
    """Test _normalize_endpoint auto-appends /chat/completions."""

    def test_full_url_unchanged(self):
        adapter = LLMAdapter(
            api_endpoint="https://api.openai.com/v1/chat/completions",
            api_key="k", model="m",
        )
        assert adapter.api_endpoint == "https://api.openai.com/v1/chat/completions"

    def test_base_url_gets_path_appended(self):
        adapter = LLMAdapter(
            api_endpoint="https://api.deepseek.com",
            api_key="k", model="m",
        )
        assert adapter.api_endpoint == "https://api.deepseek.com/chat/completions"

    def test_base_url_with_trailing_slash(self):
        adapter = LLMAdapter(
            api_endpoint="https://api.deepseek.com/",
            api_key="k", model="m",
        )
        assert adapter.api_endpoint == "https://api.deepseek.com/chat/completions"

    def test_base_url_with_v1(self):
        adapter = LLMAdapter(
            api_endpoint="https://api.openai.com/v1",
            api_key="k", model="m",
        )
        assert adapter.api_endpoint == "https://api.openai.com/v1/chat/completions"

    def test_empty_endpoint_stays_empty(self):
        adapter = LLMAdapter(
            api_endpoint="", api_key="k", model="m",
        )
        assert adapter.api_endpoint == ""
