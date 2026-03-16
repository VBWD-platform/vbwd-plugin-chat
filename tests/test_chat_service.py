"""Tests for ChatService."""
import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from plugins.chat.src.chat_service import ChatService
from plugins.chat.src.token_counting import WordCountStrategy
from plugins.chat.src.llm_adapter import LLMError


@pytest.fixture
def chat_service(mock_token_service, chat_config):
    """ChatService with mocked dependencies."""
    adapter = MagicMock()
    adapter.chat.return_value = "This is the assistant response text."
    strategy = WordCountStrategy()
    return ChatService(mock_token_service, adapter, strategy, chat_config)


class TestChatServiceSendMessage:
    def test_returns_response(self, chat_service):
        result = chat_service.send_message(
            user_id=uuid4(),
            message="Hello world",
            history=[],
        )
        assert result["response"] == "This is the assistant response text."

    def test_returns_tokens_used(self, chat_service):
        result = chat_service.send_message(
            user_id=uuid4(),
            message="Hello world test",
            history=[],
        )
        assert result["tokens_used"] > 0
        assert isinstance(result["tokens_used"], int)

    def test_returns_updated_balance(self, chat_service):
        result = chat_service.send_message(
            user_id=uuid4(),
            message="Hello",
            history=[],
        )
        assert "balance" in result

    def test_deducts_tokens(self, chat_service, mock_token_service):
        chat_service.send_message(
            user_id=uuid4(),
            message="Hello world",
            history=[],
        )
        mock_token_service.debit_tokens.assert_called_once()
        call_kwargs = mock_token_service.debit_tokens.call_args
        assert call_kwargs.kwargs["amount"] > 0

    def test_counts_both_request_and_response_tokens(self, chat_service, mock_token_service):
        chat_service.send_message(
            user_id=uuid4(),
            message=" ".join(["word"] * 15),  # 15 words -> 2 tokens
            history=[],
        )
        call_kwargs = mock_token_service.debit_tokens.call_args
        # Request tokens + response tokens > request tokens alone
        total = call_kwargs.kwargs["amount"]
        assert total >= 2  # at least request cost

    def test_insufficient_balance_raises(self, mock_token_service, chat_config):
        mock_token_service.get_balance.return_value = 0
        adapter = MagicMock()
        strategy = WordCountStrategy()
        service = ChatService(mock_token_service, adapter, strategy, chat_config)

        with pytest.raises(ValueError, match="Insufficient"):
            service.send_message(uuid4(), "Hello", [])

        adapter.chat.assert_not_called()

    def test_validates_max_length(self, chat_service):
        long_msg = "a" * 5000
        with pytest.raises(ValueError, match="maximum length"):
            chat_service.send_message(uuid4(), long_msg, [])

    def test_trims_history(self, chat_service):
        history = [
            {"role": "user", "content": f"msg {i}"}
            for i in range(30)
        ]
        chat_service.send_message(uuid4(), "Hello", history)

        adapter_call = chat_service.llm_adapter.chat.call_args[0][0]
        # max_history_messages=20, plus the new message = 21
        assert len(adapter_call) == 21

    def test_llm_error_no_deduction(self, mock_token_service, chat_config):
        adapter = MagicMock()
        adapter.chat.side_effect = LLMError("API error")
        strategy = WordCountStrategy()
        service = ChatService(mock_token_service, adapter, strategy, chat_config)

        with pytest.raises(LLMError):
            service.send_message(uuid4(), "Hello", [])

        mock_token_service.debit_tokens.assert_not_called()

    def test_passes_history_to_llm(self, chat_service):
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
        ]
        chat_service.send_message(uuid4(), "Second", history)

        messages = chat_service.llm_adapter.chat.call_args[0][0]
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Reply"
        assert messages[2]["content"] == "Second"


class TestChatServiceEstimateCost:
    def test_returns_token_count(self, chat_service):
        cost = chat_service.estimate_cost("Hello world test message")
        assert isinstance(cost, int)
        assert cost >= 1

    def test_minimum_one(self, chat_service):
        cost = chat_service.estimate_cost("")
        assert cost >= 1

    def test_longer_text_costs_more(self, chat_service):
        short_cost = chat_service.estimate_cost("Hi")
        long_cost = chat_service.estimate_cost(" ".join(["word"] * 50))
        assert long_cost >= short_cost
