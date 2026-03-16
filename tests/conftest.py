"""Shared fixtures for chat plugin tests."""
import pytest
from unittest.mock import MagicMock

from src.plugins.config_store import PluginConfigEntry


@pytest.fixture
def chat_config():
    """Chat plugin configuration dict."""
    return {
        "llm_api_endpoint": "https://api.openai.com/v1/chat/completions",
        "llm_api_key": "sk-test-key-123",
        "llm_model": "gpt-4o-mini",
        "counting_mode": "words",
        "words_per_token": 10,
        "mb_per_token": 0.001,
        "tokens_per_token": 100,
        "system_prompt": "You are a helpful assistant.",
        "max_message_length": 4000,
        "max_history_messages": 20,
    }


@pytest.fixture
def mock_config_store(mocker, chat_config):
    """Mock PluginConfigStore with enabled chat entry."""
    store = mocker.MagicMock()
    store.get_by_name.return_value = PluginConfigEntry(
        plugin_name="chat",
        status="enabled",
        config=chat_config,
    )
    store.get_config.return_value = chat_config
    return store


@pytest.fixture
def mock_config_store_disabled(mocker):
    """Config store returning disabled chat plugin."""
    store = mocker.MagicMock()
    store.get_by_name.return_value = PluginConfigEntry(
        plugin_name="chat", status="disabled"
    )
    return store


@pytest.fixture
def mock_token_service():
    """Mock TokenService."""
    service = MagicMock()
    balance_obj = MagicMock()
    balance_obj.balance = 1000
    service.get_balance.return_value = 1000
    service.debit_tokens.return_value = balance_obj
    return service
