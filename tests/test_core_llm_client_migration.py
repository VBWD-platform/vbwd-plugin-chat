"""S97.5 RED — chat routes its reply through the CORE LLM client.

Engineering requirements (binding, restated): TDD-first (RED before the
migration); DevOps-first (no DB/network — token service + core client mocked);
SOLID (Dependency inversion — chat depends on the core ``llm_client`` port, not
its own HTTP adapter); DI (resolved from the container); DRY (one LLM client, in
core); Liskov; clean code; no overengineering. Quality guard:
``bin/pre-commit-check.sh --plugin chat --full``.

What these lock in:
* ``build_chat_service`` consumes a core ``LlmClient`` (``.chat(messages,
  system_prompt=…)``) instead of building its own ``LLMAdapter``,
* the web route + bot consumer both resolve the client from
  ``container.llm_client(slug=cfg.get('llm_connection_slug') or None)``
  (empty ⇒ default; a set slug ⇒ that connection),
* no ``llm_api_endpoint`` / ``llm_api_key`` reader remains in chat source.
"""

from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

chat_service_module = import_module("plugins.chat.src.chat_service")
build_chat_service = chat_service_module.build_chat_service

CHAT_SRC_DIR = Path(chat_service_module.__file__).resolve().parent


def _config():
    return {
        "system_prompt": "You are a Tarot-free helpful assistant.",
        "counting_mode": "words",
        "words_per_token": 10,
        "max_message_length": 4000,
        "max_history_messages": 20,
    }


def _token_service():
    token_service = MagicMock()
    token_service.get_balance.return_value = 1000
    balance = MagicMock()
    balance.balance = 990
    token_service.debit_tokens.return_value = balance
    return token_service


def test_build_chat_service_routes_through_core_client():
    """The reply text comes from the injected core client's ``.chat``."""
    core_client = MagicMock()
    core_client.chat.return_value = "Reply from the core LLM client."

    service = build_chat_service(
        _token_service(),
        _config(),
        llm_client=core_client,
    )
    result = service.send_message(user_id=uuid4(), message="Hello", history=[])

    assert result["response"] == "Reply from the core LLM client."
    core_client.chat.assert_called_once()
    # The system prompt is threaded per-call (core client takes it per call).
    _messages, call_kwargs = core_client.chat.call_args
    assert call_kwargs["system_prompt"] == "You are a Tarot-free helpful assistant."


def test_no_legacy_llm_api_keys_read_in_chat_source():
    """No ``llm_api_endpoint`` / ``llm_api_key`` reader remains in chat source."""
    offenders = []
    for python_file in CHAT_SRC_DIR.rglob("*.py"):
        text = python_file.read_text(encoding="utf-8")
        for legacy_key in ("llm_api_endpoint", "llm_api_key"):
            if legacy_key in text:
                offenders.append(f"{python_file.name}: {legacy_key}")
    assert offenders == [], f"legacy LLM config readers remain in chat: {offenders}"
