"""Integration: chat bot-consumer round-trip over the fake Telegram client.

Boots chat + bot-base + bot-telegram, links a real vbwd user, then drives an
inbound ``/hello-llm`` (greeting) followed by free text (LLM answer) through the
webhook. Asserts the round-trip replies are delivered through bot-base and that
the free-text turn debits tokens exactly as the web-chat path. The LLM call and
the Telegram transport are faked (no network); all data is created through core
services / admin routes.
"""
import uuid

import pytest

from vbwd.models.enums import TokenTransactionType, UserRole

CHAT_CONFIG = {
    "llm_api_endpoint": "https://api.fake-llm.local/v1/chat/completions",
    "llm_api_key": "sk-fake",
    "llm_model": "gpt-4o-mini",
    "counting_mode": "words",
    "words_per_token": 10,
    "mb_per_token": 0.001,
    "tokens_per_token": 100,
    "system_prompt": "You are a helpful assistant.",
    "max_message_length": 4000,
    "max_history_messages": 20,
    "bot_enabled": True,
    "bot_greeting": "Hello, I am {model}. How may I help you today?",
}

FAKE_LLM_ANSWER = "Paris is the capital of France."
SENDER_REF = "555000"
CHAT_ID = "909090"


def _register_user(app, email: str):
    from vbwd.extensions import db
    from vbwd.repositories.user_repository import UserRepository

    auth_service = app.container.auth_service()
    unique_email = email.replace("@", f"+{uuid.uuid4().hex[:8]}@")
    result = auth_service.register(email=unique_email, password="ChatBot123@")
    db.session.commit()
    user = UserRepository(db.session).find_by_id(result.user_id)
    return str(user.id), result.token


def _promote_to_admin(app, user_id: str) -> None:
    from vbwd.extensions import db
    from vbwd.repositories.user_repository import UserRepository

    repository = UserRepository(db.session)
    user = repository.find_by_id(user_id)
    user.role = UserRole.ADMIN
    db.session.commit()


def _admin_headers(app):
    with app.app_context():
        user_id, token = _register_user(app, "chatbotadmin@example.com")
        _promote_to_admin(app, user_id)
    return {"Authorization": f"Bearer {token}"}


def _create_bot(app, client):
    headers = _admin_headers(app)
    body = {
        "name": f"chatbot-{uuid.uuid4().hex[:6]}",
        "username": "vbwd_chat_bot",
        "token": "111:CHAT_BOT_TOKEN",
        "default": True,
        "webhook_secret": "chat-wh-secret",
        "enabled": True,
    }
    response = client.post(
        "/api/v1/plugins/bot-telegram/admin/bots", json=body, headers=headers
    )
    assert response.status_code == 201
    return body["name"], body["webhook_secret"]


def _enable_chat_bot(app, monkeypatch):
    """Turn on the chat bot seam on the live plugin + config store (test-scoped).

    Mutates the already-enabled plugin's config in place (``set_config``) so its
    ENABLED status is preserved — ``initialize`` would reset status to
    INITIALIZED and drop it from ``get_enabled_plugins``.
    """
    plugin = app.plugin_manager.get_plugin("chat")
    for key, value in CHAT_CONFIG.items():
        plugin.set_config(key, value)

    original_get_config = app.config_store.get_config

    def _patched_get_config(plugin_name):
        if plugin_name == "chat":
            return dict(CHAT_CONFIG)
        return original_get_config(plugin_name)

    monkeypatch.setattr(app.config_store, "get_config", _patched_get_config)


def _grant_tokens(app, user_id, amount):
    from vbwd.extensions import db

    app.container.token_service().credit_tokens(
        user_id=uuid.UUID(user_id),
        amount=amount,
        transaction_type=TokenTransactionType.PURCHASE,
        description="integration seed",
    )
    db.session.commit()


def _issue_and_redeem_link(app, user_id):
    """Link the Telegram sender to the vbwd user via a one-time link token (D3)."""
    from vbwd.extensions import db
    from plugins.bot_base.bot_base.repositories.bot_link_repository import (
        BotLinkRepository,
    )
    from plugins.bot_base.bot_base.repositories.bot_link_token_repository import (
        BotLinkTokenRepository,
    )
    from plugins.bot_base.bot_base.services.link_service import LinkService

    link_service = LinkService(
        BotLinkRepository(db.session), BotLinkTokenRepository(db.session)
    )
    token = link_service.issue_token(uuid.UUID(user_id))
    link_service.redeem_token(
        token.token, provider_id="telegram", external_user_id=SENDER_REF
    )
    db.session.commit()


def _webhook_update(client, bot_name, secret, *, text):
    return client.post(
        f"/api/v1/plugins/bot-telegram/webhook/{bot_name}",
        json={
            "message": {
                "chat": {"id": int(CHAT_ID)},
                "from": {"id": int(SENDER_REF)},
                "text": text,
            }
        },
        headers={"X-Telegram-Bot-Api-Secret-Token": secret},
    )


@pytest.mark.integration
def test_hello_llm_then_free_text_round_trip_debits_tokens(
    app, client, _inject_fake_telegram_client, monkeypatch
):
    fake_telegram = _inject_fake_telegram_client

    # Fake the LLM transport so no network is touched but the billed path runs.
    from plugins.chat.src.llm_adapter import LLMAdapter

    monkeypatch.setattr(LLMAdapter, "chat", lambda self, messages: FAKE_LLM_ANSWER)

    _enable_chat_bot(app, monkeypatch)
    bot_name, secret = _create_bot(app, client)

    with app.app_context():
        user_id, _token = _register_user(app, "chatbotuser@example.com")
        _grant_tokens(app, user_id, 1000)
        _issue_and_redeem_link(app, user_id)
        starting_balance = app.container.token_service().get_balance(uuid.UUID(user_id))

    # 1) /hello-llm → greeting (claims the chat conversation), no debit.
    hello = _webhook_update(client, bot_name, secret, text="/hello-llm")
    assert hello.status_code == 200
    greeting = fake_telegram.sent_messages[-1]["payload"]["text"]
    assert "gpt-4o-mini" in greeting

    with app.app_context():
        after_hello = app.container.token_service().get_balance(uuid.UUID(user_id))
    assert after_hello == starting_balance

    # 2) free text → routed to chat (active owner) → LLM answer, debit.
    answer = _webhook_update(
        client, bot_name, secret, text="What is the capital of France?"
    )
    assert answer.status_code == 200
    delivered = fake_telegram.sent_messages[-1]["payload"]["text"]
    assert delivered == FAKE_LLM_ANSWER

    with app.app_context():
        final_balance = app.container.token_service().get_balance(uuid.UUID(user_id))
    assert final_balance < after_hello


@pytest.mark.integration
def test_unlinked_free_text_prompts_link_and_bills_nothing(
    app, client, _inject_fake_telegram_client, monkeypatch
):
    fake_telegram = _inject_fake_telegram_client

    from plugins.chat.src.llm_adapter import LLMAdapter

    llm_calls = []
    monkeypatch.setattr(
        LLMAdapter,
        "chat",
        lambda self, messages: llm_calls.append(messages) or FAKE_LLM_ANSWER,
    )

    _enable_chat_bot(app, monkeypatch)
    bot_name, secret = _create_bot(app, client)

    unlinked_sender = "999111"
    # Claim the chat conversation with /hello-llm from an UNLINKED sender, then
    # send free text — the consumer must refuse to bill and prompt to link.
    client.post(
        f"/api/v1/plugins/bot-telegram/webhook/{bot_name}",
        json={
            "message": {
                "chat": {"id": 424242},
                "from": {"id": int(unlinked_sender)},
                "text": "/hello-llm",
            }
        },
        headers={"X-Telegram-Bot-Api-Secret-Token": secret},
    )
    response = client.post(
        f"/api/v1/plugins/bot-telegram/webhook/{bot_name}",
        json={
            "message": {
                "chat": {"id": 424242},
                "from": {"id": int(unlinked_sender)},
                "text": "Tell me a joke",
            }
        },
        headers={"X-Telegram-Bot-Api-Secret-Token": secret},
    )
    assert response.status_code == 200

    prompt = fake_telegram.sent_messages[-1]["payload"]["text"]
    assert "/start" in prompt
    assert llm_calls == []
