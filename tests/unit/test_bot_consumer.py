"""Unit tests for the chat bot-consumer seam (S45.2).

``chat`` implements ``BotCommandProvider`` (bot_namespace="chat") so its
commands light up over every bot adapter unchanged. The bridge is **optional**:
the bot methods lazily import bot-base's neutral DTOs inside the method body, so
``chat`` imports cleanly even when bot-base is absent (no hard dependency).

These specs exercise the consumer in isolation with a fake ``ChatService`` (no
DB, no network) following the existing chat unit-test idiom (MagicMock repos).
"""
import sys
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from plugins.bot_base.bot_base.ports import BotCommandProvider
from plugins.bot_base.bot_base.services.command_registry import CommandRegistry
from plugins.bot_base.bot_base.types import BotIdentity, BotInbound, BotReply, ChatRef
from plugins.bot_base.tests.unit.fakes import FakePluginManager
from plugins.chat import ChatPlugin


def _make_inbound(*, command=None, text=None, identity=None, args=None):
    chat_ref = ChatRef(provider_id="telegram", chat_id="4242")
    return BotInbound(
        provider_id="telegram",
        chat_ref=chat_ref,
        sender_ref="7",
        text=text,
        command=command,
        args=args or [],
        identity=identity,
    )


def _linked_identity():
    return BotIdentity(
        provider_id="telegram",
        external_user_id="7",
        vbwd_user_id=uuid4(),
    )


@pytest.fixture
def enabled_chat_plugin(chat_config):
    """A ChatPlugin initialized with bot_enabled=True and a fake ChatService."""
    plugin = ChatPlugin()
    config = {**chat_config, "bot_enabled": True}
    plugin.initialize(config)
    return plugin


# ── D1 inversion: collected only when bot_enabled=True ───────────────────────
class TestCommandRegistryCollection:
    def test_plugin_structurally_implements_provider_seam(self, enabled_chat_plugin):
        assert isinstance(enabled_chat_plugin, BotCommandProvider)

    def test_registry_collects_chat_command_when_bot_enabled(self, enabled_chat_plugin):
        registry = CommandRegistry(FakePluginManager([enabled_chat_plugin]))

        assert enabled_chat_plugin in registry.get_command_providers()
        assert "hello-llm" in registry.command_index()
        assert registry.command_index()["hello-llm"] is enabled_chat_plugin

    def test_registry_surfaces_no_chat_command_when_bot_disabled(self, chat_config):
        """bot_enabled=false → /hello-llm is never collected → never claimable.

        The plugin still declares ``bot_namespace`` (a class attribute), but it
        contributes no command, so the dispatcher's command index has no entry
        and free text can never be routed to chat (web chat untouched).
        """
        plugin = ChatPlugin()
        plugin.initialize({**chat_config, "bot_enabled": False})
        registry = CommandRegistry(FakePluginManager([plugin]))

        assert registry.command_index() == {}
        assert registry.collect_commands() == []
        assert plugin.get_bot_commands() == []

    def test_bot_disabled_is_the_default(self, chat_config):
        plugin = ChatPlugin()
        config = {**chat_config}
        config.pop("bot_enabled", None)
        plugin.initialize(config)

        assert plugin.get_bot_commands() == []


# ── /hello-llm greeting ──────────────────────────────────────────────────────
class TestHelloLlmCommand:
    def test_exposes_hello_llm_command(self, enabled_chat_plugin):
        commands = enabled_chat_plugin.get_bot_commands()

        names = [command.name for command in commands]
        assert names == ["hello-llm"]
        assert commands[0].namespace == "chat"

    def test_greeting_interpolates_model(self, enabled_chat_plugin):
        inbound = _make_inbound(command="hello-llm")

        reply = enabled_chat_plugin.handle_action(inbound)

        assert isinstance(reply, BotReply)
        assert "gpt-4o-mini" in reply.text
        assert "{model}" not in reply.text

    def test_greeting_uses_custom_template(self, chat_config):
        plugin = ChatPlugin()
        plugin.initialize(
            {
                **chat_config,
                "bot_enabled": True,
                "llm_model": "claude-3-haiku",
                "bot_greeting": "Ready with {model}.",
            }
        )

        reply = plugin.handle_action(_make_inbound(command="hello-llm"))

        assert reply.text == "Ready with claude-3-haiku."

    def test_hello_llm_does_not_call_chat_service(
        self, enabled_chat_plugin, monkeypatch
    ):
        build_spy = MagicMock()
        monkeypatch.setattr(enabled_chat_plugin, "_build_bot_chat_service", build_spy)

        enabled_chat_plugin.handle_action(_make_inbound(command="hello-llm"))

        build_spy.assert_not_called()


# ── free-text → linked user → ChatService.send_message (token debit) ─────────
class TestFreeTextLinkedUser:
    def test_delegates_to_chat_service_with_linked_user_id(
        self, enabled_chat_plugin, monkeypatch
    ):
        identity = _linked_identity()
        fake_service = MagicMock()
        fake_service.send_message.return_value = {
            "response": "The model output.",
            "tokens_used": 5,
            "balance": 995,
        }
        monkeypatch.setattr(
            enabled_chat_plugin,
            "_build_bot_chat_service",
            lambda: fake_service,
        )

        reply = enabled_chat_plugin.handle_action(
            _make_inbound(text="What is the weather?", identity=identity)
        )

        fake_service.send_message.assert_called_once()
        call = fake_service.send_message.call_args
        assert call.kwargs["user_id"] == identity.vbwd_user_id
        assert call.kwargs["message"] == "What is the weather?"
        assert call.kwargs["history"] == []
        assert reply.text == "The model output."

    def test_token_debited_exactly_as_web_chat(self, mock_token_service, chat_config):
        """Free-text uses the same ChatService path → same debit behavior."""
        from plugins.chat.src.chat_service import ChatService
        from plugins.chat.src.token_counting import WordCountStrategy

        adapter = MagicMock()
        adapter.chat.return_value = "assistant words here"
        service = ChatService(
            mock_token_service, adapter, WordCountStrategy(), chat_config
        )

        plugin = ChatPlugin()
        plugin.initialize({**chat_config, "bot_enabled": True})
        plugin._build_bot_chat_service = lambda: service

        identity = _linked_identity()
        reply = plugin.handle_action(
            _make_inbound(text="hello there friend", identity=identity)
        )

        mock_token_service.debit_tokens.assert_called_once()
        debit = mock_token_service.debit_tokens.call_args
        assert debit.kwargs["amount"] > 0
        assert debit.kwargs["user_id"] == identity.vbwd_user_id
        assert reply.text == "assistant words here"


# ── unlinked sender → link prompt, no ChatService call, no debit ─────────────
class TestFreeTextUnlinkedSender:
    def test_unlinked_returns_link_prompt(self, enabled_chat_plugin):
        reply = enabled_chat_plugin.handle_action(
            _make_inbound(text="hello", identity=None)
        )

        assert isinstance(reply, BotReply)
        assert "/start" in reply.text

    def test_unlinked_never_builds_chat_service(self, enabled_chat_plugin, monkeypatch):
        build_spy = MagicMock()
        monkeypatch.setattr(enabled_chat_plugin, "_build_bot_chat_service", build_spy)

        enabled_chat_plugin.handle_action(_make_inbound(text="hello", identity=None))

        build_spy.assert_not_called()


# ── optional bridge: module imports with bot-base absent ─────────────────────
class TestBridgeOptional:
    def test_chat_imports_without_bot_base_on_path(self):
        """The chat package must import even when bot_base is unavailable.

        Simulate bot-base absent by blocking its import and re-importing the
        chat package: a top-level ``import bot_base`` would raise here.
        """
        import importlib

        blocked_prefixes = ("plugins.bot_base", "plugins.chat")
        saved = {
            name: module
            for name, module in sys.modules.items()
            if name.startswith(blocked_prefixes)
        }
        for name in list(saved):
            del sys.modules[name]

        class _BlockBotBase:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "plugins.bot_base" or fullname.startswith(
                    "plugins.bot_base."
                ):
                    raise ImportError("bot_base is absent in this scenario")
                return None

        finder = _BlockBotBase()
        sys.meta_path.insert(0, finder)
        try:
            chat_module = importlib.import_module("plugins.chat")
            # Plugin instantiates and runs web-chat seams with no bridge present.
            plugin = chat_module.ChatPlugin()
            assert plugin.metadata.name == "chat"
            assert "bot-base" not in (plugin.metadata.dependencies or [])
        finally:
            sys.meta_path.remove(finder)
            for name in list(sys.modules):
                if name.startswith(blocked_prefixes):
                    del sys.modules[name]
            sys.modules.update(saved)

    def test_bot_base_not_a_hard_dependency(self, enabled_chat_plugin):
        assert "bot-base" not in (enabled_chat_plugin.metadata.dependencies or [])
        assert "bot_base" not in (enabled_chat_plugin.metadata.dependencies or [])
