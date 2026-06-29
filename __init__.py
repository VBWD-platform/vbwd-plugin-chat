"""LLM Chat plugin with token-based billing."""
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from vbwd.plugins.base import BasePlugin, PluginMetadata

if TYPE_CHECKING:
    from flask import Blueprint

    # Bot-base is an OPTIONAL bridge (S45.2 / D1 inversion). It is imported only
    # for type checking here; at runtime the bot-consumer methods lazily import
    # the neutral DTOs inside their bodies, so ``chat`` imports cleanly even when
    # bot-base is absent (no hard dependency, no top-level ``bot_base`` import).
    from plugins.bot_base.bot_base.types import BotCommand, BotInbound, BotReply


DEFAULT_CONFIG = {
    # The model/endpoint/key now live in a CORE "LLM Connection" (S97). chat
    # keeps only the optional slug of the connection to use; empty ⇒ the active
    # default connection.
    "llm_connection_slug": "",
    "counting_mode": "words",
    "words_per_token": 10,
    "mb_per_token": 0.001,
    "tokens_per_token": 100,
    "system_prompt": "You are a helpful assistant.",
    "max_message_length": 4000,
    "max_history_messages": 20,
    "bot_enabled": False,
    "bot_greeting": "Hello, I am {model}. How may I help you today?",
}

BOT_NAMESPACE = "chat"
HELLO_LLM_COMMAND = "hello-llm"
LINK_PROMPT = (
    "Please connect your account first. Request a link token from your "
    "profile, then send /start <token> here to start chatting."
)


class ChatPlugin(BasePlugin):
    """LLM chat with token-based billing.

    Class MUST be defined in __init__.py (not re-exported) due to
    discovery check obj.__module__ != full_module in manager.py.

    Also a bot-base **consumer** (S45.2): it structurally implements the
    ``BotCommandProvider`` seam (``bot_namespace`` + ``get_bot_commands`` +
    ``handle_action``) so its commands light up over every bot adapter
    (Telegram now, meinchat later) with no consumer change. The bridge is
    optional — see the lazy imports below.
    """

    #: The owning namespace bot-base routes commands / free text to (D1/D7).
    bot_namespace = BOT_NAMESPACE

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="chat",
            version="26.6",
            author="VBWD Team",
            description="LLM chat with token-based billing",
            # bot-base is intentionally NOT a dependency: the bridge is optional
            # and the bot methods never import it at module load time (D1).
            dependencies=[],
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with defaults merged with provided config."""
        merged = {**DEFAULT_CONFIG}
        if config:
            merged.update(config)
        super().initialize(merged)

    def get_blueprint(self) -> Optional["Blueprint"]:
        from plugins.chat.src.routes import chat_bp

        return chat_bp

    def get_url_prefix(self) -> Optional[str]:
        return "/api/v1/plugins/chat"

    def on_enable(self) -> None:
        pass

    def on_disable(self) -> None:
        pass

    # ── bot-base consumer seam (S45.2) ───────────────────────────────────────
    def get_bot_commands(self) -> List["BotCommand"]:
        """The commands ``chat`` contributes to the bot menu.

        Returns ``[]`` when ``bot_enabled`` is false so bot-base's
        ``CommandRegistry`` never surfaces ``/hello-llm`` — web chat stays
        entirely untouched. The neutral ``BotCommand`` DTO is imported lazily so
        this module loads even when bot-base is absent.
        """
        if not self.get_config("bot_enabled", False):
            return []

        from plugins.bot_base.bot_base.types import BotCommand

        return [
            BotCommand(
                name=HELLO_LLM_COMMAND,
                description="Chat with the LLM assistant",
                namespace=BOT_NAMESPACE,
            )
        ]

    def handle_action(self, context: "BotInbound") -> "BotReply":
        """Handle a ``/hello-llm`` command or free text routed to ``chat``.

        bot-base claims the conversation for ``chat`` when ``/hello-llm`` runs,
        so subsequent free text arrives here with ``command is None``.
        """
        from plugins.bot_base.bot_base.types import BotReply

        if context.command == HELLO_LLM_COMMAND:
            return BotReply(text=self._greeting())

        return self._handle_free_text(context)

    def _greeting(self) -> str:
        template = self.get_config("bot_greeting", DEFAULT_CONFIG["bot_greeting"])
        # The concrete model lives in the central LLM connection now; surface the
        # connection slug (blank ⇒ the active default) where the greeting names
        # the "model".
        model = self.get_config("llm_connection_slug", "") or "default"
        return template.format(model=model)

    def _handle_free_text(self, context: "BotInbound") -> "BotReply":
        from plugins.bot_base.bot_base.types import BotReply

        if context.identity is None:
            # Unlinked sender: prompt to link (D3). No ChatService call, no
            # token debit, no identity mutation.
            return BotReply(text=LINK_PROMPT)

        message = context.text or ""
        service = self._build_bot_chat_service()
        result = service.send_message(
            user_id=context.identity.vbwd_user_id,
            message=message,
            # Persisting full bot-chat history is out of scope; each turn uses a
            # fresh window (reuses the service's ``max_history_messages`` trim).
            history=[],
        )
        return BotReply(text=result["response"])

    def _build_bot_chat_service(self):
        """Build the same token-billed ChatService the web route uses (DRY).

        Reads the live config from the running app's config store (single source
        of truth) and resolves the token service from the DI container — exactly
        the web-chat path, so bot chat bills tokens identically.
        """
        from flask import current_app

        from plugins.chat.src.chat_service import build_chat_service

        config = current_app.config_store.get_config("chat")
        token_service = current_app.container.token_service()
        slug = config.get("llm_connection_slug") or None
        llm_client = current_app.container.llm_client(slug=slug)
        return build_chat_service(token_service, config, llm_client=llm_client)
