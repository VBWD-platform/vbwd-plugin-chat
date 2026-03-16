"""LLM Chat plugin with token-based billing."""
from typing import Optional, Dict, Any, TYPE_CHECKING
from src.plugins.base import BasePlugin, PluginMetadata

if TYPE_CHECKING:
    from flask import Blueprint


DEFAULT_CONFIG = {
    "llm_api_endpoint": "",
    "llm_api_key": "",
    "llm_model": "gpt-4o-mini",
    "counting_mode": "words",
    "words_per_token": 10,
    "mb_per_token": 0.001,
    "tokens_per_token": 100,
    "system_prompt": "You are a helpful assistant.",
    "max_message_length": 4000,
    "max_history_messages": 20,
}


class ChatPlugin(BasePlugin):
    """LLM chat with token-based billing.

    Class MUST be defined in __init__.py (not re-exported) due to
    discovery check obj.__module__ != full_module in manager.py.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="chat",
            version="1.0.0",
            author="VBWD Team",
            description="LLM chat with token-based billing",
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
