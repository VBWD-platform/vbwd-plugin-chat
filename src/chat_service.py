"""Chat service — orchestrates the core LLM call + token deduction.

The LLM call routes through the CORE connection client (``vbwd.llm`` /
``container.llm_client``) since S97.5: chat no longer owns an API key, endpoint
or HTTP adapter. The route + bot consumer resolve the client from the active
LLM connection (by ``llm_connection_slug``, else the default) and inject it
here. The system prompt is threaded per call (the core client takes it per
``.chat``), and a successful call advances the connection's ``last_active_at``.
"""
import logging
from typing import Dict, List
from uuid import UUID

from vbwd.llm.errors import LlmError
from vbwd.models.enums import TokenTransactionType
from plugins.chat.src.token_counting import TokenCountingStrategy, get_counting_strategy

logger = logging.getLogger(__name__)

# Re-exported so the route + bot consumer keep one error type to catch (DRY).
# The core client raises ``LlmError``; chat surfaces it under this stable name.
LLMError = LlmError

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def build_chat_service(token_service, config: dict, *, llm_client) -> "ChatService":
    """Assemble a :class:`ChatService` from a token service + the core LLM client.

    Single home (DRY) for the web-chat route and the bot-consumer free-text
    handler — both must bill tokens identically, so both build the service the
    same way and pass the same connection-resolved ``llm_client``.
    """
    strategy = get_counting_strategy(config.get("counting_mode", "words"))
    system_prompt = config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
    return ChatService(token_service, llm_client, strategy, config, system_prompt)


class ChatService:
    """Orchestrates LLM communication and token billing."""

    def __init__(
        self,
        token_service,
        llm_client,
        counting_strategy: TokenCountingStrategy,
        config: dict,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ):
        self.token_service = token_service
        self.llm_client = llm_client
        self.counting_strategy = counting_strategy
        self.config = config
        self.system_prompt = system_prompt

    def send_message(
        self,
        user_id: UUID,
        message: str,
        history: List[Dict[str, str]],
    ) -> dict:
        """Send a message to the LLM, deduct tokens, return the response.

        Returns:
            {"response": str, "tokens_used": int, "balance": int}

        Raises:
            ValueError: If insufficient balance or message too long.
            LLMError: If the LLM call fails.
        """
        max_length = self.config.get("max_message_length", 4000)
        if len(message) > max_length:
            raise ValueError(
                f"Message exceeds maximum length of {max_length} characters"
            )

        request_cost = self.counting_strategy.calculate_tokens(message, self.config)

        balance = self.token_service.get_balance(user_id)
        if balance < request_cost:
            raise ValueError("Insufficient token balance")

        max_history = self.config.get("max_history_messages", 20)
        trimmed_history = history[-max_history:]

        messages = trimmed_history + [{"role": "user", "content": message}]
        response_text = self.llm_client.chat(messages, system_prompt=self.system_prompt)

        response_cost = self.counting_strategy.calculate_tokens(
            response_text, self.config
        )
        total_cost = request_cost + response_cost

        updated_balance = self.token_service.debit_tokens(
            user_id=user_id,
            amount=total_cost,
            transaction_type=TokenTransactionType.USAGE,
            reference_id=None,
            description=(
                f"Chat: {total_cost} tokens "
                f"({request_cost} sent + {response_cost} received)"
            ),
        )

        return {
            "response": response_text,
            "tokens_used": total_cost,
            "balance": updated_balance.balance,
        }

    def estimate_cost(self, message: str) -> int:
        """Estimate token cost for a message (before sending)."""
        return self.counting_strategy.calculate_tokens(message, self.config)
