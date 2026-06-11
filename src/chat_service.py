"""Chat service — orchestrates LLM call + token deduction."""
import logging
from typing import List, Dict
from uuid import UUID

from vbwd.models.enums import TokenTransactionType
from plugins.chat.src.token_counting import TokenCountingStrategy, get_counting_strategy
from plugins.chat.src.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


def build_chat_service(token_service, config: dict) -> "ChatService":
    """Assemble a :class:`ChatService` from a token service and plugin config.

    Single home (DRY) for the web-chat route and the bot-consumer free-text
    handler — both must bill tokens identically, so both must build the service
    the same way.
    """
    adapter = LLMAdapter(
        api_endpoint=config.get("llm_api_endpoint", ""),
        api_key=config.get("llm_api_key", ""),
        model=config.get("llm_model", "gpt-4o-mini"),
        system_prompt=config.get("system_prompt", "You are a helpful assistant."),
    )
    strategy = get_counting_strategy(config.get("counting_mode", "words"))
    return ChatService(token_service, adapter, strategy, config)


class ChatService:
    """Orchestrates LLM communication and token billing."""

    def __init__(
        self,
        token_service,
        llm_adapter: LLMAdapter,
        counting_strategy: TokenCountingStrategy,
        config: dict,
    ):
        self.token_service = token_service
        self.llm_adapter = llm_adapter
        self.counting_strategy = counting_strategy
        self.config = config

    def send_message(
        self,
        user_id: UUID,
        message: str,
        history: List[Dict[str, str]],
    ) -> dict:
        """Send a message to LLM, deduct tokens, return response.

        Returns:
            {"response": str, "tokens_used": int, "balance": int}

        Raises:
            ValueError: If insufficient balance or message too long.
            LLMError: If LLM API call fails.
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
        response_text = self.llm_adapter.chat(messages)

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
