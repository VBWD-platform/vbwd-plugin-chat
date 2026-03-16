"""Token counting strategies for chat plugin.

Each strategy converts text into platform token cost using a different method.
Strategy pattern — admin selects the counting mode in plugin config.
"""
import math
from abc import ABC, abstractmethod


class TokenCountingStrategy(ABC):
    """Base class for token counting strategies."""

    @abstractmethod
    def calculate_tokens(self, text: str, config: dict) -> int:
        """Return platform token cost for the given text. Minimum 1."""
        pass


class WordCountStrategy(TokenCountingStrategy):
    """Count tokens based on word count."""

    def calculate_tokens(self, text: str, config: dict) -> int:
        words_per_token = config.get("words_per_token", 10)
        word_count = len(text.split())
        return max(1, math.ceil(word_count / words_per_token))


class DataVolumeStrategy(TokenCountingStrategy):
    """Count tokens based on UTF-8 byte size."""

    def calculate_tokens(self, text: str, config: dict) -> int:
        mb_per_token = config.get("mb_per_token", 0.001)
        byte_count = len(text.encode("utf-8"))
        mb_count = byte_count / (1024 * 1024)
        return max(1, math.ceil(mb_count / mb_per_token))


class TokenizerStrategy(TokenCountingStrategy):
    """Count tokens using LLM token approximation (~4 chars per LLM token)."""

    def calculate_tokens(self, text: str, config: dict) -> int:
        tokens_per_token = config.get("tokens_per_token", 100)
        estimated_llm_tokens = max(1, len(text) // 4)
        return max(1, math.ceil(estimated_llm_tokens / tokens_per_token))


def get_counting_strategy(mode: str) -> TokenCountingStrategy:
    """Get the counting strategy for the given mode. Defaults to words."""
    strategies = {
        "words": WordCountStrategy(),
        "data_volume": DataVolumeStrategy(),
        "tokenizer": TokenizerStrategy(),
    }
    return strategies.get(mode, WordCountStrategy())
