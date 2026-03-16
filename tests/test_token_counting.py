"""Tests for token counting strategies."""
import pytest
from plugins.chat.src.token_counting import (
    WordCountStrategy,
    DataVolumeStrategy,
    TokenizerStrategy,
    get_counting_strategy,
)


class TestWordCountStrategy:
    def setup_method(self):
        self.strategy = WordCountStrategy()
        self.config = {"words_per_token": 10}

    def test_simple_sentence(self):
        text = "Hello world this is a test"
        result = self.strategy.calculate_tokens(text, self.config)
        # 6 words / 10 = 0.6 -> ceil -> 1
        assert result == 1

    def test_longer_text(self):
        text = " ".join(["word"] * 25)
        result = self.strategy.calculate_tokens(text, self.config)
        # 25 / 10 = 2.5 -> ceil -> 3
        assert result == 3

    def test_respects_config(self):
        text = " ".join(["word"] * 20)
        config = {"words_per_token": 5}
        result = self.strategy.calculate_tokens(text, config)
        # 20 / 5 = 4
        assert result == 4

    def test_minimum_one_token(self):
        text = ""
        result = self.strategy.calculate_tokens(text, self.config)
        assert result == 1

    def test_single_word(self):
        text = "Hello"
        result = self.strategy.calculate_tokens(text, self.config)
        assert result == 1

    def test_exact_boundary(self):
        text = " ".join(["word"] * 10)
        result = self.strategy.calculate_tokens(text, self.config)
        # 10 / 10 = 1.0 -> ceil -> 1
        assert result == 1


class TestDataVolumeStrategy:
    def setup_method(self):
        self.strategy = DataVolumeStrategy()
        self.config = {"mb_per_token": 0.001}

    def test_small_message(self):
        text = "Hello"  # 5 bytes
        result = self.strategy.calculate_tokens(text, self.config)
        # 5 / (0.001 * 1024 * 1024) = 5 / 1048.576 ≈ 0.00477 -> ceil -> 1
        assert result == 1

    def test_larger_message(self):
        # 2000 bytes ~ 0.00191 MB / 0.001 = 1.91 -> 2
        text = "a" * 2000
        result = self.strategy.calculate_tokens(text, self.config)
        assert result == 2

    def test_utf8_multibyte(self):
        # Each emoji is ~4 bytes in UTF-8
        text = "🎉" * 300  # 300 * 4 = 1200 bytes
        result = self.strategy.calculate_tokens(text, self.config)
        # 1200 / 1048.576 ≈ 1.14 -> ceil -> 2
        assert result == 2

    def test_respects_config(self):
        text = "a" * 1024  # 1KB
        config = {"mb_per_token": 0.0005}
        result = self.strategy.calculate_tokens(text, config)
        # 1024 / (0.0005 * 1048576) = 1024 / 524.288 ≈ 1.95 -> 2
        assert result == 2

    def test_minimum_one_token(self):
        text = ""
        result = self.strategy.calculate_tokens(text, self.config)
        assert result == 1


class TestTokenizerStrategy:
    def setup_method(self):
        self.strategy = TokenizerStrategy()
        self.config = {"tokens_per_token": 100}

    def test_approximation(self):
        text = "a" * 400  # 400 chars / 4 = 100 LLM tokens / 100 = 1
        result = self.strategy.calculate_tokens(text, self.config)
        assert result == 1

    def test_longer_text(self):
        text = "a" * 800  # 800 / 4 = 200 LLM tokens / 100 = 2
        result = self.strategy.calculate_tokens(text, self.config)
        assert result == 2

    def test_respects_config(self):
        text = "a" * 400
        config = {"tokens_per_token": 50}
        result = self.strategy.calculate_tokens(text, config)
        # 400 / 4 = 100 LLM tokens / 50 = 2
        assert result == 2

    def test_minimum_one_token(self):
        text = "Hi"  # 2 chars / 4 = 0 -> max(1, ...) = 1 LLM token / 100 -> ceil -> 1
        result = self.strategy.calculate_tokens(text, self.config)
        assert result == 1

    def test_empty_string(self):
        result = self.strategy.calculate_tokens("", self.config)
        assert result == 1


class TestGetCountingStrategy:
    def test_words(self):
        strategy = get_counting_strategy("words")
        assert isinstance(strategy, WordCountStrategy)

    def test_data_volume(self):
        strategy = get_counting_strategy("data_volume")
        assert isinstance(strategy, DataVolumeStrategy)

    def test_tokenizer(self):
        strategy = get_counting_strategy("tokenizer")
        assert isinstance(strategy, TokenizerStrategy)

    def test_unknown_defaults_to_words(self):
        strategy = get_counting_strategy("nonexistent")
        assert isinstance(strategy, WordCountStrategy)

    def test_empty_string_defaults_to_words(self):
        strategy = get_counting_strategy("")
        assert isinstance(strategy, WordCountStrategy)
