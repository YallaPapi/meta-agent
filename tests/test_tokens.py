"""Tests for token estimation utilities."""

from __future__ import annotations

import pytest

from metaagent.tokens import estimate_tokens, format_token_count, estimate_cost


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_string_returns_zero(self) -> None:
        """Test that empty string returns zero tokens."""
        assert estimate_tokens("") == 0

    def test_none_input_returns_zero(self) -> None:
        """Test that None-like empty input returns zero."""
        assert estimate_tokens("") == 0

    def test_single_word(self) -> None:
        """Test single word estimation."""
        result = estimate_tokens("hello")
        assert result >= 1  # At least 1 token

    def test_increases_with_length(self) -> None:
        """Test that token count increases with text length."""
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world this is a longer sentence with more words")
        assert long > short

    def test_deterministic(self) -> None:
        """Test that same input produces same output."""
        text = "The quick brown fox jumps over the lazy dog"
        result1 = estimate_tokens(text)
        result2 = estimate_tokens(text)
        assert result1 == result2

    def test_reasonable_estimate(self) -> None:
        """Test that estimate is reasonable (1-2 tokens per word on average)."""
        words = "one two three four five six seven eight nine ten"  # 10 words
        tokens = estimate_tokens(words)
        # Should be between 10 and 20 tokens for 10 words
        assert 10 <= tokens <= 20


class TestFormatTokenCount:
    """Tests for format_token_count function."""

    def test_small_number(self) -> None:
        """Test formatting small token counts."""
        assert format_token_count(100) == "100 tokens"

    def test_exactly_1000(self) -> None:
        """Test formatting exactly 1000 tokens."""
        result = format_token_count(1000)
        assert "1.0K" in result

    def test_large_number(self) -> None:
        """Test formatting large token counts."""
        result = format_token_count(15000)
        assert "15.0K" in result

    def test_decimal_thousands(self) -> None:
        """Test formatting with decimal thousands."""
        result = format_token_count(1500)
        assert "1.5K" in result


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_zero_tokens_zero_cost(self) -> None:
        """Test that zero tokens cost zero."""
        assert estimate_cost(0) == 0.0

    def test_1000_tokens_default_rate(self) -> None:
        """Test cost for 1000 tokens at default rate."""
        # Default rate is $0.003 per 1K tokens
        cost = estimate_cost(1000)
        assert cost == pytest.approx(0.003)

    def test_custom_rate(self) -> None:
        """Test cost with custom rate."""
        cost = estimate_cost(1000, cost_per_1k=0.01)
        assert cost == pytest.approx(0.01)
