"""Token estimation utilities for meta-agent.

Provides lightweight token estimation for previewing LLM costs
without requiring a full tokenizer library.
"""

from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple heuristic based on word count. This is an approximation
    and may differ from actual tokenizer counts by 10-20%.

    For more accurate estimates, consider integrating tiktoken or similar.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Heuristic: ~1.3 tokens per word on average
    # This approximates GPT-style tokenization
    words = text.split()
    return int(math.ceil(len(words) * 1.3))


def format_token_count(tokens: int) -> str:
    """Format a token count for display.

    Args:
        tokens: Number of tokens.

    Returns:
        Human-readable string like "1.2K tokens" or "15 tokens".
    """
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}K tokens"
    return f"{tokens} tokens"


def estimate_cost(tokens: int, cost_per_1k: float = 0.003) -> float:
    """Estimate the cost for a given token count.

    Default cost is based on typical API pricing (~$3 per 1M tokens).

    Args:
        tokens: Number of tokens.
        cost_per_1k: Cost per 1000 tokens.

    Returns:
        Estimated cost in dollars.
    """
    return (tokens / 1000) * cost_per_1k
