"""Grok API client for error diagnosis and PRD evaluation.

This module provides a client for the xAI Grok API, used as the primary
evaluator for the autonomous development loop.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Grok API endpoint
GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"

# Default settings
DEFAULT_MODEL = "grok-3-latest"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120


class GrokClientError(Exception):
    """Exception raised for Grok API errors."""

    pass


class GrokRateLimitError(GrokClientError):
    """Exception raised when Grok API rate limit is exceeded."""

    pass


def query_grok(
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    api_key: Optional[str] = None,
) -> str:
    """Query the Grok API with a chat completion request.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Grok model to use (default: grok-3-latest).
        temperature: Sampling temperature (default: 0.3).
        max_tokens: Maximum tokens in response (default: 4096).
        timeout: Request timeout in seconds (default: 120).
        api_key: Optional API key override. If not provided, uses GROK_API_KEY env var.

    Returns:
        The content of the assistant's response.

    Raises:
        GrokClientError: If the API key is missing or the request fails.
        GrokRateLimitError: If the API rate limit is exceeded.
    """
    # Load environment variables if not already loaded
    load_dotenv()

    # Get API key
    key = api_key or os.getenv("GROK_API_KEY")
    if not key:
        raise GrokClientError(
            "GROK_API_KEY environment variable is not set. "
            "Set it in your .env file or pass api_key parameter."
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.debug(f"Querying Grok API with model={model}, temp={temperature}")

    try:
        response = requests.post(
            GROK_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise GrokClientError(
            f"Grok API request timed out after {timeout} seconds"
        ) from exc
    except requests.ConnectionError as exc:
        raise GrokClientError(
            f"Failed to connect to Grok API: {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise GrokClientError(
            f"Grok API request failed: {exc}"
        ) from exc

    # Handle rate limiting
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        raise GrokRateLimitError(
            f"Grok API rate limit exceeded (HTTP 429). Retry after: {retry_after}"
        )

    # Handle other errors
    if not response.ok:
        error_detail = ""
        try:
            error_data = response.json()
            error_detail = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_detail = response.text[:500]

        raise GrokClientError(
            f"Grok API error {response.status_code}: {error_detail}"
        )

    # Parse response
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Log token usage if available
        usage = data.get("usage", {})
        if usage:
            logger.debug(
                f"Grok API usage: {usage.get('prompt_tokens', 0)} input, "
                f"{usage.get('completion_tokens', 0)} output tokens"
            )

        return content

    except (KeyError, IndexError, TypeError) as exc:
        raise GrokClientError(
            f"Unexpected Grok API response format: {exc}"
        ) from exc


class GrokClient:
    """Client wrapper for Grok API with configurable defaults.

    This class provides a stateful wrapper around the query_grok function,
    allowing configuration to be set once and reused across multiple calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initialize the Grok client.

        Args:
            api_key: Optional API key. If not provided, uses GROK_API_KEY env var.
            model: Default model to use.
            temperature: Default sampling temperature.
            max_tokens: Default maximum tokens in response.
            timeout: Default request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def query(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Query the Grok API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Override default model.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            timeout: Override default timeout.

        Returns:
            The content of the assistant's response.
        """
        return query_grok(
            messages=messages,
            model=model or self.model,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            timeout=timeout or self.timeout,
            api_key=self.api_key,
        )

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simple chat interface for single-turn conversations.

        Args:
            prompt: The user's prompt/question.
            system_prompt: Optional system prompt to set context.

        Returns:
            The assistant's response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.query(messages)


class MockGrokClient:
    """Mock Grok client for testing without API calls."""

    def __init__(self):
        """Initialize the mock client."""
        self.call_count = 0
        self.last_messages: List[Dict[str, Any]] = []
        self.mock_response: str = "Mock Grok response"
        self.should_fail: bool = False
        self.fail_error: str = "Mock failure"

    def query(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Mock query that returns predefined response.

        Args:
            messages: List of message dicts (stored for inspection).
            model: Ignored in mock.
            temperature: Ignored in mock.
            max_tokens: Ignored in mock.
            timeout: Ignored in mock.

        Returns:
            The mock response string.

        Raises:
            GrokClientError: If should_fail is True.
        """
        self.call_count += 1
        self.last_messages = messages

        if self.should_fail:
            raise GrokClientError(self.fail_error)

        return self.mock_response

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Mock chat interface.

        Args:
            prompt: The user's prompt (stored for inspection).
            system_prompt: Optional system prompt.

        Returns:
            The mock response string.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.query(messages)
