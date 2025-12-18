"""Unit tests for grok_client.py."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.metaagent.grok_client import (
    GrokClient,
    GrokClientError,
    GrokRateLimitError,
    MockGrokClient,
    query_grok,
)


class TestQueryGrok:
    """Tests for the query_grok function."""

    @patch("src.metaagent.grok_client.requests.post")
    @patch.dict("os.environ", {"GROK_API_KEY": "test-key"})
    def test_successful_query(self, mock_post):
        """Test successful API query returns content."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Grok!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_post.return_value = mock_response

        result = query_grok([{"role": "user", "content": "Hello"}])

        assert result == "Hello from Grok!"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "Authorization" in call_args.kwargs["headers"]
        assert call_args.kwargs["json"]["model"] == "grok-3-latest"

    @patch("src.metaagent.grok_client.requests.post")
    @patch.dict("os.environ", {"GROK_API_KEY": "test-key"})
    def test_custom_model_and_params(self, mock_post):
        """Test custom model and parameters are passed correctly."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_post.return_value = mock_response

        query_grok(
            messages=[{"role": "user", "content": "Test"}],
            model="grok-4",
            temperature=0.7,
            max_tokens=8192,
            timeout=60,
        )

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == "grok-4"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 8192
        assert call_args.kwargs["timeout"] == 60

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises GrokClientError."""
        with pytest.raises(GrokClientError) as exc_info:
            query_grok([{"role": "user", "content": "Hello"}])

        assert "GROK_API_KEY" in str(exc_info.value)

    @patch("src.metaagent.grok_client.requests.post")
    @patch.dict("os.environ", {"GROK_API_KEY": "test-key"})
    def test_timeout_raises_error(self, mock_post):
        """Test that timeout raises GrokClientError."""
        mock_post.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(GrokClientError) as exc_info:
            query_grok([{"role": "user", "content": "Hello"}], timeout=1)

        assert "timed out" in str(exc_info.value)

    @patch("src.metaagent.grok_client.requests.post")
    @patch.dict("os.environ", {"GROK_API_KEY": "test-key"})
    def test_connection_error_raises_error(self, mock_post):
        """Test that connection error raises GrokClientError."""
        mock_post.side_effect = requests.ConnectionError("Connection failed")

        with pytest.raises(GrokClientError) as exc_info:
            query_grok([{"role": "user", "content": "Hello"}])

        assert "connect" in str(exc_info.value).lower()

    @patch("src.metaagent.grok_client.requests.post")
    @patch.dict("os.environ", {"GROK_API_KEY": "test-key"})
    def test_rate_limit_raises_specific_error(self, mock_post):
        """Test that 429 status raises GrokRateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.ok = False
        mock_response.headers = {"Retry-After": "60"}
        mock_post.return_value = mock_response

        with pytest.raises(GrokRateLimitError) as exc_info:
            query_grok([{"role": "user", "content": "Hello"}])

        assert "429" in str(exc_info.value)
        assert "60" in str(exc_info.value)

    @patch("src.metaagent.grok_client.requests.post")
    @patch.dict("os.environ", {"GROK_API_KEY": "test-key"})
    def test_api_error_includes_detail(self, mock_post):
        """Test that API errors include error detail."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.ok = False
        mock_response.json.return_value = {
            "error": {"message": "Invalid request format"}
        }
        mock_post.return_value = mock_response

        with pytest.raises(GrokClientError) as exc_info:
            query_grok([{"role": "user", "content": "Hello"}])

        assert "400" in str(exc_info.value)
        assert "Invalid request format" in str(exc_info.value)

    @patch("src.metaagent.grok_client.requests.post")
    @patch.dict("os.environ", {"GROK_API_KEY": "test-key"})
    def test_malformed_response_raises_error(self, mock_post):
        """Test that malformed response raises GrokClientError."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"unexpected": "format"}
        mock_post.return_value = mock_response

        with pytest.raises(GrokClientError) as exc_info:
            query_grok([{"role": "user", "content": "Hello"}])

        assert "Unexpected" in str(exc_info.value)

    @patch("src.metaagent.grok_client.requests.post")
    def test_explicit_api_key_overrides_env(self, mock_post):
        """Test that explicit api_key parameter overrides env var."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_post.return_value = mock_response

        query_grok(
            [{"role": "user", "content": "Hello"}],
            api_key="explicit-key",
        )

        call_args = mock_post.call_args
        assert "explicit-key" in call_args.kwargs["headers"]["Authorization"]


class TestGrokClient:
    """Tests for the GrokClient class."""

    @patch("src.metaagent.grok_client.query_grok")
    def test_client_uses_defaults(self, mock_query):
        """Test that client uses configured defaults."""
        mock_query.return_value = "Response"

        client = GrokClient(
            api_key="test-key",
            model="grok-4",
            temperature=0.5,
            max_tokens=2048,
            timeout=30,
        )
        client.query([{"role": "user", "content": "Hello"}])

        mock_query.assert_called_once_with(
            messages=[{"role": "user", "content": "Hello"}],
            model="grok-4",
            temperature=0.5,
            max_tokens=2048,
            timeout=30,
            api_key="test-key",
        )

    @patch("src.metaagent.grok_client.query_grok")
    def test_client_query_overrides(self, mock_query):
        """Test that query-time params override defaults."""
        mock_query.return_value = "Response"

        client = GrokClient(model="grok-3-latest", temperature=0.3)
        client.query(
            [{"role": "user", "content": "Hello"}],
            model="grok-4",
            temperature=0.9,
        )

        call_args = mock_query.call_args
        assert call_args.kwargs["model"] == "grok-4"
        assert call_args.kwargs["temperature"] == 0.9

    @patch("src.metaagent.grok_client.query_grok")
    def test_chat_interface(self, mock_query):
        """Test the simple chat interface."""
        mock_query.return_value = "Hello!"

        client = GrokClient(api_key="test")
        result = client.chat("Hi there", system_prompt="Be helpful")

        assert result == "Hello!"
        call_args = mock_query.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi there"

    @patch("src.metaagent.grok_client.query_grok")
    def test_chat_without_system_prompt(self, mock_query):
        """Test chat without system prompt."""
        mock_query.return_value = "Response"

        client = GrokClient(api_key="test")
        client.chat("Hello")

        call_args = mock_query.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestMockGrokClient:
    """Tests for the MockGrokClient class."""

    def test_mock_returns_response(self):
        """Test that mock returns configured response."""
        mock = MockGrokClient()
        mock.mock_response = "Custom response"

        result = mock.query([{"role": "user", "content": "Test"}])

        assert result == "Custom response"

    def test_mock_tracks_calls(self):
        """Test that mock tracks call count and messages."""
        mock = MockGrokClient()

        mock.query([{"role": "user", "content": "First"}])
        mock.query([{"role": "user", "content": "Second"}])

        assert mock.call_count == 2
        assert mock.last_messages == [{"role": "user", "content": "Second"}]

    def test_mock_can_fail(self):
        """Test that mock can simulate failure."""
        mock = MockGrokClient()
        mock.should_fail = True
        mock.fail_error = "Simulated failure"

        with pytest.raises(GrokClientError) as exc_info:
            mock.query([{"role": "user", "content": "Test"}])

        assert "Simulated failure" in str(exc_info.value)

    def test_mock_chat_interface(self):
        """Test mock chat interface."""
        mock = MockGrokClient()
        mock.mock_response = "Chat response"

        result = mock.chat("Hello", system_prompt="Be helpful")

        assert result == "Chat response"
        assert len(mock.last_messages) == 2
