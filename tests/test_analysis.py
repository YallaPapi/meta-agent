"""Tests for analysis engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from metaagent.analysis import (
    AnalysisResult,
    MockAnalysisEngine,
    PerplexityAnalysisEngine,
    create_analysis_engine,
    extract_json_from_response,
    sanitize_error,
    validate_analysis_response,
)


class TestMockAnalysisEngine:
    """Tests for MockAnalysisEngine."""

    def test_analyze_returns_result(self) -> None:
        """Test that analyze returns a valid result."""
        engine = MockAnalysisEngine()
        result = engine.analyze("Test prompt")

        assert isinstance(result, AnalysisResult)
        assert result.success is True
        assert result.summary != ""
        assert len(result.recommendations) > 0
        assert len(result.tasks) > 0

    def test_tracks_call_count(self) -> None:
        """Test that call count is tracked."""
        engine = MockAnalysisEngine()

        assert engine.call_count == 0

        engine.analyze("First call")
        assert engine.call_count == 1

        engine.analyze("Second call")
        assert engine.call_count == 2

    def test_stores_last_prompt(self) -> None:
        """Test that last prompt is stored."""
        engine = MockAnalysisEngine()
        engine.analyze("My test prompt")

        assert engine.last_prompt == "My test prompt"

    def test_predefined_responses(self) -> None:
        """Test using predefined responses."""
        custom_result = AnalysisResult(
            summary="Custom summary",
            recommendations=["Custom rec"],
            tasks=[{"title": "Custom task"}],
        )

        engine = MockAnalysisEngine(responses={"keyword": custom_result})
        result = engine.analyze("Prompt with keyword in it")

        assert result.summary == "Custom summary"


class TestPerplexityAnalysisEngine:
    """Tests for PerplexityAnalysisEngine."""

    def test_parse_response_json(self) -> None:
        """Test parsing a JSON response."""
        engine = PerplexityAnalysisEngine(api_key="test")

        content = json.dumps(
            {
                "summary": "Test summary",
                "recommendations": ["rec1", "rec2"],
                "tasks": [{"title": "task1"}],
            }
        )

        result = engine._parse_response(content)

        assert result.summary == "Test summary"
        assert result.recommendations == ["rec1", "rec2"]
        assert len(result.tasks) == 1

    def test_parse_response_json_block(self) -> None:
        """Test parsing a JSON block in markdown."""
        engine = PerplexityAnalysisEngine(api_key="test")

        content = """Here's my analysis:

```json
{
    "summary": "Block summary",
    "recommendations": ["rec"],
    "tasks": []
}
```

Additional notes here."""

        result = engine._parse_response(content)

        assert result.summary == "Block summary"

    def test_parse_response_strict_mode_fails_on_invalid_json(self) -> None:
        """Test that strict mode returns failure for invalid JSON."""
        engine = PerplexityAnalysisEngine(api_key="test")

        content = "This is not JSON, just plain text analysis."

        result = engine._parse_response(content, strict_mode=True)

        assert result.success is False
        assert "JSON extraction failed" in result.error

    def test_parse_response_non_strict_fallback(self) -> None:
        """Test fallback when JSON parsing fails in non-strict mode."""
        engine = PerplexityAnalysisEngine(api_key="test")

        content = "This is not JSON, just plain text analysis."

        result = engine._parse_response(content, strict_mode=False)

        assert result.success is True
        assert result.summary == content
        assert result.recommendations == []
        assert result.tasks == []


class TestCreateAnalysisEngine:
    """Tests for the factory function."""

    def test_create_mock_engine(self) -> None:
        """Test creating a mock engine."""
        engine = create_analysis_engine(mock_mode=True)

        assert isinstance(engine, MockAnalysisEngine)

    def test_create_perplexity_engine(self) -> None:
        """Test creating a Perplexity engine."""
        engine = create_analysis_engine(api_key="test-key", mock_mode=False)

        assert isinstance(engine, PerplexityAnalysisEngine)

    def test_requires_api_key(self) -> None:
        """Test that API key is required when not in mock mode."""
        with pytest.raises(ValueError, match="API key is required"):
            create_analysis_engine(mock_mode=False)

    def test_create_engine_with_retry_config(self) -> None:
        """Test creating engine with custom retry configuration."""
        engine = create_analysis_engine(
            api_key="test-key",
            mock_mode=False,
            retry_max_attempts=5,
            retry_backoff_base=1.5,
            retry_backoff_max=30.0,
        )

        assert isinstance(engine, PerplexityAnalysisEngine)
        assert engine.retry_max_attempts == 5
        assert engine.retry_backoff_base == 1.5
        assert engine.retry_backoff_max == 30.0


class TestPerplexityRetryLogic:
    """Tests for retry logic in PerplexityAnalysisEngine."""

    def test_should_retry_timeout(self) -> None:
        """Test that timeouts are retryable."""
        engine = PerplexityAnalysisEngine(api_key="test", retry_max_attempts=3)
        error = httpx.TimeoutException("timeout")

        assert engine._should_retry(error, attempt=1) is True
        assert engine._should_retry(error, attempt=2) is True
        assert engine._should_retry(error, attempt=3) is False  # max reached

    def test_should_retry_429_rate_limit(self) -> None:
        """Test that 429 rate limit errors are retryable."""
        engine = PerplexityAnalysisEngine(api_key="test", retry_max_attempts=3)
        response = httpx.Response(429)
        request = httpx.Request("POST", "https://api.example.com")
        error = httpx.HTTPStatusError("rate limit", request=request, response=response)

        assert engine._should_retry(error, attempt=1) is True

    def test_should_retry_5xx_errors(self) -> None:
        """Test that 5xx server errors are retryable."""
        engine = PerplexityAnalysisEngine(api_key="test", retry_max_attempts=3)

        for status in [500, 502, 503, 504]:
            response = httpx.Response(status)
            request = httpx.Request("POST", "https://api.example.com")
            error = httpx.HTTPStatusError("server error", request=request, response=response)
            assert engine._should_retry(error, attempt=1) is True, f"Status {status} should be retryable"

    def test_should_not_retry_400_client_error(self) -> None:
        """Test that 400 client errors are not retryable."""
        engine = PerplexityAnalysisEngine(api_key="test", retry_max_attempts=3)
        response = httpx.Response(400)
        request = httpx.Request("POST", "https://api.example.com")
        error = httpx.HTTPStatusError("bad request", request=request, response=response)

        assert engine._should_retry(error, attempt=1) is False

    def test_should_not_retry_401_unauthorized(self) -> None:
        """Test that 401 unauthorized errors are not retryable."""
        engine = PerplexityAnalysisEngine(api_key="test", retry_max_attempts=3)
        response = httpx.Response(401)
        request = httpx.Request("POST", "https://api.example.com")
        error = httpx.HTTPStatusError("unauthorized", request=request, response=response)

        assert engine._should_retry(error, attempt=1) is False

    def test_should_retry_connect_error(self) -> None:
        """Test that connection errors are retryable."""
        engine = PerplexityAnalysisEngine(api_key="test", retry_max_attempts=3)
        error = httpx.ConnectError("connection refused")

        assert engine._should_retry(error, attempt=1) is True

    def test_get_backoff_time_exponential(self) -> None:
        """Test exponential backoff calculation."""
        engine = PerplexityAnalysisEngine(
            api_key="test",
            retry_backoff_base=2.0,
            retry_backoff_max=60.0,
        )

        # Test multiple attempts - backoff should be within range [0, base * 2^(attempt-1)]
        backoff1 = engine._get_backoff_time(attempt=1)
        assert 0 <= backoff1 <= 2.0  # base * 2^0 = 2.0

        backoff2 = engine._get_backoff_time(attempt=2)
        assert 0 <= backoff2 <= 4.0  # base * 2^1 = 4.0

        backoff3 = engine._get_backoff_time(attempt=3)
        assert 0 <= backoff3 <= 8.0  # base * 2^2 = 8.0

    def test_get_backoff_time_respects_max(self) -> None:
        """Test that backoff respects maximum limit."""
        engine = PerplexityAnalysisEngine(
            api_key="test",
            retry_backoff_base=10.0,
            retry_backoff_max=5.0,  # Cap at 5 seconds
        )

        # With base=10 and attempt=1, raw backoff would be 10, but capped at 5
        backoff = engine._get_backoff_time(attempt=1)
        assert 0 <= backoff <= 5.0

    def test_get_backoff_time_uses_retry_after(self) -> None:
        """Test that Retry-After header is respected."""
        engine = PerplexityAnalysisEngine(api_key="test")

        backoff = engine._get_backoff_time(attempt=1, retry_after=30)
        assert backoff == 30.0

    @patch.object(PerplexityAnalysisEngine, "client", create=True)
    def test_retry_on_500_then_success(self, mock_client: MagicMock) -> None:
        """Test that engine retries on 500 error and succeeds on second attempt."""
        engine = PerplexityAnalysisEngine(
            api_key="test",
            retry_max_attempts=3,
            retry_backoff_base=0.01,  # Very small for fast tests
        )

        # First call raises 500 error
        error_response = MagicMock()
        error_response.status_code = 500
        error_response.headers = {}
        request = MagicMock()
        error = httpx.HTTPStatusError("Server Error", request=request, response=error_response)

        # Second call succeeds
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "ok", "tasks": []}'}}]
        }
        success_response.raise_for_status.return_value = None

        # Set up mock to fail first, succeed second
        engine.client = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise error
            return success_response

        engine.client.post.side_effect = side_effect

        result = engine.analyze("test prompt")

        assert result.success is True
        assert engine.client.post.call_count == 2

    @patch.object(PerplexityAnalysisEngine, "client", create=True)
    def test_no_retry_on_400(self, mock_client: MagicMock) -> None:
        """Test that engine does not retry on 400 error."""
        engine = PerplexityAnalysisEngine(
            api_key="test",
            retry_max_attempts=3,
        )

        # 400 error - should not retry
        error_response = MagicMock()
        error_response.status_code = 400
        error_response.text = "Bad Request"
        error_response.headers = {}
        request = MagicMock()
        error = httpx.HTTPStatusError("Bad Request", request=request, response=error_response)

        engine.client = MagicMock()
        engine.client.post.side_effect = error

        result = engine.analyze("test prompt")

        assert result.success is False
        assert "400" in result.error
        assert engine.client.post.call_count == 1  # No retries

    @patch.object(PerplexityAnalysisEngine, "client", create=True)
    def test_max_retries_exhausted(self, mock_client: MagicMock) -> None:
        """Test that engine gives up after max retries."""
        engine = PerplexityAnalysisEngine(
            api_key="test",
            retry_max_attempts=3,
            retry_backoff_base=0.01,
        )

        # Always return 500 error
        error_response = MagicMock()
        error_response.status_code = 500
        error_response.text = "Server Error"
        error_response.headers = {}
        request = MagicMock()
        error = httpx.HTTPStatusError("Server Error", request=request, response=error_response)

        engine.client = MagicMock()
        engine.client.post.side_effect = error

        result = engine.analyze("test prompt")

        assert result.success is False
        assert "3 attempts" in result.error
        assert engine.client.post.call_count == 3  # All retries used


class TestJSONExtraction:
    """Tests for extract_json_from_response function."""

    def test_extract_plain_json(self) -> None:
        """Test extracting plain JSON."""
        content = '{"summary": "test", "tasks": []}'
        data, error = extract_json_from_response(content)

        assert data is not None
        assert data["summary"] == "test"
        assert error == ""

    def test_extract_json_from_markdown_block(self) -> None:
        """Test extracting JSON from markdown code block."""
        content = '''Here's my analysis:

```json
{"summary": "test", "tasks": []}
```

Additional notes.'''
        data, error = extract_json_from_response(content)

        assert data is not None
        assert data["summary"] == "test"

    def test_extract_json_with_nested_braces(self) -> None:
        """Test extracting JSON with nested braces in strings."""
        content = 'Intro {"summary": "has {braces} inside", "tasks": [{"nested": true}]} outro'
        data, error = extract_json_from_response(content)

        assert data is not None
        assert "braces" in data["summary"]

    def test_extract_json_with_code_in_strings(self) -> None:
        """Test extracting JSON with code containing braces in strings."""
        content = '{"summary": "code: if (x) { return; }", "tasks": []}'
        data, error = extract_json_from_response(content)

        assert data is not None
        assert "if (x)" in data["summary"]

    def test_extract_invalid_json_returns_none(self) -> None:
        """Test that invalid JSON returns None with error."""
        content = "This is not JSON at all"
        data, error = extract_json_from_response(content)

        assert data is None
        assert error != ""

    def test_extract_empty_content_returns_none(self) -> None:
        """Test that empty content returns None."""
        data, error = extract_json_from_response("")
        assert data is None
        assert "Empty" in error

    def test_extract_json_with_trailing_commas(self) -> None:
        """Test that trailing commas are handled."""
        content = '{"summary": "test", "tasks": [], }'
        data, error = extract_json_from_response(content)

        assert data is not None
        assert data["summary"] == "test"


class TestSchemaValidation:
    """Tests for validate_analysis_response function."""

    def test_valid_response(self) -> None:
        """Test validation of valid response."""
        data = {"summary": "test", "recommendations": ["a"], "tasks": [{"title": "t"}]}
        is_valid, error = validate_analysis_response(data)

        assert is_valid is True
        assert error == ""

    def test_missing_summary(self) -> None:
        """Test validation fails for missing summary."""
        data = {"recommendations": [], "tasks": []}
        is_valid, error = validate_analysis_response(data)

        assert is_valid is False
        assert "summary" in error

    def test_invalid_summary_type(self) -> None:
        """Test validation fails for non-string summary."""
        data = {"summary": 123, "tasks": []}
        is_valid, error = validate_analysis_response(data)

        assert is_valid is False
        assert "summary" in error

    def test_invalid_recommendations_type(self) -> None:
        """Test validation fails for non-list recommendations."""
        data = {"summary": "test", "recommendations": "not a list"}
        is_valid, error = validate_analysis_response(data)

        assert is_valid is False
        assert "recommendations" in error

    def test_invalid_tasks_type(self) -> None:
        """Test validation fails for non-list tasks."""
        data = {"summary": "test", "tasks": "not a list"}
        is_valid, error = validate_analysis_response(data)

        assert is_valid is False
        assert "tasks" in error

    def test_invalid_task_item(self) -> None:
        """Test validation fails for non-dict task items."""
        data = {"summary": "test", "tasks": ["string instead of dict"]}
        is_valid, error = validate_analysis_response(data)

        assert is_valid is False
        assert "Task at index" in error

    def test_minimal_valid_response(self) -> None:
        """Test validation passes for minimal response (just summary)."""
        data = {"summary": "test"}
        is_valid, error = validate_analysis_response(data)

        assert is_valid is True


class TestSanitizeError:
    """Tests for sanitize_error function."""

    def test_sanitize_bearer_token(self) -> None:
        """Test that Bearer tokens are sanitized."""
        message = "Error with Authorization: Bearer pplx-abc123xyz"
        result = sanitize_error(message)

        assert "pplx-abc123xyz" not in result
        assert "[REDACTED]" in result

    def test_sanitize_perplexity_api_key(self) -> None:
        """Test that Perplexity API keys are sanitized."""
        message = "Failed with key pplx-1234567890abcdef"
        result = sanitize_error(message)

        assert "pplx-1234567890abcdef" not in result
        assert "pplx-[REDACTED]" in result

    def test_sanitize_openai_style_key(self) -> None:
        """Test that OpenAI-style keys are sanitized."""
        message = "Error: sk-abc123xyz890"
        result = sanitize_error(message)

        assert "sk-abc123xyz890" not in result
        assert "sk-[REDACTED]" in result

    def test_sanitize_api_key_in_json(self) -> None:
        """Test sanitizing API key in JSON-like string."""
        message = '{"api_key": "secret123", "error": "failed"}'
        result = sanitize_error(message)

        assert "secret123" not in result
        assert "[REDACTED]" in result

    def test_preserve_non_sensitive_content(self) -> None:
        """Test that non-sensitive content is preserved."""
        message = "HTTP 500: Internal server error occurred"
        result = sanitize_error(message)

        assert result == message

    def test_sanitize_multiple_patterns(self) -> None:
        """Test sanitizing multiple sensitive patterns."""
        message = "Bearer abc123 and sk-xyz789 failed"
        result = sanitize_error(message)

        assert "abc123" not in result
        assert "xyz789" not in result
        assert result.count("[REDACTED]") == 2
