"""Tests for analysis engine."""

from __future__ import annotations

import json

import pytest

from metaagent.analysis import (
    AnalysisResult,
    MockAnalysisEngine,
    PerplexityAnalysisEngine,
    create_analysis_engine,
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

    def test_parse_response_fallback(self) -> None:
        """Test fallback when JSON parsing fails."""
        engine = PerplexityAnalysisEngine(api_key="test")

        content = "This is not JSON, just plain text analysis."

        result = engine._parse_response(content)

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
