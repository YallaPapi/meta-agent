"""Analysis engine for running LLM analysis on codebases."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class AnalysisResult:
    """Result from an analysis stage."""

    summary: str
    recommendations: list[str] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    success: bool = True
    error: Optional[str] = None


class AnalysisEngine(ABC):
    """Abstract base class for analysis engines."""

    @abstractmethod
    def analyze(self, prompt: str) -> AnalysisResult:
        """Run analysis with the given prompt.

        Args:
            prompt: The rendered prompt to send to the LLM.

        Returns:
            AnalysisResult with the analysis output.
        """
        pass


class MockAnalysisEngine(AnalysisEngine):
    """Mock analysis engine for testing without API calls."""

    def __init__(self, responses: Optional[dict[str, AnalysisResult]] = None):
        """Initialize mock engine with optional predefined responses.

        Args:
            responses: Optional dict mapping prompt substrings to responses.
        """
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt: Optional[str] = None

    def analyze(self, prompt: str) -> AnalysisResult:
        """Return a mock analysis result.

        Args:
            prompt: The rendered prompt (stored for inspection).

        Returns:
            Mock AnalysisResult.
        """
        self.call_count += 1
        self.last_prompt = prompt

        # Check for predefined responses
        for key, response in self.responses.items():
            if key in prompt:
                return response

        # Default mock response
        return AnalysisResult(
            summary="Mock analysis completed successfully.",
            recommendations=[
                "This is a mock recommendation for testing.",
                "Consider running with real API keys for actual analysis.",
            ],
            tasks=[
                {
                    "id": f"mock-task-{self.call_count}",
                    "title": "Mock task from analysis",
                    "description": "This is a placeholder task from mock analysis.",
                    "priority": "medium",
                    "file": "src/example.py",
                },
            ],
            raw_response="Mock response",
            success=True,
        )


class PerplexityAnalysisEngine(AnalysisEngine):
    """Analysis engine using Perplexity API."""

    API_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: str, timeout: int = 120, model: str = "llama-3.1-sonar-large-128k-online"):
        """Initialize Perplexity analysis engine.

        Args:
            api_key: Perplexity API key.
            timeout: Request timeout in seconds.
            model: Model to use for analysis.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    def analyze(self, prompt: str) -> AnalysisResult:
        """Run analysis using Perplexity API.

        Args:
            prompt: The rendered prompt to send.

        Returns:
            AnalysisResult with the analysis output.
        """
        try:
            response = self.client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a code analysis expert. Analyze codebases and provide structured feedback in JSON format with keys: summary, recommendations, tasks.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                },
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return self._parse_response(content)

        except httpx.HTTPStatusError as e:
            return AnalysisResult(
                summary="",
                success=False,
                error=f"HTTP error from Perplexity API: {e.response.status_code}",
                raw_response=str(e),
            )
        except httpx.TimeoutException:
            return AnalysisResult(
                summary="",
                success=False,
                error=f"Request to Perplexity API timed out after {self.timeout}s",
            )
        except Exception as e:
            return AnalysisResult(
                summary="",
                success=False,
                error=f"Unexpected error calling Perplexity API: {e}",
            )

    def _parse_response(self, content: str) -> AnalysisResult:
        """Parse the LLM response into structured result.

        Args:
            content: Raw response content from the LLM.

        Returns:
            Parsed AnalysisResult.
        """
        # Try to extract JSON from the response
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = content

        try:
            data = json.loads(json_str)
            return AnalysisResult(
                summary=data.get("summary", ""),
                recommendations=data.get("recommendations", []),
                tasks=data.get("tasks", []),
                raw_response=content,
                success=True,
            )
        except json.JSONDecodeError:
            # Fallback: treat the whole response as summary
            return AnalysisResult(
                summary=content,
                recommendations=[],
                tasks=[],
                raw_response=content,
                success=True,
            )


def create_analysis_engine(
    api_key: Optional[str] = None,
    mock_mode: bool = False,
    timeout: int = 120,
) -> AnalysisEngine:
    """Factory function to create the appropriate analysis engine.

    Args:
        api_key: Perplexity API key (required if not mock mode).
        mock_mode: If True, return a mock engine.
        timeout: Request timeout in seconds.

    Returns:
        AnalysisEngine instance.

    Raises:
        ValueError: If api_key is missing and not in mock mode.
    """
    if mock_mode:
        return MockAnalysisEngine()

    if not api_key:
        raise ValueError("API key is required when not in mock mode")

    return PerplexityAnalysisEngine(api_key=api_key, timeout=timeout)
