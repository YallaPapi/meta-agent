"""Analysis engine for running LLM analysis on codebases."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        responses: Optional[dict[str, AnalysisResult]] = None,
        done_after_iterations: int = 3,
    ):
        """Initialize mock engine with optional predefined responses.

        Args:
            responses: Optional dict mapping prompt substrings to responses.
            done_after_iterations: Number of triage calls before returning done=true.
        """
        self.responses = responses or {}
        self.call_count = 0
        self.triage_count = 0
        self.done_after_iterations = done_after_iterations
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

        # Handle triage prompts specially - check for the actual triage prompt markers
        # Be specific to avoid matching history that contains previous triage results
        if "codebase triage and prompt selection" in prompt.lower() or \
           "determine which analysis prompts should be run next" in prompt.lower():
            return self._mock_triage_response()

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

    def _mock_triage_response(self) -> AnalysisResult:
        """Return a mock triage response.

        Returns:
            AnalysisResult with triage JSON.
        """
        self.triage_count += 1

        # Cycle through different prompts for variety
        prompt_sets = [
            ["quality_error_analysis", "architecture_layer_identification"],
            ["quality_code_complexity_analysis", "testing_unit_test_generation"],
            ["improvement_best_practice_analysis"],
        ]

        # After enough iterations, say we're done
        if self.triage_count >= self.done_after_iterations:
            triage_data = {
                "assessment": "The codebase meets PRD requirements and is production-ready.",
                "priority_issues": [],
                "selected_prompts": [],
                "reasoning": "All major issues have been addressed in previous iterations.",
                "done": True,
            }
        else:
            prompt_index = (self.triage_count - 1) % len(prompt_sets)
            selected = prompt_sets[prompt_index]

            triage_data = {
                "assessment": f"Mock triage iteration {self.triage_count}: Found areas needing improvement.",
                "priority_issues": [
                    f"Mock issue {self.triage_count}.1: Code quality needs review",
                    f"Mock issue {self.triage_count}.2: Architecture could be improved",
                ],
                "selected_prompts": selected,
                "reasoning": f"Selected {len(selected)} prompts for mock iteration {self.triage_count}.",
                "done": False,
            }

        return AnalysisResult(
            summary=json.dumps(triage_data),
            recommendations=[],
            tasks=[],
            raw_response=json.dumps(triage_data),
            success=True,
        )


class PerplexityAnalysisEngine(AnalysisEngine):
    """Analysis engine using Perplexity API."""

    API_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: str, timeout: int = 120, model: str = "sonar-pro"):
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
            error_body = ""
            try:
                error_body = e.response.text
            except Exception:
                pass
            return AnalysisResult(
                summary="",
                success=False,
                error=f"HTTP error from Perplexity API: {e.response.status_code} - {error_body}",
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

        Uses multiple extraction strategies for robustness.

        Args:
            content: Raw response content from the LLM.

        Returns:
            Parsed AnalysisResult.
        """
        json_str = None

        # Strategy 1: Look for ```json ... ``` block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug("Found JSON in code block")

        # Strategy 2: Look for raw JSON object (first { to last })
        if not json_str:
            brace_match = re.search(r'(\{[\s\S]*\})', content)
            if brace_match:
                json_str = brace_match.group(1)
                logger.debug("Found JSON by brace matching")

        if json_str:
            try:
                # Clean up common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # trailing commas before }
                json_str = re.sub(r',\s*]', ']', json_str)  # trailing commas before ]

                data = json.loads(json_str)
                return AnalysisResult(
                    summary=data.get("summary", ""),
                    recommendations=data.get("recommendations", []),
                    tasks=self._normalize_tasks(data.get("tasks", [])),
                    raw_response=content,
                    success=True,
                )
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}")

        # Fallback: Create structured result from raw text
        logger.debug("Using fallback text parsing")
        return self._create_fallback_result(content)

    def _normalize_tasks(self, tasks: list) -> list[dict[str, Any]]:
        """Ensure tasks have required fields.

        Args:
            tasks: Raw task list from JSON.

        Returns:
            Normalized task list with all required fields.
        """
        normalized = []
        for task in tasks:
            if isinstance(task, dict):
                normalized.append({
                    'title': task.get('title', 'Untitled task'),
                    'description': task.get('description', str(task)),
                    'priority': task.get('priority', 'medium').lower(),
                    'file': task.get('file', ''),
                })
            elif isinstance(task, str):
                normalized.append({
                    'title': task[:80] + '...' if len(task) > 80 else task,
                    'description': task,
                    'priority': 'medium',
                    'file': '',
                })
        return normalized

    def _create_fallback_result(self, content: str) -> AnalysisResult:
        """Create a structured result from unstructured text.

        Args:
            content: Raw response content.

        Returns:
            AnalysisResult with extracted information.
        """
        # Extract bullet points as potential tasks
        tasks = []
        bullet_pattern = re.compile(r'^\s*[-*\u2022]\s+(.+)$', re.MULTILINE)
        for match in bullet_pattern.finditer(content):
            text = match.group(1).strip()
            if len(text) > 10:  # Skip very short bullets
                tasks.append({
                    'title': text[:80] + '...' if len(text) > 80 else text,
                    'description': text,
                    'priority': 'medium',
                    'file': '',
                })

        # Also look for numbered items
        numbered_pattern = re.compile(r'^\s*\d+[.)]\s+(.+)$', re.MULTILINE)
        for match in numbered_pattern.finditer(content):
            text = match.group(1).strip()
            if len(text) > 10:
                tasks.append({
                    'title': text[:80] + '...' if len(text) > 80 else text,
                    'description': text,
                    'priority': 'medium',
                    'file': '',
                })

        # Use first paragraph as summary
        paragraphs = content.split('\n\n')
        summary = paragraphs[0][:500] if paragraphs else content[:500]

        # Extract recommendations from sections that mention "recommend"
        recommendations = []
        recommend_pattern = re.compile(r'recommend[^\n]*:?\s*([^\n]+)', re.IGNORECASE)
        for match in recommend_pattern.finditer(content):
            rec = match.group(1).strip()
            if rec and len(rec) > 5:
                recommendations.append(rec)

        return AnalysisResult(
            summary=summary,
            recommendations=recommendations[:10],  # Limit recommendations
            tasks=tasks[:10],  # Limit to 10 tasks
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
