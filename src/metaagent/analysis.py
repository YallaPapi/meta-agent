"""Analysis engine for running LLM analysis on codebases."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Patterns for sensitive data that should be masked in error messages
SENSITIVE_PATTERNS = [
    (re.compile(r'(Bearer\s+)[A-Za-z0-9_-]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(api[_-]?key["\s:=]+)[A-Za-z0-9_-]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(Authorization["\s:=]+)[A-Za-z0-9_-]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(pplx-)[A-Za-z0-9]+'), r'\1[REDACTED]'),  # Perplexity API key format
    (re.compile(r'(sk-)[A-Za-z0-9]+'), r'\1[REDACTED]'),  # OpenAI/Anthropic key format
]


def sanitize_error(message: str) -> str:
    """Sanitize error messages to remove sensitive data.

    Masks API keys, bearer tokens, and other sensitive patterns
    to prevent them from being logged or shown to users.

    Args:
        message: Error message that may contain sensitive data.

    Returns:
        Sanitized message with sensitive data masked.
    """
    result = message
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


@dataclass
class AnalysisResult:
    """Result from an analysis stage."""

    summary: str
    recommendations: list[str] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    success: bool = True
    error: Optional[str] = None


def extract_json_from_response(content: str) -> tuple[Optional[dict], str]:
    """Extract JSON from LLM response using multiple strategies.

    Tries multiple strategies in order:
    1. Parse entire response as JSON
    2. Extract from ```json code blocks
    3. Find balanced braces (handles nested objects and strings)

    Args:
        content: Raw response content from LLM.

    Returns:
        Tuple of (parsed_dict or None, error_message).
    """
    if not content or not content.strip():
        return None, "Empty response content"

    # Strategy 1: Try parsing entire response as JSON
    try:
        return json.loads(content), ""
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    json_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1)), ""
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find balanced braces (handles nested objects and strings)
    try:
        start = content.index('{')
        depth = 0
        in_string = False
        escape = False

        for i, char in enumerate(content[start:], start):
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        json_str = content[start:i + 1]
                        # Clean up common JSON issues
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        return json.loads(json_str), ""
                    except json.JSONDecodeError:
                        break
    except ValueError:
        pass

    return None, "Could not extract valid JSON from response"


def validate_analysis_response(data: dict) -> tuple[bool, str]:
    """Validate that response has required structure.

    Checks:
    - Response is a dict
    - Has 'summary' field (string)
    - 'recommendations' is a list if present
    - 'tasks' is a list of dicts if present

    Args:
        data: Parsed JSON data to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Response is not a JSON object"

    if "summary" not in data:
        return False, "Missing required field: summary"

    if not isinstance(data.get("summary", ""), str):
        return False, "Field 'summary' must be a string"

    if "recommendations" in data and not isinstance(data["recommendations"], list):
        return False, "Field 'recommendations' must be a list"

    if "tasks" in data:
        if not isinstance(data["tasks"], list):
            return False, "Field 'tasks' must be a list"
        for i, task in enumerate(data["tasks"]):
            if not isinstance(task, dict):
                return False, f"Task at index {i} must be an object"

    return True, ""


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

    # Status codes eligible for retry (transient errors)
    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        api_key: str,
        timeout: int = 120,
        model: str = "sonar-pro",
        retry_max_attempts: int = 3,
        retry_backoff_base: float = 2.0,
        retry_backoff_max: float = 60.0,
    ):
        """Initialize Perplexity analysis engine.

        Args:
            api_key: Perplexity API key.
            timeout: Request timeout in seconds.
            model: Model to use for analysis.
            retry_max_attempts: Maximum number of retry attempts for transient errors.
            retry_backoff_base: Base seconds for exponential backoff.
            retry_backoff_max: Maximum backoff time in seconds.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.model = model
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_max = retry_backoff_max
        self.client = httpx.Client(timeout=timeout)

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error is transient and should be retried.

        Args:
            error: The exception that occurred.
            attempt: Current attempt number (1-indexed).

        Returns:
            True if the error is transient and retry is allowed.
        """
        if attempt >= self.retry_max_attempts:
            return False

        # Timeout errors are always retryable
        if isinstance(error, httpx.TimeoutException):
            return True

        # HTTP errors - check for retryable status codes
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in self.RETRYABLE_STATUS_CODES

        # Connection errors are retryable
        if isinstance(error, (httpx.ConnectError, httpx.RemoteProtocolError)):
            return True

        return False

    def _get_backoff_time(self, attempt: int, retry_after: Optional[int] = None) -> float:
        """Calculate exponential backoff time with jitter.

        Uses exponential backoff with full jitter strategy:
        backoff = random(0, min(cap, base * 2^attempt))

        Args:
            attempt: Current attempt number (1-indexed).
            retry_after: Optional Retry-After header value from server.

        Returns:
            Backoff time in seconds.
        """
        if retry_after is not None:
            return float(retry_after)

        exp_backoff = self.retry_backoff_base * (2 ** (attempt - 1))
        capped_backoff = min(exp_backoff, self.retry_backoff_max)
        # Add jitter: random value between 0 and capped_backoff
        return random.uniform(0, capped_backoff)

    def analyze(self, prompt: str) -> AnalysisResult:
        """Run analysis using Perplexity API with retry logic.

        Args:
            prompt: The rendered prompt to send.

        Returns:
            AnalysisResult with the analysis output.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_max_attempts + 1):
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

            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError,
                    httpx.RemoteProtocolError) as e:
                last_error = e

                if self._should_retry(e, attempt):
                    # Extract Retry-After header if present
                    retry_after = None
                    if isinstance(e, httpx.HTTPStatusError):
                        retry_after_str = e.response.headers.get("Retry-After")
                        if retry_after_str:
                            try:
                                retry_after = int(retry_after_str)
                            except ValueError:
                                pass

                    backoff = self._get_backoff_time(attempt, retry_after)
                    error_type = type(e).__name__
                    status_info = ""
                    if isinstance(e, httpx.HTTPStatusError):
                        status_info = f" (status {e.response.status_code})"

                    logger.warning(
                        f"Transient error{status_info} on attempt {attempt}/{self.retry_max_attempts}: "
                        f"{error_type}. Retrying in {backoff:.2f}s..."
                    )
                    time.sleep(backoff)
                    continue

                # Non-retryable error or max attempts reached
                break

            except Exception as e:
                # Unexpected errors are not retried - sanitize to avoid leaking secrets
                return AnalysisResult(
                    summary="",
                    success=False,
                    error=sanitize_error(f"Unexpected error calling Perplexity API: {e}"),
                )

        # Handle the final error after all retries exhausted
        if isinstance(last_error, httpx.HTTPStatusError):
            error_body = ""
            try:
                error_body = last_error.response.text
            except Exception:
                pass
            # Sanitize error body to prevent API key leakage
            sanitized_body = sanitize_error(error_body)
            return AnalysisResult(
                summary="",
                success=False,
                error=f"HTTP error from Perplexity API after {self.retry_max_attempts} attempts: "
                      f"{last_error.response.status_code} - {sanitized_body}",
                raw_response=sanitize_error(str(last_error)),
            )
        elif isinstance(last_error, httpx.TimeoutException):
            return AnalysisResult(
                summary="",
                success=False,
                error=f"Request to Perplexity API timed out after {self.retry_max_attempts} attempts "
                      f"(timeout: {self.timeout}s)",
            )
        else:
            return AnalysisResult(
                summary="",
                success=False,
                error=sanitize_error(f"Error calling Perplexity API after {self.retry_max_attempts} attempts: {last_error}"),
            )

    def _parse_response(self, content: str, strict_mode: bool = True) -> AnalysisResult:
        """Parse the LLM response into structured result.

        Uses the shared extract_json_from_response() function with optional
        schema validation in strict mode.

        Args:
            content: Raw response content from the LLM.
            strict_mode: If True, return failure for invalid JSON/schema.
                        If False, fall back to text parsing.

        Returns:
            Parsed AnalysisResult.
        """
        # Use the shared JSON extraction function
        data, extract_error = extract_json_from_response(content)

        if data is None:
            if strict_mode:
                logger.warning(f"JSON extraction failed: {extract_error}")
                return AnalysisResult(
                    summary="",
                    success=False,
                    error=f"JSON extraction failed: {extract_error}",
                    raw_response=content,
                )
            else:
                # Legacy fallback behavior
                logger.debug("Using fallback text parsing")
                return self._create_fallback_result(content)

        # Validate the response schema
        is_valid, validation_error = validate_analysis_response(data)
        if not is_valid:
            if strict_mode:
                logger.warning(f"Schema validation failed: {validation_error}")
                return AnalysisResult(
                    summary="",
                    success=False,
                    error=f"Schema validation failed: {validation_error}",
                    raw_response=content,
                )
            # In non-strict mode, continue with partial data

        return AnalysisResult(
            summary=data.get("summary", ""),
            recommendations=data.get("recommendations", []),
            tasks=self._normalize_tasks(data.get("tasks", [])),
            raw_response=content,
            success=True,
        )

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
    retry_max_attempts: int = 3,
    retry_backoff_base: float = 2.0,
    retry_backoff_max: float = 60.0,
) -> AnalysisEngine:
    """Factory function to create the appropriate analysis engine.

    Args:
        api_key: Perplexity API key (required if not mock mode).
        mock_mode: If True, return a mock engine.
        timeout: Request timeout in seconds.
        retry_max_attempts: Maximum number of retry attempts for transient errors.
        retry_backoff_base: Base seconds for exponential backoff.
        retry_backoff_max: Maximum backoff time in seconds.

    Returns:
        AnalysisEngine instance.

    Raises:
        ValueError: If api_key is missing and not in mock mode.
    """
    if mock_mode:
        return MockAnalysisEngine()

    if not api_key:
        raise ValueError("API key is required when not in mock mode")

    return PerplexityAnalysisEngine(
        api_key=api_key,
        timeout=timeout,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_base=retry_backoff_base,
        retry_backoff_max=retry_backoff_max,
    )
