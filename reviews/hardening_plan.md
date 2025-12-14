# LLM Pipeline Hardening Plan

**Project:** Meta-Agent for Automated Codebase Refinement  
**Focus:** Making the LLM-driven analysis pipeline robust and production-worthy  
**Date:** 2024-12-14

---

## Executive Summary

The meta-agent's LLM pipeline has functional error handling for network failures but lacks robustness in JSON response parsing, retry logic, and structured logging. The `_parse_response()` method silently falls back to treating malformed JSON as a summary (marking it as `success=True`), while the orchestrator's triage parsing uses fragile substring extraction. There are no retry mechanisms for transient failures, no validation of parsed response structures, and potential for secret leakage in error logs.

**Key hardening steps:**
1. Add retry with exponential backoff
2. Implement strict JSON schema validation with explicit partial-failure states
3. Add structured logging with sanitized context
4. Make timeout/retry behavior configurable

---

## Findings

### Critical Severity

#### 1. Malformed JSON Silently Marked as Success
- **File:** `src/metaagent/analysis.py`
- **Location:** `PerplexityAnalysisEngine._parse_response()` (lines 6138-6171)
- **Issue:** The method treats ALL JSON parse failures as success by falling back to raw content as summary. This masks genuine failures and provides no way for the orchestrator to distinguish between valid non-JSON responses and malformed structured data. The fallback returns `success=True` even when the LLM failed to follow the expected format.

```python
# Current problematic code:
except json.JSONDecodeError:
    # Fallback: treat the whole response as summary
    return AnalysisResult(
        summary=content,
        recommendations=[],
        tasks=[],
        raw_response=content,
        success=True,  # <-- THIS IS WRONG
    )
```

---

### High Severity

#### 2. No Retry Logic for Transient HTTP Errors
- **File:** `src/metaagent/analysis.py`
- **Location:** `PerplexityAnalysisEngine.analyze()` (lines 6081-6136)
- **Issue:** No retry logic exists for transient HTTP errors (5xx, 429 rate limits, connection resets). A single network hiccup fails the entire stage with no recovery attempt. The httpx client is configured with a timeout but no retry adapter.

#### 3. Fragile Triage JSON Extraction
- **File:** `src/metaagent/orchestrator.py`
- **Location:** `Orchestrator._run_triage()` (lines 7171-7242)
- **Issue:** The method extracts JSON using naive string slicing (`first '{'` to `last '}'`) which fails on responses with multiple JSON objects, nested strings containing braces, or markdown code blocks. There's also no schema validation of the extracted JSON structure.

```python
# Current problematic code:
json_start = response_text.find("{")
json_end = response_text.rfind("}") + 1

if json_start != -1 and json_end > json_start:
    json_str = response_text[json_start:json_end]  # <-- FRAGILE
    data = json.loads(json_str)
```

#### 4. No Response Schema Validation
- **Files:** `src/metaagent/analysis.py`, `src/metaagent/orchestrator.py`
- **Issue:** No validation that parsed JSON responses contain required fields (summary, recommendations, tasks). The code uses `.get()` with empty defaults but never validates that critical fields like `tasks` have the expected list-of-dict structure, which could cause downstream TypeErrors.

---

### Medium Severity

#### 5. Potential Secret Leakage in Error Messages
- **File:** `src/metaagent/analysis.py`
- **Issue:** Error messages in `AnalysisResult` can include raw exception strings which may leak sensitive context. The error field captures `str(e)` which could include partial API responses, headers, or request details.

#### 6. No Partial-Success State
- **File:** `src/metaagent/orchestrator.py`
- **Issue:** The orchestrator continues running stages after failures but uses a simple boolean success determination (`stages_failed == 0`). There's no partial-success state, no per-stage retry budget, and no way to mark a run as "completed with warnings". A single failed stage makes the entire run report as failed even if 5/6 stages succeeded.

#### 7. Silent Exception Swallowing in Prompt Loading
- **File:** `src/metaagent/prompts.py`
- **Location:** `PromptLibrary._parse_markdown_prompt()` (lines 7757-7801)
- **Issue:** Returns `None` on any exception without logging. This makes debugging prompt loading issues extremely difficult as errors are completely hidden.

```python
# Current problematic code:
except Exception:
    return None  # <-- SILENT FAILURE
```

#### 8. Missing Configuration for Hardening Parameters
- **File:** `src/metaagent/config.py`
- **Issue:** Config lacks parameters for retry behavior (max_retries, backoff_factor, retry_status_codes) and response validation (require_json, strict_schema). All hardening behaviors would need code changes rather than configuration.

---

### Low Severity

#### 9. No Structured Logging with Correlation IDs
- **Files:** `src/metaagent/analysis.py`, `src/metaagent/orchestrator.py`
- **Issue:** Current logging uses basic `logger.info/error` without request IDs, making it difficult to trace which LLM call corresponds to which stage in logs.

#### 10. HTTP Client Connection Leak
- **File:** `src/metaagent/analysis.py`
- **Issue:** The `httpx.Client` is instantiated in `__init__` but never explicitly closed. Long-running processes could leak connections.

---

## Implementation Tasks

Execute these tasks in order. Each task includes the specific files to modify and implementation guidance.

---

### Task 1: Add Retry Logic with Exponential Backoff (MUST-HAVE)

**Files to modify:**
- `src/metaagent/analysis.py`
- `src/metaagent/config.py`

**Implementation:**

1. Add new config fields to `Config` dataclass in `config.py`:
```python
# Add to Config dataclass
retry_max_attempts: int = 3
retry_backoff_base: float = 1.0
retry_backoff_max: float = 30.0
```

2. Update `Config.from_env()` to load these values:
```python
retry_max_attempts=int(os.getenv("METAAGENT_RETRY_MAX_ATTEMPTS", "3")),
retry_backoff_base=float(os.getenv("METAAGENT_RETRY_BACKOFF_BASE", "1.0")),
retry_backoff_max=float(os.getenv("METAAGENT_RETRY_BACKOFF_MAX", "30.0")),
```

3. In `analysis.py`, add retry logic to `PerplexityAnalysisEngine`:
```python
import time
import random

def __init__(self, api_key: str, timeout: int = 120, model: str = "...",
             max_retries: int = 3, backoff_base: float = 1.0, backoff_max: float = 30.0):
    # ... existing init ...
    self.max_retries = max_retries
    self.backoff_base = backoff_base
    self.backoff_max = backoff_max

def _should_retry(self, status_code: int) -> bool:
    """Determine if request should be retried based on status code."""
    return status_code in (429, 500, 502, 503, 504)

def _get_backoff_time(self, attempt: int, retry_after: Optional[int] = None) -> float:
    """Calculate backoff time with exponential backoff and jitter."""
    if retry_after:
        return float(retry_after)
    backoff = min(self.backoff_base * (2 ** attempt) + random.uniform(0, 1), self.backoff_max)
    return backoff

def analyze(self, prompt: str) -> AnalysisResult:
    last_error = None
    
    for attempt in range(self.max_retries + 1):
        try:
            response = self.client.post(...)
            response.raise_for_status()
            # ... success handling ...
            
        except httpx.HTTPStatusError as e:
            if self._should_retry(e.response.status_code) and attempt < self.max_retries:
                retry_after = e.response.headers.get("Retry-After")
                backoff = self._get_backoff_time(attempt, int(retry_after) if retry_after else None)
                logger.warning(f"Retrying after {backoff:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(backoff)
                continue
            last_error = e
            break
        except httpx.TimeoutException as e:
            if attempt < self.max_retries:
                backoff = self._get_backoff_time(attempt)
                logger.warning(f"Timeout, retrying after {backoff:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(backoff)
                continue
            last_error = e
            break
    
    # Return error result if all retries exhausted
    return AnalysisResult(success=False, error=self._sanitize_error(str(last_error)), ...)
```

4. Update `create_analysis_engine()` factory to accept retry parameters.

---

### Task 2: Implement Strict JSON Parsing with Schema Validation (MUST-HAVE)

**Files to modify:**
- `src/metaagent/analysis.py`
- `src/metaagent/config.py`

**Implementation:**

1. Add config field:
```python
strict_json_mode: bool = True
```

2. Create validation function in `analysis.py`:
```python
def validate_analysis_response(data: dict) -> tuple[bool, str]:
    """Validate that response has required structure.
    
    Returns:
        Tuple of (is_valid, error_message)
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
```

3. Create robust JSON extraction function:
```python
def extract_json_from_response(content: str) -> tuple[Optional[dict], str]:
    """Extract JSON from LLM response.
    
    Tries multiple strategies:
    1. Parse entire response as JSON
    2. Extract from ```json code blocks
    3. Find balanced braces
    
    Returns:
        Tuple of (parsed_dict or None, error_message)
    """
    # Strategy 1: Try parsing entire response
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
    
    # Strategy 3: Find outermost balanced braces
    # (implement proper brace matching that handles strings)
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
                        return json.loads(content[start:i+1]), ""
                    except json.JSONDecodeError:
                        break
    except ValueError:
        pass
    
    return None, "Could not extract valid JSON from response"
```

4. Update `_parse_response()`:
```python
def _parse_response(self, content: str, strict_mode: bool = True) -> AnalysisResult:
    data, extract_error = extract_json_from_response(content)
    
    if data is None:
        if strict_mode:
            return AnalysisResult(
                summary="",
                success=False,
                error=f"JSON extraction failed: {extract_error}",
                raw_response=content,
            )
        else:
            # Legacy fallback behavior
            return AnalysisResult(
                summary=content,
                raw_response=content,
                success=True,
            )
    
    is_valid, validation_error = validate_analysis_response(data)
    if not is_valid:
        if strict_mode:
            return AnalysisResult(
                summary="",
                success=False,
                error=f"Schema validation failed: {validation_error}",
                raw_response=content,
            )
    
    return AnalysisResult(
        summary=data.get("summary", ""),
        recommendations=data.get("recommendations", []),
        tasks=data.get("tasks", []),
        raw_response=content,
        success=True,
    )
```

---

### Task 3: Harden Triage JSON Extraction (MUST-HAVE)

**Files to modify:**
- `src/metaagent/orchestrator.py`

**Implementation:**

1. Import and use the shared `extract_json_from_response()` function from analysis.py (or create a shared utils module).

2. Replace `_run_triage()` JSON extraction:
```python
def _run_triage(self, prd_content: str, code_context: str, history: RunHistory) -> TriageResult:
    # ... existing prompt rendering and API call ...
    
    if not analysis_result.success:
        return TriageResult(success=False, error=analysis_result.error)
    
    response_text = analysis_result.raw_response or analysis_result.summary
    
    # Use robust JSON extraction
    data, extract_error = extract_json_from_response(response_text)
    
    if data is None:
        # Check for plain text "done" indicator as fallback
        if "done" in response_text.lower() and "no further" in response_text.lower():
            return TriageResult(success=True, done=True, assessment=response_text)
        
        return TriageResult(
            success=False,
            error=f"Could not parse triage response: {extract_error}",
        )
    
    # Validate required triage fields
    if "selected_prompts" not in data and not data.get("done", False):
        return TriageResult(
            success=False,
            error="Triage response missing 'selected_prompts' field",
        )
    
    return TriageResult(
        success=True,
        done=data.get("done", False),
        assessment=data.get("assessment", ""),
        priority_issues=data.get("priority_issues", []),
        selected_prompts=data.get("selected_prompts", []),
        reasoning=data.get("reasoning", ""),
    )
```

---

### Task 4: Add Comprehensive Hardening Tests (MUST-HAVE)

**Files to create/modify:**
- `tests/test_hardening.py` (new file)
- `tests/test_analysis.py` (add tests)

**Implementation:**

Create `tests/test_hardening.py`:
```python
"""Tests for pipeline hardening: retries, JSON parsing, error handling."""

import json
import pytest
import httpx
from unittest.mock import patch, MagicMock

from metaagent.analysis import (
    PerplexityAnalysisEngine,
    extract_json_from_response,
    validate_analysis_response,
)


class TestJSONExtraction:
    """Tests for robust JSON extraction."""
    
    def test_extract_plain_json(self):
        content = '{"summary": "test", "tasks": []}'
        data, error = extract_json_from_response(content)
        assert data is not None
        assert data["summary"] == "test"
    
    def test_extract_json_from_markdown_block(self):
        content = '''Here's my analysis:
        
```json
{"summary": "test", "tasks": []}
```

Additional notes.'''
        data, error = extract_json_from_response(content)
        assert data is not None
        assert data["summary"] == "test"
    
    def test_extract_json_with_nested_braces(self):
        content = 'Intro {"summary": "has {braces} inside", "tasks": [{"nested": true}]} outro'
        data, error = extract_json_from_response(content)
        assert data is not None
        assert "braces" in data["summary"]
    
    def test_extract_json_with_braces_in_strings(self):
        content = '{"summary": "code: if (x) { return; }", "tasks": []}'
        data, error = extract_json_from_response(content)
        assert data is not None
    
    def test_extract_invalid_json_returns_none(self):
        content = "This is not JSON at all"
        data, error = extract_json_from_response(content)
        assert data is None
        assert error != ""


class TestSchemaValidation:
    """Tests for response schema validation."""
    
    def test_valid_response(self):
        data = {"summary": "test", "recommendations": ["a"], "tasks": [{"title": "t"}]}
        is_valid, error = validate_analysis_response(data)
        assert is_valid
    
    def test_missing_summary(self):
        data = {"recommendations": [], "tasks": []}
        is_valid, error = validate_analysis_response(data)
        assert not is_valid
        assert "summary" in error
    
    def test_invalid_tasks_type(self):
        data = {"summary": "test", "tasks": "not a list"}
        is_valid, error = validate_analysis_response(data)
        assert not is_valid
        assert "tasks" in error
    
    def test_invalid_task_item(self):
        data = {"summary": "test", "tasks": ["string instead of dict"]}
        is_valid, error = validate_analysis_response(data)
        assert not is_valid


class TestRetryBehavior:
    """Tests for retry logic."""
    
    @patch("metaagent.analysis.httpx.Client")
    def test_retries_on_500(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # First call returns 500, second succeeds
        error_response = MagicMock()
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=error_response
        )
        
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "ok", "tasks": []}'}}]
        }
        success_response.raise_for_status.return_value = None
        
        mock_client.post.side_effect = [error_response, success_response]
        
        engine = PerplexityAnalysisEngine(api_key="test", max_retries=2, backoff_base=0.01)
        result = engine.analyze("test prompt")
        
        assert result.success
        assert mock_client.post.call_count == 2
    
    @patch("metaagent.analysis.httpx.Client")
    def test_no_retry_on_400(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        error_response = MagicMock()
        error_response.status_code = 400
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=error_response
        )
        
        mock_client.post.return_value = error_response
        
        engine = PerplexityAnalysisEngine(api_key="test", max_retries=3)
        result = engine.analyze("test prompt")
        
        assert not result.success
        assert mock_client.post.call_count == 1  # No retries


class TestErrorSanitization:
    """Tests for secret sanitization in errors."""
    
    def test_api_key_redacted(self):
        # Implementation depends on sanitize_error function
        pass
    
    def test_bearer_token_redacted(self):
        pass
```

Add to `tests/test_analysis.py`:
```python
def test_parse_response_strict_mode_fails_on_invalid_json(self):
    """Test that strict mode returns success=False for invalid JSON."""
    engine = PerplexityAnalysisEngine(api_key="test")
    result = engine._parse_response("Not valid JSON", strict_mode=True)
    
    assert result.success is False
    assert "JSON" in result.error

def test_parse_response_validates_schema(self):
    """Test that schema validation catches missing fields."""
    engine = PerplexityAnalysisEngine(api_key="test")
    result = engine._parse_response('{"not_summary": "wrong field"}', strict_mode=True)
    
    assert result.success is False
    assert "summary" in result.error.lower()
```

---

### Task 5: Add Partial-Success State to Orchestrator (SHOULD-HAVE)

**Files to modify:**
- `src/metaagent/orchestrator.py`
- `src/metaagent/cli.py`

**Implementation:**

1. Update `RefinementResult` dataclass:
```python
@dataclass
class RefinementResult:
    success: bool
    partial_success: bool = False  # New field
    profile_name: str
    stages_completed: int
    stages_failed: int
    warnings: list[str] = field(default_factory=list)  # New field
    plan_path: Optional[Path] = None
    error: Optional[str] = None
    stage_results: list[StageResult] = field(default_factory=list)
    iterations: list[IterationResult] = field(default_factory=list)
```

2. Update success logic in `refine()`:
```python
return RefinementResult(
    success=stages_failed == 0 and stages_completed > 0,
    partial_success=stages_failed > 0 and stages_completed > 0,
    # ... rest of fields
)
```

3. Update CLI to display partial success differently:
```python
if result.partial_success:
    console.print(f"[yellow]⚠ Completed with warnings: {result.stages_completed} succeeded, {result.stages_failed} failed[/yellow]")
elif result.success:
    console.print(f"[green]✓ All {result.stages_completed} stages completed successfully[/green]")
else:
    console.print(f"[red]✗ Failed: {result.error}[/red]")
```

---

### Task 6: Implement Structured Logging with Correlation IDs (SHOULD-HAVE)

**Files to modify:**
- `src/metaagent/orchestrator.py`
- `src/metaagent/analysis.py`
- `src/metaagent/cli.py`

**Implementation:**

1. Create logging utilities:
```python
# In a new file src/metaagent/logging_utils.py or at top of orchestrator.py
import uuid
import logging
from contextvars import ContextVar

correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id.get() or '-'
        return True

def setup_logging(log_level: str = "INFO", json_format: bool = False):
    formatter = logging.Formatter(
        '%(asctime)s [%(correlation_id)s] %(levelname)s %(name)s: %(message)s'
    )
    # ... configure handlers with formatter and filter
```

2. Set correlation ID at start of `refine()`:
```python
def refine(self, max_iterations: int = 10) -> RefinementResult:
    run_id = str(uuid.uuid4())[:8]
    correlation_id.set(run_id)
    logger.info(f"Starting refinement run {run_id}")
    # ... rest of method
```

---

### Task 7: Sanitize Error Messages (SHOULD-HAVE)

**Files to modify:**
- `src/metaagent/analysis.py`

**Implementation:**

```python
import re

def sanitize_error(error: str) -> str:
    """Remove potential secrets from error messages."""
    # Remove Bearer tokens
    error = re.sub(r'Bearer\s+[A-Za-z0-9\-_]+', 'Bearer [REDACTED]', error)
    
    # Remove API key patterns
    error = re.sub(r'(api[_-]?key["\s:=]+)[A-Za-z0-9\-_]+', r'\1[REDACTED]', error, flags=re.I)
    error = re.sub(r'(key["\s:=]+)[A-Za-z0-9]{20,}', r'\1[REDACTED]', error, flags=re.I)
    
    # Truncate long errors
    if len(error) > 500:
        error = error[:500] + "... [truncated]"
    
    return error
```

Apply in all except blocks:
```python
except Exception as e:
    return AnalysisResult(
        summary="",
        success=False,
        error=sanitize_error(f"Unexpected error: {e}"),
    )
```

---

### Task 8: Add Logging to Prompt Library Loading (SHOULD-HAVE)

**Files to modify:**
- `src/metaagent/prompts.py`

**Implementation:**

```python
import logging

logger = logging.getLogger(__name__)

def _parse_markdown_prompt(self, file_path: Path) -> Optional[Prompt]:
    try:
        content = file_path.read_text(encoding="utf-8")
        # ... parsing logic ...
        return Prompt(...)
    except Exception as e:
        logger.warning(f"Failed to parse prompt file {file_path}: {e}")
        return None
```

---

### Task 9: Add Configuration for All Hardening Parameters (SHOULD-HAVE)

**Files to modify:**
- `src/metaagent/config.py`
- `.env.example`

**Implementation:**

1. Update `Config` dataclass with all new fields:
```python
@dataclass
class Config:
    # ... existing fields ...
    
    # Retry settings
    retry_max_attempts: int = 3
    retry_backoff_base: float = 1.0
    retry_backoff_max: float = 30.0
    
    # Validation settings
    strict_json_mode: bool = True
    stage_max_retries: int = 2
    
    # Logging settings
    log_format: str = "text"  # "text" or "json"
```

2. Update `.env.example`:
```bash
# Retry Configuration
METAAGENT_RETRY_MAX_ATTEMPTS=3
METAAGENT_RETRY_BACKOFF_BASE=1.0
METAAGENT_RETRY_BACKOFF_MAX=30.0

# Validation
METAAGENT_STRICT_JSON_MODE=true
METAAGENT_STAGE_MAX_RETRIES=2

# Logging
METAAGENT_LOG_FORMAT=text
```

---

### Task 10: Add HTTP Client Lifecycle Management (NICE-TO-HAVE)

**Files to modify:**
- `src/metaagent/analysis.py`
- `src/metaagent/orchestrator.py`

**Implementation:**

```python
class PerplexityAnalysisEngine(AnalysisEngine):
    def __init__(self, ...):
        # ... existing init ...
        self._client: Optional[httpx.Client] = None
    
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client
    
    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
```

Update orchestrator to close engine:
```python
def refine(self, ...) -> RefinementResult:
    try:
        # ... existing logic ...
    finally:
        if hasattr(self.analysis_engine, 'close'):
            self.analysis_engine.close()
```

---

## Execution Order

1. **Task 1** - Retry logic (foundation for reliability)
2. **Task 2** - JSON parsing/validation (fixes critical silent failure bug)
3. **Task 3** - Triage hardening (uses Task 2's utilities)
4. **Task 4** - Tests (validates Tasks 1-3)
5. **Task 9** - Configuration (makes all above configurable)
6. **Task 5** - Partial success state
7. **Task 7** - Error sanitization
8. **Task 8** - Prompt loading logs
9. **Task 6** - Structured logging
10. **Task 10** - Client lifecycle

---

## Verification

After implementing all tasks, run:

```bash
# Run all tests including new hardening tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=metaagent --cov-report=term-missing

# Test in mock mode to verify error handling
metaagent refine --mock --verbose

# Test retry behavior (requires network)
METAAGENT_LOG_LEVEL=DEBUG metaagent refine --profile quick_review
```
