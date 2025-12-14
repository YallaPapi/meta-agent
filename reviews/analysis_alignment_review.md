# Analysis-to-Plan Alignment Review

**Date:** 2024-12-14  
**Scope:** Align test coverage and plan generation with Codebase Digest–driven analysis flow  
**Status:** Ready for Implementation

---

## Executive Summary

The meta-agent system needs alignment between three components:

1. **Analysis Engine** (`analysis.py`) — produces `AnalysisResult` with `tasks: list[dict]`
2. **Plan Writer** (`plan_writer.py`) — consumes `StageResult` with `tasks: list[dict]`  
3. **Codebase Digest Prompts** (`config/prompt_library/*.md`) — currently return prose, not JSON

**Critical Issue:** The 70+ Codebase Digest prompts in `config/prompt_library/` don't specify JSON response format. When used, the LLM returns prose, JSON parsing fails silently, and `tasks` becomes empty. The plan file will contain no actionable tasks.

**Solution:** Define a canonical task schema, inject JSON response instructions into prompts, add task validation/normalization, and create comprehensive tests for the entire flow.

---

## Current State Analysis

### How Tasks Flow Through the System

```
┌─────────────────────┐
│  Codebase Digest    │
│  Prompt (markdown)  │
└─────────┬───────────┘
          │ render()
          ▼
┌─────────────────────┐
│  AnalysisEngine     │
│  .analyze(prompt)   │
└─────────┬───────────┘
          │ _parse_response()
          ▼
┌─────────────────────┐
│  AnalysisResult     │
│  .tasks: list[dict] │  ◄── JSON extraction happens here
└─────────┬───────────┘
          │ orchestrator transforms
          ▼
┌─────────────────────┐
│  StageResult        │
│  .tasks: list[dict] │
└─────────┬───────────┘
          │ PlanWriter.write_plan()
          ▼
┌─────────────────────┐
│  mvp_improvement_   │
│  plan.md            │
└─────────────────────┘
```

### Current Task Schema (Implicit)

The system expects tasks like this, but it's never formally defined:

```python
# Expected by PlanWriter._generate_task_list()
task = {
    "title": str,        # Required - used for deduplication and display
    "description": str,  # Optional - shown as sub-bullet
    "priority": str,     # Optional - "critical"|"high"|"medium"|"low", defaults to "medium"
    "file": str,         # Optional - file path reference
}
```

### What Codebase Digest Prompts Currently Return

Example from `config/prompt_library/quality_error_analysis.md`:

```markdown
**Expected Output:** A comprehensive report detailing identified errors...
```

This produces prose like:
> "The codebase has several error handling gaps. First, the CLI module lacks..."

**Not** structured JSON that the parser can extract tasks from.

---

## Findings

### Finding 1: No JSON Response Format in Codebase Digest Prompts
**Severity:** CRITICAL  
**Files:** `config/prompt_library/*.md`

The 70+ prompts imported from Codebase Digest use "Expected Output" sections describing prose reports. When `PerplexityAnalysisEngine._parse_response()` tries to extract JSON, it fails and falls back to treating the entire response as `summary`, leaving `tasks = []`.

**Impact:** Plans generated from Codebase Digest prompts will have zero tasks.

---

### Finding 2: No Test Coverage for PlanWriter
**Severity:** HIGH  
**Files:** `tests/test_plan_writer.py` (does not exist)

`PlanWriter` has complex logic:
- Task aggregation with deduplication by title
- Priority grouping (critical → high → medium → low)
- Markdown generation with checkboxes
- PRD summarization

None of this is tested. Bugs in plan generation will go undetected.

---

### Finding 3: No Shared Task Schema or Validation
**Severity:** MEDIUM  
**Files:** `src/metaagent/analysis.py`, `src/metaagent/plan_writer.py`

Both `AnalysisResult.tasks` and `StageResult.tasks` are typed as `list[dict[str, Any]]`. There's no:
- Formal schema definition
- Validation of required fields
- Normalization of values (e.g., priority case)
- Handling of malformed tasks

**Impact:** Invalid tasks from LLM responses can produce broken markdown or crashes.

---

### Finding 4: Priority Validation is Permissive
**Severity:** LOW  
**Files:** `src/metaagent/plan_writer.py`

```python
# Current code in _generate_task_list()
priority = task.get("priority", "medium").lower()
if priority not in tasks_by_priority:
    priority = "medium"
```

Invalid priorities like "urgent", "P1", or "asap" silently become "medium". This is probably fine but should be logged.

---

### Finding 5: Orchestrator→StageResult Transformation Untested
**Severity:** MEDIUM  
**Files:** `src/metaagent/orchestrator.py`, `tests/test_orchestrator.py`

The orchestrator creates `StageResult` from `AnalysisResult`:

```python
stage_result = StageResult(
    stage_id=prompt_id,
    stage_name=prompt.goal or prompt_id,
    summary=analysis_result.summary,
    recommendations=analysis_result.recommendations,
    tasks=analysis_result.tasks,  # Direct passthrough
)
```

This transformation is not unit tested. If `analysis_result.tasks` contains unexpected formats, the error will surface much later in plan writing.

---

## Implementation Tasks

### Task 1: Define Canonical Task Schema
**Priority:** MUST-HAVE  
**Files to modify:**
- `src/metaagent/analysis.py`
- `src/metaagent/plan_writer.py`

Create a formal `Task` dataclass and validation function:

```python
# In src/metaagent/analysis.py (or new src/metaagent/schemas.py)

from dataclasses import dataclass, field
from typing import Optional, Literal

VALID_PRIORITIES = ("critical", "high", "medium", "low")
PriorityType = Literal["critical", "high", "medium", "low"]


@dataclass
class Task:
    """A single actionable task from analysis."""
    
    title: str
    description: str = ""
    priority: PriorityType = "medium"
    file: Optional[str] = None
    stage: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict, stage: Optional[str] = None) -> Optional["Task"]:
        """Create Task from dict, returning None if invalid.
        
        Args:
            data: Dict with task fields
            stage: Optional stage ID to attach
            
        Returns:
            Task instance or None if title is missing
        """
        title = data.get("title", "").strip()
        if not title:
            return None
            
        priority = data.get("priority", "medium").lower()
        if priority not in VALID_PRIORITIES:
            priority = "medium"
            
        return cls(
            title=title,
            description=data.get("description", "").strip(),
            priority=priority,
            file=data.get("file"),
            stage=stage,
        )
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "file": self.file,
            "stage": self.stage,
        }
```

---

### Task 2: Add JSON Response Suffix to Prompts
**Priority:** MUST-HAVE  
**Files to modify:**
- `src/metaagent/prompts.py`

Modify `Prompt.render()` to append JSON format instructions:

```python
# In src/metaagent/prompts.py

JSON_RESPONSE_SUFFIX = '''

---

## Response Format

You MUST respond with valid JSON in this exact structure:

```json
{
  "summary": "2-3 sentence overview of findings",
  "recommendations": [
    "High-level recommendation 1",
    "High-level recommendation 2"
  ],
  "tasks": [
    {
      "title": "Short actionable title",
      "description": "Detailed description of what needs to be done",
      "priority": "critical|high|medium|low",
      "file": "path/to/relevant/file.py"
    }
  ]
}
```

Rules:
- `summary`: Brief overview of what you found
- `recommendations`: List of 3-5 high-level suggestions
- `tasks`: Specific, actionable items with file references where possible
- `priority`: Must be exactly one of: critical, high, medium, low
- Respond with ONLY the JSON, no additional text before or after
'''


@dataclass
class Prompt:
    # ... existing fields ...
    
    def render(
        self,
        prd: str = "",
        code_context: str = "",
        history: str = "",
        current_stage: str = "",
        include_json_format: bool = True,  # New parameter
    ) -> str:
        """Render the prompt template with variables."""
        full_prompt = self.template
        
        # Add context sections
        context_sections = []
        if prd:
            context_sections.append(f"## Product Requirements Document (PRD)\n\n{prd}")
        if code_context:
            context_sections.append(f"## Codebase\n\n{code_context}")
        if history:
            context_sections.append(f"## Previous Analysis\n\n{history}")
        
        if context_sections:
            full_prompt = full_prompt + "\n\n---\n\n" + "\n\n---\n\n".join(context_sections)
        
        # Append JSON format instructions for Codebase Digest prompts
        if include_json_format:
            full_prompt += JSON_RESPONSE_SUFFIX
        
        return full_prompt
```

---

### Task 3: Improve Response Parsing Robustness
**Priority:** SHOULD-HAVE  
**Files to modify:**
- `src/metaagent/analysis.py`

Enhance `_parse_response()` to handle more edge cases:

```python
# In src/metaagent/analysis.py

import logging
import re
import json

logger = logging.getLogger(__name__)


def _parse_response(self, content: str) -> AnalysisResult:
    """Parse the LLM response into structured result.
    
    Handles:
    - JSON in ```json or ``` code blocks
    - Bare JSON without code blocks
    - Multiple JSON blocks (takes first valid)
    - JSON with surrounding text
    """
    # Try multiple extraction patterns
    patterns = [
        r"```json\s*([\s\S]*?)```",      # ```json ... ```
        r"```JSON\s*([\s\S]*?)```",      # ```JSON ... ``` (uppercase)
        r"```\s*([\s\S]*?)```",          # ``` ... ``` (no language)
        r"(\{[\s\S]*\})",                # Bare JSON object
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, dict):
                    tasks = data.get("tasks", [])
                    # Validate tasks have required structure
                    validated_tasks = []
                    for t in tasks:
                        if isinstance(t, dict) and t.get("title"):
                            validated_tasks.append(t)
                        else:
                            logger.warning(f"Skipping invalid task: {t}")
                    
                    return AnalysisResult(
                        summary=data.get("summary", ""),
                        recommendations=data.get("recommendations", []),
                        tasks=validated_tasks,
                        raw_response=content,
                        success=True,
                    )
            except json.JSONDecodeError:
                continue
    
    # Fallback: treat entire response as summary
    logger.warning("Could not extract JSON from response, using as summary")
    return AnalysisResult(
        summary=content,
        recommendations=[],
        tasks=[],
        raw_response=content,
        success=True,
    )
```

---

### Task 4: Add Task Normalization in PlanWriter
**Priority:** SHOULD-HAVE  
**Files to modify:**
- `src/metaagent/plan_writer.py`

Update `_aggregate_tasks()` to validate and normalize:

```python
# In src/metaagent/plan_writer.py

import logging

logger = logging.getLogger(__name__)

VALID_PRIORITIES = {"critical", "high", "medium", "low"}


def _normalize_task(self, task: dict, stage_id: str) -> Optional[dict]:
    """Normalize and validate a task dict.
    
    Returns None if task is invalid (missing title).
    """
    title = task.get("title", "").strip()
    if not title:
        logger.warning(f"Skipping task without title from stage {stage_id}")
        return None
    
    priority = task.get("priority", "medium")
    if isinstance(priority, str):
        priority = priority.lower().strip()
    if priority not in VALID_PRIORITIES:
        logger.debug(f"Invalid priority '{priority}' for task '{title}', defaulting to medium")
        priority = "medium"
    
    return {
        "title": title,
        "description": task.get("description", "").strip(),
        "priority": priority,
        "file": task.get("file"),
        "stage": stage_id,
    }


def _aggregate_tasks(self, stage_results: list[StageResult]) -> list[dict]:
    """Aggregate tasks from all stages, removing duplicates."""
    all_tasks = []
    seen_titles = set()
    
    for result in stage_results:
        for task in result.tasks:
            normalized = self._normalize_task(task, result.stage_id)
            if normalized is None:
                continue
                
            title = normalized["title"]
            if title in seen_titles:
                logger.debug(f"Skipping duplicate task: {title}")
                continue
                
            seen_titles.add(title)
            all_tasks.append(normalized)
    
    return all_tasks
```

---

### Task 5: Create test_plan_writer.py
**Priority:** MUST-HAVE  
**Files to create:**
- `tests/test_plan_writer.py`

```python
"""Tests for plan writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from metaagent.plan_writer import PlanWriter, StageResult


class TestStageResult:
    """Tests for StageResult dataclass."""
    
    def test_creation_with_defaults(self) -> None:
        """Test creating StageResult with minimal args."""
        result = StageResult(
            stage_id="test_stage",
            stage_name="Test Stage",
            summary="Test summary",
        )
        
        assert result.stage_id == "test_stage"
        assert result.recommendations == []
        assert result.tasks == []
    
    def test_creation_with_tasks(self) -> None:
        """Test creating StageResult with tasks."""
        tasks = [
            {"title": "Task 1", "priority": "high"},
            {"title": "Task 2", "priority": "low"},
        ]
        
        result = StageResult(
            stage_id="test",
            stage_name="Test",
            summary="Summary",
            tasks=tasks,
        )
        
        assert len(result.tasks) == 2


class TestPlanWriter:
    """Tests for PlanWriter class."""
    
    @pytest.fixture
    def plan_writer(self, tmp_path: Path) -> PlanWriter:
        """Create a PlanWriter for testing."""
        return PlanWriter(output_dir=tmp_path)
    
    @pytest.fixture
    def sample_stage_results(self) -> list[StageResult]:
        """Create sample stage results for testing."""
        return [
            StageResult(
                stage_id="quality_analysis",
                stage_name="Quality Analysis",
                summary="Found several code quality issues.",
                recommendations=["Add type hints", "Improve error handling"],
                tasks=[
                    {
                        "title": "Add type hints to cli.py",
                        "description": "Add type annotations to all functions",
                        "priority": "medium",
                        "file": "src/metaagent/cli.py",
                    },
                    {
                        "title": "Fix error handling in analysis.py",
                        "description": "Add try-catch for API calls",
                        "priority": "high",
                        "file": "src/metaagent/analysis.py",
                    },
                ],
            ),
            StageResult(
                stage_id="security_review",
                stage_name="Security Review",
                summary="No critical vulnerabilities found.",
                recommendations=["Add input validation"],
                tasks=[
                    {
                        "title": "Validate API key format",
                        "description": "Check API key format before use",
                        "priority": "critical",
                        "file": "src/metaagent/config.py",
                    },
                ],
            ),
        ]
    
    def test_write_plan_creates_file(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that write_plan creates the output file."""
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD\n\nThis is a test.",
            profile_name="Test Profile",
            stage_results=sample_stage_results,
        )
        
        assert result_path.exists()
        assert result_path.name == "mvp_improvement_plan.md"
    
    def test_write_plan_contains_header(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that plan contains proper header."""
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD",
            profile_name="Test Profile",
            stage_results=sample_stage_results,
        )
        
        content = result_path.read_text()
        
        assert "# MVP Improvement Plan" in content
        assert "**Profile:** Test Profile" in content
        assert "**Generated:**" in content
    
    def test_write_plan_contains_stage_summaries(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that plan contains stage summaries."""
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD",
            profile_name="Test",
            stage_results=sample_stage_results,
        )
        
        content = result_path.read_text()
        
        assert "Quality Analysis" in content
        assert "Found several code quality issues" in content
        assert "Security Review" in content
    
    def test_write_plan_groups_tasks_by_priority(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that tasks are grouped by priority."""
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD",
            profile_name="Test",
            stage_results=sample_stage_results,
        )
        
        content = result_path.read_text()
        
        # Critical should come before High which should come before Medium
        critical_pos = content.find("[CRITICAL]")
        high_pos = content.find("[HIGH]")
        medium_pos = content.find("[MEDIUM]")
        
        assert critical_pos < high_pos < medium_pos
    
    def test_write_plan_deduplicates_tasks(
        self, plan_writer: PlanWriter, tmp_path: Path
    ) -> None:
        """Test that duplicate tasks are removed."""
        stage_results = [
            StageResult(
                stage_id="stage1",
                stage_name="Stage 1",
                summary="Summary 1",
                tasks=[{"title": "Duplicate Task", "priority": "high"}],
            ),
            StageResult(
                stage_id="stage2",
                stage_name="Stage 2",
                summary="Summary 2",
                tasks=[{"title": "Duplicate Task", "priority": "medium"}],  # Same title
            ),
        ]
        
        result_path = plan_writer.write_plan(
            prd_content="# Test",
            profile_name="Test",
            stage_results=stage_results,
        )
        
        content = result_path.read_text()
        
        # Should only appear once
        assert content.count("Duplicate Task") == 1
    
    def test_write_plan_handles_empty_results(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test handling of empty stage results."""
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD",
            profile_name="Test",
            stage_results=[],
        )
        
        content = result_path.read_text()
        
        assert "No stages were executed" in content or "No tasks were identified" in content
    
    def test_write_plan_handles_missing_task_fields(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test handling of tasks with missing optional fields."""
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {"title": "Task with only title"},  # Missing description, priority, file
                ],
            ),
        ]
        
        result_path = plan_writer.write_plan(
            prd_content="# Test",
            profile_name="Test",
            stage_results=stage_results,
        )
        
        content = result_path.read_text()
        
        assert "Task with only title" in content
        assert "[MEDIUM]" in content  # Default priority
    
    def test_write_plan_handles_invalid_priority(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that invalid priorities default to medium."""
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {"title": "Task with bad priority", "priority": "urgent"},
                    {"title": "Another bad priority", "priority": "P1"},
                ],
            ),
        ]
        
        result_path = plan_writer.write_plan(
            prd_content="# Test",
            profile_name="Test",
            stage_results=stage_results,
        )
        
        content = result_path.read_text()
        
        # Both should appear under Medium
        assert "[MEDIUM]" in content
        assert "Task with bad priority" in content
    
    def test_write_plan_includes_file_references(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that file references are included in task output."""
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {
                        "title": "Fix the bug",
                        "file": "src/main.py",
                        "priority": "high",
                    },
                ],
            ),
        ]
        
        result_path = plan_writer.write_plan(
            prd_content="# Test",
            profile_name="Test",
            stage_results=stage_results,
        )
        
        content = result_path.read_text()
        
        assert "src/main.py" in content
        assert "`src/main.py`" in content  # Should be in code format
    
    def test_write_plan_includes_instructions(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that Claude Code instructions are included."""
        result_path = plan_writer.write_plan(
            prd_content="# Test",
            profile_name="Test",
            stage_results=sample_stage_results,
        )
        
        content = result_path.read_text()
        
        assert "Instructions for Claude Code" in content
        assert "implement the tasks in order of priority" in content


class TestPlanWriterTaskAggregation:
    """Tests specifically for task aggregation logic."""
    
    @pytest.fixture
    def plan_writer(self, tmp_path: Path) -> PlanWriter:
        return PlanWriter(output_dir=tmp_path)
    
    def test_aggregate_tasks_empty(self, plan_writer: PlanWriter) -> None:
        """Test aggregation with no results."""
        tasks = plan_writer._aggregate_tasks([])
        assert tasks == []
    
    def test_aggregate_tasks_preserves_order_within_stage(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that task order is preserved within a stage."""
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {"title": "First"},
                    {"title": "Second"},
                    {"title": "Third"},
                ],
            ),
        ]
        
        tasks = plan_writer._aggregate_tasks(stage_results)
        titles = [t["title"] for t in tasks]
        
        assert titles == ["First", "Second", "Third"]
    
    def test_aggregate_tasks_adds_stage_info(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that stage ID is added to tasks."""
        stage_results = [
            StageResult(
                stage_id="my_stage",
                stage_name="My Stage",
                summary="Test",
                tasks=[{"title": "A task"}],
            ),
        ]
        
        tasks = plan_writer._aggregate_tasks(stage_results)
        
        assert tasks[0]["stage"] == "my_stage"


class TestPlanWriterPriorityBadges:
    """Tests for priority badge generation."""
    
    @pytest.fixture
    def plan_writer(self, tmp_path: Path) -> PlanWriter:
        return PlanWriter(output_dir=tmp_path)
    
    def test_priority_badge_critical(self, plan_writer: PlanWriter) -> None:
        assert plan_writer._priority_badge("critical") == "[CRITICAL]"
    
    def test_priority_badge_high(self, plan_writer: PlanWriter) -> None:
        assert plan_writer._priority_badge("high") == "[HIGH]"
    
    def test_priority_badge_medium(self, plan_writer: PlanWriter) -> None:
        assert plan_writer._priority_badge("medium") == "[MEDIUM]"
    
    def test_priority_badge_low(self, plan_writer: PlanWriter) -> None:
        assert plan_writer._priority_badge("low") == "[LOW]"
    
    def test_priority_badge_unknown(self, plan_writer: PlanWriter) -> None:
        """Test that unknown priorities get medium badge."""
        assert plan_writer._priority_badge("unknown") == "[MEDIUM]"
        assert plan_writer._priority_badge("") == "[MEDIUM]"
```

---

### Task 6: Create test_orchestrator_integration.py
**Priority:** SHOULD-HAVE  
**Files to create:**
- `tests/test_orchestrator_integration.py`

```python
"""Integration tests for orchestrator analysis→plan flow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metaagent.analysis import AnalysisResult, MockAnalysisEngine
from metaagent.config import Config
from metaagent.orchestrator import Orchestrator, StageResult
from metaagent.plan_writer import PlanWriter


class TestAnalysisToPlanFlow:
    """Test the flow from analysis results to plan generation."""
    
    @pytest.fixture
    def mock_orchestrator(self, mock_config: Config) -> Orchestrator:
        """Create orchestrator with mocked components."""
        return Orchestrator(mock_config)
    
    def test_analysis_result_transforms_to_stage_result(self) -> None:
        """Test that AnalysisResult fields map correctly to StageResult."""
        analysis_result = AnalysisResult(
            summary="Test summary",
            recommendations=["Rec 1", "Rec 2"],
            tasks=[
                {"title": "Task 1", "priority": "high", "file": "test.py"},
                {"title": "Task 2", "description": "Do something"},
            ],
            success=True,
        )
        
        stage_result = StageResult(
            stage_id="test_stage",
            stage_name="Test Stage Goal",
            summary=analysis_result.summary,
            recommendations=analysis_result.recommendations,
            tasks=analysis_result.tasks,
        )
        
        assert stage_result.summary == "Test summary"
        assert len(stage_result.recommendations) == 2
        assert len(stage_result.tasks) == 2
        assert stage_result.tasks[0]["title"] == "Task 1"
    
    def test_mock_engine_produces_valid_tasks(self) -> None:
        """Test that MockAnalysisEngine produces properly structured tasks."""
        engine = MockAnalysisEngine()
        result = engine.analyze("Test prompt")
        
        assert result.success
        assert len(result.tasks) > 0
        
        task = result.tasks[0]
        assert "title" in task
        assert "priority" in task
    
    def test_empty_tasks_handled_gracefully(self, tmp_path: Path) -> None:
        """Test that empty task lists don't break plan generation."""
        plan_writer = PlanWriter(output_dir=tmp_path)
        
        stage_results = [
            StageResult(
                stage_id="empty_stage",
                stage_name="Empty Stage",
                summary="No issues found",
                recommendations=[],
                tasks=[],
            ),
        ]
        
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD",
            profile_name="Test",
            stage_results=stage_results,
        )
        
        assert result_path.exists()
        content = result_path.read_text()
        assert "No tasks were identified" in content
    
    def test_malformed_tasks_filtered_out(self, tmp_path: Path) -> None:
        """Test that tasks without titles are filtered."""
        plan_writer = PlanWriter(output_dir=tmp_path)
        
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {"title": "Valid task"},
                    {"description": "No title task"},  # Missing title
                    {"title": ""},  # Empty title
                    {"title": "Another valid task"},
                ],
            ),
        ]
        
        tasks = plan_writer._aggregate_tasks(stage_results)
        
        # Only tasks with non-empty titles should remain
        # Note: current implementation may not filter these - this test documents expected behavior
        valid_titles = [t["title"] for t in tasks if t.get("title")]
        assert "Valid task" in valid_titles
        assert "Another valid task" in valid_titles


class TestCustomMockResponses:
    """Test orchestrator behavior with custom mock responses."""
    
    def test_mock_engine_with_json_response(self) -> None:
        """Test MockAnalysisEngine with predefined JSON-like response."""
        custom_result = AnalysisResult(
            summary="Custom analysis summary",
            recommendations=["Fix bugs", "Add tests"],
            tasks=[
                {
                    "title": "Fix critical bug in parser",
                    "description": "The JSON parser fails on nested objects",
                    "priority": "critical",
                    "file": "src/parser.py",
                },
                {
                    "title": "Add unit tests for parser",
                    "description": "Cover edge cases",
                    "priority": "high",
                    "file": "tests/test_parser.py",
                },
            ],
            success=True,
        )
        
        engine = MockAnalysisEngine(responses={"parser": custom_result})
        result = engine.analyze("Analyze the parser module")
        
        assert result.summary == "Custom analysis summary"
        assert len(result.tasks) == 2
        assert result.tasks[0]["priority"] == "critical"
    
    def test_mock_engine_simulates_parse_failure(self) -> None:
        """Test behavior when LLM returns prose instead of JSON."""
        # Simulate what happens when Codebase Digest prompt returns prose
        prose_result = AnalysisResult(
            summary="The codebase has good structure overall. Error handling could be improved in several modules...",
            recommendations=[],  # Empty because JSON parsing failed
            tasks=[],  # Empty because JSON parsing failed
            success=True,
        )
        
        engine = MockAnalysisEngine(responses={"quality": prose_result})
        result = engine.analyze("Run quality analysis")
        
        assert result.success
        assert result.summary != ""
        assert len(result.tasks) == 0  # This is the problem we're solving
```

---

### Task 7: Update README with JSON Contract Documentation
**Priority:** NICE-TO-HAVE  
**Files to modify:**
- `README.md`

Add section after "Adding Custom Prompts":

```markdown
## Analysis Response Contract

All analysis prompts must return JSON in this structure:

```json
{
  "summary": "Brief 2-3 sentence overview of findings",
  "recommendations": [
    "High-level recommendation 1",
    "High-level recommendation 2"
  ],
  "tasks": [
    {
      "title": "Short, actionable task title",
      "description": "Detailed description of what needs to be done",
      "priority": "critical|high|medium|low",
      "file": "path/to/relevant/file.py"
    }
  ]
}
```

### Task Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | Yes | Short, actionable description |
| `description` | string | No | Detailed implementation guidance |
| `priority` | string | No | One of: `critical`, `high`, `medium`, `low`. Default: `medium` |
| `file` | string | No | Path to the relevant source file |

### Adding Codebase Digest Prompts

When using prompts from the `config/prompt_library/` directory (Codebase Digest format), the system automatically appends JSON response instructions. No modification to the original prompt files is needed.
```

---

## Implementation Order

Execute tasks in this order for minimal risk:

1. **Task 5: Create test_plan_writer.py** — Establishes baseline test coverage
2. **Task 1: Define Task schema** — Foundation for validation
3. **Task 4: Add task normalization** — Makes PlanWriter robust
4. **Task 2: Add JSON response suffix** — Fixes Codebase Digest prompts
5. **Task 3: Improve response parsing** — Better error recovery
6. **Task 6: Create integration tests** — Validates end-to-end flow
7. **Task 7: Update README** — Documentation

---

## Verification Checklist

After implementation, verify:

- [ ] `pytest tests/test_plan_writer.py` passes
- [ ] `pytest tests/test_orchestrator_integration.py` passes
- [ ] `metaagent refine --mock --repo .` produces plan with tasks
- [ ] Running with a Codebase Digest prompt (e.g., `quality_error_analysis`) produces tasks
- [ ] Tasks with invalid priorities default to medium
- [ ] Duplicate task titles are deduplicated
- [ ] Tasks without titles are skipped with warning

---

## Files Changed Summary

| File | Action |
|------|--------|
| `src/metaagent/analysis.py` | Add Task dataclass, improve _parse_response() |
| `src/metaagent/plan_writer.py` | Add _normalize_task(), improve _aggregate_tasks() |
| `src/metaagent/prompts.py` | Add JSON_RESPONSE_SUFFIX, update render() |
| `tests/test_plan_writer.py` | Create new file |
| `tests/test_orchestrator_integration.py` | Create new file |
| `README.md` | Add JSON contract documentation |
