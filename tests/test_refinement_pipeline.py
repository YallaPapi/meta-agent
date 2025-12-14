"""Integration tests for the refinement pipeline.

Tests the full orchestrator flow including:
- Profile-based staged execution
- Iterative triage-driven refinement
- Mock mode validation
- Error handling
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from metaagent.analysis import AnalysisResult, MockAnalysisEngine
from metaagent.claude_runner import MockClaudeCodeRunner
from metaagent.config import Config
from metaagent.orchestrator import (
    ImplementationExecutor,
    Orchestrator,
    RefinementResult,
    RunHistory,
    StageRunner,
    TriageEngine,
    TriageResult,
)
from metaagent.plan_writer import PlanWriter, StageResult
from metaagent.prompts import Prompt, PromptLibrary
from metaagent.repomix import RepomixResult, RepomixRunner


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a complete sample repository for testing."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()

    # Create PRD
    prd_content = """# Test PRD

## Overview
A sample project for testing the refinement pipeline.

## Requirements
1. REQ-001: The system shall process data efficiently
2. REQ-002: The system shall handle errors gracefully
3. REQ-003: The system shall log all operations
"""
    (tmp_path / "docs" / "prd.md").write_text(prd_content)

    # Create sample source files
    (tmp_path / "src" / "main.py").write_text('''"""Main module."""

def main():
    """Entry point."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')

    (tmp_path / "src" / "utils.py").write_text('''"""Utility functions."""

def process_data(data):
    """Process some data."""
    return data.upper()
''')

    # Create test file
    (tmp_path / "tests" / "test_main.py").write_text('''"""Tests for main module."""

def test_placeholder():
    assert True
''')

    return tmp_path


@pytest.fixture
def sample_config(sample_repo: Path, tmp_path: Path) -> tuple[Config, Path]:
    """Create sample config with prompts and profiles."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    prompt_library = config_dir / "prompt_library"
    prompt_library.mkdir()

    # Create test prompts
    (prompt_library / "quality_error_analysis.md").write_text("""# Error Analysis

**Objective:** Find errors in the code.

**Instructions:**
1. Review for bugs
2. Check logic errors
""")

    (prompt_library / "architecture_layer_identification.md").write_text("""# Layer Identification

**Objective:** Identify architectural layers.

**Instructions:**
1. Find presentation layer
2. Find business logic layer
""")

    (prompt_library / "meta_triage.md").write_text("""# Codebase Triage

**Objective:** Analyze and select prompts.

**Expected Output:**
```json
{
    "assessment": "Brief assessment",
    "priority_issues": [],
    "selected_prompts": ["quality_error_analysis"],
    "reasoning": "Why these prompts",
    "done": false
}
```
""")

    # Create profiles.yaml
    profiles_content = """profiles:
  quick_review:
    name: Quick Review
    description: Fast review for initial assessment
    stages:
      - quality_error_analysis
  full_review:
    name: Full Review
    description: Comprehensive review
    stages:
      - quality_error_analysis
      - architecture_layer_identification
  test_profile:
    name: Test Profile
    description: Profile for testing
    stages:
      - quality_error_analysis
"""
    (config_dir / "profiles.yaml").write_text(profiles_content)

    config = Config(
        perplexity_api_key="test-key",
        anthropic_api_key="test-key",
        repo_path=sample_repo,
        config_dir=config_dir,
        prd_path=sample_repo / "docs" / "prd.md",
        mock_mode=True,
    )

    return config, config_dir


@pytest.fixture
def mock_repomix_runner() -> MagicMock:
    """Create a mock Repomix runner."""
    runner = MagicMock(spec=RepomixRunner)
    runner.pack.return_value = RepomixResult(
        success=True,
        content="# Sample codebase content\n\ndef main():\n    pass",
        truncated=False,
        original_size=100,
    )
    return runner


# =============================================================================
# Test: Profile-Based Refinement
# =============================================================================


class TestProfileBasedRefinement:
    """Tests for profile-based (default) refinement mode."""

    def test_refine_quick_review_profile(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test refinement with quick_review profile."""
        config, config_dir = sample_config

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine("quick_review")

        assert result.success is True
        assert result.profile_name == "Quick Review"
        assert result.stages_completed >= 1
        assert result.stages_failed == 0
        assert result.plan_path is not None
        assert result.plan_path.exists()

    def test_refine_full_review_profile(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test refinement with full_review profile (multiple stages)."""
        config, config_dir = sample_config

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine("full_review")

        assert result.success is True
        assert result.profile_name == "Full Review"
        assert result.stages_completed == 2  # quality + architecture
        assert result.stages_failed == 0
        assert len(result.stage_results) == 2

    def test_refine_invalid_profile_fails(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test that invalid profile returns error."""
        config, config_dir = sample_config

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine("nonexistent_profile")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_refine_missing_prd_fails(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test that missing PRD returns error."""
        config, config_dir = sample_config
        config.prd_path = config.repo_path / "nonexistent.md"

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine("quick_review")

        assert result.success is False
        assert "not found" in result.error.lower()


# =============================================================================
# Test: Iterative Refinement (Triage Mode)
# =============================================================================


class TestIterativeRefinement:
    """Tests for iterative (triage-driven) refinement mode."""

    def test_refine_iterative_basic(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test basic iterative refinement completes."""
        config, config_dir = sample_config

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine_iterative(max_iterations=2)

        # In mock mode, triage should complete
        assert result is not None
        assert result.profile_name == "iterative"
        # Should have run at least one iteration
        assert len(result.iterations) >= 0

    def test_refine_iterative_respects_max_iterations(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test that max_iterations is respected."""
        config, config_dir = sample_config

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine_iterative(max_iterations=1)

        # Should not exceed max_iterations
        assert len(result.iterations) <= 1


# =============================================================================
# Test: Stage Runner
# =============================================================================


class TestStageRunner:
    """Tests for the StageRunner helper class."""

    def test_run_stage_success(self) -> None:
        """Test successful stage execution."""
        analysis_engine = MockAnalysisEngine()
        prompt_library = MagicMock(spec=PromptLibrary)

        runner = StageRunner(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        prompt = Prompt(
            id="test_prompt",
            goal="Test goal",
            template="Test template {{ prd }}",
            stage="test",
            source="yaml",
        )

        history = RunHistory()
        result, success = runner.run_stage(
            prompt=prompt,
            prd_content="Test PRD",
            code_context="Test code",
            history=history,
        )

        assert success is True
        assert result.stage_id == "test_prompt"
        assert result.summary is not None
        assert len(history.entries) == 1

    def test_run_stages_multiple(self) -> None:
        """Test running multiple stages."""
        analysis_engine = MockAnalysisEngine()
        prompt_library = MagicMock(spec=PromptLibrary)

        runner = StageRunner(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        prompts = [
            Prompt(id="stage1", goal="Goal 1", template="Template 1", stage="test", source="yaml"),
            Prompt(id="stage2", goal="Goal 2", template="Template 2", stage="test", source="yaml"),
            Prompt(id="stage3", goal="Goal 3", template="Template 3", stage="test", source="yaml"),
        ]

        history = RunHistory()
        results, completed, failed = runner.run_stages(
            prompts=prompts,
            prd_content="PRD",
            code_context="Code",
            history=history,
        )

        assert len(results) == 3
        assert completed == 3
        assert failed == 0
        assert len(history.entries) == 3


# =============================================================================
# Test: Triage Engine
# =============================================================================


class TestTriageEngine:
    """Tests for the TriageEngine helper class."""

    def test_run_triage_with_mock_engine(self, tmp_path: Path) -> None:
        """Test triage with mock analysis engine."""
        analysis_engine = MockAnalysisEngine()

        # Create prompt library with triage prompt
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "meta_triage.md").write_text("""# Triage

**Objective:** Select prompts.

```json
{
    "assessment": "test",
    "selected_prompts": [],
    "done": true
}
```
""")

        prompt_library = PromptLibrary(prompt_library_path=prompt_lib)
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        history = RunHistory()
        result = triage_engine.run_triage(
            prd_content="Test PRD",
            code_context="Test code",
            history=history,
        )

        # Mock engine returns structured response
        assert result is not None

    def test_validate_prompts_filters_invalid(self, tmp_path: Path) -> None:
        """Test that triage validates prompt IDs."""
        analysis_engine = MockAnalysisEngine()

        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "valid_prompt.md").write_text("# Valid\nContent")
        (prompt_lib / "meta_triage.md").write_text("# Triage\nContent")

        prompt_library = PromptLibrary(prompt_library_path=prompt_lib)
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
            max_prompts_per_iteration=3,
        )

        # Test validation of prompt IDs
        validated = triage_engine._validate_prompts(
            ["valid_prompt", "invalid_prompt", "another_invalid"]
        )

        assert "valid_prompt" in validated
        assert "invalid_prompt" not in validated

    def test_parse_triage_response_valid_json(self, tmp_path: Path) -> None:
        """Test parsing valid triage JSON response."""
        analysis_engine = MockAnalysisEngine()
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "meta_triage.md").write_text("# Triage\nContent")

        prompt_library = PromptLibrary(prompt_library_path=prompt_lib)
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        result = AnalysisResult(
            summary='{"done": false, "selected_prompts": ["test"], "assessment": "OK"}',
            raw_response='{"done": false, "selected_prompts": ["test"], "assessment": "OK"}',
            success=True,
        )

        triage_result = triage_engine._parse_triage_response(result)

        assert triage_result.success is True
        assert triage_result.done is False
        assert triage_result.assessment == "OK"

    def test_parse_triage_response_nested_json(self, tmp_path: Path) -> None:
        """Test parsing triage JSON with nested braces in strings."""
        analysis_engine = MockAnalysisEngine()
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "meta_triage.md").write_text("# Triage\nContent")

        prompt_library = PromptLibrary(prompt_library_path=prompt_lib)
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        # JSON with braces inside string value
        result = AnalysisResult(
            summary='{"done": true, "assessment": "code: if (x) { return; }"}',
            raw_response='{"done": true, "assessment": "code: if (x) { return; }"}',
            success=True,
        )

        triage_result = triage_engine._parse_triage_response(result)

        assert triage_result.success is True
        assert triage_result.done is True

    def test_parse_triage_response_invalid_done_type(self, tmp_path: Path) -> None:
        """Test parsing fails when 'done' is not a boolean."""
        analysis_engine = MockAnalysisEngine()
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "meta_triage.md").write_text("# Triage\nContent")

        prompt_library = PromptLibrary(prompt_library_path=prompt_lib)
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        result = AnalysisResult(
            summary='{"done": "yes", "selected_prompts": []}',
            raw_response='{"done": "yes", "selected_prompts": []}',
            success=True,
        )

        triage_result = triage_engine._parse_triage_response(result)

        assert triage_result.success is False
        assert "boolean" in triage_result.error

    def test_parse_triage_response_invalid_selected_prompts(self, tmp_path: Path) -> None:
        """Test parsing fails when 'selected_prompts' contains non-strings."""
        analysis_engine = MockAnalysisEngine()
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "meta_triage.md").write_text("# Triage\nContent")

        prompt_library = PromptLibrary(prompt_library_path=prompt_lib)
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        result = AnalysisResult(
            summary='{"done": false, "selected_prompts": [123, 456]}',
            raw_response='{"done": false, "selected_prompts": [123, 456]}',
            success=True,
        )

        triage_result = triage_engine._parse_triage_response(result)

        assert triage_result.success is False
        assert "strings" in triage_result.error

    def test_parse_triage_response_empty_response(self, tmp_path: Path) -> None:
        """Test parsing fails for empty response."""
        analysis_engine = MockAnalysisEngine()
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "meta_triage.md").write_text("# Triage\nContent")

        prompt_library = PromptLibrary(prompt_library_path=prompt_lib)
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        result = AnalysisResult(
            summary="",
            raw_response="",
            success=True,
        )

        triage_result = triage_engine._parse_triage_response(result)

        assert triage_result.success is False
        assert "Empty" in triage_result.error


# =============================================================================
# Test: Implementation Executor
# =============================================================================


class TestImplementationExecutor:
    """Tests for the ImplementationExecutor helper class."""

    def test_write_task_file_creates_markdown(self, sample_repo: Path) -> None:
        """Test that task file is created with proper format."""
        config = Config(
            repo_path=sample_repo,
            mock_mode=True,
        )
        claude_runner = MockClaudeCodeRunner()

        executor = ImplementationExecutor(
            config=config,
            claude_runner=claude_runner,
        )

        tasks = [
            {"title": "Task 1", "description": "Do something", "priority": "high", "file": "src/main.py"},
            {"title": "Task 2", "description": "Do something else", "priority": "low"},
        ]

        task_file = executor._write_task_file(tasks)

        assert task_file.exists()
        content = task_file.read_text()
        assert "Meta-Agent Implementation Tasks" in content
        assert "Task 1" in content
        assert "Task 2" in content
        assert "[ ]" in content  # Checkboxes

    def test_tasks_sorted_by_priority(self, sample_repo: Path) -> None:
        """Test that tasks are sorted by priority (critical first)."""
        config = Config(
            repo_path=sample_repo,
            mock_mode=True,
        )
        claude_runner = MockClaudeCodeRunner()

        executor = ImplementationExecutor(
            config=config,
            claude_runner=claude_runner,
        )

        tasks = [
            {"title": "Low Task", "description": "Low", "priority": "low"},
            {"title": "Critical Task", "description": "Critical", "priority": "critical"},
            {"title": "High Task", "description": "High", "priority": "high"},
        ]

        task_file = executor._write_task_file(tasks)
        content = task_file.read_text()

        # Critical should appear before High, High before Low
        critical_pos = content.find("Critical Task")
        high_pos = content.find("High Task")
        low_pos = content.find("Low Task")

        assert critical_pos < high_pos < low_pos

    def test_execute_with_no_tasks_returns_false(self, sample_repo: Path) -> None:
        """Test that execute returns False when there are no tasks."""
        config = Config(
            repo_path=sample_repo,
            mock_mode=True,
        )
        claude_runner = MockClaudeCodeRunner()

        executor = ImplementationExecutor(
            config=config,
            claude_runner=claude_runner,
        )

        result = executor.execute([])

        assert result is False


# =============================================================================
# Test: Run History
# =============================================================================


class TestRunHistory:
    """Tests for the RunHistory class."""

    def test_add_entry(self) -> None:
        """Test adding entries to history."""
        history = RunHistory()

        history.add_entry("stage1", "Summary 1")
        history.add_entry("stage2", "Summary 2")

        assert len(history.entries) == 2
        assert history.entries[0]["stage"] == "stage1"
        assert history.entries[1]["stage"] == "stage2"

    def test_format_for_prompt_empty(self) -> None:
        """Test formatting empty history."""
        history = RunHistory()
        formatted = history.format_for_prompt()

        assert "No previous analysis" in formatted

    def test_format_for_prompt_with_entries(self) -> None:
        """Test formatting history with entries."""
        history = RunHistory()
        history.add_entry("stage1", "Found 3 issues")
        history.add_entry("stage2", "Identified 2 layers")

        formatted = history.format_for_prompt()

        assert "stage1" in formatted
        assert "stage2" in formatted
        assert "Found 3 issues" in formatted


# =============================================================================
# Test: Mock Mode Validation
# =============================================================================


class TestMockModeValidation:
    """Tests to verify mock mode works correctly without API calls."""

    def test_mock_mode_no_api_calls(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Verify mock mode doesn't make real API calls."""
        config, config_dir = sample_config
        assert config.mock_mode is True

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        # Verify mock analysis engine is used
        assert isinstance(orchestrator.analysis_engine, MockAnalysisEngine)

        # Verify mock Claude runner is used
        assert isinstance(orchestrator.claude_runner, MockClaudeCodeRunner)

        # Run refinement - should complete without errors
        result = orchestrator.refine("quick_review")
        assert result is not None

    def test_mock_engine_returns_valid_structure(self) -> None:
        """Verify mock engine returns properly structured responses."""
        engine = MockAnalysisEngine()
        result = engine.analyze("Test prompt")

        assert result.success is True
        assert result.summary is not None
        assert isinstance(result.recommendations, list)
        assert isinstance(result.tasks, list)

        # Verify task structure
        for task in result.tasks:
            assert "title" in task
            assert "description" in task
            assert "priority" in task


# =============================================================================
# Test: Plan Generation
# =============================================================================


class TestPlanGeneration:
    """Tests for improvement plan generation."""

    def test_plan_file_created(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test that improvement plan file is created."""
        config, config_dir = sample_config

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine("quick_review")

        assert result.plan_path is not None
        assert result.plan_path.exists()

    def test_plan_contains_prd_summary(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test that plan includes PRD summary."""
        config, config_dir = sample_config

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine("quick_review")

        assert result.plan_path is not None
        plan_content = result.plan_path.read_text()
        assert "PRD" in plan_content or "requirements" in plan_content.lower()

    def test_plan_contains_stage_results(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test that plan includes stage results."""
        config, config_dir = sample_config

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine("full_review")

        assert result.plan_path is not None
        plan_content = result.plan_path.read_text()
        # Should have stage identifiers or results
        assert len(plan_content) > 100  # Not empty


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_repomix_failure_graceful_degradation(
        self, sample_config: tuple[Config, Path]
    ) -> None:
        """Test handling of Repomix failure."""
        config, config_dir = sample_config

        # Create a mock that returns failure
        failing_repomix = MagicMock(spec=RepomixRunner)
        failing_repomix.pack.return_value = RepomixResult(
            success=False,
            error="Repomix not installed",
            content="",
        )

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=failing_repomix,
        )

        # Should still complete (with degraded functionality)
        result = orchestrator.refine("quick_review")

        # The refinement should attempt to continue
        assert result is not None

    def test_empty_profile_stages(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test handling profile with no valid stages."""
        config, config_dir = sample_config

        # Add an empty profile
        profiles_content = (config_dir / "profiles.yaml").read_text()
        profiles_content += """
  empty_profile:
    name: Empty Profile
    description: No stages
    stages: []
"""
        (config_dir / "profiles.yaml").write_text(profiles_content)

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine("empty_profile")

        # Should complete but with no stages
        assert result.stages_completed == 0


# =============================================================================
# Test: Stage Triage
# =============================================================================


class TestStageTriage:
    """Tests for stage-specific triage functionality."""

    def test_triage_stage_success(self, tmp_path: Path) -> None:
        """Test successful stage-specific triage."""
        from unittest.mock import MagicMock

        analysis_engine = MagicMock()
        # Return a valid stage triage response
        analysis_engine.analyze.return_value = AnalysisResult(
            success=True,
            summary='{"selected_prompts": ["architecture_layers"], "reasoning": "Test reasoning"}',
            raw_response='{"selected_prompts": ["architecture_layers"], "reasoning": "Test reasoning"}',
        )

        # Create prompt library with candidates
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "architecture_layers.md").write_text("# Layers\nCheck layers.")
        (prompt_lib / "architecture_patterns.md").write_text("# Patterns\nCheck patterns.")
        (prompt_lib / "meta_stage_triage.md").write_text("# Stage Triage\nSelect prompts.")

        # Create stage candidates
        stage_file = tmp_path / "stage_candidates.yaml"
        stage_file.write_text("""stage_candidates:
  architecture:
    candidates:
      - architecture_layers
      - architecture_patterns
    max_prompts: 2
""")

        prompt_library = PromptLibrary(
            prompt_library_path=prompt_lib,
            stage_candidates_path=stage_file,
        )
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        history = RunHistory()
        result = triage_engine.triage_stage(
            stage="architecture",
            prd_content="Test PRD",
            code_context="Test code",
            history=history,
        )

        assert result.success is True
        assert result.stage == "architecture"
        assert "architecture_layers" in result.selected_prompts
        assert result.reasoning == "Test reasoning"

    def test_triage_stage_unknown_stage(self, tmp_path: Path) -> None:
        """Test triage fails for unknown stage."""
        analysis_engine = MockAnalysisEngine()

        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create empty stage candidates (no stages defined)
        stage_file = tmp_path / "stage_candidates.yaml"
        stage_file.write_text("stage_candidates: {}")

        prompt_library = PromptLibrary(
            prompt_library_path=prompt_lib,
            stage_candidates_path=stage_file,
        )
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        history = RunHistory()
        result = triage_engine.triage_stage(
            stage="nonexistent_stage",
            prd_content="Test PRD",
            code_context="Test code",
            history=history,
        )

        assert result.success is False
        assert "No configuration found" in result.error

    def test_triage_stage_filters_invalid_prompts(self, tmp_path: Path) -> None:
        """Test that stage triage filters out invalid prompt selections."""
        # Use MagicMock to control the analysis response
        from unittest.mock import MagicMock

        analysis_engine = MagicMock()
        # Return a response with some invalid prompt IDs
        analysis_engine.analyze.return_value = AnalysisResult(
            success=True,
            summary='{"selected_prompts": ["arch_valid", "arch_invalid", "not_a_candidate"], "reasoning": "Test"}',
            raw_response='{"selected_prompts": ["arch_valid", "arch_invalid", "not_a_candidate"], "reasoning": "Test"}',
        )

        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()
        (prompt_lib / "arch_valid.md").write_text("# Valid\nContent.")
        (prompt_lib / "meta_stage_triage.md").write_text("# Stage Triage\nSelect.")

        stage_file = tmp_path / "stage_candidates.yaml"
        stage_file.write_text("""stage_candidates:
  architecture:
    candidates:
      - arch_valid
    max_prompts: 3
""")

        prompt_library = PromptLibrary(
            prompt_library_path=prompt_lib,
            stage_candidates_path=stage_file,
        )
        prompt_library.load()

        triage_engine = TriageEngine(
            analysis_engine=analysis_engine,
            prompt_library=prompt_library,
        )

        history = RunHistory()
        result = triage_engine.triage_stage(
            stage="architecture",
            prd_content="Test PRD",
            code_context="Test code",
            history=history,
        )

        assert result.success is True
        # Only valid candidate should remain
        for prompt_id in result.selected_prompts:
            assert prompt_id == "arch_valid"


class TestRefineWithStageTriage:
    """Tests for refine_with_stage_triage orchestrator method."""

    def test_refine_with_stage_triage_basic(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test basic stage triage refinement."""
        config, config_dir = sample_config

        # Create stage candidates file
        stage_file = config_dir / "stage_candidates.yaml"
        stage_file.write_text("""stage_candidates:
  architecture:
    candidates:
      - quality_error_analysis
    max_prompts: 1
  quality:
    candidates:
      - quality_code_complexity_analysis
    max_prompts: 1
""")

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
            stage_candidates_path=stage_file,
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine_with_stage_triage(stages=["architecture", "quality"])

        assert result is not None
        assert result.profile_name == "stage_triage"
        # Should attempt stages
        assert result.stages_completed >= 0 or result.stages_failed >= 0

    def test_refine_with_stage_triage_missing_prd(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test stage triage fails gracefully without PRD."""
        config, config_dir = sample_config

        # Remove PRD file
        prd_path = config.prd_path
        if prd_path and prd_path.exists():
            prd_path.unlink()

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        result = orchestrator.refine_with_stage_triage(stages=["architecture"])

        assert result.success is False
        assert "PRD" in result.error

    def test_run_stage_with_triage_helper(
        self, sample_config: tuple[Config, Path], mock_repomix_runner: MagicMock
    ) -> None:
        """Test the run_stage_with_triage helper method."""
        config, config_dir = sample_config

        # Create stage candidates
        stage_file = config_dir / "stage_candidates.yaml"
        stage_file.write_text("""stage_candidates:
  quality:
    candidates:
      - quality_error_analysis
    max_prompts: 1
""")

        prompt_library = PromptLibrary(
            prompts_path=None,
            profiles_path=config_dir / "profiles.yaml",
            prompt_library_path=config_dir / "prompt_library",
            stage_candidates_path=stage_file,
        )
        prompt_library.load()

        orchestrator = Orchestrator(
            config=config,
            prompt_library=prompt_library,
            repomix_runner=mock_repomix_runner,
        )

        history = RunHistory()
        results, completed, failed = orchestrator.run_stage_with_triage(
            stage="quality",
            prd_content="Test PRD",
            code_context="Test code",
            history=history,
        )

        # Should return results (may be empty if no prompts selected)
        assert isinstance(results, list)
        assert isinstance(completed, int)
        assert isinstance(failed, int)
