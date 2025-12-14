"""Tests for plan writer module."""

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
        assert result.stage_name == "Test Stage"
        assert result.summary == "Test summary"
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
        assert result.tasks[0]["title"] == "Task 1"

    def test_creation_with_recommendations(self) -> None:
        """Test creating StageResult with recommendations."""
        result = StageResult(
            stage_id="test",
            stage_name="Test",
            summary="Summary",
            recommendations=["Rec 1", "Rec 2", "Rec 3"],
        )

        assert len(result.recommendations) == 3
        assert "Rec 1" in result.recommendations


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

    def test_write_plan_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that write_plan creates the output directory if needed."""
        output_dir = tmp_path / "docs" / "plans"
        plan_writer = PlanWriter(output_dir=output_dir)

        result_path = plan_writer.write_plan(
            prd_content="# Test",
            profile_name="Test",
            stage_results=[],
        )

        assert output_dir.exists()
        assert result_path.exists()

    def test_write_plan_custom_filename(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that custom output filename works."""
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD",
            profile_name="Test Profile",
            stage_results=sample_stage_results,
            output_filename="custom_plan.md",
        )

        assert result_path.name == "custom_plan.md"

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
        assert "**Status:** Ready for implementation" in content

    def test_write_plan_contains_prd_summary(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that plan contains PRD summary section."""
        result_path = plan_writer.write_plan(
            prd_content="# My Project PRD\n\nThis project does amazing things.",
            profile_name="Test",
            stage_results=sample_stage_results,
        )

        content = result_path.read_text()

        assert "## PRD Summary" in content
        assert "My Project PRD" in content
        assert "amazing things" in content

    def test_write_plan_truncates_long_prd(self, plan_writer: PlanWriter) -> None:
        """Test that long PRD content is truncated."""
        long_prd = "\n".join([f"Line {i}" for i in range(100)])

        result_path = plan_writer.write_plan(
            prd_content=long_prd,
            profile_name="Test",
            stage_results=[],
        )

        content = result_path.read_text()

        assert "[PRD truncated for brevity]" in content

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

        assert "## Analysis Stages" in content
        assert "### Quality Analysis" in content
        assert "Found several code quality issues" in content
        assert "### Security Review" in content
        assert "No critical vulnerabilities found" in content

    def test_write_plan_contains_recommendations(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that recommendations are included in stage summaries."""
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD",
            profile_name="Test",
            stage_results=sample_stage_results,
        )

        content = result_path.read_text()

        assert "**Recommendations:**" in content
        assert "Add type hints" in content
        assert "Improve error handling" in content

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

        # Check priority sections exist
        assert "[CRITICAL]" in content
        assert "[HIGH]" in content
        assert "[MEDIUM]" in content

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

    def test_write_plan_handles_empty_results(self, plan_writer: PlanWriter) -> None:
        """Test handling of empty stage results."""
        result_path = plan_writer.write_plan(
            prd_content="# Test PRD",
            profile_name="Test",
            stage_results=[],
        )

        content = result_path.read_text()

        assert "No stages were executed" in content
        assert "No tasks were identified" in content

    def test_write_plan_handles_empty_tasks(self, plan_writer: PlanWriter) -> None:
        """Test handling of stage with no tasks."""
        stage_results = [
            StageResult(
                stage_id="empty",
                stage_name="Empty Stage",
                summary="All good, nothing to do.",
                tasks=[],
            ),
        ]

        result_path = plan_writer.write_plan(
            prd_content="# Test",
            profile_name="Test",
            stage_results=stage_results,
        )

        content = result_path.read_text()

        assert "No tasks were identified" in content
        assert "Empty Stage" in content

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

        # Both should appear under Medium (invalid priorities default to medium)
        assert "Task with bad priority" in content
        assert "Another bad priority" in content

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

    def test_write_plan_includes_descriptions(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that task descriptions are included."""
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {
                        "title": "Refactor module",
                        "description": "Split into smaller functions for better maintainability",
                        "priority": "medium",
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

        assert "Refactor module" in content
        assert "Split into smaller functions" in content

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

    def test_write_plan_checkboxes_present(
        self, plan_writer: PlanWriter, sample_stage_results: list[StageResult]
    ) -> None:
        """Test that task checkboxes are present."""
        result_path = plan_writer.write_plan(
            prd_content="# Test",
            profile_name="Test",
            stage_results=sample_stage_results,
        )

        content = result_path.read_text()

        assert "- [ ]" in content


class TestPlanWriterTaskAggregation:
    """Tests specifically for task aggregation logic."""

    @pytest.fixture
    def plan_writer(self, tmp_path: Path) -> PlanWriter:
        return PlanWriter(output_dir=tmp_path)

    def test_aggregate_tasks_empty(self, plan_writer: PlanWriter) -> None:
        """Test aggregation with no results."""
        tasks = plan_writer._aggregate_tasks([])
        assert tasks == []

    def test_aggregate_tasks_single_stage(self, plan_writer: PlanWriter) -> None:
        """Test aggregation with a single stage."""
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {"title": "Task 1"},
                    {"title": "Task 2"},
                ],
            ),
        ]

        tasks = plan_writer._aggregate_tasks(stage_results)

        assert len(tasks) == 2

    def test_aggregate_tasks_multiple_stages(self, plan_writer: PlanWriter) -> None:
        """Test aggregation across multiple stages."""
        stage_results = [
            StageResult(
                stage_id="stage1",
                stage_name="Stage 1",
                summary="Summary 1",
                tasks=[{"title": "Task A"}],
            ),
            StageResult(
                stage_id="stage2",
                stage_name="Stage 2",
                summary="Summary 2",
                tasks=[{"title": "Task B"}],
            ),
        ]

        tasks = plan_writer._aggregate_tasks(stage_results)

        assert len(tasks) == 2
        titles = [t["title"] for t in tasks]
        assert "Task A" in titles
        assert "Task B" in titles

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

    def test_aggregate_tasks_adds_stage_info(self, plan_writer: PlanWriter) -> None:
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

    def test_aggregate_tasks_skips_empty_titles(self, plan_writer: PlanWriter) -> None:
        """Test that tasks with empty titles are skipped."""
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {"title": "Valid task"},
                    {"title": ""},  # Empty title
                    {"title": "Another valid task"},
                ],
            ),
        ]

        tasks = plan_writer._aggregate_tasks(stage_results)
        titles = [t["title"] for t in tasks]

        assert "Valid task" in titles
        assert "Another valid task" in titles
        assert "" not in titles

    def test_aggregate_tasks_skips_missing_titles(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that tasks without title key are skipped."""
        stage_results = [
            StageResult(
                stage_id="test",
                stage_name="Test",
                summary="Test",
                tasks=[
                    {"title": "Valid task"},
                    {"description": "No title here"},  # Missing title key
                ],
            ),
        ]

        tasks = plan_writer._aggregate_tasks(stage_results)

        assert len(tasks) == 1
        assert tasks[0]["title"] == "Valid task"


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
        assert plan_writer._priority_badge("urgent") == "[MEDIUM]"
        assert plan_writer._priority_badge("P1") == "[MEDIUM]"

    def test_priority_badge_case_insensitive(self, plan_writer: PlanWriter) -> None:
        """Test that priority badge is case insensitive."""
        assert plan_writer._priority_badge("CRITICAL") == "[CRITICAL]"
        assert plan_writer._priority_badge("High") == "[HIGH]"
        assert plan_writer._priority_badge("MEDIUM") == "[MEDIUM]"

    def test_priority_badge_empty_string(self, plan_writer: PlanWriter) -> None:
        """Test that empty string defaults to medium."""
        assert plan_writer._priority_badge("") == "[MEDIUM]"


class TestPlanWriterTaskNormalization:
    """Tests for task normalization logic."""

    @pytest.fixture
    def plan_writer(self, tmp_path: Path) -> PlanWriter:
        return PlanWriter(output_dir=tmp_path)

    def test_normalize_valid_task(self, plan_writer: PlanWriter) -> None:
        """Test normalization of a valid task."""
        task = {
            "title": "Fix bug",
            "description": "Fix the bug in module X",
            "priority": "high",
            "file": "src/module.py",
        }

        result = plan_writer._normalize_task(task, "test_stage")

        assert result is not None
        assert result["title"] == "Fix bug"
        assert result["description"] == "Fix the bug in module X"
        assert result["priority"] == "high"
        assert result["file"] == "src/module.py"
        assert result["stage"] == "test_stage"

    def test_normalize_strips_whitespace(self, plan_writer: PlanWriter) -> None:
        """Test that whitespace is stripped from title and description."""
        task = {
            "title": "  Task with spaces  ",
            "description": "  Description with spaces  ",
            "priority": "medium",
        }

        result = plan_writer._normalize_task(task, "test")

        assert result["title"] == "Task with spaces"
        assert result["description"] == "Description with spaces"

    def test_normalize_invalid_priority(self, plan_writer: PlanWriter) -> None:
        """Test that invalid priorities default to medium."""
        task = {"title": "Task", "priority": "urgent"}

        result = plan_writer._normalize_task(task, "test")

        assert result["priority"] == "medium"

    def test_normalize_priority_case_insensitive(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that priority is normalized to lowercase."""
        task = {"title": "Task", "priority": "HIGH"}

        result = plan_writer._normalize_task(task, "test")

        assert result["priority"] == "high"

    def test_normalize_missing_title_returns_none(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that tasks without titles return None."""
        task = {"description": "No title here", "priority": "high"}

        result = plan_writer._normalize_task(task, "test")

        assert result is None

    def test_normalize_empty_title_returns_none(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that tasks with empty titles return None."""
        task = {"title": "", "priority": "high"}

        result = plan_writer._normalize_task(task, "test")

        assert result is None

    def test_normalize_whitespace_only_title_returns_none(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that tasks with whitespace-only titles return None."""
        task = {"title": "   ", "priority": "high"}

        result = plan_writer._normalize_task(task, "test")

        assert result is None

    def test_normalize_missing_optional_fields(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test that missing optional fields get defaults."""
        task = {"title": "Minimal task"}

        result = plan_writer._normalize_task(task, "test")

        assert result["title"] == "Minimal task"
        assert result["description"] == ""
        assert result["priority"] == "medium"
        assert result["file"] is None
        assert result["stage"] == "test"

    def test_normalize_non_string_description(
        self, plan_writer: PlanWriter
    ) -> None:
        """Test handling of non-string description."""
        task = {"title": "Task", "description": 123}

        result = plan_writer._normalize_task(task, "test")

        assert result["description"] == ""


class TestPlanWriterPRDExtraction:
    """Tests for PRD summary extraction."""

    @pytest.fixture
    def plan_writer(self, tmp_path: Path) -> PlanWriter:
        return PlanWriter(output_dir=tmp_path)

    def test_short_prd_not_truncated(self, plan_writer: PlanWriter) -> None:
        """Test that short PRDs are not truncated."""
        short_prd = "# Project\n\nShort description."
        summary = plan_writer._extract_prd_summary(short_prd)

        assert summary == short_prd.strip()
        assert "[truncated]" not in summary.lower()

    def test_long_prd_truncated(self, plan_writer: PlanWriter) -> None:
        """Test that long PRDs are truncated."""
        long_prd = "\n".join([f"Line {i}" for i in range(50)])
        summary = plan_writer._extract_prd_summary(long_prd, max_lines=10)

        assert "[PRD truncated for brevity]" in summary
        assert "Line 0" in summary
        assert "Line 9" in summary
        assert "Line 10" not in summary

    def test_prd_exactly_max_lines(self, plan_writer: PlanWriter) -> None:
        """Test PRD exactly at max_lines boundary."""
        prd = "\n".join([f"Line {i}" for i in range(20)])
        summary = plan_writer._extract_prd_summary(prd, max_lines=20)

        assert "[truncated]" not in summary.lower()
        assert "Line 19" in summary
