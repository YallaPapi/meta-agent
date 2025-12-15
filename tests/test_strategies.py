"""Tests for the implementation strategies module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metaagent.strategies import (
    CodeImplementationStrategy,
    CurrentSessionStrategy,
    SubprocessStrategy,
    StrategyResult,
    create_strategy,
)


class TestCodeImplementationStrategy:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify that the abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            CodeImplementationStrategy()  # type: ignore


class TestCurrentSessionStrategy:
    """Tests for CurrentSessionStrategy."""

    def test_init_default_context(self):
        """Test initialization with default session context."""
        strategy = CurrentSessionStrategy()
        assert strategy.session_context == "Current Claude Code session"

    def test_init_custom_context(self):
        """Test initialization with custom session context."""
        strategy = CurrentSessionStrategy(session_context="Custom session")
        assert strategy.session_context == "Custom session"

    def test_get_name(self):
        """Test that get_name returns correct strategy name."""
        strategy = CurrentSessionStrategy()
        assert strategy.get_name() == "current_session"

    def test_execute_with_task_plan_string(self, tmp_path: Path):
        """Test execute with a task plan string (no plan file)."""
        strategy = CurrentSessionStrategy()

        task_plan = """
- [ ] Add error handling to CLI module
- [ ] Implement logging for analysis
- [ ] Write unit tests for strategies
"""

        result = strategy.execute(
            repo_path=tmp_path,
            task_plan=task_plan,
            plan_file=None,
        )

        assert result.success is True
        assert result.action == "return_to_session"
        assert result.implementation_report is not None
        assert len(result.implementation_report.tasks) >= 1
        assert "Implementation Instructions" in result.instructions

    def test_execute_with_plan_file(self, tmp_path: Path):
        """Test execute with an existing plan file."""
        strategy = CurrentSessionStrategy()

        # Create a mock plan file
        plan_file = tmp_path / "improvement_plan.md"
        plan_file.write_text("""
# Improvement Plan

## Tasks

### [CRITICAL] Critical bugs

- [ ] **Fix authentication** (`auth.py`)
  - Security vulnerability in token validation

### [HIGH] High priority

- [ ] **Add input validation** (`cli.py`)
  - Missing validation for user input
""")

        result = strategy.execute(
            repo_path=tmp_path,
            task_plan="",
            plan_file=plan_file,
        )

        assert result.success is True
        assert result.action == "return_to_session"
        assert result.implementation_report is not None

    def test_execute_with_nonexistent_plan_file(self, tmp_path: Path):
        """Test execute with a non-existent plan file falls back to task_plan."""
        strategy = CurrentSessionStrategy()

        result = strategy.execute(
            repo_path=tmp_path,
            task_plan="- [ ] Task from string",
            plan_file=tmp_path / "nonexistent.md",
        )

        assert result.success is True
        assert result.action == "return_to_session"

    def test_generate_session_instructions(self, tmp_path: Path):
        """Test that session instructions are generated correctly."""
        strategy = CurrentSessionStrategy()

        task_plan = "- [ ] Implement feature X"
        result = strategy.execute(repo_path=tmp_path, task_plan=task_plan)

        assert "Implementation Instructions" in result.instructions
        assert "Repository:" in result.instructions
        assert "Tasks to Implement" in result.instructions
        assert "Workflow" in result.instructions


class TestSubprocessStrategy:
    """Tests for SubprocessStrategy."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        strategy = SubprocessStrategy()
        assert strategy.timeout == 600
        assert strategy.model == "claude-sonnet-4-20250514"
        assert strategy.max_turns == 50

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        strategy = SubprocessStrategy(
            timeout=300,
            model="claude-opus-4-20250514",
            max_turns=25,
        )
        assert strategy.timeout == 300
        assert strategy.model == "claude-opus-4-20250514"
        assert strategy.max_turns == 25

    def test_get_name(self):
        """Test that get_name returns correct strategy name."""
        strategy = SubprocessStrategy()
        assert strategy.get_name() == "subprocess"

    @patch("metaagent.claude_runner.ClaudeCodeRunner")
    def test_execute_calls_claude_runner(self, mock_runner_class, tmp_path: Path):
        """Test that execute calls ClaudeCodeRunner.implement."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner.implement.return_value = MagicMock(
            success=True,
            files_modified=["file1.py"],
            implementation_report=None,
            error=None,
        )
        mock_runner_class.return_value = mock_runner

        strategy = SubprocessStrategy()
        result = strategy.execute(
            repo_path=tmp_path,
            task_plan="Test task plan",
            plan_file=None,
        )

        assert result.success is True
        assert result.action == "subprocess_completed"
        mock_runner.implement.assert_called_once()

    @patch("metaagent.claude_runner.ClaudeCodeRunner")
    def test_execute_handles_failure(self, mock_runner_class, tmp_path: Path):
        """Test that execute handles ClaudeCodeRunner failure."""
        # Setup mock to fail
        mock_runner = MagicMock()
        mock_runner.implement.return_value = MagicMock(
            success=False,
            files_modified=[],
            implementation_report=None,
            error="Claude Code CLI not found",
        )
        mock_runner_class.return_value = mock_runner

        strategy = SubprocessStrategy()
        result = strategy.execute(
            repo_path=tmp_path,
            task_plan="Test task plan",
        )

        assert result.success is False
        assert result.action == "error"
        assert result.error == "Claude Code CLI not found"


class TestStrategyFactory:
    """Tests for the create_strategy factory function."""

    def test_create_current_session_strategy(self):
        """Test creating a CurrentSessionStrategy."""
        strategy = create_strategy("current_session")
        assert isinstance(strategy, CurrentSessionStrategy)
        assert strategy.get_name() == "current_session"

    def test_create_subprocess_strategy(self):
        """Test creating a SubprocessStrategy."""
        strategy = create_strategy("subprocess")
        assert isinstance(strategy, SubprocessStrategy)
        assert strategy.get_name() == "subprocess"

    def test_create_strategy_case_insensitive(self):
        """Test that strategy names are case-insensitive."""
        strategy1 = create_strategy("CURRENT_SESSION")
        strategy2 = create_strategy("Current_Session")
        assert isinstance(strategy1, CurrentSessionStrategy)
        assert isinstance(strategy2, CurrentSessionStrategy)

    def test_create_strategy_with_kwargs(self):
        """Test creating a strategy with custom kwargs."""
        strategy = create_strategy("subprocess", timeout=120, max_turns=10)
        assert isinstance(strategy, SubprocessStrategy)
        assert strategy.timeout == 120
        assert strategy.max_turns == 10

    def test_create_strategy_unknown_name(self):
        """Test that unknown strategy names raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_strategy("unknown_strategy")
        assert "Unknown strategy" in str(exc_info.value)
        assert "unknown_strategy" in str(exc_info.value)
        assert "current_session" in str(exc_info.value)
        assert "subprocess" in str(exc_info.value)


class TestStrategyResult:
    """Tests for StrategyResult dataclass."""

    def test_strategy_result_success(self):
        """Test creating a successful StrategyResult."""
        result = StrategyResult(
            success=True,
            action="return_to_session",
            summary="3 tasks prepared",
        )
        assert result.success is True
        assert result.action == "return_to_session"
        assert result.error is None

    def test_strategy_result_failure(self):
        """Test creating a failed StrategyResult."""
        result = StrategyResult(
            success=False,
            action="error",
            error="Failed to parse plan",
        )
        assert result.success is False
        assert result.action == "error"
        assert result.error == "Failed to parse plan"

    def test_strategy_result_with_files_modified(self):
        """Test StrategyResult with modified files."""
        result = StrategyResult(
            success=True,
            action="subprocess_completed",
            files_modified=["src/main.py", "tests/test_main.py"],
        )
        assert len(result.files_modified) == 2
        assert "src/main.py" in result.files_modified
