"""Tests for Claude Code runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

import pytest

from metaagent.claude_runner import (
    ClaudeCodeRunner,
    ClaudeCodeResult,
    MockClaudeCodeRunner,
)


class TestClaudeCodeResult:
    """Tests for ClaudeCodeResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a successful result."""
        result = ClaudeCodeResult(
            success=True,
            output="Implementation completed",
            files_modified=["src/main.py", "tests/test_main.py"],
            exit_code=0,
        )

        assert result.success is True
        assert result.output == "Implementation completed"
        assert len(result.files_modified) == 2
        assert result.error is None
        assert result.exit_code == 0

    def test_failure_result(self) -> None:
        """Test creating a failure result."""
        result = ClaudeCodeResult(
            success=False,
            error="Claude Code not installed",
            exit_code=-1,
        )

        assert result.success is False
        assert result.error == "Claude Code not installed"
        assert result.exit_code == -1


class TestClaudeCodeRunner:
    """Tests for ClaudeCodeRunner."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        runner = ClaudeCodeRunner()

        assert runner.timeout == 600
        assert runner.model == "claude-sonnet-4-20250514"
        assert runner.max_turns == 50

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        runner = ClaudeCodeRunner(
            timeout=300,
            model="claude-opus-4-20250514",
            max_turns=100,
        )

        assert runner.timeout == 300
        assert runner.model == "claude-opus-4-20250514"
        assert runner.max_turns == 100

    def test_check_installed_success(self) -> None:
        """Test check_installed when Claude is available."""
        runner = ClaudeCodeRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert runner.check_installed() is True

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args == ["claude", "--version"]

    def test_check_installed_not_found(self) -> None:
        """Test check_installed when Claude is not installed."""
        runner = ClaudeCodeRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert runner.check_installed() is False

    def test_check_installed_timeout(self) -> None:
        """Test check_installed when command times out."""
        runner = ClaudeCodeRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=10)
            assert runner.check_installed() is False

    def test_implement_not_installed(self, tmp_path: Path) -> None:
        """Test implement when Claude is not installed."""
        runner = ClaudeCodeRunner()

        with patch.object(runner, "check_installed", return_value=False):
            result = runner.implement(
                repo_path=tmp_path,
                prompt="Implement feature X",
            )

            assert result.success is False
            assert "not installed" in result.error.lower()
            assert result.exit_code == -1

    def test_implement_success(self, tmp_path: Path) -> None:
        """Test successful implementation."""
        runner = ClaudeCodeRunner()

        with patch.object(runner, "check_installed", return_value=True):
            with patch("subprocess.run") as mock_run:
                # Mock successful Claude Code execution
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="Implementation completed successfully",
                    stderr="",
                )

                with patch.object(runner, "_get_modified_files", return_value=["src/main.py"]):
                    result = runner.implement(
                        repo_path=tmp_path,
                        prompt="Implement feature X",
                    )

                    assert result.success is True
                    assert "completed" in result.output.lower()
                    assert result.files_modified == ["src/main.py"]
                    assert result.exit_code == 0

    def test_implement_failure(self, tmp_path: Path) -> None:
        """Test implementation failure."""
        runner = ClaudeCodeRunner()

        with patch.object(runner, "check_installed", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1,
                    stdout="",
                    stderr="Error: Something went wrong",
                )

                result = runner.implement(
                    repo_path=tmp_path,
                    prompt="Implement feature X",
                )

                assert result.success is False
                assert result.error is not None
                assert result.exit_code == 1

    def test_implement_timeout(self, tmp_path: Path) -> None:
        """Test implementation timeout."""
        runner = ClaudeCodeRunner(timeout=60)

        with patch.object(runner, "check_installed", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)

                result = runner.implement(
                    repo_path=tmp_path,
                    prompt="Implement feature X",
                )

                assert result.success is False
                assert "timed out" in result.error.lower()
                assert result.exit_code == -1

    def test_build_prompt_simple(self) -> None:
        """Test building a simple prompt."""
        runner = ClaudeCodeRunner()

        prompt = runner._build_prompt("Implement feature X", None)

        assert "Implement feature X" in prompt
        assert "verify changes" in prompt.lower()

    def test_build_prompt_with_plan_file(self, tmp_path: Path) -> None:
        """Test building a prompt with plan file reference."""
        runner = ClaudeCodeRunner()
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("# Implementation Plan")

        prompt = runner._build_prompt("Implement feature X", plan_file)

        assert "Implement feature X" in prompt
        assert str(plan_file) in prompt

    def test_get_modified_files(self, tmp_path: Path) -> None:
        """Test getting modified files from git."""
        runner = ClaudeCodeRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="M src/main.py\nA tests/test_new.py\n",
            )

            files = runner._get_modified_files(tmp_path)

            assert "src/main.py" in files
            assert "tests/test_new.py" in files


class TestMockClaudeCodeRunner:
    """Tests for MockClaudeCodeRunner."""

    def test_check_installed(self) -> None:
        """Test mock always returns True for check_installed."""
        runner = MockClaudeCodeRunner()
        assert runner.check_installed() is True

    def test_implement_success(self, tmp_path: Path) -> None:
        """Test mock implementation always succeeds."""
        runner = MockClaudeCodeRunner()

        result = runner.implement(
            repo_path=tmp_path,
            prompt="Implement feature X",
        )

        assert result.success is True
        assert runner.call_count == 1
        assert runner.last_prompt == "Implement feature X"

    def test_implement_multiple_calls(self, tmp_path: Path) -> None:
        """Test mock tracks multiple calls."""
        runner = MockClaudeCodeRunner()

        runner.implement(repo_path=tmp_path, prompt="Task 1")
        runner.implement(repo_path=tmp_path, prompt="Task 2")
        runner.implement(repo_path=tmp_path, prompt="Task 3")

        assert runner.call_count == 3
        assert runner.last_prompt == "Task 3"

    def test_mock_files_modified(self, tmp_path: Path) -> None:
        """Test mock can report modified files."""
        runner = MockClaudeCodeRunner()
        runner.mock_files_modified = ["src/main.py", "README.md"]

        result = runner.implement(
            repo_path=tmp_path,
            prompt="Implement feature X",
        )

        assert result.files_modified == ["src/main.py", "README.md"]
