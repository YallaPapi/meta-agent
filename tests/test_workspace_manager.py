"""Unit tests for workspace_manager.py."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.metaagent.workspace_manager import (
    CommitResult,
    FileChange,
    TestResult,
    WorkspaceManager,
    generate_branch_name,
)


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_create_action(self):
        change = FileChange(path="test.py", content="print('hello')", action="create")
        assert change.path == "test.py"
        assert change.content == "print('hello')"
        assert change.action == "create"

    def test_delete_action(self):
        change = FileChange(path="old.py", content=None, action="delete")
        assert change.path == "old.py"
        assert change.content is None
        assert change.action == "delete"


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_passed_result(self):
        result = TestResult(
            passed=True,
            exit_code=0,
            stdout="All tests passed",
            stderr="",
            command="pytest",
        )
        assert result.passed
        assert result.exit_code == 0
        assert "All tests passed" in result.output

    def test_failed_result_error_summary(self):
        result = TestResult(
            passed=False,
            exit_code=1,
            stdout="",
            stderr="Error: assertion failed\nTraceback...",
            command="pytest",
        )
        assert not result.passed
        summary = result.error_summary
        assert "Error" in summary or "assertion" in summary

    def test_passed_result_no_error_summary(self):
        result = TestResult(
            passed=True,
            exit_code=0,
            stdout="OK",
            stderr="",
            command="pytest",
        )
        assert result.error_summary == ""


class TestCommitResult:
    """Tests for CommitResult dataclass."""

    def test_successful_commit(self):
        result = CommitResult(
            success=True,
            commit_hash="abc123",
            message="feat: add feature",
        )
        assert result.success
        assert result.commit_hash == "abc123"
        assert result.error is None

    def test_failed_commit(self):
        result = CommitResult(
            success=False,
            commit_hash=None,
            message="feat: add feature",
            error="Nothing to commit",
        )
        assert not result.success
        assert result.commit_hash is None
        assert "Nothing" in result.error


class TestWorkspaceManager:
    """Tests for WorkspaceManager class."""

    def test_init(self, tmp_path):
        manager = WorkspaceManager(tmp_path)
        assert manager.workspace_path == tmp_path

    def test_apply_changes_create(self, tmp_path):
        manager = WorkspaceManager(tmp_path)
        changes = [
            FileChange(path="new_file.py", content="# new file", action="create")
        ]

        count = manager.apply_changes(changes)

        assert count == 1
        assert (tmp_path / "new_file.py").exists()
        assert (tmp_path / "new_file.py").read_text() == "# new file"

    def test_apply_changes_update(self, tmp_path):
        # Create initial file
        initial = tmp_path / "existing.py"
        initial.write_text("# original")

        manager = WorkspaceManager(tmp_path)
        changes = [
            FileChange(path="existing.py", content="# updated", action="update")
        ]

        count = manager.apply_changes(changes)

        assert count == 1
        assert initial.read_text() == "# updated"

    def test_apply_changes_delete(self, tmp_path):
        # Create file to delete
        to_delete = tmp_path / "delete_me.py"
        to_delete.write_text("# delete this")

        manager = WorkspaceManager(tmp_path)
        changes = [
            FileChange(path="delete_me.py", content=None, action="delete")
        ]

        count = manager.apply_changes(changes)

        assert count == 1
        assert not to_delete.exists()

    def test_apply_changes_creates_parent_dirs(self, tmp_path):
        manager = WorkspaceManager(tmp_path)
        changes = [
            FileChange(
                path="nested/dir/file.py",
                content="# nested",
                action="create"
            )
        ]

        count = manager.apply_changes(changes)

        assert count == 1
        assert (tmp_path / "nested/dir/file.py").exists()

    @patch("subprocess.run")
    def test_run_tests_passing(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="All tests passed",
            stderr="",
        )

        manager = WorkspaceManager(tmp_path)
        result = manager.run_tests("pytest")

        assert result.passed
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_tests_failing(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="FAILED test_example.py",
        )

        manager = WorkspaceManager(tmp_path)
        result = manager.run_tests("pytest")

        assert not result.passed
        assert result.exit_code == 1

    @patch("subprocess.run")
    def test_run_tests_timeout(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 60)

        manager = WorkspaceManager(tmp_path)
        result = manager.run_tests("pytest", timeout=60)

        assert not result.passed
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower()

    @patch("subprocess.run")
    def test_create_branch(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        manager = WorkspaceManager(tmp_path)
        manager.create_branch("feature-branch")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "checkout" in call_args[0][0]
        assert "-b" in call_args[0][0]
        assert "feature-branch" in call_args[0][0]

    @patch("subprocess.run")
    def test_commit_with_changes(self, mock_run, tmp_path):
        # Setup mocks for add, status, commit, and rev-parse
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=0, stdout="M file.py"),  # git status
            MagicMock(returncode=0),  # git commit
            MagicMock(returncode=0, stdout="abc1234567890"),  # git rev-parse
        ]

        manager = WorkspaceManager(tmp_path)
        result = manager.commit("test commit")

        assert result.success
        assert result.commit_hash == "abc12345"

    @patch("subprocess.run")
    def test_commit_no_changes(self, mock_run, tmp_path):
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=0, stdout=""),  # git status (empty = no changes)
        ]

        manager = WorkspaceManager(tmp_path)
        result = manager.commit("test commit")

        assert result.success
        assert result.commit_hash is None
        assert "No changes" in result.message


class TestGenerateBranchName:
    """Tests for generate_branch_name function."""

    def test_default_pattern(self):
        name = generate_branch_name()
        assert name.startswith("meta-agent-loop/")
        # Should contain a timestamp
        assert len(name) > len("meta-agent-loop/")

    def test_custom_pattern(self):
        name = generate_branch_name("feature/{timestamp}")
        assert name.startswith("feature/")

    def test_no_placeholder(self):
        name = generate_branch_name("fixed-branch-name")
        assert name == "fixed-branch-name"
