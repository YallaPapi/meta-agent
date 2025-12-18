"""Unit tests for local_manager.py."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import pytest

from src.metaagent.config import LoopConfig
from src.metaagent.local_manager import (
    CommitResult,
    LocalRepoError,
    LocalRepoManager,
    MockLocalRepoManager,
    TestResult,
)


@pytest.fixture
def loop_config():
    """Create a test LoopConfig."""
    return LoopConfig(
        branch_prefix="test-loop",
        test_command="python -c \"print('ok')\"",
    )


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository."""
    repo = git.Repo.init(tmp_path)

    # Configure git user for commits
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Create initial file and commit
    test_file = tmp_path / "test.py"
    test_file.write_text("# Initial file\n")
    repo.index.add(["test.py"])
    repo.index.commit("Initial commit")

    return repo


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_output_combined(self):
        """Test combined output property."""
        result = TestResult(
            success=True,
            stdout="stdout content",
            stderr="stderr content",
            returncode=0,
            command="pytest",
        )
        assert "stdout content" in result.output
        assert "stderr content" in result.output

    def test_error_summary_on_failure(self):
        """Test error summary on failure."""
        result = TestResult(
            success=False,
            stdout="",
            stderr="Error: test failed\nTraceback...",
            returncode=1,
            command="pytest",
        )
        assert "Error" in result.error_summary

    def test_error_summary_empty_on_success(self):
        """Test error summary is empty on success."""
        result = TestResult(
            success=True,
            stdout="OK",
            stderr="",
            returncode=0,
            command="pytest",
        )
        assert result.error_summary == ""


class TestCommitResult:
    """Tests for CommitResult dataclass."""

    def test_successful_commit(self):
        """Test successful commit result."""
        result = CommitResult(
            success=True,
            commit_hash="abc12345",
            message="feat: add feature",
        )
        assert result.success
        assert result.commit_hash == "abc12345"
        assert result.error is None

    def test_failed_commit(self):
        """Test failed commit result."""
        result = CommitResult(
            success=False,
            commit_hash=None,
            message="feat: add feature",
            error="Nothing to commit",
        )
        assert not result.success
        assert result.commit_hash is None
        assert "Nothing" in result.error


class TestLocalRepoManager:
    """Tests for LocalRepoManager class."""

    def test_init_valid_repo(self, git_repo, loop_config, tmp_path):
        """Test initialization with valid git repository."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        assert manager.repo_path == tmp_path
        assert manager.config == loop_config

    def test_init_invalid_repo(self, loop_config, tmp_path):
        """Test initialization with non-git directory."""
        non_git_dir = tmp_path / "not_a_repo"
        non_git_dir.mkdir()

        with pytest.raises(LocalRepoError) as exc_info:
            LocalRepoManager(loop_config, repo_path=non_git_dir)

        assert "Not a git repository" in str(exc_info.value)

    def test_get_current_branch(self, git_repo, loop_config, tmp_path):
        """Test getting current branch name."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        # Default branch might be 'master' or 'main' depending on git config
        assert manager.get_current_branch() in ("master", "main")

    def test_is_on_protected_branch(self, git_repo, loop_config, tmp_path):
        """Test protected branch detection."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        assert manager.is_on_protected_branch()

        # Create and checkout feature branch
        git_repo.create_head("feature-test").checkout()
        assert not manager.is_on_protected_branch()

    def test_create_branch_from_main(self, git_repo, loop_config, tmp_path):
        """Test branch creation from main/master."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        branch_name = manager.create_branch()

        assert branch_name.startswith("test-loop-")
        assert manager.get_current_branch() == branch_name
        assert not manager.is_on_protected_branch()

    def test_create_branch_custom_name(self, git_repo, loop_config, tmp_path):
        """Test branch creation with custom name."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        branch_name = manager.create_branch("custom-branch")

        assert branch_name == "custom-branch"
        assert manager.get_current_branch() == "custom-branch"

    def test_create_branch_stays_on_feature(self, git_repo, loop_config, tmp_path):
        """Test that branch creation stays on feature branch."""
        # Create feature branch first
        git_repo.create_head("existing-feature").checkout()

        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        branch_name = manager.create_branch()

        # Should stay on existing feature branch
        assert branch_name == "existing-feature"

    def test_has_uncommitted_changes_clean(self, git_repo, loop_config, tmp_path):
        """Test uncommitted changes detection on clean repo."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        assert not manager.has_uncommitted_changes()

    def test_has_uncommitted_changes_dirty(self, git_repo, loop_config, tmp_path):
        """Test uncommitted changes detection on dirty repo."""
        # Make a change
        test_file = tmp_path / "test.py"
        test_file.write_text("# Modified\n")

        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        assert manager.has_uncommitted_changes()

    def test_commit_changes_success(self, git_repo, loop_config, tmp_path):
        """Test successful commit."""
        # Make a change
        test_file = tmp_path / "new_file.py"
        test_file.write_text("# New file\n")

        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        result = manager.commit_changes("Add new file")

        assert result.success
        assert result.commit_hash is not None
        assert len(result.commit_hash) == 8
        assert not manager.has_uncommitted_changes()

    def test_commit_changes_no_changes(self, git_repo, loop_config, tmp_path):
        """Test commit with no changes."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        result = manager.commit_changes("Empty commit")

        assert result.success
        assert result.commit_hash is None
        assert "No changes" in result.message

    def test_run_tests_success(self, git_repo, loop_config, tmp_path):
        """Test successful test run."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        result = manager.run_tests("python -c \"print('ok')\"")

        assert result.success
        assert result.returncode == 0
        assert "ok" in result.stdout

    def test_run_tests_failure(self, git_repo, loop_config, tmp_path):
        """Test failed test run."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        result = manager.run_tests("python -c \"import sys; sys.exit(1)\"")

        assert not result.success
        assert result.returncode == 1

    def test_run_tests_command_not_found(self, git_repo, loop_config, tmp_path):
        """Test test run with non-existent command."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        result = manager.run_tests("nonexistent_command_xyz")

        assert not result.success
        assert result.returncode == -1
        assert "not found" in result.stderr.lower() or "nonexistent" in result.stderr.lower()

    def test_run_tests_uses_config_command(self, git_repo, loop_config, tmp_path):
        """Test that run_tests uses config command by default."""
        loop_config.test_command = "python -c \"print('from_config')\""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        result = manager.run_tests()

        assert result.success
        assert "from_config" in result.stdout

    def test_get_recent_commits(self, git_repo, loop_config, tmp_path):
        """Test getting recent commits."""
        # Add a few more commits
        for i in range(3):
            test_file = tmp_path / f"file{i}.py"
            test_file.write_text(f"# File {i}\n")
            git_repo.index.add([f"file{i}.py"])
            git_repo.index.commit(f"Add file {i}")

        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        commits = manager.get_recent_commits(count=3)

        assert len(commits) == 3
        assert "Add file 2" in commits[0]

    def test_get_diff_summary_clean(self, git_repo, loop_config, tmp_path):
        """Test diff summary on clean repo."""
        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        summary = manager.get_diff_summary()
        assert "No uncommitted changes" in summary

    def test_get_diff_summary_dirty(self, git_repo, loop_config, tmp_path):
        """Test diff summary on dirty repo."""
        # Make a change
        test_file = tmp_path / "test.py"
        test_file.write_text("# Modified\n")

        manager = LocalRepoManager(loop_config, repo_path=tmp_path)
        summary = manager.get_diff_summary()

        assert "Changed files" in summary
        assert "test.py" in summary


class TestMockLocalRepoManager:
    """Tests for MockLocalRepoManager class."""

    def test_mock_create_branch(self):
        """Test mock branch creation."""
        mock = MockLocalRepoManager()
        branch = mock.create_branch()

        assert branch.startswith("meta-loop-")
        assert mock.current_branch == branch

    def test_mock_stays_on_feature_branch(self):
        """Test mock stays on feature branch."""
        mock = MockLocalRepoManager()
        mock.current_branch = "feature-existing"

        branch = mock.create_branch()
        assert branch == "feature-existing"

    def test_mock_commit_changes(self):
        """Test mock commit."""
        mock = MockLocalRepoManager()
        result = mock.commit_changes("Test commit")

        assert result.success
        assert result.commit_hash is not None
        assert len(mock.commits) == 1

    def test_mock_run_tests_success(self):
        """Test mock successful tests."""
        mock = MockLocalRepoManager()
        result = mock.run_tests()

        assert result.success
        assert len(mock.test_results) == 1

    def test_mock_run_tests_failure(self):
        """Test mock failed tests."""
        mock = MockLocalRepoManager()
        mock.set_test_failure(True, "Test failed")

        result = mock.run_tests()

        assert not result.success
        assert "Test failed" in result.stdout or "Test failure" in result.stderr

    def test_mock_tracks_calls(self):
        """Test mock tracks call count."""
        mock = MockLocalRepoManager()
        mock.create_branch()
        mock.commit_changes("msg")
        mock.run_tests()

        assert mock.call_count == 3
