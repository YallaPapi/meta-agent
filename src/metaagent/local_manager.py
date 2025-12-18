"""Local repository manager using GitPython for the autonomous development loop.

This module provides git operations and test execution for the local-first
development loop. It operates directly on the current repository without
workspace isolation.
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import git
from git.exc import GitCommandError, InvalidGitRepositoryError

# Platform-specific settings
IS_WINDOWS = platform.system() == "Windows"

from .config import LoopConfig

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running tests."""

    success: bool
    stdout: str
    stderr: str
    returncode: int
    command: str

    @property
    def output(self) -> str:
        """Combined stdout and stderr output."""
        return f"{self.stdout}\n{self.stderr}".strip()

    @property
    def error_summary(self) -> str:
        """Get a summary of the error for diagnosis."""
        if self.success:
            return ""
        # Return last 50 lines of combined output
        lines = self.output.split("\n")
        return "\n".join(lines[-50:])


@dataclass
class CommitResult:
    """Result of a git commit operation."""

    success: bool
    commit_hash: Optional[str]
    message: str
    error: Optional[str] = None


class LocalRepoError(Exception):
    """Exception raised for local repository operations."""

    pass


class LocalRepoManager:
    """Manages local git operations using GitPython.

    This class provides git operations for the autonomous development loop,
    working directly in the current repository (local-first approach).
    """

    def __init__(
        self,
        config: LoopConfig,
        repo_path: Optional[Path] = None,
    ):
        """Initialize the local repository manager.

        Args:
            config: Loop configuration with branch prefix, test command, etc.
            repo_path: Optional path to repository. Defaults to current directory.

        Raises:
            LocalRepoError: If the directory is not a git repository.
        """
        self.config = config
        self.repo_path = repo_path or Path.cwd()
        self._original_branch: Optional[str] = None
        self._created_branch: Optional[str] = None

        try:
            self.repo = git.Repo(self.repo_path)
        except InvalidGitRepositoryError as exc:
            raise LocalRepoError(
                f"Not a git repository: {self.repo_path}"
            ) from exc

        logger.info(f"LocalRepoManager initialized at: {self.repo_path}")

    def _generate_branch_name(self) -> str:
        """Generate a unique branch name with timestamp.

        Returns:
            Branch name like 'meta-loop-20231218-1430'.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        return f"{self.config.branch_prefix}-{timestamp}"

    def get_current_branch(self) -> str:
        """Get the current branch name.

        Returns:
            Current branch name.
        """
        return self.repo.active_branch.name

    def is_on_protected_branch(self) -> bool:
        """Check if currently on a protected branch (main/master).

        Returns:
            True if on main or master branch.
        """
        current = self.get_current_branch()
        return current in ("main", "master")

    def create_branch(self, branch_name: Optional[str] = None) -> str:
        """Create and checkout a new branch for loop work.

        This method ensures we never work directly on main/master.

        Args:
            branch_name: Optional branch name. Auto-generated if not provided.

        Returns:
            The name of the created/checked-out branch.

        Raises:
            LocalRepoError: If branch creation fails.
        """
        self._original_branch = self.get_current_branch()

        # If already on a feature branch (not main/master), stay on it
        if not self.is_on_protected_branch():
            logger.info(f"Already on feature branch: {self._original_branch}")
            return self._original_branch

        # Generate branch name if not provided
        branch_name = branch_name or self._generate_branch_name()

        try:
            # Create and checkout new branch
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            self._created_branch = branch_name
            logger.info(f"Created and checked out branch: {branch_name}")
            return branch_name
        except GitCommandError as exc:
            raise LocalRepoError(
                f"Failed to create branch {branch_name}: {exc}"
            ) from exc

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes (staged or unstaged).

        Returns:
            True if there are uncommitted changes.
        """
        return self.repo.is_dirty(untracked_files=True)

    def stage_all_changes(self) -> None:
        """Stage all changes (including untracked files)."""
        self.repo.git.add(all=True)

    def commit_changes(self, message: str) -> CommitResult:
        """Commit staged changes with the given message.

        Args:
            message: Commit message.

        Returns:
            CommitResult with success status and commit hash.
        """
        try:
            # Stage all changes
            self.stage_all_changes()

            # Check if there are changes to commit
            if not self.has_uncommitted_changes():
                logger.info("No changes to commit")
                return CommitResult(
                    success=True,
                    commit_hash=None,
                    message="No changes to commit",
                )

            # Commit
            commit = self.repo.index.commit(message)
            commit_hash = commit.hexsha[:8]
            logger.info(f"Committed: {commit_hash} - {message[:50]}")

            return CommitResult(
                success=True,
                commit_hash=commit_hash,
                message=message,
            )

        except GitCommandError as exc:
            logger.error(f"Commit failed: {exc}")
            return CommitResult(
                success=False,
                commit_hash=None,
                message=message,
                error=str(exc),
            )

    def run_tests(
        self,
        command: Optional[str] = None,
        timeout: int = 300,
    ) -> TestResult:
        """Run tests and capture output.

        Args:
            command: Test command. Uses config.test_command if not provided.
            timeout: Timeout in seconds (default: 5 minutes).

        Returns:
            TestResult with success status and output.
        """
        cmd = command or self.config.test_command

        # Convert string command to list for subprocess
        # Note: We use posix=True even on Windows because:
        # 1. Typical test commands like "pytest -q" work fine with posix=True
        # 2. posix=False keeps quotes around arguments which breaks "python -c ..."
        # 3. If users have Windows-specific paths with backslashes, they should
        #    use forward slashes (Python handles this) or provide a list
        if isinstance(cmd, str):
            cmd_list = shlex.split(cmd, posix=True)
        else:
            cmd_list = list(cmd)

        logger.info(f"Running tests: {' '.join(cmd_list)}")

        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.repo_path,
            )

            return TestResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                command=cmd if isinstance(cmd, str) else " ".join(cmd),
            )

        except subprocess.TimeoutExpired as exc:
            logger.warning(f"Tests timed out after {timeout}s")
            return TestResult(
                success=False,
                stdout=exc.stdout.decode() if exc.stdout else "",
                stderr=f"Tests timed out after {timeout} seconds",
                returncode=-1,
                command=cmd if isinstance(cmd, str) else " ".join(cmd),
            )

        except FileNotFoundError as exc:
            logger.error(f"Test command not found: {exc}")
            return TestResult(
                success=False,
                stdout="",
                stderr=f"Command not found: {cmd_list[0]}",
                returncode=-1,
                command=cmd if isinstance(cmd, str) else " ".join(cmd),
            )

        except Exception as exc:
            logger.error(f"Test execution failed: {exc}")
            return TestResult(
                success=False,
                stdout="",
                stderr=str(exc),
                returncode=-1,
                command=cmd if isinstance(cmd, str) else " ".join(cmd),
            )

    def get_recent_commits(self, count: int = 5) -> List[str]:
        """Get recent commit messages.

        Args:
            count: Number of recent commits to retrieve.

        Returns:
            List of commit messages.
        """
        commits = list(self.repo.iter_commits(max_count=count))
        return [c.message.split("\n")[0] for c in commits]

    def get_diff_summary(self) -> str:
        """Get a summary of current uncommitted changes.

        Returns:
            Diff summary as string.
        """
        if not self.has_uncommitted_changes():
            return "No uncommitted changes"

        # Get list of changed files
        changed = []
        if self.repo.index.diff(None):  # Unstaged
            changed.extend([d.a_path for d in self.repo.index.diff(None)])
        if self.repo.index.diff("HEAD"):  # Staged
            changed.extend([d.a_path for d in self.repo.index.diff("HEAD")])
        if self.repo.untracked_files:
            changed.extend(self.repo.untracked_files)

        unique_changed = list(set(changed))
        return f"Changed files ({len(unique_changed)}): {', '.join(unique_changed[:10])}"

    def cleanup(self) -> None:
        """Cleanup resources (no-op for local manager but kept for API consistency)."""
        logger.debug("LocalRepoManager cleanup (no-op)")


class MockLocalRepoManager:
    """Mock LocalRepoManager for testing without actual git operations."""

    def __init__(self, config: Optional[LoopConfig] = None):
        """Initialize mock manager."""
        self.config = config
        self.current_branch = "main"
        self.commits: List[CommitResult] = []
        self.test_results: List[TestResult] = []
        self.call_count = 0
        self._should_fail_tests = False
        self._test_output = "All tests passed"

    def get_current_branch(self) -> str:
        """Get mock current branch."""
        return self.current_branch

    def is_on_protected_branch(self) -> bool:
        """Check if on mock protected branch."""
        return self.current_branch in ("main", "master")

    def create_branch(self, branch_name: Optional[str] = None) -> str:
        """Mock branch creation."""
        self.call_count += 1
        if not self.is_on_protected_branch():
            return self.current_branch

        new_name = branch_name or f"meta-loop-{datetime.now().strftime('%Y%m%d-%H%M')}"
        self.current_branch = new_name
        return new_name

    def has_uncommitted_changes(self) -> bool:
        """Mock uncommitted changes check."""
        return True  # Always has changes for testing

    def commit_changes(self, message: str) -> CommitResult:
        """Mock commit."""
        self.call_count += 1
        result = CommitResult(
            success=True,
            commit_hash=f"mock{self.call_count:04d}",
            message=message,
        )
        self.commits.append(result)
        return result

    def run_tests(
        self,
        command: Optional[str] = None,
        timeout: int = 300,
    ) -> TestResult:
        """Mock test run."""
        self.call_count += 1
        result = TestResult(
            success=not self._should_fail_tests,
            stdout=self._test_output,
            stderr="" if not self._should_fail_tests else "Test failure",
            returncode=0 if not self._should_fail_tests else 1,
            command=command or "pytest -q",
        )
        self.test_results.append(result)
        return result

    def set_test_failure(self, should_fail: bool, output: str = "Test failure") -> None:
        """Configure mock to fail tests."""
        self._should_fail_tests = should_fail
        self._test_output = output

    def cleanup(self) -> None:
        """Mock cleanup."""
        pass
