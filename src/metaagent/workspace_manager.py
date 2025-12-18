"""Workspace management utilities for Git operations and test execution.

This module provides utilities for managing workspaces during the autonomous
development loop, including Git operations and test execution.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running tests in a workspace."""

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    command: str

    @property
    def output(self) -> str:
        """Combined stdout and stderr output."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        return "\n".join(parts)

    @property
    def error_summary(self) -> str:
        """Extract error summary from test output."""
        if self.passed:
            return ""
        # Try to extract the most relevant error information
        output = self.output
        lines = output.split("\n")
        # Look for common error patterns
        error_lines = []
        in_error = False
        for line in lines:
            if any(
                pattern in line.lower()
                for pattern in ["error", "failed", "exception", "traceback"]
            ):
                in_error = True
            if in_error:
                error_lines.append(line)
                if len(error_lines) > 50:  # Limit error output
                    error_lines.append("... (truncated)")
                    break
        return "\n".join(error_lines) if error_lines else output[:1000]


@dataclass
class CommitResult:
    """Result of a git commit operation."""

    success: bool
    commit_hash: Optional[str]
    message: str
    error: Optional[str] = None


@dataclass
class FileChange:
    """Represents a file change to apply."""

    path: str
    content: Optional[str]  # None for deletions
    action: str  # "create", "update", or "delete"


class WorkspaceManager:
    """Manages workspace operations for the autonomous development loop."""

    def __init__(self, workspace_path: Path):
        """Initialize workspace manager.

        Args:
            workspace_path: Path to the workspace directory.
        """
        self.workspace_path = Path(workspace_path)

    def clone_repo(self, source: str, target: Optional[Path] = None) -> Path:
        """Clone a repository to a target directory.

        Args:
            source: Source repository URL or path.
            target: Target directory. If None, uses workspace_path.

        Returns:
            Path to the cloned repository.

        Raises:
            subprocess.CalledProcessError: If git clone fails.
        """
        target_path = target or self.workspace_path
        logger.info(f"Cloning {source} to {target_path}")

        # Create parent directory if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["git", "clone", "--depth", "1", source, str(target_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"Clone output: {result.stdout}")
        return target_path

    def create_branch(self, branch_name: str) -> None:
        """Create and checkout a new branch.

        Args:
            branch_name: Name of the branch to create.

        Raises:
            subprocess.CalledProcessError: If git checkout fails.
        """
        logger.info(f"Creating branch: {branch_name}")
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=self.workspace_path,
            capture_output=True,
            text=True,
            check=True,
        )

    def get_current_branch(self) -> str:
        """Get the current branch name.

        Returns:
            Current branch name.
        """
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.workspace_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def apply_changes(self, changes: list[FileChange]) -> int:
        """Apply file changes to the workspace.

        Args:
            changes: List of FileChange objects to apply.

        Returns:
            Number of files changed.
        """
        files_changed = 0
        for change in changes:
            file_path = self.workspace_path / change.path
            if change.action == "delete":
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted: {change.path}")
                    files_changed += 1
            elif change.action in ("create", "update"):
                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(change.content or "")
                logger.info(f"{change.action.title()}d: {change.path}")
                files_changed += 1
            else:
                logger.warning(f"Unknown action: {change.action} for {change.path}")
        return files_changed

    def commit(self, message: str) -> CommitResult:
        """Stage all changes and create a commit.

        Args:
            message: Commit message.

        Returns:
            CommitResult with success status and commit hash.
        """
        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                check=True,
            )

            # Check if there are changes to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                check=True,
            )

            if not status_result.stdout.strip():
                logger.info("No changes to commit")
                return CommitResult(
                    success=True,
                    commit_hash=None,
                    message="No changes to commit",
                )

            # Create commit
            commit_result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                check=True,
            )

            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                check=True,
            )
            commit_hash = hash_result.stdout.strip()[:8]

            logger.info(f"Created commit {commit_hash}: {message}")
            return CommitResult(
                success=True,
                commit_hash=commit_hash,
                message=message,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Commit failed: {e.stderr}")
            return CommitResult(
                success=False,
                commit_hash=None,
                message=message,
                error=e.stderr,
            )

    def run_tests(
        self, command: str = "pytest -q", timeout: int = 300
    ) -> TestResult:
        """Run tests in the workspace.

        Args:
            command: Test command to run.
            timeout: Maximum time to wait for tests in seconds.

        Returns:
            TestResult with pass/fail status and output.
        """
        logger.info(f"Running tests: {command}")

        try:
            # Use shell=True for command strings, handle both Windows and Unix
            result = subprocess.run(
                command,
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
            )

            passed = result.returncode == 0
            logger.info(f"Tests {'passed' if passed else 'failed'} (exit code: {result.returncode})")

            return TestResult(
                passed=passed,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=command,
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Tests timed out after {timeout}s")
            return TestResult(
                passed=False,
                exit_code=-1,
                stdout="",
                stderr=f"Tests timed out after {timeout} seconds",
                command=command,
            )
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return TestResult(
                passed=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                command=command,
            )

    def get_git_status(self) -> str:
        """Get current git status.

        Returns:
            Git status output.
        """
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.workspace_path,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def get_git_diff(self) -> str:
        """Get current git diff.

        Returns:
            Git diff output.
        """
        result = subprocess.run(
            ["git", "diff"],
            cwd=self.workspace_path,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def cleanup(self) -> None:
        """Clean up the workspace directory.

        Use with caution - this deletes the entire workspace.
        """
        if self.workspace_path.exists():
            logger.warning(f"Cleaning up workspace: {self.workspace_path}")
            shutil.rmtree(self.workspace_path)


def generate_branch_name(pattern: str = "meta-agent-loop/{timestamp}") -> str:
    """Generate a branch name from a pattern.

    Args:
        pattern: Branch name pattern with {timestamp} placeholder.

    Returns:
        Generated branch name.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return pattern.replace("{timestamp}", timestamp)


def create_workspace(
    source_path: Path,
    workspace_dir: Optional[Path] = None,
    branch_pattern: str = "meta-agent-loop/{timestamp}",
    create_branch: bool = True,
) -> WorkspaceManager:
    """Create a new workspace from a source repository.

    Args:
        source_path: Path to the source repository.
        workspace_dir: Directory for the workspace. If None, uses temp directory.
        branch_pattern: Pattern for the new branch name.
        create_branch: Whether to create a new branch.

    Returns:
        WorkspaceManager instance for the new workspace.
    """
    import tempfile

    if workspace_dir is None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="meta-agent-"))

    manager = WorkspaceManager(workspace_dir)

    # If source is a git repo, clone it; otherwise copy files
    source_git = source_path / ".git"
    if source_git.exists():
        manager.clone_repo(str(source_path), workspace_dir)
    else:
        # Copy files directly
        shutil.copytree(source_path, workspace_dir, dirs_exist_ok=True)
        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            check=True,
        )

    if create_branch:
        branch_name = generate_branch_name(branch_pattern)
        manager.create_branch(branch_name)

    return manager
