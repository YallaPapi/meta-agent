"""Claude Code CLI integration for automated implementation."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaudeCodeResult:
    """Result from a Claude Code execution."""

    success: bool
    output: str = ""
    error: Optional[str] = None
    files_modified: list[str] = field(default_factory=list)
    exit_code: int = 0


class ClaudeCodeRunner:
    """Runs Claude Code CLI to implement changes in a repository."""

    def __init__(
        self,
        timeout: int = 600,  # 10 minutes default for implementation
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 50,
    ):
        """Initialize the Claude Code runner.

        Args:
            timeout: Maximum time in seconds for Claude Code to run.
            model: Claude model to use for implementation.
            max_turns: Maximum conversation turns for the agentic loop.
        """
        self.timeout = timeout
        self.model = model
        self.max_turns = max_turns

    def check_installed(self) -> bool:
        """Check if Claude Code CLI is installed and accessible.

        Returns:
            True if claude command is available, False otherwise.
        """
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def implement(
        self,
        repo_path: Path,
        prompt: str,
        plan_file: Optional[Path] = None,
    ) -> ClaudeCodeResult:
        """Run Claude Code to implement changes.

        Args:
            repo_path: Path to the target repository.
            prompt: Implementation prompt to send to Claude Code.
            plan_file: Optional path to the improvement plan file.

        Returns:
            ClaudeCodeResult with execution outcome.
        """
        if not self.check_installed():
            return ClaudeCodeResult(
                success=False,
                error="Claude Code CLI not installed. Install with: npm install -g @anthropic-ai/claude-code",
                exit_code=-1,
            )

        # Build the implementation prompt
        full_prompt = self._build_prompt(prompt, plan_file)

        try:
            logger.info("Invoking Claude Code CLI...")
            logger.debug(f"Prompt: {full_prompt[:200]}...")

            # Run Claude Code in non-interactive mode with the prompt
            result = subprocess.run(
                [
                    "claude",
                    "--print",  # Non-interactive mode, output only
                    "--model", self.model,
                    "--max-turns", str(self.max_turns),
                    "--dangerously-skip-permissions",  # Auto-approve for automation
                    "-p", full_prompt,  # Pass prompt directly
                ],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                # Parse output to find modified files
                files_modified = self._get_modified_files(repo_path)

                logger.info(f"Claude Code completed. Files modified: {len(files_modified)}")

                return ClaudeCodeResult(
                    success=True,
                    output=result.stdout,
                    files_modified=files_modified,
                    exit_code=result.returncode,
                )
            else:
                error_msg = result.stderr or f"Claude Code exited with code {result.returncode}"
                logger.error(f"Claude Code failed: {error_msg}")

                return ClaudeCodeResult(
                    success=False,
                    output=result.stdout,
                    error=error_msg,
                    exit_code=result.returncode,
                )

        except subprocess.TimeoutExpired:
            logger.error(f"Claude Code timed out after {self.timeout} seconds")
            return ClaudeCodeResult(
                success=False,
                error=f"Claude Code timed out after {self.timeout} seconds",
                exit_code=-1,
            )
        except FileNotFoundError:
            return ClaudeCodeResult(
                success=False,
                error="Claude Code CLI not found in PATH",
                exit_code=-1,
            )
        except Exception as e:
            logger.error(f"Unexpected error running Claude Code: {e}")
            return ClaudeCodeResult(
                success=False,
                error=f"Unexpected error running Claude Code: {e}",
                exit_code=-1,
            )

    def _build_prompt(self, prompt: str, plan_file: Optional[Path]) -> str:
        """Build the full implementation prompt.

        Args:
            prompt: Base implementation prompt.
            plan_file: Optional path to improvement plan.

        Returns:
            Full prompt string for Claude Code.
        """
        parts = []

        if plan_file and plan_file.exists():
            parts.append(f"Read the improvement plan at {plan_file} for context.")

        parts.append(prompt)
        parts.append(
            "\nAfter implementing each task:\n"
            "1. Run relevant tests to verify changes work\n"
            "2. Fix any issues before moving to the next task\n"
            "3. Keep changes focused and incremental"
        )

        return "\n\n".join(parts)

    def _get_modified_files(self, repo_path: Path) -> list[str]:
        """Get list of modified files from git status.

        Args:
            repo_path: Repository path.

        Returns:
            List of modified file paths relative to repo.
        """
        modified = []

        try:
            # Check git status for modifications (staged and unstaged)
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        # Format: "XY filename" where XY is status
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            modified.append(parts[1].strip())
        except Exception as e:
            logger.warning(f"Failed to get modified files: {e}")

        return modified


class MockClaudeCodeRunner(ClaudeCodeRunner):
    """Mock Claude Code runner for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize mock runner."""
        super().__init__(*args, **kwargs)
        self.call_count = 0
        self.last_prompt: Optional[str] = None
        self.mock_files_modified: list[str] = []

    def check_installed(self) -> bool:
        """Always return True for mock."""
        return True

    def implement(
        self,
        repo_path: Path,
        prompt: str,
        plan_file: Optional[Path] = None,
    ) -> ClaudeCodeResult:
        """Return mock implementation result.

        Args:
            repo_path: Path to the target repository.
            prompt: Implementation prompt.
            plan_file: Optional path to improvement plan.

        Returns:
            Mock ClaudeCodeResult.
        """
        self.call_count += 1
        self.last_prompt = prompt

        return ClaudeCodeResult(
            success=True,
            output="Mock implementation completed successfully.",
            files_modified=self.mock_files_modified,
            exit_code=0,
        )
