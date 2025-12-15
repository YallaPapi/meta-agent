"""Repomix integration for codebase packing."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RepomixResult:
    """Result from running Repomix on a repository."""

    content: str
    success: bool
    error: Optional[str] = None
    truncated: bool = False
    original_size: int = 0


class RepomixRunner:
    """Runs Repomix to pack a codebase into a single file."""

    def __init__(self, timeout: int = 120, max_chars: int = 400000):
        """Initialize the Repomix runner.

        Args:
            timeout: Timeout in seconds for Repomix execution.
            max_chars: Maximum characters to keep (approximate token budget).
        """
        self.timeout = timeout
        self.max_chars = max_chars

    def pack(self, repo_path: Path) -> RepomixResult:
        """Pack a repository using Repomix.

        Args:
            repo_path: Path to the repository to pack.

        Returns:
            RepomixResult with packed content or error information.
        """
        if not repo_path.exists():
            return RepomixResult(
                content="",
                success=False,
                error=f"Repository path does not exist: {repo_path}",
            )

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False
            ) as tmp_file:
                output_path = Path(tmp_file.name)

            try:
                # Try repomix directly first (if installed globally)
                # Fall back to npx repomix if direct call fails
                # Use list-based commands to avoid shell injection vulnerabilities
                #
                # Note: No default ignores - we pack the full codebase.
                # Ollama handles the full context locally (free, no token limits),
                # then sends only relevant findings to Perplexity.
                cmd_options = [
                    ["repomix", "--output", str(output_path), "--style", "markdown"],
                    ["npx", "repomix", "--output", str(output_path), "--style", "markdown"],
                ]

                result = None
                last_error = None

                for cmd in cmd_options:
                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=repo_path,
                            capture_output=True,
                            text=True,
                            timeout=self.timeout,
                            shell=False,  # Explicit shell=False for security
                        )
                        if result.returncode == 0:
                            break
                        last_error = result.stderr
                    except FileNotFoundError:
                        last_error = f"Command not found: {cmd[0]}"
                        continue

                if result is None or result.returncode != 0:
                    return RepomixResult(
                        content="",
                        success=False,
                        error=f"Repomix failed: {last_error}",
                    )

                content = output_path.read_text(encoding="utf-8")
                original_size = len(content)
                truncated = False

                if len(content) > self.max_chars:
                    content = self._truncate_content(content)
                    truncated = True

                return RepomixResult(
                    content=content,
                    success=True,
                    truncated=truncated,
                    original_size=original_size,
                )

            finally:
                output_path.unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return RepomixResult(
                content="",
                success=False,
                error=f"Repomix timed out after {self.timeout} seconds",
            )
        except FileNotFoundError:
            return RepomixResult(
                content="",
                success=False,
                error="Repomix not found. Install with: npm install -g repomix",
            )
        except Exception as e:
            return RepomixResult(
                content="",
                success=False,
                error=f"Unexpected error running Repomix: {e}",
            )

    def _truncate_content(self, content: str) -> str:
        """Truncate content to fit within max_chars while preserving structure.

        Args:
            content: The content to truncate.

        Returns:
            Truncated content with a notice appended.
        """
        # Try to truncate at a sensible boundary
        truncated = content[: self.max_chars]

        # Find the last complete line
        last_newline = truncated.rfind("\n")
        if last_newline > self.max_chars * 0.8:
            truncated = truncated[:last_newline]

        truncated += "\n\n[... content truncated due to size limits ...]"
        return truncated
