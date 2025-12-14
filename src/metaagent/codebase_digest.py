"""Codebase-digest integration for directory tree and metrics."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DigestResult:
    """Result from running codebase-digest on a repository."""

    tree: str  # Directory tree structure
    metrics: str  # File counts, sizes, token estimates
    content: str  # Full output including file contents (optional)
    success: bool
    error: Optional[str] = None
    format: str = "markdown"


class CodebaseDigestRunner:
    """Runs codebase-digest to generate directory tree and metrics."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        output_format: str = "markdown",
        include_content: bool = False,
        max_size_kb: int = 500,
    ):
        """Initialize the codebase-digest runner.

        Args:
            max_depth: Maximum directory traversal depth (None for unlimited).
            output_format: Output format (text, json, markdown, xml, html).
            include_content: Whether to include file contents in output.
            max_size_kb: Maximum file size to include (in KB).
        """
        self.max_depth = max_depth
        self.output_format = output_format
        self.include_content = include_content
        self.max_size_kb = max_size_kb

    def analyze(self, repo_path: Path) -> DigestResult:
        """Analyze a repository using codebase-digest.

        Args:
            repo_path: Path to the repository to analyze.

        Returns:
            DigestResult with tree, metrics, and optionally full content.
        """
        if not repo_path.exists():
            return DigestResult(
                tree="",
                metrics="",
                content="",
                success=False,
                error=f"Repository path does not exist: {repo_path}",
            )

        try:
            # Create temp file for output
            ext = self._get_extension()
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=ext, delete=False
            ) as tmp_file:
                output_path = Path(tmp_file.name)

            # Build command
            cmd = ["cdigest", str(repo_path)]

            if self.max_depth is not None:
                cmd.extend(["-d", str(self.max_depth)])

            cmd.extend(["-o", self.output_format])
            cmd.extend(["--max-size", str(self.max_size_kb)])
            cmd.append("--show-size")

            if not self.include_content:
                cmd.append("--no-content")

            cmd.extend(["-f", str(output_path)])

            # Run codebase-digest (suppress debug output by redirecting stderr)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=repo_path,
            )

            # cdigest may return 0 even with warnings, check if output file exists
            content = ""
            if output_path.exists():
                try:
                    content = output_path.read_text(encoding="utf-8")
                except Exception:
                    pass
                finally:
                    output_path.unlink(missing_ok=True)

            # If we got content, consider it a success regardless of return code
            if content and len(content) > 50:
                # Filter out debug lines if they made it into content
                lines = content.split("\n")
                filtered_lines = [
                    line for line in lines
                    if not line.startswith("Debug:") and not line.startswith("Analyzing:")
                ]
                content = "\n".join(filtered_lines)

                # Parse the output to extract tree and metrics
                tree, metrics = self._parse_output(content)

                return DigestResult(
                    tree=tree,
                    metrics=metrics,
                    content=content if self.include_content else "",
                    success=True,
                    format=self.output_format,
                )

            # No content - check for errors
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return DigestResult(
                    tree="",
                    metrics="",
                    content="",
                    success=False,
                    error=f"codebase-digest failed: {error_msg}",
                )

            return DigestResult(
                tree="",
                metrics="",
                content="",
                success=False,
                error="codebase-digest produced no output",
            )

        except subprocess.TimeoutExpired:
            return DigestResult(
                tree="",
                metrics="",
                content="",
                success=False,
                error="codebase-digest timed out after 60 seconds",
            )
        except FileNotFoundError:
            return DigestResult(
                tree="",
                metrics="",
                content="",
                success=False,
                error="codebase-digest not found. Install with: pip install codebase-digest",
            )
        except Exception as e:
            return DigestResult(
                tree="",
                metrics="",
                content="",
                success=False,
                error=f"Unexpected error running codebase-digest: {e}",
            )

    def _get_extension(self) -> str:
        """Get file extension for output format."""
        extensions = {
            "text": ".txt",
            "json": ".json",
            "markdown": ".md",
            "xml": ".xml",
            "html": ".html",
        }
        return extensions.get(self.output_format, ".txt")

    def _parse_output(self, content: str) -> tuple[str, str]:
        """Parse output to extract tree and metrics sections.

        Args:
            content: Full codebase-digest output.

        Returns:
            Tuple of (tree_section, metrics_section).
        """
        # For markdown format, look for sections
        if self.output_format == "markdown":
            tree = ""
            metrics = ""

            lines = content.split("\n")
            current_section = None
            section_lines: list[str] = []

            for line in lines:
                if line.startswith("## ") or line.startswith("# "):
                    # Save previous section
                    if current_section == "tree":
                        tree = "\n".join(section_lines)
                    elif current_section == "metrics":
                        metrics = "\n".join(section_lines)

                    # Start new section
                    section_lines = [line]
                    lower_line = line.lower()
                    if "tree" in lower_line or "structure" in lower_line or "directory" in lower_line:
                        current_section = "tree"
                    elif "metric" in lower_line or "statistic" in lower_line or "summary" in lower_line:
                        current_section = "metrics"
                    else:
                        current_section = None
                else:
                    section_lines.append(line)

            # Save last section
            if current_section == "tree":
                tree = "\n".join(section_lines)
            elif current_section == "metrics":
                metrics = "\n".join(section_lines)

            # If parsing didn't find sections, return full content as tree
            if not tree and not metrics:
                return content, ""

            return tree, metrics

        # For other formats, return full content
        return content, ""


def get_codebase_context(
    repo_path: Path,
    include_tree: bool = True,
    include_metrics: bool = True,
    max_depth: int = 5,
) -> str:
    """Get codebase context from codebase-digest.

    Convenience function for getting formatted codebase context.

    Args:
        repo_path: Path to the repository.
        include_tree: Include directory tree in output.
        include_metrics: Include metrics in output.
        max_depth: Maximum directory depth.

    Returns:
        Formatted string with codebase context.
    """
    runner = CodebaseDigestRunner(
        max_depth=max_depth,
        output_format="markdown",
        include_content=False,
    )

    result = runner.analyze(repo_path)

    if not result.success:
        return f"[Codebase-digest failed: {result.error}]"

    sections = []

    if include_tree and result.tree:
        sections.append("### Directory Structure\n")
        sections.append(result.tree)

    if include_metrics and result.metrics:
        sections.append("\n### Codebase Metrics\n")
        sections.append(result.metrics)

    return "\n".join(sections) if sections else "[No codebase context available]"
