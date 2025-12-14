"""Tests for codebase-digest integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metaagent.codebase_digest import (
    CodebaseDigestRunner,
    DigestResult,
    get_codebase_context,
)


class TestCodebaseDigestRunner:
    """Tests for CodebaseDigestRunner."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        runner = CodebaseDigestRunner()

        assert runner.max_depth is None
        assert runner.output_format == "markdown"
        assert runner.include_content is False

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        runner = CodebaseDigestRunner(
            max_depth=5,
            output_format="json",
            include_content=True,
            max_size_kb=1000,
        )

        assert runner.max_depth == 5
        assert runner.output_format == "json"
        assert runner.include_content is True
        assert runner.max_size_kb == 1000

    def test_analyze_nonexistent_path(self, tmp_path: Path) -> None:
        """Test analyzing a non-existent path."""
        runner = CodebaseDigestRunner()
        result = runner.analyze(tmp_path / "nonexistent")

        assert not result.success
        assert "does not exist" in result.error

    @patch("metaagent.codebase_digest.subprocess.run")
    def test_analyze_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test successful analysis with mocked subprocess."""
        # Create temp dir
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")

        # Mock subprocess result
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="## Directory Structure\n```\nsrc/\n  main.py\n```",
            stderr="",
        )

        runner = CodebaseDigestRunner()
        result = runner.analyze(tmp_path)

        # Should call subprocess
        assert mock_run.called

    @patch("metaagent.codebase_digest.subprocess.run")
    def test_analyze_failure(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test handling of subprocess failure."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )

        runner = CodebaseDigestRunner()
        result = runner.analyze(tmp_path)

        assert not result.success
        assert "failed" in result.error.lower()

    @patch("metaagent.codebase_digest.subprocess.run")
    def test_analyze_timeout(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test handling of subprocess timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="cdigest", timeout=60)

        runner = CodebaseDigestRunner()
        result = runner.analyze(tmp_path)

        assert not result.success
        assert "timed out" in result.error.lower()

    @patch("metaagent.codebase_digest.subprocess.run")
    def test_analyze_not_found(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test handling when codebase-digest is not installed."""
        mock_run.side_effect = FileNotFoundError()

        runner = CodebaseDigestRunner()
        result = runner.analyze(tmp_path)

        assert not result.success
        assert "not found" in result.error.lower()

    def test_get_extension(self) -> None:
        """Test file extension mapping."""
        runner = CodebaseDigestRunner(output_format="markdown")
        assert runner._get_extension() == ".md"

        runner = CodebaseDigestRunner(output_format="json")
        assert runner._get_extension() == ".json"

        runner = CodebaseDigestRunner(output_format="text")
        assert runner._get_extension() == ".txt"

    def test_parse_output_markdown(self) -> None:
        """Test parsing markdown output."""
        runner = CodebaseDigestRunner(output_format="markdown")

        content = """## Directory Structure
```
src/
  main.py
```

## Statistics
Files: 10
Lines: 500
"""
        tree, metrics = runner._parse_output(content)

        assert "Directory Structure" in tree or "src/" in tree


class TestGetCodebaseContext:
    """Tests for get_codebase_context helper function."""

    @patch("metaagent.codebase_digest.CodebaseDigestRunner.analyze")
    def test_success(self, mock_analyze: MagicMock, tmp_path: Path) -> None:
        """Test successful context retrieval."""
        mock_analyze.return_value = DigestResult(
            tree="src/\n  main.py",
            metrics="Files: 1",
            content="",
            success=True,
        )

        result = get_codebase_context(tmp_path)

        assert "Directory Structure" in result
        assert "Codebase Metrics" in result

    @patch("metaagent.codebase_digest.CodebaseDigestRunner.analyze")
    def test_failure(self, mock_analyze: MagicMock, tmp_path: Path) -> None:
        """Test handling of analysis failure."""
        mock_analyze.return_value = DigestResult(
            tree="",
            metrics="",
            content="",
            success=False,
            error="Test error",
        )

        result = get_codebase_context(tmp_path)

        assert "failed" in result.lower()
        assert "Test error" in result
