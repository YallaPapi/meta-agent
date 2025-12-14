"""Tests for CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from metaagent.cli import app

runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_version(self) -> None:
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "metaagent version" in result.stdout

    def test_help(self) -> None:
        """Test --help flag."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Meta-agent for automated codebase refinement" in result.stdout

    def test_refine_help(self) -> None:
        """Test refine command help."""
        result = runner.invoke(app, ["refine", "--help"])

        assert result.exit_code == 0
        assert "--repo" in result.stdout
        assert "--mock" in result.stdout
        assert "--max-iterations" in result.stdout
        assert "iterative refinement" in result.stdout.lower()

    def test_refine_invalid_repo(self, tmp_path: Path) -> None:
        """Test refine command with invalid repository path."""
        result = runner.invoke(
            app,
            ["refine", "--repo", str(tmp_path / "nonexistent")],
        )

        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, 'output') else "")
        assert "does not exist" in output

    def test_refine_missing_config(self, tmp_path: Path) -> None:
        """Test refine command with missing config directory."""
        result = runner.invoke(
            app,
            ["refine", "--repo", str(tmp_path)],
        )

        assert result.exit_code != 0

    def test_list_profiles_missing_config(self, tmp_path: Path) -> None:
        """Test list-profiles with missing config."""
        result = runner.invoke(
            app,
            ["list-profiles", "--config-dir", str(tmp_path / "nonexistent")],
        )

        assert result.exit_code != 0

    def test_refine_mock_mode(self, mock_config, tmp_path: Path) -> None:
        """Test refine command in mock mode."""
        repo_path = mock_config.repo_path

        # Mock the orchestrator to avoid actual execution
        with patch("metaagent.cli.Orchestrator") as mock_orch:
            from metaagent.orchestrator import RefinementResult
            mock_instance = mock_orch.return_value
            mock_instance.refine.return_value = RefinementResult(
                success=True,
                profile_name="iterative",
                stages_completed=1,
                stages_failed=0,
                iterations=[],
            )

            result = runner.invoke(
                app,
                [
                    "refine",
                    "--repo",
                    str(repo_path),
                    "--mock",
                ],
            )

            # Should succeed with mock mode
            assert result.exit_code == 0 or "completed" in result.stdout.lower()

    def test_list_profiles(self, mock_config) -> None:
        """Test list-profiles command."""
        config_dir = mock_config.config_dir

        result = runner.invoke(
            app,
            ["list-profiles", "--config-dir", str(config_dir)],
        )

        assert result.exit_code == 0
        assert "Test Profile" in result.stdout

    def test_list_prompts(self, mock_config) -> None:
        """Test list-prompts command."""
        config_dir = mock_config.config_dir

        result = runner.invoke(
            app,
            ["list-prompts", "--config-dir", str(config_dir)],
        )

        # Should work even with empty prompt library
        assert result.exit_code == 0
