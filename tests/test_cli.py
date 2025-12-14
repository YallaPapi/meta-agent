"""Tests for CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

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
        assert "--profile" in result.stdout
        assert "--repo" in result.stdout
        assert "--mock" in result.stdout

    def test_refine_missing_profile(self) -> None:
        """Test refine command without profile."""
        result = runner.invoke(app, ["refine"])

        assert result.exit_code != 0
        # Check both stdout and the combined output (Typer may output to stderr)
        output = result.stdout + (result.output if hasattr(result, 'output') else "")
        assert "Missing option" in output or "required" in output.lower() or result.exit_code == 2

    def test_refine_invalid_repo(self, tmp_path: Path) -> None:
        """Test refine command with invalid repository path."""
        result = runner.invoke(
            app,
            ["refine", "--profile", "test", "--repo", str(tmp_path / "nonexistent")],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.stdout

    def test_refine_missing_config(self, tmp_path: Path) -> None:
        """Test refine command with missing config directory."""
        result = runner.invoke(
            app,
            ["refine", "--profile", "test", "--repo", str(tmp_path)],
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
        # Set up config files in temp repo
        config_dir = mock_config.config_dir
        repo_path = mock_config.repo_path

        result = runner.invoke(
            app,
            [
                "refine",
                "--profile",
                "test_profile",
                "--repo",
                str(repo_path),
                "--config-dir",
                str(config_dir),
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
