"""Tests for CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

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
        assert "--config-dir" in result.stdout
        assert "--mock" in result.stdout

    def test_refine_missing_profile(self) -> None:
        """Test refine command without profile."""
        result = runner.invoke(app, ["refine"])

        # Should fail because --profile is required
        assert result.exit_code != 0

    def test_refine_invalid_repo(self, tmp_path: Path) -> None:
        """Test refine command with invalid repository path."""
        result = runner.invoke(
            app,
            ["refine", "--profile", "test", "--repo", str(tmp_path / "nonexistent")],
        )

        assert result.exit_code != 0
        output = result.stdout + (result.output if hasattr(result, 'output') else "")
        assert "does not exist" in output

    def test_refine_missing_config(self, tmp_path: Path) -> None:
        """Test refine command with missing config directory."""
        # Create a repo dir but no config
        result = runner.invoke(
            app,
            ["refine", "--profile", "test", "--repo", str(tmp_path), "--config-dir", str(tmp_path / "no_config")],
        )

        assert result.exit_code != 0

    def test_list_profiles_missing_config(self, tmp_path: Path) -> None:
        """Test list-profiles with missing config."""
        result = runner.invoke(
            app,
            ["list-profiles", "--config-dir", str(tmp_path / "nonexistent")],
        )

        assert result.exit_code != 0

    def test_refine_mock_mode(self, mock_config) -> None:
        """Test refine command in mock mode."""
        repo_path = mock_config.repo_path
        config_dir = mock_config.config_dir

        # Mock the orchestrator to avoid actual execution
        with patch("metaagent.cli.Orchestrator") as mock_orch:
            from metaagent.orchestrator import RefinementResult
            mock_instance = mock_orch.return_value
            mock_instance.refine.return_value = RefinementResult(
                success=True,
                profile_name="test_profile",
                stages_completed=1,
                stages_failed=0,
            )

            result = runner.invoke(
                app,
                [
                    "refine",
                    "--profile", "test_profile",
                    "--repo", str(repo_path),
                    "--config-dir", str(config_dir),
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

        # Should work even with minimal prompt library
        assert result.exit_code == 0


class TestIterativeMode:
    """Tests for iterative refinement mode."""

    def test_refine_iterative_help(self) -> None:
        """Test refine-iterative command help."""
        result = runner.invoke(app, ["refine-iterative", "--help"])

        assert result.exit_code == 0
        assert "--max-iterations" in result.stdout
        assert "--repo" in result.stdout
        assert "--config-dir" in result.stdout
        assert "--mock" in result.stdout

    def test_refine_iterative_mock_mode(self, mock_config) -> None:
        """Test refine-iterative command in mock mode."""
        repo_path = mock_config.repo_path
        config_dir = mock_config.config_dir

        with patch("metaagent.cli.Orchestrator") as mock_orch:
            from metaagent.orchestrator import RefinementResult, IterationResult
            mock_instance = mock_orch.return_value
            mock_instance.refine_iterative.return_value = RefinementResult(
                success=True,
                profile_name="iterative",
                stages_completed=3,
                stages_failed=0,
                iterations=[
                    IterationResult(
                        iteration=1,
                        prompts_run=["quality_error_analysis"],
                        changes_made=True,
                        committed=False,
                    ),
                ],
            )

            result = runner.invoke(
                app,
                [
                    "refine-iterative",
                    "--repo", str(repo_path),
                    "--config-dir", str(config_dir),
                    "--mock",
                    "--max-iterations", "5",
                ],
            )

            assert result.exit_code == 0
            mock_instance.refine_iterative.assert_called_once_with(max_iterations=5)

    def test_refine_iterative_missing_triage_prompt(self, tmp_path: Path) -> None:
        """Test refine-iterative with missing triage prompt."""
        # Create minimal config without meta_triage prompt
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        # prompts must be a dict, not a list
        (config_dir / "prompts.yaml").write_text("prompts:\n  other_prompt:\n    id: other\n    goal: test\n    template: test")
        (config_dir / "profiles.yaml").write_text("profiles: {}")

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "docs").mkdir()
        (repo / "docs" / "prd.md").write_text("# PRD")

        result = runner.invoke(
            app,
            [
                "refine-iterative",
                "--repo", str(repo),
                "--config-dir", str(config_dir),
                "--mock",
            ],
        )

        assert result.exit_code != 0
        assert "meta_triage" in result.stdout


class TestRepoAgnostic:
    """Tests for repo-agnostic behavior."""

    def test_refine_different_repo(self, mock_config, tmp_path: Path) -> None:
        """Test refining a different repo using meta-agent's config."""
        # Create a separate target repo
        target_repo = tmp_path / "target_repo"
        target_repo.mkdir()
        (target_repo / "docs").mkdir()
        (target_repo / "docs" / "prd.md").write_text("# Target PRD\nThis is a different repo.")
        (target_repo / "src").mkdir()
        (target_repo / "src" / "main.py").write_text("print('hello')")

        config_dir = mock_config.config_dir

        with patch("metaagent.cli.Orchestrator") as mock_orch:
            from metaagent.orchestrator import RefinementResult
            mock_instance = mock_orch.return_value
            mock_instance.refine.return_value = RefinementResult(
                success=True,
                profile_name="test_profile",
                stages_completed=1,
                stages_failed=0,
            )

            result = runner.invoke(
                app,
                [
                    "refine",
                    "--profile", "test_profile",
                    "--repo", str(target_repo),
                    "--config-dir", str(config_dir),
                    "--mock",
                ],
            )

            # Verify the orchestrator was called with correct paths
            assert mock_orch.called
            # Check that target repo is used
            call_config = mock_orch.call_args[0][0]
            assert call_config.repo_path == target_repo

    def test_config_dir_resolution(self, mock_config) -> None:
        """Test that config dir is properly resolved."""
        config_dir = mock_config.config_dir

        result = runner.invoke(
            app,
            ["list-profiles", "--config-dir", str(config_dir)],
        )

        assert result.exit_code == 0
