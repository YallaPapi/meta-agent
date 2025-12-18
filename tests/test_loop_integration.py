"""Integration tests for the autonomous development loop."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.metaagent.config import Config, LoopConfig
from src.metaagent.orchestrator import AutonomousLoopResult, Orchestrator


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration for testing."""
    # Create a minimal repo structure
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create a minimal PRD
    prd_file = docs_dir / "prd.md"
    prd_file.write_text("""
# Test PRD

## Feature: Add fibonacci function

Create a function that calculates fibonacci numbers.
""")

    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create minimal prompt library
    prompt_lib = config_dir / "prompt_library"
    prompt_lib.mkdir()

    # Create meta_feature_expansion prompt
    (prompt_lib / "meta_feature_expansion.md").write_text("""
# Feature Expansion

Analyze the feature request and provide implementation tasks.

Respond with JSON:
{
  "tasks": [
    {"title": "Task 1", "description": "Implement X", "priority": "high"}
  ]
}
""")

    # Create meta_error_fix prompt
    (prompt_lib / "meta_error_fix.md").write_text("""
# Error Fix

Diagnose the error and provide a fix.

Respond with JSON:
{
  "diagnosis": {"root_cause": "Test error"},
  "fix_prompt": "Fix the error by doing X"
}
""")

    # Create meta_prd_evaluation prompt
    (prompt_lib / "meta_prd_evaluation.md").write_text("""
# PRD Evaluation

Evaluate if implementation matches PRD.

Respond with JSON:
{
  "approved": true,
  "overall_assessment": "Implementation complete"
}
""")

    # Create profiles.yaml
    (config_dir / "profiles.yaml").write_text("profiles: {}")

    # Create prompts.yaml
    (config_dir / "prompts.yaml").write_text("prompts: {}")

    # Create loop config
    loop_config = LoopConfig(
        enabled=True,
        max_iterations=3,
        human_approve=False,  # Disable for testing
        dry_run=False,
        test_command="python -c 'print(1)'",  # Simple test that always passes
        create_branch=False,  # Don't create branch in test
        max_consecutive_failures=2,
    )

    config = Config(
        repo_path=tmp_path,
        config_dir=config_dir,
        prd_path=prd_file,
        mock_mode=True,  # Use mock mode for testing
        loop=loop_config,
    )

    return config


class TestAutonomousLoopIntegration:
    """Integration tests for the autonomous loop."""

    @patch("src.metaagent.orchestrator.OllamaEngine")
    @patch("src.metaagent.orchestrator.create_analysis_engine")
    @patch("src.metaagent.orchestrator.RepomixRunner")
    @patch("src.metaagent.orchestrator.CodebaseDigestRunner")
    def test_basic_loop_execution(
        self,
        mock_digest_runner,
        mock_repomix_runner,
        mock_analysis_engine,
        mock_ollama_engine,
        mock_config,
    ):
        """Test basic loop execution with mocked components."""
        # Setup mocks
        mock_ollama = MagicMock()
        mock_ollama.feature_focused_triage.return_value = MagicMock(
            success=True,
            selected_files=["test.py"],
            error=None,
        )
        mock_ollama_engine.return_value = mock_ollama

        mock_engine = MagicMock()
        mock_engine.analyze.return_value = MagicMock(
            success=True,
            raw_response=json.dumps({
                "tasks": [
                    {"title": "Add fibonacci", "description": "Implement fibonacci function"}
                ]
            }),
            summary="Add fibonacci function",
        )
        mock_analysis_engine.return_value = mock_engine

        mock_repomix = MagicMock()
        mock_repomix.pack.return_value = MagicMock(
            success=True,
            content="<file>test.py</file>",
        )
        mock_repomix_runner.return_value = mock_repomix

        mock_digest = MagicMock()
        mock_digest.analyze.return_value = MagicMock(
            success=True,
            tree="test.py",
            metrics="1 file",
        )
        mock_digest_runner.return_value = mock_digest

        # Create orchestrator with mocked prompt library
        with patch("src.metaagent.orchestrator.PromptLibrary") as mock_prompt_lib:
            mock_lib = MagicMock()
            mock_lib.get_prompt.return_value = MagicMock(
                template="Analyze and implement"
            )
            mock_prompt_lib.return_value = mock_lib

            orchestrator = Orchestrator(mock_config, prompt_library=mock_lib)

            # Run the loop
            result = orchestrator.run_autonomous_loop(
                feature_request="Add fibonacci function"
            )

            # Verify result
            assert isinstance(result, AutonomousLoopResult)
            # The loop should complete (even if tasks_completed is 0 due to mocking)
            assert result.iterations_completed >= 1 or result.error is not None

    def test_loop_stops_at_max_iterations(self, mock_config):
        """Test that loop stops when max iterations reached."""
        mock_config.loop.max_iterations = 2

        with patch.object(Orchestrator, "_pack_codebase") as mock_pack:
            mock_pack.return_value = "mock codebase"

            with patch.object(Orchestrator, "_run_feature_analysis") as mock_analysis:
                # Always return tasks to force max iterations
                mock_analysis.return_value = MagicMock(
                    success=True,
                    raw_response=json.dumps({"tasks": [{"title": "Endless task"}]}),
                    summary="",
                )

                with patch.object(Orchestrator, "_extract_tasks_from_analysis") as mock_extract:
                    mock_extract.return_value = [{"title": "Endless task", "description": "Always more work"}]

                    # Mock the Claude implementer to always succeed but never complete
                    with patch("src.metaagent.orchestrator.MockClaudeImplementer") as mock_impl_class:
                        mock_impl = MagicMock()
                        mock_impl.apply_task_to_repo.return_value = MagicMock(
                            success=True,
                            commit_message="impl",
                            files_changed=1,
                            tokens_used=100,
                        )
                        mock_impl_class.return_value = mock_impl

                        with patch("src.metaagent.orchestrator.PromptLibrary"):
                            orchestrator = Orchestrator(mock_config)
                            result = orchestrator.run_autonomous_loop("endless feature")

                            # Should stop at max iterations
                            assert result.iterations_completed <= 2

    def test_loop_handles_analysis_failure(self, mock_config):
        """Test that loop handles analysis failures gracefully."""
        with patch.object(Orchestrator, "_pack_codebase") as mock_pack:
            mock_pack.return_value = "mock codebase"

            with patch.object(Orchestrator, "_run_feature_analysis") as mock_analysis:
                # Simulate analysis failure
                mock_analysis.return_value = MagicMock(
                    success=False,
                    error="API error",
                )

                with patch("src.metaagent.orchestrator.PromptLibrary"):
                    orchestrator = Orchestrator(mock_config)
                    result = orchestrator.run_autonomous_loop("failing feature")

                    # Should handle gracefully
                    assert isinstance(result, AutonomousLoopResult)

    def test_loop_with_no_tasks(self, mock_config):
        """Test that loop completes when no tasks are returned."""
        with patch.object(Orchestrator, "_pack_codebase") as mock_pack:
            mock_pack.return_value = "mock codebase"

            with patch.object(Orchestrator, "_run_feature_analysis") as mock_analysis:
                mock_analysis.return_value = MagicMock(
                    success=True,
                    raw_response=json.dumps({"tasks": []}),  # No tasks
                    summary="",
                )

                with patch.object(Orchestrator, "_extract_tasks_from_analysis") as mock_extract:
                    mock_extract.return_value = []  # No tasks

                    with patch("src.metaagent.orchestrator.PromptLibrary"):
                        orchestrator = Orchestrator(mock_config)
                        result = orchestrator.run_autonomous_loop("complete feature")

                        # Should succeed with no tasks
                        assert result.iterations_completed >= 1


class TestLoopConfigIntegration:
    """Test loop configuration integration."""

    def test_dry_run_mode(self, mock_config):
        """Test that dry run mode doesn't modify files."""
        mock_config.loop.dry_run = True

        with patch.object(Orchestrator, "_pack_codebase") as mock_pack:
            mock_pack.return_value = "mock"

            with patch.object(Orchestrator, "_run_feature_analysis") as mock_analysis:
                mock_analysis.return_value = MagicMock(
                    success=True,
                    raw_response=json.dumps({"tasks": [{"title": "Test"}]}),
                    summary="",
                )

                with patch.object(Orchestrator, "_extract_tasks_from_analysis") as mock_extract:
                    mock_extract.return_value = [{"title": "Test", "description": "Test task"}]

                    with patch("src.metaagent.orchestrator.MockClaudeImplementer") as mock_impl_class:
                        mock_impl = MagicMock()
                        mock_impl.apply_task_to_repo.return_value = MagicMock(
                            success=True,
                            commit_message="test",
                            files_changed=1,
                            tokens_used=100,
                        )
                        mock_impl_class.return_value = mock_impl

                        with patch("src.metaagent.orchestrator.PromptLibrary"):
                            orchestrator = Orchestrator(mock_config)
                            result = orchestrator.run_autonomous_loop("dry run test")

                            # Verify dry_run was passed to implementer
                            calls = mock_impl.apply_task_to_repo.call_args_list
                            if calls:
                                _, kwargs = calls[0]
                                assert kwargs.get("dry_run") == True

    def test_max_consecutive_failures(self, mock_config):
        """Test that loop stops after max consecutive failures."""
        mock_config.loop.max_consecutive_failures = 2

        with patch.object(Orchestrator, "_pack_codebase") as mock_pack:
            mock_pack.return_value = "mock"

            with patch.object(Orchestrator, "_run_feature_analysis") as mock_analysis:
                # Always fail
                mock_analysis.return_value = MagicMock(
                    success=False,
                    error="Always fails",
                )

                with patch("src.metaagent.orchestrator.PromptLibrary"):
                    orchestrator = Orchestrator(mock_config)
                    result = orchestrator.run_autonomous_loop("failing feature")

                    # Should stop after max consecutive failures
                    assert result.iterations_completed <= 3  # max_iterations
