"""Unit tests for claude_impl.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.metaagent.claude_impl import (
    ClaudeImplementer,
    FixPromptResult,
    ImplementationResult,
    MockClaudeImplementer,
)
from src.metaagent.workspace_manager import FileChange, WorkspaceManager


class TestImplementationResult:
    """Tests for ImplementationResult dataclass."""

    def test_successful_result(self):
        result = ImplementationResult(
            success=True,
            commit_message="feat: add feature",
            files_changed=3,
            changes=[
                FileChange(path="a.py", content="# a", action="create"),
                FileChange(path="b.py", content="# b", action="update"),
            ],
            tokens_used=500,
        )
        assert result.success
        assert result.files_changed == 3
        assert len(result.changes) == 2
        assert result.tokens_used == 500
        assert result.error is None

    def test_failed_result(self):
        result = ImplementationResult(
            success=False,
            commit_message="",
            files_changed=0,
            error="API error",
        )
        assert not result.success
        assert result.error == "API error"


class TestFixPromptResult:
    """Tests for FixPromptResult dataclass."""

    def test_successful_fix(self):
        result = FixPromptResult(
            success=True,
            commit_message="fix: resolve error",
            files_changed=1,
        )
        assert result.success
        assert "fix" in result.commit_message

    def test_failed_fix(self):
        result = FixPromptResult(
            success=False,
            commit_message="",
            files_changed=0,
            error="Could not parse response",
        )
        assert not result.success


class TestClaudeImplementerParseFileChanges:
    """Tests for _parse_file_changes method."""

    def test_parse_valid_json(self):
        impl = ClaudeImplementer()
        response = json.dumps({
            "files": [
                {"path": "test.py", "content": "# test", "action": "create"}
            ],
            "commit_message": "add test file"
        })

        changes, message = impl._parse_file_changes(response)

        assert len(changes) == 1
        assert changes[0].path == "test.py"
        assert changes[0].action == "create"
        assert message == "add test file"

    def test_parse_json_with_extra_text(self):
        impl = ClaudeImplementer()
        response = """
        Here's the implementation:
        {"files": [{"path": "x.py", "content": "#x", "action": "update"}], "commit_message": "update"}
        Done!
        """

        changes, message = impl._parse_file_changes(response)

        assert len(changes) == 1
        assert changes[0].path == "x.py"

    def test_parse_invalid_json(self):
        impl = ClaudeImplementer()
        response = "This is not valid JSON"

        changes, message = impl._parse_file_changes(response)

        assert len(changes) == 0
        assert message == ""

    def test_parse_missing_files(self):
        impl = ClaudeImplementer()
        response = json.dumps({"commit_message": "no files"})

        changes, message = impl._parse_file_changes(response)

        assert len(changes) == 0

    def test_parse_normalizes_action(self):
        impl = ClaudeImplementer()
        response = json.dumps({
            "files": [
                {"path": "a.py", "content": "#a", "action": "CREATE"},
                {"path": "b.py", "content": "#b", "action": "invalid"}
            ],
            "commit_message": "test"
        })

        changes, _ = impl._parse_file_changes(response)

        assert changes[0].action == "create"
        assert changes[1].action == "update"  # invalid normalized to update


class TestMockClaudeImplementer:
    """Tests for MockClaudeImplementer."""

    def test_mock_apply_task_success(self, tmp_path):
        mock_impl = MockClaudeImplementer()
        workspace = WorkspaceManager(tmp_path)

        result = mock_impl.apply_task_to_repo(
            task_description="Add a hello world function",
            workspace=workspace,
        )

        assert result.success
        assert mock_impl.call_count == 1
        assert mock_impl.last_task == "Add a hello world function"

    def test_mock_apply_task_with_custom_response(self, tmp_path):
        mock_impl = MockClaudeImplementer()
        mock_impl.mock_response = {
            "files": [
                {"path": "custom.py", "content": "# custom", "action": "create"}
            ],
            "commit_message": "custom commit"
        }
        workspace = WorkspaceManager(tmp_path)

        result = mock_impl.apply_task_to_repo(
            task_description="Custom task",
            workspace=workspace,
        )

        assert result.success
        assert result.commit_message == "custom commit"
        assert (tmp_path / "custom.py").exists()

    def test_mock_apply_task_failure(self, tmp_path):
        mock_impl = MockClaudeImplementer()
        mock_impl.should_fail = True
        workspace = WorkspaceManager(tmp_path)

        result = mock_impl.apply_task_to_repo(
            task_description="Failing task",
            workspace=workspace,
        )

        assert not result.success
        assert result.error == "Mock failure"

    def test_mock_dry_run(self, tmp_path):
        mock_impl = MockClaudeImplementer()
        workspace = WorkspaceManager(tmp_path)

        result = mock_impl.apply_task_to_repo(
            task_description="Dry run task",
            workspace=workspace,
            dry_run=True,
        )

        assert result.success
        # In dry run mode, files should NOT be created
        assert not (tmp_path / "mock_file.py").exists()

    def test_mock_apply_fix_prompt_success(self, tmp_path):
        mock_impl = MockClaudeImplementer()
        workspace = WorkspaceManager(tmp_path)

        result = mock_impl.apply_fix_prompt(
            fix_prompt="Fix the type error",
            workspace=workspace,
        )

        assert result.success
        assert "fix" in result.commit_message

    def test_mock_apply_fix_prompt_failure(self, tmp_path):
        mock_impl = MockClaudeImplementer()
        mock_impl.should_fail = True
        workspace = WorkspaceManager(tmp_path)

        result = mock_impl.apply_fix_prompt(
            fix_prompt="Fix that fails",
            workspace=workspace,
        )

        assert not result.success


class TestClaudeImplementerIntegration:
    """Integration-style tests (with mocked API)."""

    @patch("anthropic.Anthropic")
    def test_apply_task_with_mocked_api(self, mock_anthropic_class, tmp_path):
        # Setup mock response
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps({
                    "files": [
                        {"path": "impl.py", "content": "def impl(): pass", "action": "create"}
                    ],
                    "commit_message": "implement feature"
                })
            )
        ]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response

        impl = ClaudeImplementer(api_key="test-key")
        workspace = WorkspaceManager(tmp_path)

        result = impl.apply_task_to_repo(
            task_description="Implement feature",
            workspace=workspace,
        )

        assert result.success
        assert result.files_changed == 1
        assert result.tokens_used == 150
        assert (tmp_path / "impl.py").exists()

    @patch("anthropic.Anthropic")
    def test_apply_task_api_error(self, mock_anthropic_class, tmp_path):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        impl = ClaudeImplementer(api_key="test-key")
        workspace = WorkspaceManager(tmp_path)

        result = impl.apply_task_to_repo(
            task_description="Will fail",
            workspace=workspace,
        )

        assert not result.success
        assert "API error" in result.error
