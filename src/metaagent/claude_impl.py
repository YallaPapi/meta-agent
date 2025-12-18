"""Claude API client for autonomous code implementation.

This module provides direct Anthropic API integration for the autonomous
development loop. Unlike claude_runner.py which uses the CLI, this module
calls the API directly and parses file change responses.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .workspace_manager import FileChange, WorkspaceManager

logger = logging.getLogger(__name__)


@dataclass
class ImplementationResult:
    """Result of applying a task implementation."""

    success: bool
    commit_message: str
    files_changed: int
    changes: list[FileChange] = field(default_factory=list)
    error: Optional[str] = None
    raw_response: str = ""
    tokens_used: int = 0


@dataclass
class FixPromptResult:
    """Result from applying a fix prompt."""

    success: bool
    commit_message: str
    files_changed: int
    error: Optional[str] = None


class ClaudeImplementer:
    """Client for Claude API to implement code changes."""

    # System prompt for code implementation
    IMPL_SYSTEM_PROMPT = """You are an expert software engineer implementing changes to a codebase.

When given a task, analyze the context and generate the exact file changes needed.

IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:
{
  "files": [
    {
      "path": "relative/path/to/file.py",
      "content": "full file content here",
      "action": "create|update|delete"
    }
  ],
  "commit_message": "Short description of changes"
}

Rules:
- Use "create" for new files
- Use "update" for modifying existing files (provide FULL content, not diffs)
- Use "delete" for removing files (content can be null)
- Paths should be relative to the repository root
- Include ALL code in the file, not just changes
- Write clean, well-documented code
- Follow existing code style and patterns
- Do NOT include any text outside the JSON object"""

    # System prompt for fixing errors
    FIX_SYSTEM_PROMPT = """You are an expert debugging engineer fixing code issues.

You will be given:
1. The current codebase context
2. Test errors or failures
3. A diagnosis/fix suggestion

Apply the fix and respond with ONLY a valid JSON object in this exact format:
{
  "files": [
    {
      "path": "relative/path/to/file.py",
      "content": "full file content here",
      "action": "update"
    }
  ],
  "commit_message": "fix: description of the fix"
}

Rules:
- Focus on fixing the specific error
- Provide complete file contents, not diffs
- Keep changes minimal and focused
- Do NOT include any text outside the JSON object"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        timeout: int = 120,
    ):
        """Initialize the Claude implementer.

        Args:
            api_key: Anthropic API key. If None, reads from environment.
            model: Claude model to use.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None
        self._api_key = api_key

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
        return self._client

    def apply_task_to_repo(
        self,
        task_description: str,
        workspace: WorkspaceManager,
        repomix_xml: Optional[str] = None,
        dry_run: bool = False,
    ) -> ImplementationResult:
        """Apply a task implementation to the repository.

        Args:
            task_description: Description of the task to implement.
            workspace: WorkspaceManager for the target workspace.
            repomix_xml: Optional Repomix XML context of the codebase.
            dry_run: If True, don't apply changes, just return what would change.

        Returns:
            ImplementationResult with success status and changes.
        """
        logger.info(f"Implementing task: {task_description[:100]}...")

        # Build the prompt
        prompt_parts = [f"## Task to Implement\n{task_description}"]

        if repomix_xml:
            # Truncate if too long
            max_context = 100000
            if len(repomix_xml) > max_context:
                repomix_xml = repomix_xml[:max_context] + "\n... (truncated)"
            prompt_parts.insert(0, f"## Current Codebase\n```xml\n{repomix_xml}\n```")

        user_prompt = "\n\n".join(prompt_parts)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.IMPL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract text content
            raw_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    raw_response += block.text

            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # Parse the JSON response
            changes, commit_message = self._parse_file_changes(raw_response)

            if not changes:
                return ImplementationResult(
                    success=False,
                    commit_message="",
                    files_changed=0,
                    error="No valid file changes found in response",
                    raw_response=raw_response,
                    tokens_used=tokens_used,
                )

            # Apply changes unless dry run
            if not dry_run:
                files_changed = workspace.apply_changes(changes)
            else:
                files_changed = len(changes)
                logger.info(f"[DRY RUN] Would change {files_changed} files")

            return ImplementationResult(
                success=True,
                commit_message=commit_message,
                files_changed=files_changed,
                changes=changes,
                raw_response=raw_response,
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.error(f"Implementation failed: {e}")
            return ImplementationResult(
                success=False,
                commit_message="",
                files_changed=0,
                error=str(e),
            )

    def apply_fix_prompt(
        self,
        fix_prompt: str,
        workspace: WorkspaceManager,
        repomix_xml: Optional[str] = None,
        dry_run: bool = False,
    ) -> FixPromptResult:
        """Apply a fix based on a Perplexity-generated prompt.

        Args:
            fix_prompt: The fix prompt from Perplexity error analysis.
            workspace: WorkspaceManager for the target workspace.
            repomix_xml: Optional Repomix XML context.
            dry_run: If True, don't apply changes.

        Returns:
            FixPromptResult with success status.
        """
        logger.info("Applying fix prompt...")

        prompt_parts = [f"## Fix Instructions\n{fix_prompt}"]

        if repomix_xml:
            max_context = 100000
            if len(repomix_xml) > max_context:
                repomix_xml = repomix_xml[:max_context] + "\n... (truncated)"
            prompt_parts.insert(0, f"## Current Codebase\n```xml\n{repomix_xml}\n```")

        user_prompt = "\n\n".join(prompt_parts)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.FIX_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    raw_response += block.text

            changes, commit_message = self._parse_file_changes(raw_response)

            if not changes:
                return FixPromptResult(
                    success=False,
                    commit_message="",
                    files_changed=0,
                    error="No valid file changes found in fix response",
                )

            if not dry_run:
                files_changed = workspace.apply_changes(changes)
            else:
                files_changed = len(changes)
                logger.info(f"[DRY RUN] Would change {files_changed} files")

            return FixPromptResult(
                success=True,
                commit_message=commit_message or "fix: apply automated fix",
                files_changed=files_changed,
            )

        except Exception as e:
            logger.error(f"Fix application failed: {e}")
            return FixPromptResult(
                success=False,
                commit_message="",
                files_changed=0,
                error=str(e),
            )

    def _parse_file_changes(
        self, response: str
    ) -> tuple[list[FileChange], str]:
        """Parse file changes from Claude's JSON response.

        Args:
            response: Raw response text from Claude.

        Returns:
            Tuple of (list of FileChange, commit message).
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.warning("No JSON object found in response")
            return [], ""

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return [], ""

        files = data.get("files", [])
        commit_message = data.get("commit_message", "implement: automated changes")

        changes = []
        for file_data in files:
            path = file_data.get("path", "")
            content = file_data.get("content")
            action = file_data.get("action", "update")

            if not path:
                continue

            # Normalize action
            action = action.lower()
            if action not in ("create", "update", "delete"):
                action = "update"

            changes.append(FileChange(
                path=path,
                content=content,
                action=action,
            ))

        return changes, commit_message


class MockClaudeImplementer(ClaudeImplementer):
    """Mock implementer for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize mock implementer."""
        super().__init__(*args, **kwargs)
        self.call_count = 0
        self.last_task: Optional[str] = None
        self.mock_response: Optional[dict] = None
        self.should_fail: bool = False

    @property
    def client(self):
        """Return None for mock."""
        return None

    def apply_task_to_repo(
        self,
        task_description: str,
        workspace: WorkspaceManager,
        repomix_xml: Optional[str] = None,
        dry_run: bool = False,
    ) -> ImplementationResult:
        """Return mock implementation result."""
        self.call_count += 1
        self.last_task = task_description

        if self.should_fail:
            return ImplementationResult(
                success=False,
                commit_message="",
                files_changed=0,
                error="Mock failure",
            )

        # Generate mock changes based on task
        if self.mock_response:
            changes = [
                FileChange(
                    path=f["path"],
                    content=f.get("content"),
                    action=f.get("action", "update"),
                )
                for f in self.mock_response.get("files", [])
            ]
            commit_message = self.mock_response.get(
                "commit_message", "mock: implement task"
            )
        else:
            changes = [
                FileChange(
                    path="mock_file.py",
                    content="# Mock implementation\npass\n",
                    action="create",
                )
            ]
            commit_message = "mock: implement task"

        if not dry_run:
            workspace.apply_changes(changes)

        return ImplementationResult(
            success=True,
            commit_message=commit_message,
            files_changed=len(changes),
            changes=changes,
            raw_response=json.dumps(self.mock_response or {}),
            tokens_used=100,
        )

    def apply_fix_prompt(
        self,
        fix_prompt: str,
        workspace: WorkspaceManager,
        repomix_xml: Optional[str] = None,
        dry_run: bool = False,
    ) -> FixPromptResult:
        """Return mock fix result."""
        self.call_count += 1

        if self.should_fail:
            return FixPromptResult(
                success=False,
                commit_message="",
                files_changed=0,
                error="Mock fix failure",
            )

        return FixPromptResult(
            success=True,
            commit_message="fix: mock fix applied",
            files_changed=1,
        )
