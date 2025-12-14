"""Claude Code CLI integration for automated implementation.

This module provides structured report generation for the meta-agent pipeline.
Instead of spawning separate Claude Code subprocesses, it returns structured
data that can be consumed by the calling Claude Code session.

The key insight: the meta-agent generates analysis and task reports that should
flow back to the CURRENT Claude Code session (the one running this code),
not spawn a separate process.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Structured Data Classes for Task Analysis
# =============================================================================


@dataclass
class TaskAnalysis:
    """Analysis of a single implementation task."""

    task_id: str
    title: str
    description: str
    priority: str = "medium"  # critical, high, medium, low
    estimated_complexity: str = "medium"  # low, medium, high
    affected_files: list[str] = field(default_factory=list)
    implementation_steps: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass
class ImplementationRecommendation:
    """Recommendation for implementing a specific change."""

    target_file: str
    change_type: str  # create, modify, delete
    description: str
    code_snippet: Optional[str] = None
    rationale: str = ""
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class ImplementationReport:
    """Structured report from implementation analysis.

    This is the key return type that replaces subprocess spawning.
    The report contains all the information needed for the CURRENT
    Claude Code session to implement the changes.
    """

    tasks: list[TaskAnalysis] = field(default_factory=list)
    recommendations: list[ImplementationRecommendation] = field(default_factory=list)
    implementation_plan: str = ""  # Markdown formatted plan
    estimated_changes: dict[str, int] = field(default_factory=dict)  # file -> estimated lines
    success: bool = True
    error: Optional[str] = None
    total_estimated_effort: str = ""  # e.g., "2-4 hours"

    def to_markdown(self) -> str:
        """Generate markdown summary of the implementation report."""
        lines = [
            "# Implementation Report",
            "",
            f"**Status:** {'Success' if self.success else f'Failed: {self.error}'}",
            f"**Tasks:** {len(self.tasks)}",
            f"**Estimated Effort:** {self.total_estimated_effort or 'Not estimated'}",
            "",
        ]

        if self.tasks:
            lines.append("## Tasks")
            lines.append("")
            for task in self.tasks:
                priority_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(task.priority.lower(), "⚪")
                lines.append(f"### {priority_emoji} {task.title}")
                lines.append("")
                lines.append(f"**Priority:** {task.priority} | **Complexity:** {task.estimated_complexity}")
                lines.append("")
                lines.append(task.description)
                if task.affected_files:
                    lines.append("")
                    lines.append(f"**Files:** {', '.join(task.affected_files)}")
                lines.append("")

        if self.implementation_plan:
            lines.append("## Implementation Plan")
            lines.append("")
            lines.append(self.implementation_plan)

        return "\n".join(lines)


@dataclass
class ClaudeCodeResult:
    """Result from a Claude Code execution (legacy compatibility)."""

    success: bool
    output: str = ""
    error: Optional[str] = None
    files_modified: list[str] = field(default_factory=list)
    exit_code: int = 0

    # New field for structured report
    implementation_report: Optional[ImplementationReport] = None


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


    def analyze_plan(self, plan_file: Path) -> ImplementationReport:
        """Analyze an improvement plan and return structured report.

        This method replaces subprocess spawning. Instead of running Claude Code
        as a subprocess, it parses the plan and returns structured data that
        the CURRENT Claude Code session can use to implement changes.

        Args:
            plan_file: Path to the improvement plan markdown file.

        Returns:
            ImplementationReport with parsed tasks and recommendations.
        """
        if not plan_file.exists():
            return ImplementationReport(
                success=False,
                error=f"Plan file not found: {plan_file}",
            )

        content = plan_file.read_text(encoding="utf-8")
        tasks = self._extract_tasks_from_plan(content)
        recommendations = self._generate_recommendations(tasks)
        plan_text = self._create_implementation_plan(tasks)
        effort = self._estimate_effort(tasks)

        return ImplementationReport(
            tasks=tasks,
            recommendations=recommendations,
            implementation_plan=plan_text,
            estimated_changes=self._estimate_file_changes(tasks),
            success=True,
            total_estimated_effort=effort,
        )

    def _extract_tasks_from_plan(self, content: str) -> list[TaskAnalysis]:
        """Extract tasks from improvement plan markdown.

        Parses markdown checkboxes and task descriptions from the plan file.

        Args:
            content: Plan file content.

        Returns:
            List of TaskAnalysis objects.
        """
        tasks = []

        # Find priority sections
        priority_sections = {
            "critical": re.findall(
                r'### \[CRITICAL\].*?\n(.*?)(?=### \[|## |$)',
                content,
                re.DOTALL | re.IGNORECASE
            ),
            "high": re.findall(
                r'### \[HIGH\].*?\n(.*?)(?=### \[|## |$)',
                content,
                re.DOTALL | re.IGNORECASE
            ),
            "medium": re.findall(
                r'### \[MEDIUM\].*?\n(.*?)(?=### \[|## |$)',
                content,
                re.DOTALL | re.IGNORECASE
            ),
            "low": re.findall(
                r'### \[LOW\].*?\n(.*?)(?=### \[|## |$)',
                content,
                re.DOTALL | re.IGNORECASE
            ),
        }

        # Parse tasks from each section
        task_pattern = re.compile(
            r'- \[ \] \*\*(.+?)\*\*(?:\s*\(`([^`]+)`\))?\s*\n\s*-\s*(.+?)(?=\n- \[|$)',
            re.DOTALL
        )

        task_id = 1
        for priority, sections in priority_sections.items():
            for section in sections:
                for match in task_pattern.finditer(section):
                    title = match.group(1).strip()
                    file_ref = match.group(2) or ""
                    description = match.group(3).strip()

                    task = TaskAnalysis(
                        task_id=f"task-{task_id}",
                        title=title,
                        description=description,
                        priority=priority,
                        estimated_complexity=self._estimate_complexity(title, description),
                        affected_files=[file_ref] if file_ref else [],
                        implementation_steps=self._generate_steps(title, description),
                        risks=self._identify_risks(description),
                    )
                    tasks.append(task)
                    task_id += 1

        # Fallback: try simpler checkbox pattern if no priority sections found
        if not tasks:
            simple_pattern = re.compile(
                r'- \[ \] \*?\*?(.+?)\*?\*?(?:\s*\(`([^`]+)`\))?(?:\n\s*-\s*(.+?))?(?=\n- \[|$)',
                re.DOTALL
            )
            for match in simple_pattern.finditer(content):
                title = match.group(1).strip()
                file_ref = match.group(2) or ""
                description = match.group(3).strip() if match.group(3) else title

                task = TaskAnalysis(
                    task_id=f"task-{task_id}",
                    title=title,
                    description=description,
                    priority="medium",
                    estimated_complexity="medium",
                    affected_files=[file_ref] if file_ref else [],
                )
                tasks.append(task)
                task_id += 1

        return tasks

    def _estimate_complexity(self, title: str, description: str) -> str:
        """Estimate task complexity based on title and description."""
        text = (title + " " + description).lower()

        high_indicators = ["refactor", "architecture", "redesign", "rewrite", "migration"]
        low_indicators = ["fix typo", "update comment", "rename", "simple"]

        if any(ind in text for ind in high_indicators):
            return "high"
        elif any(ind in text for ind in low_indicators):
            return "low"
        return "medium"

    def _generate_steps(self, title: str, description: str) -> list[str]:
        """Generate implementation steps for a task."""
        steps = [
            f"1. Read and understand: {title}",
            "2. Identify all affected code locations",
            "3. Make the required changes",
            "4. Run tests to verify",
            "5. Review and refine",
        ]
        return steps

    def _identify_risks(self, description: str) -> list[str]:
        """Identify potential risks from task description."""
        risks = []
        text = description.lower()

        if "breaking change" in text or "breaking" in text:
            risks.append("Potential breaking change")
        if "refactor" in text:
            risks.append("Refactoring may affect multiple files")
        if "database" in text or "schema" in text:
            risks.append("Database changes require migration")
        if "api" in text:
            risks.append("API changes may affect clients")

        return risks

    def _generate_recommendations(self, tasks: list[TaskAnalysis]) -> list[ImplementationRecommendation]:
        """Generate implementation recommendations from tasks."""
        recommendations = []

        for task in tasks:
            for file_path in task.affected_files:
                rec = ImplementationRecommendation(
                    target_file=file_path,
                    change_type="modify",
                    description=task.description,
                    rationale=f"Task: {task.title}",
                    prerequisites=[],
                )
                recommendations.append(rec)

        return recommendations

    def _create_implementation_plan(self, tasks: list[TaskAnalysis]) -> str:
        """Create markdown implementation plan from tasks."""
        if not tasks:
            return "No tasks identified."

        lines = [
            "## Recommended Implementation Order",
            "",
            "Tasks are sorted by priority (critical → high → medium → low).",
            "",
        ]

        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_tasks = sorted(tasks, key=lambda t: priority_order.get(t.priority, 2))

        for i, task in enumerate(sorted_tasks, 1):
            lines.append(f"{i}. **[{task.priority.upper()}]** {task.title}")
            if task.affected_files:
                lines.append(f"   - Files: {', '.join(task.affected_files)}")

        return "\n".join(lines)

    def _estimate_file_changes(self, tasks: list[TaskAnalysis]) -> dict[str, int]:
        """Estimate file changes from tasks."""
        changes = {}
        for task in tasks:
            for file_path in task.affected_files:
                # Rough estimate based on complexity
                lines = {"low": 10, "medium": 50, "high": 200}.get(
                    task.estimated_complexity, 50
                )
                changes[file_path] = changes.get(file_path, 0) + lines
        return changes

    def _estimate_effort(self, tasks: list[TaskAnalysis]) -> str:
        """Estimate total effort for all tasks."""
        if not tasks:
            return "No effort required"

        # Rough effort mapping
        effort_map = {"low": 0.5, "medium": 2, "high": 8}
        total_hours = sum(
            effort_map.get(t.estimated_complexity, 2)
            for t in tasks
        )

        if total_hours <= 1:
            return "< 1 hour"
        elif total_hours <= 4:
            return "1-4 hours"
        elif total_hours <= 8:
            return "4-8 hours"
        elif total_hours <= 24:
            return "1-3 days"
        else:
            return f"{int(total_hours / 8)}-{int(total_hours / 4)} days"


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

        # Generate structured report for mock
        report = None
        if plan_file and plan_file.exists():
            report = self.analyze_plan(plan_file)

        return ClaudeCodeResult(
            success=True,
            output="Mock implementation completed successfully.",
            files_modified=self.mock_files_modified,
            exit_code=0,
            implementation_report=report,
        )
