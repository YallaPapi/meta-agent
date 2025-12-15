"""Implementation strategies for code changes.

This module implements the Strategy design pattern to decouple code implementation
logic from the orchestrator. It enables runtime selection between different
implementation approaches:

- CurrentSessionStrategy: Returns structured results for the current Claude session
- ClaudeCodeStrategy: Wraps subprocess-based ClaudeCodeRunner (legacy)

The key insight: the meta-agent should return structured reports to the CURRENT
Claude Code session instead of spawning separate processes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from metaagent.claude_runner import ImplementationReport

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result from executing an implementation strategy.

    This is the common result type for all strategies, providing
    a consistent interface for the orchestrator.
    """
    success: bool
    action: str  # "return_to_session", "subprocess_completed", "error"
    summary: str = ""
    implementation_report: Optional["ImplementationReport"] = None
    files_modified: list[str] = field(default_factory=list)
    error: Optional[str] = None
    instructions: str = ""  # Instructions for current session


class CodeImplementationStrategy(ABC):
    """Abstract base class for code implementation strategies.

    Implementations of this class define HOW code changes are executed:
    - In the current session (return structured data)
    - Via subprocess (spawn Claude Code CLI)
    - Via API (future: direct Anthropic API calls)

    This follows the Strategy pattern to allow runtime selection
    of implementation approach.
    """

    @abstractmethod
    def execute(
        self,
        repo_path: Path,
        task_plan: str,
        plan_file: Optional[Path] = None,
    ) -> StrategyResult:
        """Execute the implementation strategy.

        Args:
            repo_path: Path to the target repository.
            task_plan: Markdown task plan or prompt for implementation.
            plan_file: Optional path to an improvement plan file.

        Returns:
            StrategyResult with execution outcome and any reports.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the strategy name for logging and configuration."""
        pass


class CurrentSessionStrategy(CodeImplementationStrategy):
    """Strategy that returns structured results for the current Claude session.

    This strategy does NOT spawn subprocesses. Instead, it:
    1. Parses the improvement plan
    2. Generates a structured ImplementationReport
    3. Returns instructions for the CURRENT session to implement

    This is the preferred strategy for iterative refinement where
    the meta-agent is running within a Claude Code session.
    """

    def __init__(self, session_context: Optional[str] = None):
        """Initialize the CurrentSessionStrategy.

        Args:
            session_context: Optional context string identifying the session.
        """
        self.session_context = session_context or "Current Claude Code session"

    def execute(
        self,
        repo_path: Path,
        task_plan: str,
        plan_file: Optional[Path] = None,
    ) -> StrategyResult:
        """Execute by returning structured report for current session.

        This method analyzes the plan and returns structured data that
        the calling Claude Code session can use to implement changes.
        No subprocess is spawned.

        Args:
            repo_path: Path to the target repository.
            task_plan: Markdown task plan or prompt for implementation.
            plan_file: Optional path to an improvement plan file.

        Returns:
            StrategyResult with action="return_to_session" and structured report.
        """
        logger.info(f"CurrentSessionStrategy: Processing plan for {repo_path}")

        # Import here to avoid circular dependency
        from metaagent.claude_runner import ClaudeCodeRunner, ImplementationReport

        # Use the existing analyze_plan method if plan_file is provided
        report: Optional[ImplementationReport] = None
        if plan_file and plan_file.exists():
            runner = ClaudeCodeRunner()
            report = runner.analyze_plan(plan_file)
            logger.info(f"Analyzed plan: {len(report.tasks)} tasks found")
        else:
            # Create report from task_plan string
            report = self._create_report_from_plan(task_plan)

        # Generate instructions for the current session
        instructions = self._generate_session_instructions(report, repo_path)

        return StrategyResult(
            success=True,
            action="return_to_session",
            summary=f"Prepared {len(report.tasks)} tasks for current session implementation",
            implementation_report=report,
            instructions=instructions,
        )

    def get_name(self) -> str:
        """Return strategy name."""
        return "current_session"

    def _create_report_from_plan(self, task_plan: str) -> "ImplementationReport":
        """Create ImplementationReport from a task plan string.

        Args:
            task_plan: Task plan content as a string.

        Returns:
            ImplementationReport with parsed tasks.
        """
        from metaagent.claude_runner import (
            ImplementationReport,
            TaskAnalysis,
        )

        # Parse basic tasks from the plan
        tasks = []
        lines = task_plan.strip().split('\n')
        task_id = 1

        for line in lines:
            line = line.strip()
            # Look for task-like patterns
            if line.startswith('- [ ]') or line.startswith('* [ ]'):
                title = line.replace('- [ ]', '').replace('* [ ]', '').strip()
                title = title.strip('*').strip()  # Remove any bold markers
                if title:
                    tasks.append(TaskAnalysis(
                        task_id=f"task-{task_id}",
                        title=title,
                        description=title,
                        priority="medium",
                    ))
                    task_id += 1
            elif line.startswith('- ') or line.startswith('* '):
                # Simple list item
                title = line[2:].strip()
                title = title.strip('*').strip()
                if title and len(title) > 10:  # Skip very short items
                    tasks.append(TaskAnalysis(
                        task_id=f"task-{task_id}",
                        title=title,
                        description=title,
                        priority="medium",
                    ))
                    task_id += 1

        return ImplementationReport(
            tasks=tasks,
            implementation_plan=task_plan,
            success=True,
            total_estimated_effort=f"{len(tasks) * 30} minutes" if tasks else "N/A",
        )

    def _generate_session_instructions(
        self,
        report: "ImplementationReport",
        repo_path: Path,
    ) -> str:
        """Generate instructions for the current session to implement changes.

        Args:
            report: The implementation report with tasks.
            repo_path: Path to the repository.

        Returns:
            Markdown instructions string.
        """
        lines = [
            "## Implementation Instructions for Current Session",
            "",
            f"Repository: `{repo_path}`",
            "",
            "### Tasks to Implement",
            "",
        ]

        for task in report.tasks:
            priority_marker = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }.get(task.priority.lower(), "⚪")

            lines.append(f"{priority_marker} **{task.title}**")
            if task.affected_files:
                lines.append(f"   - Files: {', '.join(task.affected_files)}")
            lines.append("")

        lines.extend([
            "### Workflow",
            "",
            "1. Work through tasks in priority order (critical → low)",
            "2. For each task:",
            "   - Understand the requirements",
            "   - Make the code changes",
            "   - Run tests to verify",
            "3. Commit changes after completing related tasks",
            "",
        ])

        return "\n".join(lines)


class SubprocessStrategy(CodeImplementationStrategy):
    """Strategy that spawns Claude Code as a subprocess.

    This wraps the existing ClaudeCodeRunner for backward compatibility.
    Use this when you need isolated subprocess execution.
    """

    def __init__(
        self,
        timeout: int = 600,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 50,
    ):
        """Initialize subprocess strategy.

        Args:
            timeout: Maximum time in seconds for execution.
            model: Claude model to use.
            max_turns: Maximum conversation turns.
        """
        self.timeout = timeout
        self.model = model
        self.max_turns = max_turns

    def execute(
        self,
        repo_path: Path,
        task_plan: str,
        plan_file: Optional[Path] = None,
    ) -> StrategyResult:
        """Execute by spawning Claude Code subprocess.

        Args:
            repo_path: Path to the target repository.
            task_plan: Prompt for Claude Code.
            plan_file: Optional path to improvement plan.

        Returns:
            StrategyResult with subprocess execution outcome.
        """
        from metaagent.claude_runner import ClaudeCodeRunner

        logger.info(f"SubprocessStrategy: Spawning Claude Code for {repo_path}")

        runner = ClaudeCodeRunner(
            timeout=self.timeout,
            model=self.model,
            max_turns=self.max_turns,
        )

        result = runner.implement(repo_path, task_plan, plan_file)

        return StrategyResult(
            success=result.success,
            action="subprocess_completed" if result.success else "error",
            summary=f"Subprocess {'completed' if result.success else 'failed'}",
            implementation_report=result.implementation_report,
            files_modified=result.files_modified,
            error=result.error,
        )

    def get_name(self) -> str:
        """Return strategy name."""
        return "subprocess"


def create_strategy(
    strategy_name: str,
    **kwargs,
) -> CodeImplementationStrategy:
    """Factory function to create implementation strategies.

    Args:
        strategy_name: Name of the strategy ("current_session", "subprocess").
        **kwargs: Additional arguments passed to strategy constructor.

    Returns:
        CodeImplementationStrategy instance.

    Raises:
        ValueError: If strategy_name is not recognized.
    """
    strategies = {
        "current_session": CurrentSessionStrategy,
        "subprocess": SubprocessStrategy,
    }

    strategy_cls = strategies.get(strategy_name.lower())
    if not strategy_cls:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown strategy: '{strategy_name}'. Available: {available}"
        )

    return strategy_cls(**kwargs)
