"""Plan writer for generating improvement plan documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Valid priority values for tasks
VALID_PRIORITIES = {"critical", "high", "medium", "low"}


@dataclass
class StageResult:
    """Result from a single analysis stage."""

    stage_id: str
    stage_name: str
    summary: str
    recommendations: list[str] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)


class PlanWriter:
    """Writes aggregated analysis results to an improvement plan document."""

    def __init__(self, output_dir: Path):
        """Initialize the plan writer.

        Args:
            output_dir: Directory to write the plan file to.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_plan(
        self,
        prd_content: str,
        profile_name: str,
        stage_results: list[StageResult],
        output_filename: str = "mvp_improvement_plan.md",
    ) -> Path:
        """Write the improvement plan to a markdown file.

        Args:
            prd_content: Original PRD content for summary.
            profile_name: Name of the profile used.
            stage_results: List of StageResult from each analysis stage.
            output_filename: Name of the output file.

        Returns:
            Path to the written file.
        """
        output_path = self.output_dir / output_filename

        sections = [
            self._generate_header(profile_name),
            self._generate_prd_summary(prd_content),
            self._generate_stage_summaries(stage_results),
            self._generate_task_list(stage_results),
            self._generate_instructions(),
        ]

        content = "\n\n".join(sections)

        output_path.write_text(content, encoding="utf-8")
        return output_path

    def _generate_header(self, profile_name: str) -> str:
        """Generate the document header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# MVP Improvement Plan

**Generated:** {timestamp}
**Profile:** {profile_name}
**Status:** Ready for implementation

---"""

    def _generate_prd_summary(self, prd_content: str) -> str:
        """Generate a summary of the PRD."""
        summary = self._extract_prd_summary(prd_content)
        return f"""## PRD Summary

{summary}"""

    def _extract_prd_summary(self, prd_content: str, max_lines: int = 20) -> str:
        """Extract a summary from the PRD content.

        Args:
            prd_content: Full PRD content.
            max_lines: Maximum lines to include.

        Returns:
            Summarized PRD content.
        """
        lines = prd_content.strip().split("\n")
        if len(lines) <= max_lines:
            return prd_content.strip()

        # Take first max_lines and add truncation notice
        summary_lines = lines[:max_lines]
        return "\n".join(summary_lines) + "\n\n*[PRD truncated for brevity]*"

    def _generate_stage_summaries(self, stage_results: list[StageResult]) -> str:
        """Generate summaries for each analysis stage."""
        if not stage_results:
            return "## Analysis Stages\n\n*No stages were executed.*"

        sections = ["## Analysis Stages"]

        for result in stage_results:
            section = f"""### {result.stage_name}

{result.summary}"""

            if result.recommendations:
                section += "\n\n**Recommendations:**\n"
                for rec in result.recommendations:
                    section += f"- {rec}\n"

            sections.append(section)

        return "\n\n".join(sections)

    def _generate_task_list(self, stage_results: list[StageResult]) -> str:
        """Generate the aggregated and prioritized task list."""
        all_tasks = self._aggregate_tasks(stage_results)

        if not all_tasks:
            return "## Implementation Tasks\n\n*No tasks were identified.*"

        sections = ["## Implementation Tasks\n"]

        # Group by priority
        priority_order = ["critical", "high", "medium", "low"]
        tasks_by_priority: dict[str, list[dict]] = {p: [] for p in priority_order}

        for task in all_tasks:
            priority = task.get("priority", "medium").lower()
            if priority not in tasks_by_priority:
                priority = "medium"
            tasks_by_priority[priority].append(task)

        for priority in priority_order:
            tasks = tasks_by_priority[priority]
            if not tasks:
                continue

            badge = self._priority_badge(priority)
            sections.append(f"### {badge} {priority.capitalize()} Priority\n")

            for task in tasks:
                title = task.get("title", "Untitled task")
                description = task.get("description", "")
                file_ref = task.get("file", "")

                task_line = f"- [ ] **{title}**"
                if file_ref:
                    task_line += f" (`{file_ref}`)"
                if description:
                    task_line += f"\n  - {description}"

                sections.append(task_line)

            sections.append("")  # Add spacing between priority groups

        return "\n".join(sections)

    def _normalize_task(self, task: dict, stage_id: str) -> Optional[dict]:
        """Normalize and validate a task dict.

        Args:
            task: Raw task dictionary from analysis.
            stage_id: ID of the stage this task came from.

        Returns:
            Normalized task dict, or None if invalid (missing title).
        """
        title = task.get("title", "")
        if isinstance(title, str):
            title = title.strip()

        if not title:
            logger.warning(f"Skipping task without title from stage {stage_id}")
            return None

        priority = task.get("priority", "medium")
        if isinstance(priority, str):
            priority = priority.lower().strip()
        if priority not in VALID_PRIORITIES:
            logger.debug(
                f"Invalid priority '{priority}' for task '{title}', defaulting to medium"
            )
            priority = "medium"

        return {
            "title": title,
            "description": task.get("description", "").strip() if isinstance(task.get("description"), str) else "",
            "priority": priority,
            "file": task.get("file"),
            "stage": stage_id,
        }

    def _aggregate_tasks(self, stage_results: list[StageResult]) -> list[dict]:
        """Aggregate tasks from all stages, removing duplicates.

        Tasks are normalized and validated. Tasks without titles are skipped
        with a warning. Duplicate titles are deduplicated (first occurrence wins).

        Args:
            stage_results: List of stage results to aggregate.

        Returns:
            List of normalized, deduplicated tasks.
        """
        all_tasks = []
        seen_titles = set()

        for result in stage_results:
            for task in result.tasks:
                normalized = self._normalize_task(task, result.stage_id)
                if normalized is None:
                    continue

                title = normalized["title"]
                if title in seen_titles:
                    logger.debug(f"Skipping duplicate task: {title}")
                    continue

                seen_titles.add(title)
                all_tasks.append(normalized)

        return all_tasks

    def _priority_badge(self, priority: str) -> str:
        """Get an emoji badge for the priority level."""
        badges = {
            "critical": "[CRITICAL]",
            "high": "[HIGH]",
            "medium": "[MEDIUM]",
            "low": "[LOW]",
        }
        return badges.get(priority.lower(), "[MEDIUM]")

    def _generate_instructions(self) -> str:
        """Generate instructions for using the plan with Claude Code."""
        return """---

## Instructions for Claude Code

### IMPORTANT: Use Taskmaster for Implementation

**You MUST use Taskmaster to implement these tasks.** Do NOT manage tasks manually.

To import and implement the tasks:

```bash
# First, import the tasks file into Taskmaster
task-master parse-prd .meta-agent-tasks.md --append

# Work through tasks using Taskmaster:
task-master list                    # See all tasks
task-master next                    # Get next task to work on
task-master set-status --id=<id> --status=in-progress
task-master set-status --id=<id> --status=done
```

### Task Workflow

1. Import tasks into Taskmaster using `parse-prd --append`
2. Use `task-master next` to get the highest priority task
3. Mark task as `in-progress` before starting work
4. Implement the task following the description
5. Run relevant tests
6. Mark task as `done` when complete
7. Commit changes after completing related tasks

### Implementation Notes

- Work through tasks systematically, starting with Critical/High priority
- Run tests after each significant change
- Commit changes incrementally with descriptive messages
- If a task is unclear, review the relevant stage summary above for context
"""
