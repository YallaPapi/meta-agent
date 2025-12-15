"""Orchestrator for the meta-agent refinement pipeline.

This module provides the main orchestration logic for running analysis
pipelines on codebases. It supports two modes:

1. Profile-based (default): Deterministic execution of stages defined in profiles.yaml
2. Iterative triage: AI decides which prompts to run each iteration

The orchestrator coordinates several subsystems:
- StageRunner: Executes individual analysis stages
- TriageEngine: AI-driven prompt selection (for iterative mode)
- ImplementationExecutor: Handles Claude Code integration and commits
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

from .analysis import (
    AnalysisEngine,
    AnalysisResult,
    create_analysis_engine,
    extract_json_from_response,
)
from .claude_runner import (
    ClaudeCodeRunner,
    ClaudeCodeResult,
    ImplementationRecommendation,
    ImplementationReport,
    MockClaudeCodeRunner,
    TaskAnalysis,
)
from .codebase_digest import CodebaseDigestRunner, DigestResult
from .config import Config
from .plan_writer import PlanWriter, StageResult
from .prompts import Prompt, PromptLibrary
from .repomix import RepomixRunner, RepomixResult
from .strategies import (
    CodeImplementationStrategy,
    CurrentSessionStrategy,
    SubprocessStrategy,
    StrategyResult,
    create_strategy,
)
from .tokens import estimate_tokens, format_token_count

logger = logging.getLogger(__name__)


@dataclass
class RunHistory:
    """History of analysis runs for context building."""

    entries: list[dict] = field(default_factory=list)

    def add_entry(self, stage_id: str, summary: str) -> None:
        """Add an entry to the history."""
        self.entries.append(
            {
                "stage": stage_id,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def format_for_prompt(self) -> str:
        """Format history for inclusion in prompts."""
        if not self.entries:
            return "No previous analysis history."

        lines = ["Previous analysis stages:"]
        for entry in self.entries:
            lines.append(f"- [{entry['stage']}]: {entry['summary'][:200]}...")
        return "\n".join(lines)


@dataclass
class TriageResult:
    """Result from the triage step."""

    success: bool
    done: bool = False
    assessment: str = ""
    priority_issues: list[str] = field(default_factory=list)
    selected_prompts: list[str] = field(default_factory=list)
    reasoning: str = ""
    error: Optional[str] = None


@dataclass
class StageTriageResult:
    """Result from stage-specific triage."""

    success: bool
    stage: str
    selected_prompts: list[str] = field(default_factory=list)
    reasoning: str = ""
    error: Optional[str] = None


@dataclass
class IterationResult:
    """Result from a single iteration of the refinement loop."""

    iteration: int
    prompts_run: list[str]
    changes_made: bool
    committed: bool
    commit_hash: Optional[str] = None
    stage_results: list[StageResult] = field(default_factory=list)


@dataclass
class PlannedCall:
    """Represents a planned LLM call for dry-run mode."""

    stage: str
    prompt_id: str
    profile: str
    rendered_prompt: str
    estimated_tokens: int


@dataclass
class RefinementResult:
    """Result from a complete refinement run.

    This dataclass includes an implementation_report field that contains
    structured TaskAnalysis and ImplementationRecommendation data. This
    enables the CURRENT Claude Code session to consume results directly
    instead of spawning a subprocess.
    """

    success: bool
    profile_name: str
    stages_completed: int
    stages_failed: int
    plan_path: Optional[Path] = None
    error: Optional[str] = None
    stage_results: list[StageResult] = field(default_factory=list)
    iterations: list[IterationResult] = field(default_factory=list)
    planned_calls: list[PlannedCall] = field(default_factory=list)  # For dry-run mode
    partial_success: bool = False  # True when some stages succeeded but not all
    # Structured report for direct Claude Code consumption (no subprocess)
    implementation_report: Optional[ImplementationReport] = None

    @property
    def status(self) -> str:
        """Get human-readable status.

        Returns:
            'success', 'partial_success', or 'failure'
        """
        if self.success:
            return "success"
        elif self.partial_success:
            return "partial_success"
        else:
            return "failure"

    def to_markdown(self) -> str:
        """Generate markdown summary including implementation report.

        Returns:
            Markdown-formatted summary of the refinement result.
        """
        lines = [
            f"# Refinement Result: {self.status.upper()}",
            "",
            f"**Profile:** {self.profile_name}",
            f"**Stages Completed:** {self.stages_completed}",
            f"**Stages Failed:** {self.stages_failed}",
        ]

        if self.plan_path:
            lines.append(f"**Plan Path:** {self.plan_path}")

        if self.error:
            lines.append(f"\n**Error:** {self.error}")

        if self.implementation_report:
            lines.append("\n---\n")
            lines.append(self.implementation_report.to_markdown())

        return "\n".join(lines)


# =============================================================================
# Helper Classes for Separation of Concerns
# =============================================================================


class StageRunner:
    """Executes individual analysis stages.

    Responsible for:
    - Rendering prompts with context
    - Calling the analysis engine
    - Recording results in history
    """

    def __init__(
        self,
        analysis_engine: AnalysisEngine,
        prompt_library: PromptLibrary,
    ):
        """Initialize the stage runner.

        Args:
            analysis_engine: Engine for running LLM analysis.
            prompt_library: Library of available prompts.
        """
        self.analysis_engine = analysis_engine
        self.prompt_library = prompt_library

    def run_stage(
        self,
        prompt: Prompt,
        prd_content: str,
        code_context: str,
        history: RunHistory,
    ) -> tuple[StageResult, bool]:
        """Execute a single analysis stage.

        Args:
            prompt: The prompt to run.
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.

        Returns:
            Tuple of (StageResult, success_bool).
        """
        logger.info(f"Running stage: {prompt.id}")

        # Render prompt with context
        rendered_prompt = prompt.render(
            prd=prd_content,
            code_context=code_context,
            history=history.format_for_prompt(),
            current_stage=prompt.id,
        )

        # Run analysis
        analysis_result = self.analysis_engine.analyze(rendered_prompt)

        if analysis_result.success:
            history.add_entry(prompt.id, analysis_result.summary)
            stage_result = StageResult(
                stage_id=prompt.id,
                stage_name=prompt.goal or prompt.id,
                summary=analysis_result.summary,
                recommendations=analysis_result.recommendations,
                tasks=analysis_result.tasks,
            )
            logger.info(f"Stage {prompt.id} completed successfully")
            return stage_result, True
        else:
            stage_result = StageResult(
                stage_id=prompt.id,
                stage_name=prompt.goal or prompt.id,
                summary=f"Stage failed: {analysis_result.error}",
                recommendations=[],
                tasks=[],
            )
            logger.error(f"Stage {prompt.id} failed: {analysis_result.error}")
            return stage_result, False

    def run_stages(
        self,
        prompts: list[Prompt],
        prd_content: str,
        code_context: str,
        history: RunHistory,
    ) -> tuple[list[StageResult], int, int]:
        """Execute multiple stages in order.

        Args:
            prompts: List of prompts to run.
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.

        Returns:
            Tuple of (stage_results, completed_count, failed_count).
        """
        stage_results: list[StageResult] = []
        completed = 0
        failed = 0

        for prompt in prompts:
            result, success = self.run_stage(prompt, prd_content, code_context, history)
            stage_results.append(result)
            if success:
                completed += 1
            else:
                failed += 1

        return stage_results, completed, failed

    def preview_stage(
        self,
        prompt: Prompt,
        prd_content: str,
        code_context: str,
        history: RunHistory,
        profile_name: str,
    ) -> PlannedCall:
        """Preview a stage for dry-run mode (no actual API call).

        Args:
            prompt: The prompt to preview.
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.
            profile_name: Name of the profile being used.

        Returns:
            PlannedCall with rendered prompt and token estimate.
        """
        logger.info(f"[DRY-RUN] Previewing stage: {prompt.id}")

        # Render prompt with context
        rendered_prompt = prompt.render(
            prd=prd_content,
            code_context=code_context,
            history=history.format_for_prompt(),
            current_stage=prompt.id,
        )

        # Estimate tokens
        tokens = estimate_tokens(rendered_prompt)

        return PlannedCall(
            stage=prompt.stage,
            prompt_id=prompt.id,
            profile=profile_name,
            rendered_prompt=rendered_prompt,
            estimated_tokens=tokens,
        )

    def preview_stages(
        self,
        prompts: list[Prompt],
        prd_content: str,
        code_context: str,
        history: RunHistory,
        profile_name: str,
    ) -> list[PlannedCall]:
        """Preview multiple stages for dry-run mode.

        Args:
            prompts: List of prompts to preview.
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.
            profile_name: Name of the profile.

        Returns:
            List of PlannedCall objects.
        """
        planned_calls = []
        for prompt in prompts:
            call = self.preview_stage(prompt, prd_content, code_context, history, profile_name)
            planned_calls.append(call)
        return planned_calls


class TriageEngine:
    """AI-driven prompt selection for iterative mode.

    Responsible for:
    - Running the meta_triage prompt
    - Parsing triage responses
    - Validating selected prompts
    """

    def __init__(
        self,
        analysis_engine: AnalysisEngine,
        prompt_library: PromptLibrary,
        max_prompts_per_iteration: int = 3,
    ):
        """Initialize the triage engine.

        Args:
            analysis_engine: Engine for running LLM analysis.
            prompt_library: Library of available prompts.
            max_prompts_per_iteration: Maximum prompts to run per iteration.
        """
        self.analysis_engine = analysis_engine
        self.prompt_library = prompt_library
        self.max_prompts_per_iteration = max_prompts_per_iteration

    def run_triage(
        self,
        prd_content: str,
        code_context: str,
        history: RunHistory,
    ) -> TriageResult:
        """Run triage to determine which prompts to run next.

        Args:
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.

        Returns:
            TriageResult with selected prompts or done flag.
        """
        triage_prompt = self.prompt_library.get_prompt("meta_triage")
        if not triage_prompt:
            return TriageResult(
                success=False,
                error="Triage prompt (meta_triage) not found in prompt library",
            )

        rendered_prompt = triage_prompt.render(
            prd=prd_content,
            code_context=code_context,
            history=history.format_for_prompt(),
        )

        analysis_result = self.analysis_engine.analyze(rendered_prompt)

        if not analysis_result.success:
            return TriageResult(success=False, error=analysis_result.error)

        return self._parse_triage_response(analysis_result)

    def _parse_triage_response(self, analysis_result: AnalysisResult) -> TriageResult:
        """Parse the triage response into a TriageResult.

        Uses the shared extract_json_from_response() function for robust
        JSON extraction with proper handling of nested braces and strings.

        Args:
            analysis_result: Raw analysis result from the LLM.

        Returns:
            Parsed TriageResult.
        """
        # Use raw_response if summary is empty
        response_text = analysis_result.summary
        if not response_text or not response_text.strip():
            response_text = analysis_result.raw_response

        if not response_text:
            return TriageResult(
                success=False,
                error="Empty triage response",
            )

        logger.debug(f"Triage response (first 500 chars): {response_text[:500]}")

        # Use shared JSON extraction function for robust parsing
        data, extract_error = extract_json_from_response(response_text)

        if data is None:
            # No JSON found, check for "done" indicator in text
            if "done" in response_text.lower() and "no further" in response_text.lower():
                return TriageResult(
                    success=True,
                    done=True,
                    assessment=response_text,
                )

            logger.warning(f"Triage JSON extraction failed: {extract_error}")
            return TriageResult(
                success=False,
                error=f"Could not parse triage response: {extract_error}. Response: {response_text[:200]}",
            )

        # Validate triage-specific fields
        validation_error = self._validate_triage_data(data)
        if validation_error:
            logger.warning(f"Triage validation failed: {validation_error}")
            return TriageResult(
                success=False,
                error=f"Invalid triage response: {validation_error}",
            )

        selected_prompts = self._validate_prompts(data.get("selected_prompts", []))

        return TriageResult(
            success=True,
            done=data.get("done", False),
            assessment=data.get("assessment", ""),
            priority_issues=data.get("priority_issues", []),
            selected_prompts=selected_prompts,
            reasoning=data.get("reasoning", ""),
        )

    def _validate_triage_data(self, data: dict) -> Optional[str]:
        """Validate triage response data structure.

        Args:
            data: Parsed JSON data from triage response.

        Returns:
            Error message if invalid, None if valid.
        """
        if not isinstance(data, dict):
            return "Response is not a JSON object"

        # Check 'done' is a boolean if present
        if "done" in data and not isinstance(data["done"], bool):
            return "'done' field must be a boolean"

        # Check 'selected_prompts' is a list if present
        if "selected_prompts" in data:
            if not isinstance(data["selected_prompts"], list):
                return "'selected_prompts' must be a list"
            for item in data["selected_prompts"]:
                if not isinstance(item, str):
                    return "'selected_prompts' must contain only strings"

        # Check 'priority_issues' is a list if present
        if "priority_issues" in data and not isinstance(data["priority_issues"], list):
            return "'priority_issues' must be a list"

        return None

    def _validate_prompts(self, prompt_ids: list[str]) -> list[str]:
        """Validate and filter selected prompts.

        Args:
            prompt_ids: List of prompt IDs from triage.

        Returns:
            Validated list of prompt IDs that exist in the library.
        """
        validated = []
        for prompt_id in prompt_ids:
            if self.prompt_library.get_prompt(prompt_id):
                validated.append(prompt_id)
            else:
                logger.warning(f"Triage selected unknown prompt: {prompt_id}")

        # Limit to max prompts per iteration
        if len(validated) > self.max_prompts_per_iteration:
            logger.warning(
                f"Triage selected {len(validated)} prompts, limiting to {self.max_prompts_per_iteration}"
            )
            validated = validated[:self.max_prompts_per_iteration]

        return validated

    def triage_stage(
        self,
        stage: str,
        prd_content: str,
        code_context: str,
        history: RunHistory,
    ) -> StageTriageResult:
        """Run triage for a specific stage to select best prompts.

        This method performs AI-driven prompt selection for a single stage,
        choosing the most relevant prompts from the stage's candidate list.

        Args:
            stage: The conceptual stage (e.g., 'architecture', 'quality').
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.

        Returns:
            StageTriageResult with selected prompts for this stage.
        """
        logger.info(f"Running triage for stage: {stage}")

        # Get stage configuration
        stage_config = self.prompt_library.get_stage_config(stage)
        if not stage_config:
            return StageTriageResult(
                success=False,
                stage=stage,
                error=f"No configuration found for stage: {stage}",
            )

        # Get candidate prompts for this stage
        candidates = self.prompt_library.get_all_candidate_prompts_for_stage(stage)
        if not candidates:
            return StageTriageResult(
                success=False,
                stage=stage,
                error=f"No candidate prompts found for stage: {stage}",
            )

        logger.debug(f"Stage {stage} has {len(candidates)} candidate prompts")

        # Build stage-specific triage prompt
        triage_prompt = self._build_stage_triage_prompt(
            stage=stage,
            candidates=candidates,
            max_prompts=stage_config.max_prompts,
            prd_content=prd_content,
            code_context=code_context,
            history=history,
        )

        # Run triage
        result = self.analysis_engine.analyze(triage_prompt)
        if not result.success:
            return StageTriageResult(
                success=False,
                stage=stage,
                error=result.error,
            )

        return self._parse_stage_triage_response(stage, result, candidates)

    def _build_stage_triage_prompt(
        self,
        stage: str,
        candidates: list[Prompt],
        max_prompts: int,
        prd_content: str,
        code_context: str,
        history: RunHistory,
    ) -> str:
        """Build a prompt for stage-specific triage.

        Args:
            stage: The stage name.
            candidates: List of candidate Prompt objects.
            max_prompts: Maximum prompts to select.
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.

        Returns:
            Rendered triage prompt string.
        """
        # Build candidates list
        candidates_list = "\n".join(
            f"- `{p.id}`: {p.goal}" for p in candidates
        )

        # Get the stage triage prompt template
        triage_prompt = self.prompt_library.get_prompt("meta_stage_triage")

        if triage_prompt:
            # Use the template with Jinja2 substitution
            from jinja2 import Template
            template = Template(triage_prompt.template)
            prompt_text = template.render(
                stage=stage,
                candidates_list=candidates_list,
                max_prompts=max_prompts,
            )
        else:
            # Fallback to inline prompt if template not found
            prompt_text = f"""# Stage-Specific Prompt Selection

**Stage:** {stage}

Select the most relevant analysis prompts for this stage from the candidates below.

## Candidate Prompts

{candidates_list}

## Instructions

1. Review the codebase and PRD
2. Select up to {max_prompts} prompts that would provide the most value
3. Prioritize prompts addressing gaps between code and PRD

## Response Format (JSON only)

```json
{{
  "selected_prompts": ["prompt_id_1", "prompt_id_2"],
  "reasoning": "Brief explanation of selections"
}}
```
"""

        # Build full context with PRD and codebase
        context_sections = []
        if prd_content:
            context_sections.append(f"## Product Requirements Document (PRD)\n\n{prd_content}")
        if code_context:
            context_sections.append(f"## Codebase\n\n{code_context}")
        if history.entries:
            context_sections.append(f"## Previous Analysis\n\n{history.format_for_prompt()}")

        context_block = "\n\n---\n\n".join(context_sections) if context_sections else ""

        # Combine context with prompt
        if context_block:
            return f"{context_block}\n\n---\n\n{prompt_text}"
        return prompt_text

    def _parse_stage_triage_response(
        self,
        stage: str,
        analysis_result: AnalysisResult,
        candidates: list[Prompt],
    ) -> StageTriageResult:
        """Parse the stage triage response.

        Args:
            stage: The stage name.
            analysis_result: Raw analysis result from the LLM.
            candidates: List of valid candidate prompts.

        Returns:
            Parsed StageTriageResult.
        """
        # Get response text
        response_text = analysis_result.summary
        if not response_text or not response_text.strip():
            response_text = analysis_result.raw_response

        if not response_text:
            return StageTriageResult(
                success=False,
                stage=stage,
                error="Empty stage triage response",
            )

        logger.debug(f"Stage triage response (first 500 chars): {response_text[:500]}")

        # Extract JSON
        data, extract_error = extract_json_from_response(response_text)

        if data is None:
            logger.warning(f"Stage triage JSON extraction failed: {extract_error}")
            return StageTriageResult(
                success=False,
                stage=stage,
                error=f"Could not parse stage triage response: {extract_error}",
            )

        # Validate selected_prompts field
        selected = data.get("selected_prompts", [])
        if not isinstance(selected, list):
            return StageTriageResult(
                success=False,
                stage=stage,
                error="'selected_prompts' must be a list",
            )

        # Filter to only valid candidate prompt IDs
        valid_ids = {p.id for p in candidates}
        validated = []
        for prompt_id in selected:
            if not isinstance(prompt_id, str):
                continue
            if prompt_id in valid_ids:
                validated.append(prompt_id)
            else:
                logger.warning(
                    f"Stage triage selected invalid prompt '{prompt_id}' "
                    f"for stage '{stage}'"
                )

        return StageTriageResult(
            success=True,
            stage=stage,
            selected_prompts=validated,
            reasoning=data.get("reasoning", ""),
        )


class ImplementationExecutor:
    """Handles Claude Code integration and git commits.

    Responsible for:
    - Writing task files for Claude Code
    - Invoking Claude Code (when auto_implement is enabled)
    - Committing changes to git
    """

    def __init__(
        self,
        config: Config,
        claude_runner: ClaudeCodeRunner,
    ):
        """Initialize the implementation executor.

        Args:
            config: Configuration settings.
            claude_runner: Claude Code runner instance.
        """
        self.config = config
        self.claude_runner = claude_runner

    def execute(self, stage_results: list[StageResult]) -> bool:
        """Execute implementation for stage results.

        Args:
            stage_results: Results from analysis stages.

        Returns:
            True if changes were made, False otherwise.
        """
        if not stage_results:
            return False

        # Collect all tasks
        tasks = []
        for result in stage_results:
            tasks.extend(result.tasks)

        if not tasks:
            logger.info("No tasks to implement")
            return False

        # Write tasks to file
        prompt_file = self._write_task_file(tasks)
        logger.info(f"Implementation tasks written to: {prompt_file}")

        # Build implementation prompt
        implementation_prompt = self._build_prompt(tasks)

        # Execute with Claude Code if enabled
        if self.config.auto_implement:
            return self._run_claude_code(implementation_prompt)
        else:
            logger.info("Please run Claude Code to implement these changes.")
            logger.info("Or use --auto-implement to run Claude Code automatically.")
            return True  # Tasks were written

    def analyze_and_report(self, stage_results: list[StageResult]) -> ImplementationReport:
        """Analyze stage results and return structured implementation report.

        This method replaces subprocess-based execution. Instead of spawning
        Claude Code as a subprocess, it returns structured data that the
        CURRENT Claude Code session can use to implement changes.

        Args:
            stage_results: Results from analysis stages.

        Returns:
            ImplementationReport with tasks and recommendations.
        """
        if not stage_results:
            return ImplementationReport(
                success=True,
                tasks=[],
                implementation_plan="No analysis results to process.",
            )

        # Collect all tasks from stage results
        all_tasks: list[TaskAnalysis] = []
        task_id = 1

        for result in stage_results:
            for task_dict in result.tasks:
                if isinstance(task_dict, dict):
                    task = TaskAnalysis(
                        task_id=f"task-{task_id}",
                        title=task_dict.get("title", "Untitled"),
                        description=task_dict.get("description", str(task_dict)),
                        priority=task_dict.get("priority", "medium"),
                        affected_files=[task_dict.get("file", "")] if task_dict.get("file") else [],
                    )
                    all_tasks.append(task)
                    task_id += 1
                else:
                    # Handle string tasks
                    task = TaskAnalysis(
                        task_id=f"task-{task_id}",
                        title=str(task_dict),
                        description=str(task_dict),
                        priority="medium",
                    )
                    all_tasks.append(task)
                    task_id += 1

        # Write tasks to file for reference
        tasks_for_file = [
            {
                "title": t.title,
                "description": t.description,
                "priority": t.priority,
                "file": t.affected_files[0] if t.affected_files else "",
            }
            for t in all_tasks
        ]
        prompt_file = self._write_task_file(tasks_for_file)
        logger.info(f"Implementation tasks written to: {prompt_file}")

        # Build implementation plan
        plan_lines = [
            "## Implementation Plan",
            "",
            "Tasks are sorted by priority.",
            "",
        ]
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_tasks = sorted(all_tasks, key=lambda t: priority_order.get(t.priority, 2))

        for i, task in enumerate(sorted_tasks, 1):
            plan_lines.append(f"{i}. **[{task.priority.upper()}]** {task.title}")
            if task.affected_files:
                plan_lines.append(f"   - Files: {', '.join(task.affected_files)}")

        # Estimate effort
        effort_map = {"low": 0.5, "medium": 2, "high": 8, "critical": 4}
        total_hours = sum(effort_map.get(t.priority, 2) for t in all_tasks)
        if total_hours <= 1:
            effort = "< 1 hour"
        elif total_hours <= 4:
            effort = "1-4 hours"
        elif total_hours <= 8:
            effort = "4-8 hours"
        else:
            effort = f"{int(total_hours / 8)}-{int(total_hours / 4)} days"

        return ImplementationReport(
            tasks=all_tasks,
            implementation_plan="\n".join(plan_lines),
            success=True,
            total_estimated_effort=effort,
        )

    def _write_task_file(self, tasks: list) -> Path:
        """Write tasks to a markdown file for Claude Code.

        Generates a well-structured task file with:
        - Claude Code instruction header
        - Tasks sorted by priority (critical > high > medium > low)
        - Checkboxes for tracking progress
        - File paths for context

        Args:
            tasks: List of task dictionaries.

        Returns:
            Path to the written file.
        """
        prompt_file = self.config.repo_path / ".meta-agent-tasks.md"

        # Sort tasks by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_tasks = sorted(
            tasks,
            key=lambda t: priority_order.get(
                t.get("priority", "medium").lower() if isinstance(t, dict) else "medium",
                2
            )
        )

        lines = [
            "# Meta-Agent Implementation Tasks",
            "",
            "## Instructions for Claude Code",
            "",
            "This file contains implementation tasks identified by meta-agent analysis.",
            "Please work through these tasks in order, checking each box when complete.",
            "",
            "### How to use this file:",
            "1. Read through all tasks to understand the scope",
            "2. Start with critical/high priority tasks first",
            "3. Check the file path for context on where to make changes",
            "4. Mark tasks complete by changing `[ ]` to `[x]`",
            "5. Commit changes after completing related tasks",
            "",
            "---",
            "",
            "## Task Summary",
            "",
        ]

        # Add summary by priority
        critical = sum(1 for t in sorted_tasks if isinstance(t, dict) and t.get("priority", "").lower() == "critical")
        high = sum(1 for t in sorted_tasks if isinstance(t, dict) and t.get("priority", "").lower() == "high")
        medium = sum(1 for t in sorted_tasks if isinstance(t, dict) and t.get("priority", "").lower() == "medium")
        low = sum(1 for t in sorted_tasks if isinstance(t, dict) and t.get("priority", "").lower() == "low")

        if critical:
            lines.append(f"- **Critical:** {critical} task(s)")
        if high:
            lines.append(f"- **High:** {high} task(s)")
        if medium:
            lines.append(f"- **Medium:** {medium} task(s)")
        if low:
            lines.append(f"- **Low:** {low} task(s)")
        lines.append(f"- **Total:** {len(sorted_tasks)} task(s)")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Tasks")
        lines.append("")

        for i, task in enumerate(sorted_tasks, 1):
            if isinstance(task, dict):
                title = task.get("title", "Untitled task")
                desc = task.get("description", str(task))
                priority = task.get("priority", "medium").lower()
                file_ref = task.get("file", "")

                # Priority emoji
                priority_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(priority, "⚪")

                lines.append(f"### {i}. {priority_emoji} {title}")
                lines.append("")
                lines.append(f"- [ ] **Status:** Not started")
                lines.append(f"- **Priority:** {priority.title()}")
                if file_ref:
                    lines.append(f"- **File:** `{file_ref}`")
                lines.append("")
                lines.append("**Description:**")
                lines.append("")
                lines.append(desc)
                lines.append("")
            else:
                lines.append(f"### {i}. {task}")
                lines.append("")
                lines.append(f"- [ ] **Status:** Not started")
                lines.append("")

        prompt_file.write_text("\n".join(lines), encoding="utf-8")
        return prompt_file

    def _build_prompt(self, tasks: list) -> str:
        """Build a prompt string for Claude Code.

        Args:
            tasks: List of task dictionaries.

        Returns:
            Formatted prompt string.
        """
        prompt = "Please implement the following improvements:\n\n"

        for i, task in enumerate(tasks, 1):
            if isinstance(task, dict):
                title = task.get("title", "")
                desc = task.get("description", str(task))
                file_ref = task.get("file", "")

                if title:
                    prompt += f"{i}. **{title}**"
                    if file_ref:
                        prompt += f" ({file_ref})"
                    prompt += f"\n   {desc}\n\n"
                else:
                    prompt += f"{i}. {desc}\n\n"
            else:
                prompt += f"{i}. {task}\n\n"

        return prompt

    def _run_claude_code(self, prompt: str) -> bool:
        """Run Claude Code to implement changes.

        Args:
            prompt: Implementation prompt.

        Returns:
            True if changes were made, False otherwise.
        """
        logger.info("Auto-implementing with Claude Code...")

        result = self.claude_runner.implement(
            repo_path=self.config.repo_path,
            prompt=prompt,
            plan_file=self.config.repo_path / "docs" / "mvp_improvement_plan.md",
        )

        if result.success:
            logger.info(f"Claude Code completed. Files modified: {len(result.files_modified)}")
            for f in result.files_modified:
                logger.debug(f"  Modified: {f}")
            return len(result.files_modified) > 0
        else:
            logger.error(f"Claude Code failed: {result.error}")
            return False

    def commit_changes(
        self,
        message: str,
        prompt_ids: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Commit changes to git.

        Args:
            message: Commit message.
            prompt_ids: Optional list of prompt IDs to reference.

        Returns:
            Commit hash if successful, None otherwise.
        """
        if not self.config.auto_commit:
            logger.info("Auto-commit disabled, skipping commit")
            return None

        try:
            # Check for changes
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.config.repo_path,
                capture_output=True,
                text=True,
            )

            if not status_result.stdout.strip():
                logger.info("No changes to commit")
                return None

            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.config.repo_path,
                check=True,
            )

            # Build commit message
            commit_message = self._build_commit_message(message, prompt_ids)
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.config.repo_path,
                check=True,
            )

            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.config.repo_path,
                capture_output=True,
                text=True,
            )

            commit_hash = hash_result.stdout.strip()[:8]
            logger.info(f"Committed changes: {commit_hash}")

            # Push if enabled
            if self.config.auto_push:
                self._push_changes()

            return commit_hash

        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e}")
            return None

    def _build_commit_message(
        self,
        message: str,
        prompt_ids: Optional[list[str]] = None,
    ) -> str:
        """Build a formatted commit message.

        Args:
            message: Base message.
            prompt_ids: Optional list of prompt IDs.

        Returns:
            Formatted commit message.
        """
        lines = [f"meta-agent: {message}"]

        if prompt_ids:
            lines.append("")
            lines.append(f"Prompts: {', '.join(prompt_ids)}")

        lines.append("")
        lines.append("Generated with meta-agent")

        return "\n".join(lines)

    def _push_changes(self) -> None:
        """Push changes to remote."""
        try:
            subprocess.run(
                ["git", "push"],
                cwd=self.config.repo_path,
                check=True,
                timeout=60,
            )
            logger.info("Pushed to remote")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to push: {e}")


class Orchestrator:
    """Orchestrates the refinement pipeline."""

    def __init__(
        self,
        config: Config,
        prompt_library: Optional[PromptLibrary] = None,
        repomix_runner: Optional[RepomixRunner] = None,
        digest_runner: Optional[CodebaseDigestRunner] = None,
        analysis_engine: Optional[AnalysisEngine] = None,
        plan_writer: Optional[PlanWriter] = None,
        claude_runner: Optional[ClaudeCodeRunner] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration settings.
            prompt_library: Optional PromptLibrary instance.
            repomix_runner: Optional RepomixRunner instance.
            digest_runner: Optional CodebaseDigestRunner instance.
            analysis_engine: Optional AnalysisEngine instance.
            plan_writer: Optional PlanWriter instance.
            claude_runner: Optional ClaudeCodeRunner instance.
        """
        self.config = config

        # Initialize components with defaults if not provided
        self.prompt_library = prompt_library or PromptLibrary(
            prompts_path=config.prompts_file,
            profiles_path=config.profiles_file,
            prompt_library_path=config.prompt_library_path,
            stage_candidates_path=config.stage_candidates_file,
        )

        self.repomix_runner = repomix_runner or RepomixRunner(
            timeout=config.timeout,
            max_chars=100000,  # ~25k tokens - reasonable limit for LLM context
        )

        self.digest_runner = digest_runner or CodebaseDigestRunner(
            max_depth=10,
            output_format="markdown",
            include_content=False,  # We get content from Repomix
        )

        self.analysis_engine = analysis_engine or create_analysis_engine(
            api_key=config.perplexity_api_key,
            mock_mode=config.mock_mode,
            timeout=config.timeout,
            retry_max_attempts=config.retry_max_attempts,
            retry_backoff_base=config.retry_backoff_base,
            retry_backoff_max=config.retry_backoff_max,
        )

        self.plan_writer = plan_writer or PlanWriter(
            output_dir=config.repo_path / "docs",
        )

        # Initialize Claude Code runner
        if claude_runner:
            self.claude_runner = claude_runner
        elif config.mock_mode:
            self.claude_runner = MockClaudeCodeRunner(
                timeout=config.claude_code_timeout,
                model=config.claude_code_model,
                max_turns=config.claude_code_max_turns,
            )
        else:
            self.claude_runner = ClaudeCodeRunner(
                timeout=config.claude_code_timeout,
                model=config.claude_code_model,
                max_turns=config.claude_code_max_turns,
            )

        # Initialize helper classes for separation of concerns
        self.stage_runner = StageRunner(
            analysis_engine=self.analysis_engine,
            prompt_library=self.prompt_library,
        )

        self.triage_engine = TriageEngine(
            analysis_engine=self.analysis_engine,
            prompt_library=self.prompt_library,
        )

        self.implementation_executor = ImplementationExecutor(
            config=self.config,
            claude_runner=self.claude_runner,
        )

        # Implementation strategy (default to current_session to return
        # structured results instead of spawning subprocesses)
        self.implementation_strategy: Optional[CodeImplementationStrategy] = None
        if not config.mock_mode and not config.auto_implement:
            # Default to current session strategy for non-auto-implement mode
            self.implementation_strategy = CurrentSessionStrategy()
        elif config.auto_implement:
            # Use subprocess strategy when auto_implement is enabled
            self.implementation_strategy = SubprocessStrategy(
                timeout=config.claude_code_timeout,
                model=config.claude_code_model,
                max_turns=config.claude_code_max_turns,
            )

    def set_implementation_strategy(
        self,
        strategy: CodeImplementationStrategy,
    ) -> None:
        """Set the implementation strategy for code changes.

        This method allows runtime switching between different implementation
        approaches following the Strategy pattern.

        Args:
            strategy: The strategy to use (CurrentSessionStrategy or SubprocessStrategy).
        """
        logger.info(f"Setting implementation strategy to: {strategy.get_name()}")
        self.implementation_strategy = strategy

    def execute_with_strategy(
        self,
        task_plan: str,
        plan_file: Optional[Path] = None,
    ) -> StrategyResult:
        """Execute implementation using the configured strategy.

        This method is the primary way to invoke code implementation,
        returning structured results for the current session.

        Args:
            task_plan: The task plan or prompt for implementation.
            plan_file: Optional path to an improvement plan file.

        Returns:
            StrategyResult with the execution outcome.
        """
        if not self.implementation_strategy:
            logger.warning("No implementation strategy set, defaulting to CurrentSessionStrategy")
            self.implementation_strategy = CurrentSessionStrategy()

        return self.implementation_strategy.execute(
            repo_path=self.config.repo_path,
            task_plan=task_plan,
            plan_file=plan_file,
        )

    def refine(self, profile_id: str) -> RefinementResult:
        """Run the refinement pipeline for a profile.

        This is the default mode - deterministic execution of stages
        defined in profiles.yaml. Runs all prompts sequentially against
        the TARGET repo (config.repo_path).

        Args:
            profile_id: ID of the profile to use (e.g., 'automation_agent').

        Returns:
            RefinementResult with the outcome.
        """
        logger.info(f"Starting refinement with profile: {profile_id}")
        logger.info(f"Target repo: {self.config.repo_path}")
        logger.info(f"Config dir: {self.config.config_dir}")

        # Load PRD from TARGET repo
        prd_content = self._load_prd()
        if prd_content is None:
            return RefinementResult(
                success=False,
                profile_name=profile_id,
                stages_completed=0,
                stages_failed=0,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        # Get profile from prompt library
        profile = self.prompt_library.get_profile(profile_id)
        if not profile:
            return RefinementResult(
                success=False,
                profile_name=profile_id,
                stages_completed=0,
                stages_failed=0,
                error=f"Profile not found: {profile_id}",
            )

        # Pack the TARGET repo codebase
        logger.info("Packing target codebase...")
        code_context = self._pack_codebase()

        # Get prompts for the profile
        prompts = self.prompt_library.get_prompts_for_profile(profile_id)
        if not prompts:
            logger.warning(f"No prompts found for profile: {profile_id}")

        # Run all stages using the StageRunner
        history = RunHistory()
        stage_results, stages_completed, stages_failed = self.stage_runner.run_stages(
            prompts=prompts,
            prd_content=prd_content,
            code_context=code_context,
            history=history,
        )

        # Write plan to TARGET repo
        plan_path = None
        if stage_results:
            logger.info("Writing improvement plan...")
            plan_path = self.plan_writer.write_plan(
                prd_content=prd_content,
                profile_name=profile.name,
                stage_results=stage_results,
            )
            logger.info(f"Plan written to: {plan_path}")

        # Generate structured implementation report for Claude Code consumption
        implementation_report = None
        if stage_results:
            implementation_report = self.implementation_executor.analyze_and_report(
                stage_results
            )
            logger.info(
                f"Generated implementation report with "
                f"{len(implementation_report.tasks)} tasks"
            )

        # Execute implementation if auto_implement is enabled
        if self.config.auto_implement and stage_results:
            logger.info("Auto-implementing changes with Claude Code...")
            changes_made = self.implementation_executor.execute(stage_results)
            if changes_made:
                logger.info("Implementation completed successfully")
            else:
                logger.warning("No changes were made during implementation")

        return RefinementResult(
            success=stages_failed == 0 and stages_completed > 0,
            profile_name=profile.name,
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            plan_path=plan_path,
            stage_results=stage_results,
            partial_success=stages_failed > 0 and stages_completed > 0,
            implementation_report=implementation_report,
        )

    def refine_dry_run(self, profile_id: str) -> RefinementResult:
        """Preview the refinement pipeline without making API calls.

        This mode shows what prompts would be sent and estimates token usage
        without actually calling the LLM.

        Args:
            profile_id: ID of the profile to preview.

        Returns:
            RefinementResult with planned_calls populated.
        """
        logger.info(f"[DRY-RUN] Previewing refinement with profile: {profile_id}")
        logger.info(f"[DRY-RUN] Target repo: {self.config.repo_path}")

        # Load PRD from TARGET repo
        prd_content = self._load_prd()
        if prd_content is None:
            return RefinementResult(
                success=False,
                profile_name=profile_id,
                stages_completed=0,
                stages_failed=0,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        # Get profile from prompt library
        profile = self.prompt_library.get_profile(profile_id)
        if not profile:
            return RefinementResult(
                success=False,
                profile_name=profile_id,
                stages_completed=0,
                stages_failed=0,
                error=f"Profile not found: {profile_id}",
            )

        # Pack the TARGET repo codebase
        logger.info("[DRY-RUN] Packing target codebase...")
        code_context = self._pack_codebase()

        # Get prompts for the profile
        prompts = self.prompt_library.get_prompts_for_profile(profile_id)
        if not prompts:
            logger.warning(f"[DRY-RUN] No prompts found for profile: {profile_id}")

        # Preview all stages (no actual API calls)
        history = RunHistory()
        planned_calls = self.stage_runner.preview_stages(
            prompts=prompts,
            prd_content=prd_content,
            code_context=code_context,
            history=history,
            profile_name=profile.name,
        )

        return RefinementResult(
            success=True,
            profile_name=profile.name,
            stages_completed=0,  # No stages actually run
            stages_failed=0,
            planned_calls=planned_calls,
        )

    def refine_iterative(self, max_iterations: int = 10) -> RefinementResult:
        """Run the iterative refinement loop with AI-driven triage.

        This is an optional mode where the AI decides which prompts to run:
        1. Pack codebase
        2. Triage (AI decides which prompts to run)
        3. Run selected prompts
        4. Implement changes with Claude Code
        5. Commit to GitHub
        6. Repeat until triage says "done" or max iterations reached

        Args:
            max_iterations: Maximum number of iterations to run.

        Returns:
            RefinementResult with the outcome.
        """
        logger.info("Starting iterative refinement loop")

        # Load PRD
        prd_content = self._load_prd()
        if prd_content is None:
            return RefinementResult(
                success=False,
                profile_name="iterative",
                stages_completed=0,
                stages_failed=0,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        history = RunHistory()
        all_stage_results: list[StageResult] = []
        iterations: list[IterationResult] = []
        stages_completed = 0
        stages_failed = 0

        for iteration in range(1, max_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{max_iterations} ===")

            # Step 1: Pack codebase
            logger.info("Step 1: Packing codebase...")
            code_context = self._pack_codebase()

            # Step 2: Triage using TriageEngine
            logger.info("Step 2: Running triage...")
            triage_result = self.triage_engine.run_triage(prd_content, code_context, history)

            if not triage_result.success:
                logger.error(f"Triage failed: {triage_result.error}")
                return RefinementResult(
                    success=False,
                    profile_name="iterative",
                    stages_completed=stages_completed,
                    stages_failed=stages_failed,
                    error=f"Triage failed: {triage_result.error}",
                    stage_results=all_stage_results,
                    iterations=iterations,
                )

            if triage_result.done:
                logger.info("Triage says we're done! Codebase meets requirements.")
                break

            logger.info(f"Triage selected prompts: {triage_result.selected_prompts}")
            logger.info(f"Assessment: {triage_result.assessment}")

            # Step 3: Run selected prompts using StageRunner
            logger.info("Step 3: Running selected prompts...")
            prompts = [
                self.prompt_library.get_prompt(pid)
                for pid in triage_result.selected_prompts
                if self.prompt_library.get_prompt(pid)
            ]

            iteration_results, completed, failed = self.stage_runner.run_stages(
                prompts=prompts,
                prd_content=prd_content,
                code_context=code_context,
                history=history,
            )

            stages_completed += completed
            stages_failed += failed
            all_stage_results.extend(iteration_results)

            # Step 4: Implement changes with Claude Code
            logger.info("Step 4: Implementing changes with Claude Code...")
            changes_made = self.implementation_executor.execute(iteration_results)

            # Step 5: Commit to GitHub
            commit_hash = None
            if changes_made:
                logger.info("Step 5: Committing changes to GitHub...")
                commit_hash = self.implementation_executor.commit_changes(
                    f"Iteration {iteration}: improvements from AI analysis",
                    prompt_ids=triage_result.selected_prompts,
                )

            iterations.append(
                IterationResult(
                    iteration=iteration,
                    prompts_run=triage_result.selected_prompts,
                    changes_made=changes_made,
                    committed=commit_hash is not None,
                    commit_hash=commit_hash,
                    stage_results=iteration_results,
                )
            )

            if not changes_made:
                logger.info("No changes made this iteration, continuing...")

        # Write final plan
        plan_path = None
        if all_stage_results:
            logger.info("Writing final improvement plan...")
            plan_path = self.plan_writer.write_plan(
                prd_content=prd_content,
                profile_name="Iterative Refinement",
                stage_results=all_stage_results,
            )
            logger.info(f"Plan written to: {plan_path}")

        # Generate final aggregated implementation report
        implementation_report = None
        if all_stage_results:
            implementation_report = self.implementation_executor.analyze_and_report(
                all_stage_results
            )
            logger.info(
                f"Final report: {len(implementation_report.tasks)} total tasks"
            )

        return RefinementResult(
            success=stages_failed == 0 and stages_completed > 0,
            profile_name="iterative",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            plan_path=plan_path,
            stage_results=all_stage_results,
            iterations=iterations,
            partial_success=stages_failed > 0 and stages_completed > 0,
            implementation_report=implementation_report,
        )

    def refine_with_stage_triage(self, stages: list[str]) -> RefinementResult:
        """Run refinement with AI-driven prompt selection per stage.

        This mode runs triage for each conceptual stage to dynamically select
        the most relevant prompts from the full prompt library.

        For each stage:
        1. Get candidate prompts for that stage from stage_candidates.yaml
        2. Run triage to select the best prompts for this codebase
        3. Execute the selected prompts
        4. Aggregate results into the final plan

        Args:
            stages: List of conceptual stage names (e.g., ['architecture', 'quality']).

        Returns:
            RefinementResult with the outcome.
        """
        logger.info(f"Starting stage-aware refinement with stages: {stages}")
        logger.info(f"Target repo: {self.config.repo_path}")

        # Load PRD from TARGET repo
        prd_content = self._load_prd()
        if prd_content is None:
            return RefinementResult(
                success=False,
                profile_name="stage_triage",
                stages_completed=0,
                stages_failed=0,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        # Pack the TARGET repo codebase
        logger.info("Packing target codebase...")
        code_context = self._pack_codebase()

        history = RunHistory()
        all_stage_results: list[StageResult] = []
        stages_completed = 0
        stages_failed = 0
        triage_calls = 0
        analysis_calls = 0

        for stage in stages:
            logger.info(f"=== Processing stage: {stage} ===")

            # Run triage for this stage
            logger.info(f"Running triage for stage '{stage}'...")
            triage_result = self.triage_engine.triage_stage(
                stage=stage,
                prd_content=prd_content,
                code_context=code_context,
                history=history,
            )
            triage_calls += 1

            if not triage_result.success:
                logger.error(f"Triage failed for stage '{stage}': {triage_result.error}")
                stages_failed += 1
                continue

            if not triage_result.selected_prompts:
                logger.info(f"No prompts selected for stage '{stage}', skipping")
                continue

            logger.info(
                f"Stage '{stage}' selected prompts: {triage_result.selected_prompts}"
            )
            logger.info(f"Reasoning: {triage_result.reasoning}")

            # Get the selected prompts
            prompts = [
                self.prompt_library.get_prompt(pid)
                for pid in triage_result.selected_prompts
                if self.prompt_library.get_prompt(pid)
            ]

            if not prompts:
                logger.warning(f"No valid prompts found for stage '{stage}'")
                continue

            # Run the selected prompts
            stage_results, completed, failed = self.stage_runner.run_stages(
                prompts=prompts,
                prd_content=prd_content,
                code_context=code_context,
                history=history,
            )

            all_stage_results.extend(stage_results)
            stages_completed += completed
            stages_failed += failed
            analysis_calls += len(prompts)

        # Write plan
        plan_path = None
        if all_stage_results:
            logger.info("Writing improvement plan...")
            plan_path = self.plan_writer.write_plan(
                prd_content=prd_content,
                profile_name=f"Stage Triage ({', '.join(stages)})",
                stage_results=all_stage_results,
            )
            logger.info(f"Plan written to: {plan_path}")

        # Generate structured implementation report
        implementation_report = None
        if all_stage_results:
            implementation_report = self.implementation_executor.analyze_and_report(
                all_stage_results
            )
            logger.info(
                f"Generated report with {len(implementation_report.tasks)} tasks"
            )

        total_calls = triage_calls + analysis_calls
        logger.info(
            f"Stage triage complete: {triage_calls} triage calls, "
            f"{analysis_calls} analysis calls, {total_calls} total API calls"
        )

        return RefinementResult(
            success=stages_failed == 0 and stages_completed > 0,
            profile_name="stage_triage",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            plan_path=plan_path,
            stage_results=all_stage_results,
            partial_success=stages_failed > 0 and stages_completed > 0,
            implementation_report=implementation_report,
        )

    def run_stage_with_triage(
        self,
        stage: str,
        prd_content: str,
        code_context: str,
        history: RunHistory,
    ) -> tuple[list[StageResult], int, int]:
        """Run a single stage with AI-driven prompt selection.

        This helper method:
        1. Runs triage to select prompts for the stage
        2. Executes the selected prompts
        3. Returns the results

        Args:
            stage: The conceptual stage name.
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.

        Returns:
            Tuple of (stage_results, completed_count, failed_count).
        """
        # Run triage for this stage
        triage_result = self.triage_engine.triage_stage(
            stage=stage,
            prd_content=prd_content,
            code_context=code_context,
            history=history,
        )

        if not triage_result.success:
            logger.error(f"Triage failed for stage '{stage}': {triage_result.error}")
            return [], 0, 1

        if not triage_result.selected_prompts:
            logger.info(f"No prompts selected for stage '{stage}'")
            return [], 0, 0

        logger.info(f"Stage '{stage}' selected: {triage_result.selected_prompts}")

        # Get the selected prompts
        prompts = [
            self.prompt_library.get_prompt(pid)
            for pid in triage_result.selected_prompts
            if self.prompt_library.get_prompt(pid)
        ]

        if not prompts:
            return [], 0, 0

        # Run the selected prompts
        return self.stage_runner.run_stages(
            prompts=prompts,
            prd_content=prd_content,
            code_context=code_context,
            history=history,
        )

    def _pack_codebase(self) -> str:
        """Pack the codebase using both codebase-digest and Repomix.

        Returns:
            Combined code context string.
        """
        # Run codebase-digest for directory tree and metrics
        digest_result = self.digest_runner.analyze(self.config.repo_path)

        if digest_result.success:
            structure_context = self._format_digest_output(digest_result)
        else:
            logger.warning(f"codebase-digest failed: {digest_result.error}")
            structure_context = ""

        # Run Repomix for full file contents
        repomix_result = self.repomix_runner.pack(self.config.repo_path)

        if not repomix_result.success:
            logger.warning(f"Repomix failed: {repomix_result.error}")
            file_contents = f"[Repomix failed: {repomix_result.error}]"
        else:
            file_contents = repomix_result.content
            if repomix_result.truncated:
                logger.warning(
                    f"Codebase was truncated from {repomix_result.original_size} chars"
                )

        return self._build_code_context(structure_context, file_contents)

    def _load_prd(self) -> Optional[str]:
        """Load the PRD file content.

        Returns:
            PRD content string or None if not found.
        """
        prd_path = self.config.prd_path
        if not prd_path or not prd_path.exists():
            return None

        return prd_path.read_text(encoding="utf-8")

    def _format_digest_output(self, digest_result: DigestResult) -> str:
        """Format codebase-digest output for inclusion in prompts.

        Args:
            digest_result: Result from codebase-digest analysis.

        Returns:
            Formatted string with directory tree and metrics.
        """
        sections = []

        if digest_result.tree:
            sections.append("## Directory Structure")
            sections.append(digest_result.tree)

        if digest_result.metrics:
            sections.append("\n## Codebase Metrics")
            sections.append(digest_result.metrics)

        return "\n".join(sections) if sections else ""

    def _build_code_context(self, structure_context: str, file_contents: str) -> str:
        """Build comprehensive code context from both tools.

        Args:
            structure_context: Directory tree and metrics from codebase-digest.
            file_contents: Full file contents from Repomix.

        Returns:
            Combined code context string.
        """
        sections = []

        if structure_context:
            sections.append("# Codebase Overview (from codebase-digest)")
            sections.append(structure_context)
            sections.append("\n---\n")

        if file_contents and not file_contents.startswith("["):
            sections.append("# File Contents (from Repomix)")
            sections.append(file_contents)
        elif file_contents:
            # Repomix failed, include the error message
            sections.append(file_contents)

        if not sections:
            return "[No codebase context available - both codebase-digest and Repomix failed]"

        return "\n".join(sections)
