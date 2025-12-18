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
from .claude_impl import ClaudeImplementer, MockClaudeImplementer, ImplementationResult
from .config import get_evaluator_name
from .grok_client import GrokClient, MockGrokClient, GrokClientError
from .local_manager import LocalRepoManager, MockLocalRepoManager, LocalRepoError
from .task_manager import TaskManager, Task, TaskManagerError, MockTaskManager
from .license import verify_pro_key, check_iteration_limit, get_tier_info, FREE_TIER_LIMIT
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
from .workspace_manager import WorkspaceManager, generate_branch_name, TestResult
from .ollama_engine import OllamaEngine, TriageOutput, check_ollama_status
from .plan_writer import PlanWriter, StageResult
from .prompts import Prompt, PromptLibrary
from .repomix import RepomixRunner, RepomixResult
from .repomix_parser import extract_files, get_file_list
from .strategies import (
    CodeImplementationStrategy,
    CurrentSessionStrategy,
    SubprocessStrategy,
    StrategyResult,
    create_strategy,
)
from .tokens import estimate_tokens, format_token_count

# Import dashboard events (optional - only if dashboard module exists)
try:
    from .dashboard.events import EventType, emit, get_emitter
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    # Stub for when dashboard isn't available
    def emit(event_type, data=None):
        pass
    class EventType:
        SESSION_START = "session_start"
        SESSION_END = "session_end"
        ITERATION_START = "iteration_start"
        LAYER_UPDATE = "layer_update"
        TASK_LIST = "task_list"
        TASK_START = "task_start"
        TASK_COMPLETE = "task_complete"
        LOG = "log"
        ERROR = "error"
        OLLAMA_START = "ollama_start"
        OLLAMA_COMPLETE = "ollama_complete"
        PERPLEXITY_START = "perplexity_start"
        PERPLEXITY_COMPLETE = "perplexity_complete"
        FILE_MODIFIED = "file_modified"

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
class LayerStatus:
    """Status of development layers in iterative refinement."""

    current_layer: int = 1
    layer_name: str = "scaffold"
    layer_progress: str = ""
    scaffold_complete: bool = False
    core_complete: bool = False
    integration_complete: bool = False
    polish_complete: bool = False


@dataclass
class IterationResult:
    """Result from a single iteration of the refinement loop."""

    iteration: int
    prompts_run: list[str]
    changes_made: bool
    committed: bool
    commit_hash: Optional[str] = None
    stage_results: list[StageResult] = field(default_factory=list)
    layer_status: Optional[LayerStatus] = None


@dataclass
class PlannedCall:
    """Represents a planned LLM call for dry-run mode."""

    stage: str
    prompt_id: str
    profile: str
    rendered_prompt: str
    estimated_tokens: int


@dataclass
class AutonomousLoopResult:
    """Result from the autonomous development loop.

    This dataclass captures the complete state of an autonomous loop run,
    including iterations, tasks, tests, and final evaluation.
    """

    success: bool
    iterations_completed: int
    max_iterations: int
    tasks_completed: int = 0
    tests_passed: int = 0
    fixes_applied: int = 0
    tokens_used: int = 0
    branch_name: Optional[str] = None
    commits_made: int = 0
    final_evaluation: Optional[str] = None
    prd_aligned: bool = False
    error: Optional[str] = None
    iteration_details: list[dict] = field(default_factory=list)


@dataclass
class LocalLoopResult:
    """Result from the Grok-powered local development loop.

    This dataclass captures the state of a local loop run using GitPython
    and Grok for error diagnosis.
    """

    success: bool
    iterations_completed: int
    max_iterations: int
    tasks_completed: int = 0
    tests_passed: int = 0
    fixes_applied: int = 0
    tokens_used: int = 0
    branch_name: Optional[str] = None
    commits_made: int = 0
    final_evaluation: Optional[str] = None
    prd_aligned: bool = False
    evaluator_used: str = "grok"
    error: Optional[str] = None
    iteration_details: list[dict] = field(default_factory=list)


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
        focus: Optional[str] = None,
    ) -> tuple[StageResult, bool]:
        """Execute a single analysis stage.

        Args:
            prompt: The prompt to run.
            prd_content: The PRD content.
            code_context: The packed codebase.
            history: Previous analysis history.
            focus: Optional development focus to steer the analysis
                (e.g., "add analytics tracking for videos processed").

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

        # Inject development focus if provided
        if focus:
            focus_instruction = f"""
## Development Focus

In addition to the standard analysis, prioritize recommendations and tasks
that help achieve the following development goal:

**{focus}**

When generating tasks, include specific implementation steps for this feature
while considering how it integrates with the existing codebase.

---

"""
            rendered_prompt = focus_instruction + rendered_prompt

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

    def build_prompt_index(self) -> str:
        """Build a lightweight index of all available prompts.

        Returns a formatted string with prompt ID and goal for each prompt,
        organized by category. This is injected into triage prompts so the
        LLM can select from ALL available prompts dynamically.

        Returns:
            Formatted string listing all prompts by category.
        """
        prompts_by_category = self.prompt_library.list_prompts_by_category()

        lines = []
        for category in sorted(prompts_by_category.keys()):
            # Skip meta prompts (triage prompts themselves)
            if category == "meta":
                continue

            prompts = prompts_by_category[category]
            # Capitalize category name for display
            category_display = category.replace("_", " ").title()
            lines.append(f"\n**{category_display}:**")

            for prompt in sorted(prompts, key=lambda p: p.id):
                goal = prompt.goal or "No description"
                # Truncate long goals
                if len(goal) > 100:
                    goal = goal[:97] + "..."
                lines.append(f"- `{prompt.id}`: {goal}")

        return "\n".join(lines)

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

        # Build dynamic prompt index from ALL available prompts
        available_prompts = self.build_prompt_index()

        rendered_prompt = triage_prompt.render(
            prd=prd_content,
            code_context=code_context,
            history=history.format_for_prompt(),
            available_prompts=available_prompts,
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
            "",
            "### IMPORTANT: Use Taskmaster for Implementation",
            "",
            "**You MUST use Taskmaster to implement these tasks.** Do NOT manage tasks manually.",
            "",
            "To import these tasks into Taskmaster:",
            "```bash",
            "# Parse this file as a PRD to add tasks to Taskmaster",
            "task-master parse-prd .meta-agent-tasks.md --append",
            "",
            "# Then work through tasks using Taskmaster commands:",
            "task-master list                    # See all tasks",
            "task-master next                    # Get next task to work on",
            "task-master set-status --id=<id> --status=in-progress",
            "task-master set-status --id=<id> --status=done",
            "```",
            "",
            "### Task Workflow:",
            "1. Import tasks into Taskmaster using `parse-prd --append`",
            "2. Use `task-master next` to get the highest priority task",
            "3. Mark task as `in-progress` before starting",
            "4. Implement the task following the description below",
            "5. Mark task as `done` when complete",
            "6. Commit changes after completing related tasks",
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

        # Initialize Ollama for local triage (if available)
        # Uses local LLM to analyze full codebase for free, then sends
        # only relevant files to Perplexity for detailed analysis
        self.ollama_engine: Optional[OllamaEngine] = None
        if not config.mock_mode:
            ollama_status = check_ollama_status()
            if ollama_status["running"] and ollama_status["recommended_model"]:
                self.ollama_engine = OllamaEngine(
                    model=ollama_status["recommended_model"],
                )
                logger.info(
                    f"Ollama available for local triage (model: {ollama_status['recommended_model']})"
                )
            else:
                logger.debug(
                    "Ollama not available - will use Perplexity for triage. "
                    "Install Ollama for lower API costs: https://ollama.com"
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

    def refine_with_ollama_triage(
        self, focus: Optional[str] = None
    ) -> RefinementResult:
        """Run refinement with Ollama-based intelligent triage.

        This mode uses Ollama (local LLM) to analyze the full codebase for FREE,
        then sends only relevant files to Perplexity for detailed analysis.

        Flow:
        1. Pack full codebase with Repomix
        2. Ollama reads full codebase + PRD + all prompts (local, free)
        3. Ollama selects relevant prompts and files
        4. Extract only those files from the packed codebase
        5. Send reduced context to Perplexity for each selected prompt

        Args:
            focus: Optional custom focus for the analysis (e.g., "electron frontend
                with gamified UX"). Steers prompt selection toward specific goals.

        Returns:
            RefinementResult with the outcome.
        """
        if not self.ollama_engine:
            return RefinementResult(
                success=False,
                profile_name="ollama_triage",
                stages_completed=0,
                stages_failed=0,
                error="Ollama not available. Install from https://ollama.com and run: ollama pull llama3.2:3b",
            )

        logger.info("Starting refinement with Ollama-based triage")
        logger.info(f"Target repo: {self.config.repo_path}")
        logger.info(f"Ollama model: {self.ollama_engine.model}")
        if focus:
            logger.info(f"Focus: {focus}")

        # Load PRD
        prd_content = self._load_prd()
        if prd_content is None:
            return RefinementResult(
                success=False,
                profile_name="ollama_triage",
                stages_completed=0,
                stages_failed=0,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        # Pack full codebase (no filtering - Ollama will handle it)
        logger.info("Packing full codebase for Ollama analysis...")
        full_code_context = self._pack_codebase()

        if not full_code_context:
            return RefinementResult(
                success=False,
                profile_name="ollama_triage",
                stages_completed=0,
                stages_failed=0,
                error="Failed to pack codebase",
            )

        # Build prompt index for Ollama
        prompt_index = self.triage_engine.build_prompt_index()

        # Run Ollama triage (FREE - local processing)
        logger.info("Running Ollama triage (local, free)...")
        triage_result = self.ollama_engine.triage(
            prd_content=prd_content,
            code_context=full_code_context,
            prompt_index=prompt_index,
            focus=focus,
        )

        if not triage_result.success:
            logger.error(f"Ollama triage failed: {triage_result.error}")
            return RefinementResult(
                success=False,
                profile_name="ollama_triage",
                stages_completed=0,
                stages_failed=0,
                error=f"Ollama triage failed: {triage_result.error}",
            )

        logger.info(f"Ollama assessment: {triage_result.assessment}")
        logger.info(f"Ollama selected {len(triage_result.selected_prompts)} prompts")

        if not triage_result.selected_prompts:
            logger.info("Ollama indicates codebase is complete - no analysis needed")
            return RefinementResult(
                success=True,
                profile_name="ollama_triage",
                stages_completed=0,
                stages_failed=0,
            )

        # Extract only relevant files for each prompt and run analysis
        history = RunHistory()
        stage_results = []
        stages_completed = 0
        stages_failed = 0

        for prompt_selection in triage_result.selected_prompts:
            prompt_id = prompt_selection.get("prompt_id", "")
            relevant_files = prompt_selection.get("relevant_files", [])
            reasoning = prompt_selection.get("reasoning", "")

            logger.info(f"Running prompt: {prompt_id}")
            logger.info(f"  Relevant files: {relevant_files}")
            logger.info(f"  Reasoning: {reasoning}")

            # Get the prompt
            prompt = self.prompt_library.get_prompt(prompt_id)
            if not prompt:
                logger.warning(f"Prompt not found: {prompt_id}")
                stages_failed += 1
                continue

            # Extract only relevant files from the full codebase
            if relevant_files:
                reduced_context = extract_files(full_code_context, relevant_files)
                tokens_saved = estimate_tokens(full_code_context) - estimate_tokens(reduced_context)
                logger.info(
                    f"  Reduced context: {format_token_count(estimate_tokens(reduced_context))} "
                    f"(saved {format_token_count(tokens_saved)} tokens)"
                )
            else:
                # If no specific files, use full context
                reduced_context = full_code_context

            # Run the analysis with reduced context (PAID - but much smaller)
            # Pass focus to Perplexity so it can do the heavy thinking
            stage_result, success = self.stage_runner.run_stage(
                prompt=prompt,
                prd_content=prd_content,
                code_context=reduced_context,
                history=history,
                focus=focus,
            )

            if success:
                stages_completed += 1
                stage_results.append(stage_result)
            else:
                stages_failed += 1

        # Write plan
        plan_path = None
        if stage_results:
            logger.info("Writing improvement plan...")
            plan_path = self.plan_writer.write_plan(
                prd_content=prd_content,
                profile_name="Ollama Intelligent Triage",
                stage_results=stage_results,
            )
            logger.info(f"Plan written to: {plan_path}")

        # Generate implementation report
        implementation_report = None
        if stage_results:
            implementation_report = self.implementation_executor.analyze_and_report(
                stage_results
            )
            logger.info(
                f"Generated implementation report with "
                f"{len(implementation_report.tasks)} tasks"
            )

        return RefinementResult(
            success=stages_failed == 0 and stages_completed > 0,
            profile_name="ollama_triage",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            plan_path=plan_path,
            stage_results=stage_results,
            partial_success=stages_failed > 0 and stages_completed > 0,
            implementation_report=implementation_report,
        )

    def refine_with_feature_focus(self, feature_request: str) -> RefinementResult:
        """Run feature-focused refinement with intelligent prompt selection and rewriting.

        This mode combines Ollama (for file selection) with Perplexity (for heavy lifting):

        Flow:
        1. Pack full codebase with Repomix
        2. Ollama reads codebase + feature request, returns relevant FILES only (lightweight)
        3. Extract those files from the packed codebase
        4. Perplexity receives:
           - Relevant code (reduced context)
           - Feature request
           - List of ALL codebase-digest prompts
        5. Perplexity:
           - Selects which prompts are relevant for this feature
           - REWRITES those prompts to be specific to the feature
           - Generates implementation tasks

        This gives Perplexity the full context to do intelligent prompt selection
        and customization, while Ollama handles cheap file filtering.

        Args:
            feature_request: The feature to implement or bug to fix
                (e.g., "add analytics: track videos processed, captions, API calls").

        Returns:
            RefinementResult with the outcome.
        """
        if not self.ollama_engine:
            return RefinementResult(
                success=False,
                profile_name="feature_focus",
                stages_completed=0,
                stages_failed=0,
                error="Ollama not available. Install from https://ollama.com and run: ollama pull llama3.2:3b",
            )

        logger.info("Starting feature-focused refinement")
        logger.info(f"Target repo: {self.config.repo_path}")
        logger.info(f"Feature request: {feature_request}")

        # Load PRD
        prd_content = self._load_prd()
        if prd_content is None:
            return RefinementResult(
                success=False,
                profile_name="feature_focus",
                stages_completed=0,
                stages_failed=0,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        # Pack full codebase
        logger.info("Packing full codebase...")
        full_code_context = self._pack_codebase()

        if not full_code_context:
            return RefinementResult(
                success=False,
                profile_name="feature_focus",
                stages_completed=0,
                stages_failed=0,
                error="Failed to pack codebase",
            )

        # Step 1: Ollama does lightweight file selection (dedicated method)
        logger.info("Step 1: Ollama selecting relevant files (local, free)...")
        relevant_files = self.ollama_engine.select_files(
            feature_request=feature_request,
            code_context=full_code_context,
            prd_content=prd_content,
        )

        if relevant_files:
            logger.info(f"Ollama selected {len(relevant_files)} relevant files: {relevant_files}")
        else:
            logger.info("Ollama returned no files, using full codebase context")

        # Step 2: Extract only relevant files (or use full context if no files selected)
        if relevant_files:
            reduced_context = extract_files(full_code_context, relevant_files)
            tokens_full = estimate_tokens(full_code_context)
            tokens_reduced = estimate_tokens(reduced_context)
            logger.info(
                f"Reduced context: {format_token_count(tokens_reduced)} "
                f"(saved {format_token_count(tokens_full - tokens_reduced)} tokens)"
            )
        else:
            reduced_context = full_code_context
            logger.info("Using full codebase context")

        # Step 3: Build prompt index for Perplexity (all available prompts)
        prompt_index = self.triage_engine.build_prompt_index()

        # Step 4: Get the feature expansion prompt
        feature_prompt = self.prompt_library.get_prompt("meta_feature_expansion")
        if not feature_prompt:
            return RefinementResult(
                success=False,
                profile_name="feature_focus",
                stages_completed=0,
                stages_failed=0,
                error="Feature expansion prompt (meta_feature_expansion) not found",
            )

        # Step 5: Send to Perplexity for heavy lifting
        logger.info("Step 2: Perplexity analyzing feature and customizing prompts...")
        rendered_prompt = feature_prompt.render(
            feature_request=feature_request,
            code_context=reduced_context,
            prd=prd_content,
            available_prompts=prompt_index,
        )

        # Use analyze_raw to avoid schema validation (feature expansion has its own schema)
        analysis_result = self.analysis_engine.analyze_raw(
            rendered_prompt,
            system_message=(
                "You are a senior software architect. Analyze the codebase and feature request. "
                "Respond with valid JSON following the exact schema specified in the prompt."
            ),
        )

        if not analysis_result.success:
            return RefinementResult(
                success=False,
                profile_name="feature_focus",
                stages_completed=0,
                stages_failed=0,
                error=f"Perplexity analysis failed: {analysis_result.error}",
            )

        # Step 6: Parse and process the response
        logger.info("Processing Perplexity response...")
        feature_result = self._parse_feature_response(analysis_result)

        if not feature_result:
            return RefinementResult(
                success=False,
                profile_name="feature_focus",
                stages_completed=0,
                stages_failed=0,
                error="Failed to parse feature expansion response",
            )

        # Build stage results from the customized prompts and tasks
        stage_results = []

        # Create a stage result for the feature analysis
        stage_result = StageResult(
            stage_id="feature_expansion",
            stage_name=f"Feature: {feature_result.get('feature_analysis', {}).get('core_functionality', feature_request)[:50]}",
            summary=self._format_feature_summary(feature_result),
            recommendations=self._extract_recommendations(feature_result),
            tasks=feature_result.get("implementation_tasks", []),
        )
        stage_results.append(stage_result)

        # Log the customized prompts
        selected_prompts = feature_result.get("selected_prompts", [])
        logger.info(f"Perplexity selected and customized {len(selected_prompts)} prompts:")
        for sp in selected_prompts:
            logger.info(f"  - {sp.get('original_prompt_id')}: {sp.get('rationale', '')[:80]}")

        # Write plan
        plan_path = None
        if stage_results:
            logger.info("Writing improvement plan...")
            plan_path = self.plan_writer.write_plan(
                prd_content=prd_content,
                profile_name=f"Feature Focus: {feature_request[:50]}",
                stage_results=stage_results,
            )
            logger.info(f"Plan written to: {plan_path}")

        # Generate implementation report
        implementation_report = self.implementation_executor.analyze_and_report(stage_results)
        logger.info(f"Generated {len(implementation_report.tasks)} implementation tasks")

        return RefinementResult(
            success=True,
            profile_name="feature_focus",
            stages_completed=1,
            stages_failed=0,
            plan_path=plan_path,
            stage_results=stage_results,
            implementation_report=implementation_report,
        )

    def refine_with_feature_focus_iterative(
        self,
        feature_request: str,
        max_iterations: int = 10,
    ) -> RefinementResult:
        """Run iterative feature-focused refinement until the feature is complete.

        This mode runs the feature-focused refinement in a loop:

        1. Pack codebase with Repomix
        2. Ollama selects relevant files (local, free)
        3. Perplexity analyzes and generates tasks
        4. Check if "done" (feature already implemented)
        5. If not done, write tasks file
        6. Claude Code implements the tasks (if auto_implement enabled)
        7. Commit changes
        8. Repeat until done or max_iterations reached

        Args:
            feature_request: The feature to implement (e.g., "add analytics tracking").
            max_iterations: Maximum iterations before stopping.

        Returns:
            RefinementResult with the outcome.
        """
        if not self.ollama_engine:
            return RefinementResult(
                success=False,
                profile_name="feature_focus_iterative",
                stages_completed=0,
                stages_failed=0,
                error="Ollama not available. Install from https://ollama.com and run: ollama pull llama3.2:3b",
            )

        logger.info("Starting iterative feature-focused refinement")
        logger.info(f"Target repo: {self.config.repo_path}")
        logger.info(f"Feature request: {feature_request}")
        logger.info(f"Max iterations: {max_iterations}")

        # Emit session start event for dashboard
        emit(EventType.SESSION_START, {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "feature_request": feature_request,
            "max_iterations": max_iterations,
            "repo_path": str(self.config.repo_path),
        })

        # Load PRD once (doesn't change between iterations)
        prd_content = self._load_prd()
        if prd_content is None:
            return RefinementResult(
                success=False,
                profile_name="feature_focus_iterative",
                stages_completed=0,
                stages_failed=0,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        # Get the feature expansion prompt once
        feature_prompt = self.prompt_library.get_prompt("meta_feature_expansion")
        if not feature_prompt:
            return RefinementResult(
                success=False,
                profile_name="feature_focus_iterative",
                stages_completed=0,
                stages_failed=0,
                error="Feature expansion prompt (meta_feature_expansion) not found",
            )

        # Build prompt index once (doesn't change)
        prompt_index = self.triage_engine.build_prompt_index()

        all_stage_results: list[StageResult] = []
        iterations: list[IterationResult] = []
        stages_completed = 0
        stages_failed = 0

        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"=== Iteration {iteration}/{max_iterations} ===")
            logger.info(f"{'='*60}\n")

            # Emit iteration start event
            emit(EventType.ITERATION_START, {"iteration": iteration, "max_iterations": max_iterations})

            # Step 1: Pack codebase (fresh each iteration to see changes)
            logger.info("Step 1: Packing codebase...")
            emit(EventType.LOG, {"message": "Packing codebase...", "level": "info"})
            full_code_context = self._pack_codebase()

            if not full_code_context:
                logger.error("Failed to pack codebase")
                stages_failed += 1
                continue

            # Step 2: Ollama selects relevant files
            logger.info("Step 2: Ollama selecting relevant files (local, free)...")
            emit(EventType.OLLAMA_START, {"message": "Selecting relevant files..."})
            relevant_files = self.ollama_engine.select_files(
                feature_request=feature_request,
                code_context=full_code_context,
                prd_content=prd_content,
            )

            if relevant_files:
                logger.info(f"Selected {len(relevant_files)} files: {relevant_files[:5]}...")
                emit(EventType.OLLAMA_COMPLETE, {"file_count": len(relevant_files), "files": relevant_files[:10]})
                reduced_context = extract_files(full_code_context, relevant_files)
                tokens_saved = estimate_tokens(full_code_context) - estimate_tokens(reduced_context)
                logger.info(f"Reduced context by {format_token_count(tokens_saved)} tokens")
                emit(EventType.LOG, {"message": f"Reduced context by {format_token_count(tokens_saved)} tokens", "level": "info"})
            else:
                logger.info("Using full codebase context")
                emit(EventType.OLLAMA_COMPLETE, {"file_count": 0, "files": []})
                reduced_context = full_code_context

            # Step 3: Perplexity analyzes
            logger.info("Step 3: Perplexity analyzing feature...")
            emit(EventType.PERPLEXITY_START, {"message": "Analyzing feature and generating tasks..."})
            rendered_prompt = feature_prompt.render(
                feature_request=feature_request,
                code_context=reduced_context,
                prd=prd_content,
                available_prompts=prompt_index,
            )

            analysis_result = self.analysis_engine.analyze_raw(
                rendered_prompt,
                system_message=(
                    "You are a senior software architect. Analyze the codebase and feature request. "
                    "Respond with valid JSON following the exact schema specified in the prompt. "
                    "IMPORTANT: If the feature is already implemented, set done=true."
                ),
            )

            if not analysis_result.success:
                logger.error(f"Analysis failed: {analysis_result.error}")
                stages_failed += 1
                continue

            # Step 4: Parse response and check if done
            emit(EventType.PERPLEXITY_COMPLETE, {"message": "Analysis complete"})
            feature_result = self._parse_feature_response(analysis_result)
            if not feature_result:
                logger.error("Failed to parse feature response")
                emit(EventType.ERROR, {"message": "Failed to parse feature response"})
                stages_failed += 1
                continue

            is_done = feature_result.get("done", False)
            done_summary = feature_result.get("done_summary", "")

            # Extract layer status
            layer_status = feature_result.get("layer_status", {})
            current_layer = layer_status.get("current_layer", 1)
            layer_name = layer_status.get("layer_name", "scaffold")
            layer_progress = layer_status.get("layer_progress", "")
            layers_complete = layer_status.get("layers_complete", {})

            # Emit layer update event
            emit(EventType.LAYER_UPDATE, {
                "current_layer": current_layer,
                "layer_name": layer_name,
                "layer_progress": layer_progress,
                "layers_complete": layers_complete,
            })

            # Create LayerStatus object
            current_layer_status = LayerStatus(
                current_layer=current_layer,
                layer_name=layer_name,
                layer_progress=layer_progress,
                scaffold_complete=layers_complete.get("scaffold", False),
                core_complete=layers_complete.get("core", False),
                integration_complete=layers_complete.get("integration", False),
                polish_complete=layers_complete.get("polish", False),
            )

            # Log layer progress
            logger.info(f"Layer: {current_layer}/4 ({layer_name.upper()})")
            if layer_progress:
                progress_preview = layer_progress[:80] + "..." if len(layer_progress) > 80 else layer_progress
                logger.info(f"Progress: {progress_preview}")

            # Show layer completion status
            def layer_icon(done: bool) -> str:
                return "[+]" if done else "[ ]"

            logger.info(
                f"  {layer_icon(current_layer_status.scaffold_complete)} Scaffold  "
                f"{layer_icon(current_layer_status.core_complete)} Core  "
                f"{layer_icon(current_layer_status.integration_complete)} Integration  "
                f"{layer_icon(current_layer_status.polish_complete)} Polish"
            )

            if is_done:
                logger.info("="*60)
                logger.info("FEATURE COMPLETE! All layers done.")
                logger.info(f"Summary: {done_summary}")
                logger.info("="*60)

                # Record final iteration with completed layer status
                iterations.append(
                    IterationResult(
                        iteration=iteration,
                        prompts_run=[],
                        changes_made=False,
                        committed=False,
                        stage_results=[],
                        layer_status=current_layer_status,
                    )
                )
                break

            # Step 5: Build stage results from tasks
            tasks = feature_result.get("implementation_tasks", [])
            logger.info(f"Generated {len(tasks)} tasks for layer: {layer_name}")

            # Emit task list event
            emit(EventType.TASK_LIST, {
                "tasks": [{"title": t.get("title", ""), "priority": t.get("priority", "medium")} for t in tasks],
                "count": len(tasks),
                "layer": layer_name,
            })

            stage_result = StageResult(
                stage_id=f"feature_iteration_{iteration}_layer_{current_layer}",
                stage_name=f"Layer {current_layer} ({layer_name.capitalize()}): {feature_result.get('feature_analysis', {}).get('core_functionality', feature_request)[:30]}",
                summary=self._format_feature_summary(feature_result),
                recommendations=self._extract_recommendations(feature_result),
                tasks=tasks,
            )
            all_stage_results.append(stage_result)
            stages_completed += 1

            # Step 6: Write tasks file and implement
            if tasks:
                # Write tasks to file for Claude Code
                implementation_report = self.implementation_executor.analyze_and_report([stage_result])
                logger.info(f"Wrote {len(implementation_report.tasks)} tasks to .meta-agent-tasks.md")

                # Execute implementation if auto_implement is enabled
                changes_made = False
                if self.config.auto_implement:
                    logger.info("Step 6: Claude Code implementing changes...")
                    changes_made = self.implementation_executor.execute([stage_result])
                    if changes_made:
                        logger.info("Implementation completed")
                    else:
                        logger.warning("No changes made during implementation")
                else:
                    logger.info("Step 6: Tasks written. Run Claude Code to implement.")
                    logger.info("       Use --auto-implement to run Claude Code automatically.")
                    # In non-auto mode, we can't loop - just return after first iteration
                    break

                # Step 7: Commit changes
                commit_hash = None
                if changes_made and self.config.auto_commit:
                    logger.info("Step 7: Committing changes...")
                    commit_hash = self.implementation_executor.commit_changes(
                        f"feat: iteration {iteration} - {feature_request[:50]}",
                        prompt_ids=["feature_focus"],
                    )
                    if commit_hash:
                        logger.info(f"Committed: {commit_hash}")

                iterations.append(
                    IterationResult(
                        iteration=iteration,
                        prompts_run=["feature_focus"],
                        changes_made=changes_made,
                        committed=commit_hash is not None,
                        commit_hash=commit_hash,
                        stage_results=[stage_result],
                        layer_status=current_layer_status,
                    )
                )

                if not changes_made:
                    logger.warning("No changes made - stopping iteration")
                    break
            else:
                logger.warning("No tasks generated - stopping iteration")
                iterations.append(
                    IterationResult(
                        iteration=iteration,
                        prompts_run=["feature_focus"],
                        changes_made=False,
                        committed=False,
                        stage_results=[stage_result],
                        layer_status=current_layer_status,
                    )
                )
                break

        # Write final plan
        plan_path = None
        if all_stage_results:
            logger.info("Writing final improvement plan...")
            plan_path = self.plan_writer.write_plan(
                prd_content=prd_content,
                profile_name=f"Feature Focus (Iterative): {feature_request[:40]}",
                stage_results=all_stage_results,
            )
            logger.info(f"Plan written to: {plan_path}")

        # Generate final implementation report
        implementation_report = None
        if all_stage_results:
            implementation_report = self.implementation_executor.analyze_and_report(
                all_stage_results
            )
            logger.info(f"Final report: {len(implementation_report.tasks)} total tasks")

        # Determine success
        feature_done = any(
            it.iteration > 0 and not it.changes_made and it.prompts_run == []
            for it in iterations
        )

        # Emit session end event
        emit(EventType.SESSION_END, {
            "success": feature_done or (stages_failed == 0 and stages_completed > 0),
            "stages_completed": stages_completed,
            "stages_failed": stages_failed,
            "iterations": len(iterations),
        })

        return RefinementResult(
            success=feature_done or (stages_failed == 0 and stages_completed > 0),
            profile_name="feature_focus_iterative",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            plan_path=plan_path,
            stage_results=all_stage_results,
            iterations=iterations,
            partial_success=stages_failed > 0 and stages_completed > 0,
            implementation_report=implementation_report,
        )

    def _parse_feature_response(self, analysis_result: AnalysisResult) -> Optional[dict]:
        """Parse the feature expansion response from Perplexity.

        Args:
            analysis_result: Raw analysis result from Perplexity.

        Returns:
            Parsed feature response dict or None if parsing fails.
        """
        response_text = analysis_result.raw_response or analysis_result.summary

        if not response_text:
            logger.error("Empty feature expansion response")
            return None

        # Log first part of response for debugging
        logger.debug(f"Feature response (first 500 chars): {response_text[:500]}")

        # Use shared JSON extraction
        data, error = extract_json_from_response(response_text)

        if data is None:
            logger.error(f"Failed to parse feature response: {error}")
            logger.debug(f"Full response: {response_text[:2000]}")
            return None

        logger.debug(f"Parsed feature data keys: {list(data.keys())}")
        return data

    def _format_feature_summary(self, feature_result: dict) -> str:
        """Format feature analysis into a summary string.

        Args:
            feature_result: Parsed feature response dict.

        Returns:
            Formatted summary string.
        """
        analysis = feature_result.get("feature_analysis", {})
        lines = []

        if analysis.get("core_functionality"):
            lines.append(f"**Core Functionality:** {analysis['core_functionality']}")

        additions = analysis.get("suggested_additions", [])
        if additions:
            lines.append(f"\n**Suggested Additions:** {len(additions)} additional features recommended")
            for add in additions[:5]:  # Show first 5
                lines.append(f"  - {add}")
            if len(additions) > 5:
                lines.append(f"  - ... and {len(additions) - 5} more")

        files = analysis.get("affected_files", [])
        if files:
            lines.append(f"\n**Affected Files:** {', '.join(files)}")

        if analysis.get("architectural_notes"):
            lines.append(f"\n**Architecture Notes:** {analysis['architectural_notes']}")

        # Add customized prompts section
        prompts = feature_result.get("selected_prompts", [])
        if prompts:
            lines.append(f"\n**Customized Analysis Prompts:** {len(prompts)} prompts tailored to this feature")
            for p in prompts:
                lines.append(f"  - {p.get('original_prompt_id')}: {p.get('rationale', '')[:60]}...")

        return "\n".join(lines)

    def _extract_recommendations(self, feature_result: dict) -> list[str]:
        """Extract recommendations from feature response.

        Args:
            feature_result: Parsed feature response dict.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Add suggested additions as recommendations
        analysis = feature_result.get("feature_analysis", {})
        for addition in analysis.get("suggested_additions", []):
            recommendations.append(f"Consider adding: {addition}")

        # Add customized prompt summaries
        for prompt in feature_result.get("selected_prompts", []):
            if prompt.get("rationale"):
                recommendations.append(f"Analysis focus: {prompt['rationale']}")

        return recommendations

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

    def run_autonomous_loop(
        self,
        feature_request: str,
    ) -> AutonomousLoopResult:
        """Run the autonomous development loop.

        This method implements the full autonomous cycle:
        1. Analyze codebase with Ollama + Perplexity
        2. Generate task plan
        3. Implement each task with Claude API
        4. Run tests
        5. On failure: get Perplexity diagnosis -> Claude fix
        6. Commit changes
        7. Repeat until complete or max iterations

        Args:
            feature_request: Description of the feature to implement.

        Returns:
            AutonomousLoopResult with success status and details.
        """
        logger.info("Starting autonomous development loop")
        emit(EventType.SESSION_START, {"mode": "autonomous_loop", "feature": feature_request})

        loop_config = self.config.loop
        max_iterations = loop_config.max_iterations

        # Initialize Claude implementer
        if self.config.mock_mode:
            claude_impl = MockClaudeImplementer(
                model=loop_config.claude_model,
            )
        else:
            claude_impl = ClaudeImplementer(
                api_key=self.config.anthropic_api_key,
                model=loop_config.claude_model,
            )

        # Initialize workspace manager
        workspace = WorkspaceManager(self.config.repo_path)

        # Create branch if configured
        branch_name = None
        if loop_config.create_branch:
            branch_name = generate_branch_name(loop_config.branch_pattern)
            try:
                workspace.create_branch(branch_name)
                logger.info(f"Created branch: {branch_name}")
            except Exception as e:
                logger.warning(f"Failed to create branch: {e}")
                branch_name = None

        # Load PRD
        prd_content = self._load_prd()
        if prd_content is None:
            return AutonomousLoopResult(
                success=False,
                iterations_completed=0,
                max_iterations=max_iterations,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        # Track loop state
        iterations_completed = 0
        tasks_completed = 0
        tests_passed = 0
        fixes_applied = 0
        tokens_used = 0
        commits_made = 0
        consecutive_failures = 0
        iteration_details = []

        try:
            while iterations_completed < max_iterations:
                iterations_completed += 1
                logger.info(f"=== Autonomous Loop Iteration {iterations_completed}/{max_iterations} ===")
                emit(EventType.ITERATION_START, {"iteration": iterations_completed, "max": max_iterations})

                # Step 1: Pack codebase and analyze
                code_context = self._pack_codebase()

                # Step 2: Get analysis and task plan using feature focus
                analysis_result = self._run_feature_analysis(
                    feature_request=feature_request,
                    prd_content=prd_content,
                    code_context=code_context,
                )

                if not analysis_result.success:
                    logger.error(f"Analysis failed: {analysis_result.error}")
                    consecutive_failures += 1
                    if consecutive_failures >= loop_config.max_consecutive_failures:
                        break
                    continue

                # Extract tasks from analysis
                tasks = self._extract_tasks_from_analysis(analysis_result)
                if not tasks:
                    logger.info("No more tasks to implement - feature complete!")
                    break

                # Step 3: Implement each task
                for task in tasks:
                    logger.info(f"Implementing task: {task.get('title', 'Unknown')}")
                    emit(EventType.TASK_START, {"task": task.get("title", "Unknown")})

                    # Get fresh repomix for implementation
                    repomix_result = self.repomix_runner.pack(self.config.repo_path)
                    repomix_xml = repomix_result.content if repomix_result.success else None

                    # Implement with Claude
                    impl_result = claude_impl.apply_task_to_repo(
                        task_description=task.get("description", task.get("title", "")),
                        workspace=workspace,
                        repomix_xml=repomix_xml,
                        dry_run=loop_config.dry_run,
                    )

                    tokens_used += impl_result.tokens_used

                    if not impl_result.success:
                        logger.error(f"Implementation failed: {impl_result.error}")
                        consecutive_failures += 1
                        continue

                    tasks_completed += 1
                    consecutive_failures = 0

                    # Step 4: Run tests
                    test_result = workspace.run_tests(loop_config.test_command)

                    if test_result.passed:
                        tests_passed += 1
                        logger.info("Tests passed!")

                        # Step 5: Commit if configured
                        if loop_config.commit_per_task and not loop_config.dry_run:
                            commit_result = workspace.commit(impl_result.commit_message)
                            if commit_result.success:
                                commits_made += 1
                                logger.info(f"Committed: {commit_result.commit_hash}")
                    else:
                        # Step 6: Get Perplexity diagnosis and fix
                        logger.info("Tests failed, getting diagnosis...")
                        fix_result = self._diagnose_and_fix(
                            task=task,
                            test_result=test_result,
                            workspace=workspace,
                            claude_impl=claude_impl,
                            prd_content=prd_content,
                        )

                        if fix_result.get("success"):
                            fixes_applied += 1
                            tokens_used += fix_result.get("tokens_used", 0)

                            # Re-run tests after fix
                            retest_result = workspace.run_tests(loop_config.test_command)
                            if retest_result.passed:
                                tests_passed += 1
                                if loop_config.commit_per_task and not loop_config.dry_run:
                                    commit_result = workspace.commit(
                                        f"fix: {impl_result.commit_message}"
                                    )
                                    if commit_result.success:
                                        commits_made += 1
                            else:
                                consecutive_failures += 1
                        else:
                            consecutive_failures += 1

                    emit(EventType.TASK_COMPLETE, {
                        "task": task.get("title", "Unknown"),
                        "success": test_result.passed,
                    })

                    # Store iteration details
                    iteration_details.append({
                        "iteration": iterations_completed,
                        "task": task.get("title", "Unknown"),
                        "tests_passed": test_result.passed,
                        "fixes_applied": 1 if not test_result.passed and fixes_applied > 0 else 0,
                        "success": test_result.passed,
                    })

                    # Check for max consecutive failures
                    if consecutive_failures >= loop_config.max_consecutive_failures:
                        logger.error(f"Max consecutive failures reached ({consecutive_failures})")
                        break

                # Human approval gate
                if loop_config.human_approve and not loop_config.dry_run:
                    try:
                        response = input("\nContinue to next iteration? (Enter=yes, 'q'=quit): ")
                        if response.lower() in ("q", "quit", "n", "no"):
                            logger.info("User requested stop")
                            break
                    except EOFError:
                        pass  # Non-interactive mode

                if consecutive_failures >= loop_config.max_consecutive_failures:
                    break

        except KeyboardInterrupt:
            logger.info("Loop interrupted by user")
        except Exception as e:
            logger.error(f"Loop error: {e}")
            return AutonomousLoopResult(
                success=False,
                iterations_completed=iterations_completed,
                max_iterations=max_iterations,
                tasks_completed=tasks_completed,
                tests_passed=tests_passed,
                fixes_applied=fixes_applied,
                tokens_used=tokens_used,
                branch_name=branch_name,
                commits_made=commits_made,
                error=str(e),
                iteration_details=iteration_details,
            )

        # Final PRD evaluation
        final_evaluation = None
        prd_aligned = False
        if tasks_completed > 0:
            eval_result = self._run_prd_evaluation(prd_content)
            if eval_result:
                final_evaluation = eval_result.get("overall_assessment", "")
                prd_aligned = eval_result.get("approved", False)

        emit(EventType.SESSION_END, {
            "success": tasks_completed > 0 and consecutive_failures < loop_config.max_consecutive_failures,
            "tasks_completed": tasks_completed,
        })

        return AutonomousLoopResult(
            success=tasks_completed > 0 and consecutive_failures < loop_config.max_consecutive_failures,
            iterations_completed=iterations_completed,
            max_iterations=max_iterations,
            tasks_completed=tasks_completed,
            tests_passed=tests_passed,
            fixes_applied=fixes_applied,
            tokens_used=tokens_used,
            branch_name=branch_name,
            commits_made=commits_made,
            final_evaluation=final_evaluation,
            prd_aligned=prd_aligned,
            iteration_details=iteration_details,
        )

    def run_local_loop(
        self,
        prd_path: str,
        evaluator_override: Optional[str] = None,
    ) -> LocalLoopResult:
        """Run the Grok-powered local development loop.

        This method implements the local-first autonomous cycle using GitPython
        and Grok for error diagnosis:
        1. Read PRD from file
        2. Initialize LocalRepoManager (creates branch)
        3. Pack current repo with Repomix
        4. Generate task plan
        5. For each task:
           a. Implement task (Claude)
           b. Commit changes (if not dry-run)
           c. Run tests
           d. If tests fail: Grok diagnosis -> Claude fix -> retry
        6. Final Grok evaluation of PRD alignment
        7. Return summary result

        Args:
            prd_path: Path to the PRD file.
            evaluator_override: Optional override for evaluator ('grok' or 'perplexity').

        Returns:
            LocalLoopResult with success status and details.
        """
        logger.info("Starting Grok-powered local development loop")
        emit(EventType.SESSION_START, {"mode": "local_loop", "prd": prd_path})

        loop_config = self.config.loop
        max_iterations = loop_config.max_iterations
        evaluator_name = get_evaluator_name(loop_config.evaluator, evaluator_override)

        # Read PRD from file
        prd_file = Path(prd_path)
        if not prd_file.exists():
            return LocalLoopResult(
                success=False,
                iterations_completed=0,
                max_iterations=max_iterations,
                evaluator_used=evaluator_name,
                error=f"PRD file not found: {prd_path}",
            )

        try:
            prd_content = prd_file.read_text(encoding="utf-8")
        except Exception as e:
            return LocalLoopResult(
                success=False,
                iterations_completed=0,
                max_iterations=max_iterations,
                evaluator_used=evaluator_name,
                error=f"Failed to read PRD file: {e}",
            )

        # Initialize Claude implementer
        if self.config.mock_mode:
            claude_impl = MockClaudeImplementer(model=loop_config.claude_model)
            grok_client = MockGrokClient()
        else:
            claude_impl = ClaudeImplementer(
                api_key=self.config.anthropic_api_key,
                model=loop_config.claude_model,
            )
            grok_client = GrokClient(
                model=loop_config.evaluator.grok.model,
                temperature=loop_config.evaluator.grok.temperature,
                max_tokens=loop_config.evaluator.grok.max_tokens,
                timeout=loop_config.evaluator.grok.timeout,
            )

        # Initialize local repo manager
        try:
            if self.config.mock_mode:
                local_manager = MockLocalRepoManager(loop_config)
            else:
                local_manager = LocalRepoManager(loop_config, repo_path=self.config.repo_path)
        except LocalRepoError as e:
            return LocalLoopResult(
                success=False,
                iterations_completed=0,
                max_iterations=max_iterations,
                evaluator_used=evaluator_name,
                error=f"Failed to initialize local repo manager: {e}",
            )

        # Create branch if configured
        branch_name = None
        if loop_config.create_branch and not loop_config.dry_run:
            try:
                branch_name = local_manager.create_branch()
                logger.info(f"Working on branch: {branch_name}")
            except Exception as e:
                logger.warning(f"Failed to create branch: {e}")
                branch_name = local_manager.get_current_branch()

        # Track loop state
        iterations_completed = 0
        tasks_completed = 0
        tests_passed = 0
        fixes_applied = 0
        tokens_used = 0
        commits_made = 0
        consecutive_failures = 0
        iteration_details = []
        final_evaluation = None
        prd_aligned = False

        try:
            while iterations_completed < max_iterations:
                iterations_completed += 1
                logger.info(f"=== Local Loop Iteration {iterations_completed}/{max_iterations} ===")
                emit(EventType.ITERATION_START, {"iteration": iterations_completed, "max": max_iterations})

                # Step 1: Pack codebase
                code_context = self._pack_codebase()
                if not code_context:
                    logger.warning("Failed to pack codebase, using empty context")
                    code_context = ""

                # Step 2: Analyze and generate tasks
                analysis_result = self._run_feature_analysis(
                    feature_request=f"Implement features from PRD: {prd_path}",
                    prd_content=prd_content,
                    code_context=code_context,
                )

                if not analysis_result.success:
                    consecutive_failures += 1
                    if consecutive_failures >= loop_config.max_consecutive_failures:
                        logger.error("Max consecutive failures reached")
                        break
                    continue

                # Extract tasks from analysis
                tasks = self._extract_tasks_from_analysis(analysis_result)
                if not tasks:
                    logger.info("No more tasks to implement - feature complete!")
                    break

                consecutive_failures = 0  # Reset on successful analysis

                # Step 3: Implement each task
                for task_idx, task in enumerate(tasks):
                    task_title = task.get("title", f"Task {task_idx + 1}")
                    task_desc = task.get("description", "")
                    logger.info(f"Implementing task: {task_title}")
                    emit(EventType.TASK_START, {"task": task_title})

                    iteration_detail = {
                        "iteration": iterations_completed,
                        "task": task_title,
                        "tests_passed": False,
                        "fixes_applied": 0,
                        "success": False,
                    }

                    # Create workspace manager for Claude impl (uses existing interface)
                    workspace = WorkspaceManager(self.config.repo_path)

                    # Implement task with Claude
                    impl_result = claude_impl.apply_task_to_repo(
                        task_description=f"{task_title}\n\n{task_desc}",
                        workspace=workspace,
                        dry_run=loop_config.dry_run,
                    )

                    if impl_result.tokens_used:
                        tokens_used += impl_result.tokens_used

                    if not impl_result.success:
                        logger.warning(f"Task implementation failed: {impl_result.error}")
                        iteration_details.append(iteration_detail)
                        continue

                    # Commit changes if not dry-run
                    if not loop_config.dry_run and loop_config.commit_per_task:
                        commit_msg = f"[local-loop] {task_title}"
                        commit_result = local_manager.commit_changes(commit_msg)
                        if commit_result.success and commit_result.commit_hash:
                            commits_made += 1
                            logger.info(f"Committed: {commit_result.commit_hash}")

                    # Step 4: Run tests
                    test_result = local_manager.run_tests()
                    if test_result.success:
                        tests_passed += 1
                        tasks_completed += 1
                        iteration_detail["tests_passed"] = True
                        iteration_detail["success"] = True
                        logger.info(f"Tests passed for task: {task_title}")
                        emit(EventType.TASK_COMPLETE, {"task": task_title, "success": True})
                    else:
                        # Step 5: Get Grok diagnosis and attempt fix
                        logger.info("Tests failed, getting Grok diagnosis...")
                        fix_count = 0
                        max_fix_attempts = 3

                        while not test_result.success and fix_count < max_fix_attempts:
                            fix_count += 1
                            try:
                                # Get diagnosis from Grok
                                diagnosis_prompt = self._build_grok_diagnosis_prompt(
                                    task=task,
                                    errors=test_result.error_summary,
                                    code_context=code_context[:50000],  # Truncate
                                )
                                grok_response = grok_client.chat(
                                    diagnosis_prompt,
                                    system_prompt="You are a senior debugging engineer. Analyze test failures and provide structured JSON diagnosis.",
                                )

                                # Parse structured JSON response from Grok
                                fix_prompt = grok_response  # Default to raw response
                                diagnosis_data = extract_json_from_response(grok_response)
                                if diagnosis_data and "fix_prompt" in diagnosis_data:
                                    fix_prompt = diagnosis_data["fix_prompt"]
                                    logger.info(f"Grok diagnosis: {diagnosis_data.get('diagnosis', {}).get('root_cause', 'unknown')}")
                                    logger.info(f"Confidence: {diagnosis_data.get('confidence', 'unknown')}")

                                # Apply fix with Claude
                                fix_result = claude_impl.apply_fix_prompt(
                                    fix_prompt=fix_prompt,
                                    workspace=workspace,
                                    dry_run=loop_config.dry_run,
                                )

                                if fix_result.tokens_used:
                                    tokens_used += fix_result.tokens_used

                                if fix_result.success and not loop_config.dry_run:
                                    commit_msg = f"[local-loop] Fix: {task_title}"
                                    commit_result = local_manager.commit_changes(commit_msg)
                                    if commit_result.success and commit_result.commit_hash:
                                        commits_made += 1

                                # Re-run tests
                                test_result = local_manager.run_tests()

                            except GrokClientError as e:
                                logger.error(f"Grok diagnosis failed: {e}")
                                break

                        iteration_detail["fixes_applied"] = fix_count
                        fixes_applied += fix_count

                        if test_result.success:
                            tests_passed += 1
                            tasks_completed += 1
                            iteration_detail["tests_passed"] = True
                            iteration_detail["success"] = True
                            logger.info(f"Tests passed after {fix_count} fix(es)")
                            emit(EventType.TASK_COMPLETE, {"task": task_title, "success": True})
                        else:
                            logger.warning(f"Tests still failing after {fix_count} attempts")
                            emit(EventType.TASK_COMPLETE, {"task": task_title, "success": False})

                    iteration_details.append(iteration_detail)

                    # Human approval check
                    if loop_config.human_approve and not loop_config.dry_run:
                        try:
                            response = input("\nContinue to next task? (Enter to continue, 'q' to quit): ")
                            if response.lower() == 'q':
                                logger.info("User requested exit")
                                break
                        except (EOFError, KeyboardInterrupt):
                            logger.info("Input interrupted, continuing...")

                # Check if feature is complete (no tasks in last iteration)
                if not tasks:
                    break

        except KeyboardInterrupt:
            logger.info("Loop interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in local loop: {e}")
            return LocalLoopResult(
                success=False,
                iterations_completed=iterations_completed,
                max_iterations=max_iterations,
                tasks_completed=tasks_completed,
                tests_passed=tests_passed,
                fixes_applied=fixes_applied,
                tokens_used=tokens_used,
                branch_name=branch_name,
                commits_made=commits_made,
                evaluator_used=evaluator_name,
                error=str(e),
                iteration_details=iteration_details,
            )
        finally:
            local_manager.cleanup()

        # Step 6: Final Grok evaluation
        if tasks_completed > 0 and not self.config.mock_mode:
            try:
                # Build implementation summary from iteration details
                impl_summary_lines = []
                for detail in iteration_details:
                    status = "✓" if detail.get("success") else "✗"
                    impl_summary_lines.append(
                        f"{status} Task: {detail.get('task_title', 'Unknown')} "
                        f"(fixes: {detail.get('fixes_applied', 0)})"
                    )
                implementation_summary = "\n".join(impl_summary_lines) if impl_summary_lines else ""

                eval_prompt = self._build_grok_evaluation_prompt(
                    prd_content=prd_content,
                    code_context=self._pack_codebase()[:50000],
                    implementation_summary=implementation_summary,
                )
                eval_response = grok_client.chat(
                    eval_prompt,
                    system_prompt="You are a senior evaluator assessing PRD implementation completeness. Provide structured JSON assessment.",
                )
                final_evaluation = eval_response

                # Try to parse as JSON for structured result
                try:
                    eval_data = extract_json_from_response(eval_response)
                    if eval_data:
                        # Handle new structured format with nested "evaluation"
                        if "evaluation" in eval_data:
                            prd_aligned = eval_data["evaluation"].get("prd_aligned", False)
                            completion_pct = eval_data["evaluation"].get("completion_percentage", 0)
                            logger.info(f"PRD completion: {completion_pct}%")
                        # Fall back to flat format
                        prd_aligned = prd_aligned or eval_data.get("approved", False)
                        logger.info(f"PRD aligned: {prd_aligned}")
                        if eval_data.get("remaining_tasks"):
                            logger.info(f"Remaining tasks: {len(eval_data['remaining_tasks'])}")
                except Exception:
                    # If not JSON, just use the raw response
                    prd_aligned = "approved" in eval_response.lower() or "complete" in eval_response.lower()

            except GrokClientError as e:
                logger.warning(f"Final Grok evaluation failed: {e}")

        emit(EventType.SESSION_END, {
            "success": tasks_completed > 0,
            "tasks_completed": tasks_completed,
            "evaluator": evaluator_name,
        })

        return LocalLoopResult(
            success=tasks_completed > 0 and consecutive_failures < loop_config.max_consecutive_failures,
            iterations_completed=iterations_completed,
            max_iterations=max_iterations,
            tasks_completed=tasks_completed,
            tests_passed=tests_passed,
            fixes_applied=fixes_applied,
            tokens_used=tokens_used,
            branch_name=branch_name,
            commits_made=commits_made,
            final_evaluation=final_evaluation,
            prd_aligned=prd_aligned,
            evaluator_used=evaluator_name,
            iteration_details=iteration_details,
        )

    def _build_grok_diagnosis_prompt(
        self,
        task: dict,
        errors: str,
        code_context: str,
    ) -> str:
        """Build a prompt for Grok to diagnose test failures.

        Uses the structured meta_error_fix prompt template for consistent
        JSON output format.

        Args:
            task: The task that failed.
            errors: Error output from tests.
            code_context: Repository context (truncated).

        Returns:
            Prompt string for Grok with structured output expectations.
        """
        task_description = f"""Title: {task.get('title', 'Unknown')}
Description: {task.get('description', '')}
Priority: {task.get('priority', 'unknown')}
Details: {task.get('details', '')}"""

        return f"""You are a senior debugging engineer analyzing code that has failed tests during automated development.

## Current Codebase
{code_context}

## Failing Task
{task_description}

## Test Errors
{errors}

**Instructions:**

1. **Analyze the error context** provided:
   - Identify which files/functions are causing the failure
   - Understand the expected vs actual behavior
   - Trace the error through the stack trace
   - Check for common issues:
     * Missing imports or dependencies
     * Type mismatches
     * Logic errors
     * Edge cases not handled
     * Missing or incorrect error handling

2. **Generate a precise fix prompt** that:
   - Clearly identifies the specific file(s) and function(s) to modify
   - Describes exactly what change is needed
   - Is actionable without requiring additional analysis

**Expected Output:**

Respond with a JSON object in this exact format:
```json
{{
  "diagnosis": {{
    "root_cause": "Brief description of the root cause",
    "affected_files": ["list", "of", "files"],
    "error_type": "type of error (syntax, logic, import, etc.)"
  }},
  "fix_prompt": "Detailed prompt for Claude to fix the issue. Be specific about what to change and where.",
  "confidence": "high|medium|low",
  "alternative_approaches": ["Optional alternative fixes if main approach fails"]
}}
```

Focus on the minimal change needed to fix the error. Provide exact file paths and function names."""

    def _build_grok_evaluation_prompt(
        self,
        prd_content: str,
        code_context: str,
        implementation_summary: str = "",
    ) -> str:
        """Build a prompt for Grok to evaluate PRD completion.

        Uses the structured meta_prd_evaluation prompt template for consistent
        JSON output format.

        Args:
            prd_content: The PRD requirements.
            code_context: Current repository state.
            implementation_summary: Optional summary of what was implemented.

        Returns:
            Prompt string for Grok with structured evaluation expectations.
        """
        return f"""You are evaluating a codebase after an autonomous development loop has attempted to implement features described in a PRD.

## Product Requirements Document
{prd_content}

## Current Codebase
{code_context}

## Implementation History
{implementation_summary if implementation_summary else "No implementation history available."}

**Instructions:**

1. **Compare the codebase to the PRD requirements:**
   - Identify which requirements are fully implemented
   - Identify which requirements are partially implemented
   - Identify which requirements are missing entirely

2. **Evaluate implementation quality:**
   - Does the implementation match the PRD intent?
   - Are there any edge cases not handled?
   - Is the code production-ready or MVP-quality?
   - Are there any obvious bugs or issues?

3. **Check integration completeness:**
   - Do all components work together correctly?
   - Are there any broken connections between modules?
   - Is error handling comprehensive?

4. **Generate final assessment:**
   - Overall completion percentage
   - Remaining critical items
   - Recommended next steps

**Expected Output:**

Respond with a JSON object in this exact format:
```json
{{
  "evaluation": {{
    "completion_percentage": 85,
    "prd_aligned": true,
    "production_ready": false,
    "mvp_ready": true
  }},
  "requirements_status": [
    {{
      "requirement": "Requirement description",
      "status": "complete|partial|missing",
      "notes": "Any relevant notes"
    }}
  ],
  "remaining_tasks": [
    {{
      "task": "Description of remaining task",
      "priority": "critical|high|medium|low",
      "estimated_complexity": "low|medium|high"
    }}
  ],
  "overall_assessment": "Summary of the current state and recommendations",
  "approved": true,
  "approval_reason": "Reason for approval/rejection"
}}
```

**Approval Criteria:**
- approved=true if all critical requirements are implemented, tests pass, and no major bugs exist
- approved=false if critical requirements are missing or major bugs prevent core functionality"""

    def _run_feature_analysis(
        self,
        feature_request: str,
        prd_content: str,
        code_context: str,
    ) -> AnalysisResult:
        """Run feature-focused analysis using Ollama + Perplexity.

        Args:
            feature_request: The feature to analyze for.
            prd_content: PRD content.
            code_context: Packed codebase content.

        Returns:
            AnalysisResult with tasks.
        """
        # Use the existing feature focus mechanism
        try:
            # First, use Ollama to select relevant files
            ollama_result = self.ollama_engine.feature_focused_triage(
                prd_content=prd_content,
                code_context=code_context,
                feature_request=feature_request,
            )

            if not ollama_result.success:
                return AnalysisResult(
                    success=False,
                    error=ollama_result.error or "Ollama triage failed",
                )

            # Filter code context to relevant files
            filtered_context = self._filter_code_context(
                code_context, ollama_result.selected_files or []
            )

            # Get meta_feature_expansion prompt
            prompt = self.prompt_library.get_prompt("meta_feature_expansion")
            if not prompt:
                return AnalysisResult(
                    success=False,
                    error="meta_feature_expansion prompt not found",
                )

            # Build the analysis prompt
            rendered_prompt = f"""
## Feature Request
{feature_request}

## PRD Context
{prd_content}

## Relevant Files
{filtered_context}

{prompt.template}
"""

            # Run Perplexity analysis
            result = self.engine.analyze(rendered_prompt)
            return result

        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            return AnalysisResult(
                success=False,
                error=str(e),
            )

    def _extract_tasks_from_analysis(self, analysis_result: AnalysisResult) -> list[dict]:
        """Extract actionable tasks from analysis result.

        Args:
            analysis_result: Result from Perplexity analysis.

        Returns:
            List of task dictionaries.
        """
        tasks = []

        # Try to parse JSON from response
        if analysis_result.raw_response:
            json_data = extract_json_from_response(analysis_result.raw_response)
            if json_data:
                # Look for tasks in various formats
                if isinstance(json_data, dict):
                    if "tasks" in json_data:
                        tasks = json_data["tasks"]
                    elif "implementation_tasks" in json_data:
                        tasks = json_data["implementation_tasks"]
                    elif "recommendations" in json_data:
                        # Convert recommendations to tasks
                        for rec in json_data.get("recommendations", []):
                            if isinstance(rec, dict):
                                tasks.append({
                                    "title": rec.get("title", rec.get("recommendation", "")),
                                    "description": rec.get("description", rec.get("details", "")),
                                    "priority": rec.get("priority", "medium"),
                                })
                            elif isinstance(rec, str):
                                tasks.append({"title": rec, "description": rec})

        # Fallback: try to extract from summary
        if not tasks and analysis_result.summary:
            # Simple extraction from numbered lists
            import re
            lines = analysis_result.summary.split("\n")
            for line in lines:
                match = re.match(r"^\d+\.\s*(.+)$", line.strip())
                if match:
                    tasks.append({
                        "title": match.group(1),
                        "description": match.group(1),
                    })

        return tasks

    def _diagnose_and_fix(
        self,
        task: dict,
        test_result: TestResult,
        workspace: WorkspaceManager,
        claude_impl: ClaudeImplementer,
        prd_content: str,
    ) -> dict:
        """Use Perplexity to diagnose error and Claude to fix.

        Args:
            task: The task that failed.
            test_result: Test failure details.
            workspace: The workspace manager.
            claude_impl: Claude implementer instance.
            prd_content: PRD content for context.

        Returns:
            Dictionary with success status and details.
        """
        try:
            # Get fresh repomix
            repomix_result = self.repomix_runner.pack(self.config.repo_path)
            repomix_xml = repomix_result.content if repomix_result.success else ""

            # Get error_fix prompt
            error_fix_prompt = self.prompt_library.get_prompt("meta_error_fix")
            if not error_fix_prompt:
                logger.warning("meta_error_fix prompt not found, using default")
                error_fix_template = "Diagnose this error and provide a fix."
            else:
                error_fix_template = error_fix_prompt.template

            # Build diagnosis prompt
            diagnosis_prompt = f"""
{error_fix_template}

## Current Codebase
{repomix_xml[:50000] if len(repomix_xml) > 50000 else repomix_xml}

## Failing Task
{task.get('title', 'Unknown')}: {task.get('description', '')}

## Test Errors
{test_result.error_summary}
"""

            # Get diagnosis from Perplexity
            emit(EventType.PERPLEXITY_START, {"purpose": "error_diagnosis"})
            diagnosis_result = self.engine.analyze(diagnosis_prompt)
            emit(EventType.PERPLEXITY_COMPLETE, {"success": diagnosis_result.success})

            if not diagnosis_result.success:
                return {"success": False, "error": diagnosis_result.error}

            # Extract fix prompt from diagnosis
            fix_prompt = diagnosis_result.summary
            json_data = extract_json_from_response(diagnosis_result.raw_response)
            if json_data and isinstance(json_data, dict):
                fix_prompt = json_data.get("fix_prompt", fix_prompt)

            # Apply fix with Claude
            fix_result = claude_impl.apply_fix_prompt(
                fix_prompt=fix_prompt,
                workspace=workspace,
                repomix_xml=repomix_xml,
                dry_run=self.config.loop.dry_run,
            )

            return {
                "success": fix_result.success,
                "tokens_used": 0,  # Would need to track from both calls
                "error": fix_result.error,
            }

        except Exception as e:
            logger.error(f"Diagnosis and fix failed: {e}")
            return {"success": False, "error": str(e)}

    def _run_prd_evaluation(self, prd_content: str) -> Optional[dict]:
        """Run final PRD alignment evaluation.

        Args:
            prd_content: Original PRD content.

        Returns:
            Evaluation result dictionary or None.
        """
        try:
            # Get fresh repomix
            repomix_result = self.repomix_runner.pack(self.config.repo_path)
            repomix_xml = repomix_result.content if repomix_result.success else ""

            # Get evaluation prompt
            eval_prompt = self.prompt_library.get_prompt("meta_prd_evaluation")
            if not eval_prompt:
                logger.warning("meta_prd_evaluation prompt not found")
                return None

            # Build evaluation request
            evaluation_request = f"""
{eval_prompt.template}

## Product Requirements Document
{prd_content}

## Current Codebase
{repomix_xml[:100000] if len(repomix_xml) > 100000 else repomix_xml}
"""

            # Run evaluation
            result = self.engine.analyze(evaluation_request)
            if not result.success:
                return None

            # Parse response
            json_data = extract_json_from_response(result.raw_response)
            if json_data and isinstance(json_data, dict):
                return json_data

            return {"overall_assessment": result.summary, "approved": True}

        except Exception as e:
            logger.error(f"PRD evaluation failed: {e}")
            return None

    def _filter_code_context(self, code_context: str, selected_files: list[str]) -> str:
        """Filter code context to only include selected files.

        Args:
            code_context: Full code context.
            selected_files: List of file paths to include.

        Returns:
            Filtered code context.
        """
        if not selected_files:
            return code_context

        # Try to extract only the relevant files from repomix output
        try:
            files = extract_files(code_context)
            filtered_files = []
            for file_path, content in files.items():
                # Check if any selected file matches
                for selected in selected_files:
                    if selected in file_path or file_path in selected:
                        filtered_files.append(f"## {file_path}\n```\n{content}\n```")
                        break

            if filtered_files:
                return "\n\n".join(filtered_files)
        except Exception as e:
            logger.warning(f"Failed to filter code context: {e}")

        return code_context

    # =========================================================================
    # Task-Based Autonomous Loop (with custom TaskManager)
    # =========================================================================

    def run_task_loop(
        self,
        prd_path: str,
        max_iterations: Optional[int] = None,
        human_approve: bool = True,
        dry_run: bool = False,
        free_tier_limit: int = 5,
        pro_key: Optional[str] = None,
    ) -> "TaskLoopResult":
        """Run the task-based autonomous development loop.

        This method uses the custom TaskManager to:
        1. Parse PRD into structured tasks using Grok
        2. Iterate through tasks: implement -> test -> fix errors
        3. Track progress with persistence for resumability
        4. Generate final report on completion

        Args:
            prd_path: Path to the PRD file.
            max_iterations: Maximum loop iterations (None = use config).
            human_approve: Require approval before each implementation.
            dry_run: Preview without making changes.
            free_tier_limit: Max iterations for free tier (default: 5).
            pro_key: Pro license key to unlock unlimited iterations.

        Returns:
            TaskLoopResult with completion status and report.
        """
        logger.info("Starting task-based autonomous loop")
        emit(EventType.SESSION_START, {"mode": "task_loop", "prd": prd_path})

        loop_config = self.config.loop
        max_iters = max_iterations or loop_config.max_iterations

        # Freemium check using license module
        tier_info = get_tier_info(pro_key)
        is_pro = tier_info["is_pro"]
        effective_limit = max_iters if is_pro else min(max_iters, FREE_TIER_LIMIT)

        # Display tier information
        print(f"\n{'='*80}")
        print(f"META-AGENT TASK LOOP")
        print(f"{'='*80}")
        print(f"License: {tier_info['tier']}")
        print(f"Iteration Limit: {'Unlimited' if is_pro else f'{FREE_TIER_LIMIT} (upgrade for unlimited)'}")
        print(f"{'='*80}\n")

        if not is_pro and max_iters > FREE_TIER_LIMIT:
            logger.warning(
                f"Free tier limited to {FREE_TIER_LIMIT} iterations. "
                "Upgrade to Pro for unlimited iterations."
            )

        # Initialize Grok client for PRD parsing and error diagnosis
        if self.config.mock_mode:
            grok_client = MockGrokClient()
        else:
            grok_client = GrokClient(
                model=loop_config.evaluator.grok.model,
                temperature=loop_config.evaluator.grok.temperature,
                max_tokens=loop_config.evaluator.grok.max_tokens,
                timeout=loop_config.evaluator.grok.timeout,
            )

        # Initialize TaskManager with Grok as query function
        tasks_file = self.config.repo_path / "meta_agent_tasks.json"
        try:
            task_manager = TaskManager(
                prd_path=prd_path,
                tasks_file=str(tasks_file),
                query_fn=lambda prompt: grok_client.chat(
                    prompt,
                    system_prompt="You are a senior software architect. Parse PRDs into implementation tasks.",
                ),
            )
        except TaskManagerError as e:
            return TaskLoopResult(
                success=False,
                error=str(e),
                tasks_total=0,
                tasks_completed=0,
            )

        # Initialize local repo manager
        try:
            if self.config.mock_mode:
                local_manager = MockLocalRepoManager(loop_config)
            else:
                local_manager = LocalRepoManager(loop_config, repo_path=self.config.repo_path)
        except LocalRepoError as e:
            return TaskLoopResult(
                success=False,
                error=f"Git initialization failed: {e}",
                tasks_total=len(task_manager.get_tasks()),
                tasks_completed=0,
            )

        # Create branch for work
        branch_name = None
        if loop_config.create_branch and not dry_run:
            try:
                branch_name = local_manager.create_branch()
                logger.info(f"Working on branch: {branch_name}")
            except Exception as e:
                logger.warning(f"Branch creation failed: {e}")
                branch_name = local_manager.get_current_branch()

        # Initialize Claude implementer
        if self.config.mock_mode:
            claude_impl = MockClaudeImplementer(model=loop_config.claude_model)
        else:
            claude_impl = ClaudeImplementer(
                api_key=self.config.anthropic_api_key,
                model=loop_config.claude_model,
            )

        # Main loop
        iteration_log = []
        while not task_manager.is_complete():
            iteration = task_manager.increment_iteration()

            # Check iteration limit
            if iteration > effective_limit:
                logger.warning(f"Iteration limit ({effective_limit}) reached")
                if not is_pro:
                    logger.info("Upgrade to Pro for unlimited iterations")
                break

            task = task_manager.get_next_task()
            if not task:
                logger.info("No pending tasks remaining")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{effective_limit} | Task {task.id}: {task.title}")
            logger.info(f"{'='*60}\n")
            emit(EventType.TASK_START, {"task": task.title, "iteration": iteration})

            # Human approval gate with clear task guidance
            if human_approve and not dry_run:
                self._display_task_for_implementation(task)
                response = input("\nProceed with implementation? [y/n/skip]: ").strip().lower()
                if response == 'n':
                    logger.info("User cancelled loop")
                    break
                elif response == 'skip':
                    task_manager.set_task_status(task.id, "blocked", "Skipped by user")
                    continue

            # Mark task in progress
            task_manager.set_task_status(task.id, "in_progress")

            # Build implementation prompt
            impl_prompt = self._build_task_impl_prompt(task)

            # Implement with Claude
            if dry_run:
                logger.info(f"[DRY RUN] Would implement: {task.title}")
                task_manager.set_task_status(task.id, "done", "Dry run - skipped")
                iteration_log.append({
                    "iteration": iteration,
                    "task_id": task.id,
                    "task_title": task.title,
                    "status": "dry_run",
                })
                continue

            workspace = WorkspaceManager(self.config.repo_path)
            impl_result = claude_impl.apply_task_to_repo(
                task_description=impl_prompt,
                workspace=workspace,
                dry_run=False,
            )

            if not impl_result.success:
                task_manager.add_task_error(task.id, impl_result.error or "Implementation failed")
                iteration_log.append({
                    "iteration": iteration,
                    "task_id": task.id,
                    "status": "impl_failed",
                    "error": impl_result.error,
                })
                continue

            # Commit changes
            if loop_config.commit_per_task:
                commit_msg = f"[meta-agent] Task {task.id}: {task.title}"
                commit_result = local_manager.commit_changes(commit_msg)
                if commit_result.success:
                    logger.info(f"Committed: {commit_result.commit_hash}")

            # Run tests
            test_result = local_manager.run_tests()

            if test_result.success:
                task_manager.set_task_status(task.id, "done", "Tests passed")
                iteration_log.append({
                    "iteration": iteration,
                    "task_id": task.id,
                    "status": "success",
                })
                logger.info(f"Task {task.id} completed successfully!")
            else:
                # Grok diagnosis for test failures
                logger.warning(f"Tests failed for task {task.id}, getting Grok diagnosis...")
                task_manager.add_task_error(task.id, test_result.error_summary)

                diagnosis_prompt = self._build_grok_diagnosis_prompt(
                    task={"title": task.title, "description": task.description},
                    errors=test_result.error_summary,
                    code_context=self._pack_codebase()[:30000],
                )

                try:
                    diagnosis_response = grok_client.chat(
                        diagnosis_prompt,
                        system_prompt="Diagnose test failures and provide fix instructions.",
                    )
                    fix_data = extract_json_from_response(diagnosis_response)
                    if fix_data and "fix_prompt" in fix_data:
                        logger.info(f"Grok diagnosis: {fix_data.get('diagnosis', {}).get('root_cause', 'unknown')}")
                        # Store fix prompt for next iteration
                        task_manager.set_task_status(
                            task.id, "pending",
                            notes=f"Fix needed: {fix_data['fix_prompt'][:200]}..."
                        )
                    else:
                        task_manager.set_task_status(task.id, "failed", "Grok could not diagnose")
                except GrokClientError as e:
                    logger.error(f"Grok diagnosis failed: {e}")
                    task_manager.set_task_status(task.id, "failed", str(e))

                iteration_log.append({
                    "iteration": iteration,
                    "task_id": task.id,
                    "status": "tests_failed",
                    "error": test_result.error_summary[:200],
                })

        # Generate final report
        progress = task_manager.get_progress()
        report = self._generate_task_loop_report(
            task_manager=task_manager,
            iteration_log=iteration_log,
            branch_name=branch_name,
            is_pro=is_pro,
            prd_path=prd_path,
            save_to_file=not dry_run,
        )

        # Final Grok evaluation
        final_evaluation = None
        if progress['completed'] > 0 and not self.config.mock_mode:
            try:
                prd_content = Path(prd_path).read_text(encoding="utf-8")
                eval_prompt = self._build_grok_evaluation_prompt(
                    prd_content=prd_content,
                    code_context=self._pack_codebase()[:30000],
                    implementation_summary=task_manager.to_markdown(),
                )
                final_evaluation = grok_client.chat(
                    eval_prompt,
                    system_prompt="Evaluate PRD implementation completeness.",
                )
            except Exception as e:
                logger.warning(f"Final evaluation failed: {e}")

        emit(EventType.SESSION_END, {
            "success": task_manager.is_complete(),
            "progress": progress,
        })

        return TaskLoopResult(
            success=task_manager.is_complete(),
            tasks_total=progress['total'],
            tasks_completed=progress['completed'],
            iterations=progress['iterations'],
            total_fixes=progress['total_fixes'],
            branch_name=branch_name,
            report=report,
            final_evaluation=final_evaluation,
            is_pro=is_pro,
        )

    def _build_task_impl_prompt(self, task: Task) -> str:
        """Build implementation prompt for a task.

        Args:
            task: Task to implement.

        Returns:
            Prompt string for Claude.
        """
        lines = [
            f"# Task: {task.title}",
            "",
            task.description,
            "",
        ]

        if task.subtasks:
            lines.append("## Subtasks:")
            for st in task.subtasks:
                lines.append(f"- {st.title}: {st.description}")
            lines.append("")

        if task.notes:
            lines.append(f"## Notes from previous attempt:")
            lines.append(task.notes)
            lines.append("")

        if task.error_log:
            lines.append("## Previous errors to avoid:")
            for err in task.error_log[-3:]:
                lines.append(f"- {err}")

        lines.append("\nImplement this task. Make minimal, focused changes.")

        return "\n".join(lines)

    def _display_task_for_implementation(self, task: Task, grok_fix_prompt: Optional[str] = None) -> None:
        """Display task details clearly for Claude Code implementation.

        This provides clear visual guidance so anyone (human or Claude) knows
        exactly what needs to be implemented.

        Args:
            task: Task to display.
            grok_fix_prompt: Optional fix suggestion from Grok (for retries).
        """
        print("\n" + "=" * 80)
        print("IMPLEMENT THIS TASK (Claude Code)")
        print("=" * 80)
        print(f"\nTask ID: {task.id}")
        print(f"Title: {task.title}")
        print(f"Priority: {task.priority.upper()}")
        print(f"\nDescription:")
        print("-" * 40)
        print(task.description or "(No description)")
        print("-" * 40)

        if task.subtasks:
            print(f"\nSubtasks ({len(task.subtasks)}):")
            for st in task.subtasks:
                status_icon = "[x]" if st.status == "done" else "[ ]"
                print(f"  {status_icon} {st.id}. {st.title}")

        if task.notes:
            print(f"\nNotes from previous attempt:")
            print("-" * 40)
            print(task.notes)
            print("-" * 40)

        if task.error_log:
            print(f"\nPrevious errors to avoid ({len(task.error_log)}):")
            for err in task.error_log[-3:]:
                print(f"  ! {err[:100]}...")

        if grok_fix_prompt:
            print("\nGrok Fix Suggestion:")
            print("-" * 40)
            print(grok_fix_prompt)
            print("-" * 40)

        print("\n" + "=" * 80)
        print("Make your changes now. When finished, type 'y' and press Enter.")
        print("Type 'skip' to skip this task, or 'n' to stop the loop.")
        print("=" * 80)

    def _generate_task_loop_report(
        self,
        task_manager: TaskManager,
        iteration_log: list,
        branch_name: Optional[str],
        is_pro: bool,
        prd_path: Optional[str] = None,
        save_to_file: bool = True,
    ) -> str:
        """Generate final report for task loop.

        Args:
            task_manager: TaskManager with completed tasks.
            iteration_log: Log of iteration results.
            branch_name: Git branch used.
            is_pro: Whether pro license was used.
            prd_path: Path to PRD file for inclusion in report.
            save_to_file: Whether to save report to meta-agent-report.md.

        Returns:
            Markdown report string.
        """
        progress = task_manager.get_progress()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# Meta-Agent Task Loop Report",
            f"Generated: {now}",
            "",
            "## Summary",
            f"- **Tasks Completed:** {progress['completed']}/{progress['total']} ({progress['percentage']}%)",
            f"- **Iterations:** {progress['iterations']}",
            f"- **Fixes Applied:** {progress['total_fixes']}",
            f"- **Branch:** {branch_name or 'N/A'}",
            f"- **License:** {'Pro' if is_pro else 'Free'}",
            "",
        ]

        # Include PRD summary if available
        if prd_path:
            try:
                prd_content = Path(prd_path).read_text(encoding='utf-8')
                prd_preview = prd_content[:500].strip()
                if len(prd_content) > 500:
                    prd_preview += "..."
                lines.extend([
                    "## PRD Summary",
                    "```",
                    prd_preview,
                    "```",
                    "",
                ])
            except Exception:
                pass

        # Task status
        lines.append("## Task Status")
        for task in task_manager.get_tasks():
            status_icon = {"done": "[x]", "pending": "[ ]", "failed": "[!]", "blocked": "[B]"}.get(task.status, "[ ]")
            lines.append(f"- {status_icon} **{task.id}. {task.title}** - {task.status}")
            if task.notes:
                lines.append(f"  - Notes: {task.notes[:100]}...")

        # Iteration log
        if iteration_log:
            lines.extend([
                "",
                "## Iteration Log",
            ])
            for entry in iteration_log[-10:]:  # Last 10 entries
                lines.append(f"- Iteration {entry['iteration']}: Task {entry.get('task_id', 'N/A')} - {entry.get('status', 'unknown')}")

        # Git diff summary
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD~1"],
                capture_output=True,
                text=True,
                cwd=str(self.config.repo_path),
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines.extend([
                    "",
                    "## Changes Made (git diff --stat)",
                    "```",
                    result.stdout.strip()[-2000:],  # Limit to last 2000 chars
                    "```",
                ])
        except Exception:
            pass

        # Upgrade prompt for free tier
        if not is_pro:
            lines.extend([
                "",
                "---",
                "*Free tier: Limited to 5 iterations. Upgrade to Pro for unlimited iterations.*",
                "*Get your pro key at: https://yoursite.gumroad.com/l/meta-agent-pro*",
            ])

        report = "\n".join(lines)

        # Save report to file
        if save_to_file:
            report_path = self.config.repo_path / "meta-agent-report.md"
            try:
                report_path.write_text(report, encoding='utf-8')
                logger.info(f"Report saved to: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to save report: {e}")

        return report


@dataclass
class TaskLoopResult:
    """Result from the task-based autonomous loop."""

    success: bool
    tasks_total: int = 0
    tasks_completed: int = 0
    iterations: int = 0
    total_fixes: int = 0
    branch_name: Optional[str] = None
    report: str = ""
    final_evaluation: Optional[str] = None
    is_pro: bool = False
    error: Optional[str] = None
