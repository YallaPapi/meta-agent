"""Orchestrator for the meta-agent refinement pipeline."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .analysis import AnalysisEngine, AnalysisResult, create_analysis_engine
from .codebase_digest import CodebaseDigestRunner, DigestResult
from .config import Config
from .plan_writer import PlanWriter, StageResult
from .prompts import PromptLibrary
from .repomix import RepomixRunner, RepomixResult

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
class IterationResult:
    """Result from a single iteration of the refinement loop."""

    iteration: int
    prompts_run: list[str]
    changes_made: bool
    committed: bool
    commit_hash: Optional[str] = None
    stage_results: list[StageResult] = field(default_factory=list)


@dataclass
class RefinementResult:
    """Result from a complete refinement run."""

    success: bool
    profile_name: str
    stages_completed: int
    stages_failed: int
    plan_path: Optional[Path] = None
    error: Optional[str] = None
    stage_results: list[StageResult] = field(default_factory=list)
    iterations: list[IterationResult] = field(default_factory=list)


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
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration settings.
            prompt_library: Optional PromptLibrary instance.
            repomix_runner: Optional RepomixRunner instance.
            digest_runner: Optional CodebaseDigestRunner instance.
            analysis_engine: Optional AnalysisEngine instance.
            plan_writer: Optional PlanWriter instance.
        """
        self.config = config

        # Initialize components with defaults if not provided
        self.prompt_library = prompt_library or PromptLibrary(
            prompts_path=config.prompts_file,
            profiles_path=config.profiles_file,
            prompt_library_path=config.prompt_library_path,
        )

        self.repomix_runner = repomix_runner or RepomixRunner(
            timeout=config.timeout,
            max_chars=config.max_tokens * 4,  # Rough char-to-token ratio
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
        )

        self.plan_writer = plan_writer or PlanWriter(
            output_dir=config.repo_path / "docs",
        )

    def refine(self, max_iterations: int = 10) -> RefinementResult:
        """Run the iterative refinement loop.

        This is the main entry point:
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

            # Step 2: Triage
            logger.info("Step 2: Running triage...")
            triage_result = self._run_triage(prd_content, code_context, history)

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

            # Step 3: Run selected prompts
            logger.info("Step 3: Running selected prompts...")
            iteration_stage_results = []

            for prompt_id in triage_result.selected_prompts:
                prompt = self.prompt_library.get_prompt(prompt_id)
                if not prompt:
                    logger.warning(f"Prompt not found: {prompt_id}")
                    continue

                logger.info(f"Running prompt: {prompt_id}")
                rendered_prompt = prompt.render(
                    prd=prd_content,
                    code_context=code_context,
                    history=history.format_for_prompt(),
                    current_stage=prompt_id,
                )

                analysis_result = self.analysis_engine.analyze(rendered_prompt)

                if analysis_result.success:
                    stages_completed += 1
                    history.add_entry(prompt_id, analysis_result.summary)

                    stage_result = StageResult(
                        stage_id=prompt_id,
                        stage_name=prompt.goal or prompt_id,
                        summary=analysis_result.summary,
                        recommendations=analysis_result.recommendations,
                        tasks=analysis_result.tasks,
                    )
                    iteration_stage_results.append(stage_result)
                    all_stage_results.append(stage_result)
                    logger.info(f"Prompt {prompt_id} completed successfully")
                else:
                    stages_failed += 1
                    logger.error(f"Prompt {prompt_id} failed: {analysis_result.error}")

            # Step 4: Implement changes with Claude Code
            logger.info("Step 4: Implementing changes with Claude Code...")
            changes_made = self._implement_with_claude(iteration_stage_results)

            # Step 5: Commit to GitHub
            commit_hash = None
            if changes_made:
                logger.info("Step 5: Committing changes to GitHub...")
                commit_hash = self._commit_changes(
                    f"Iteration {iteration}: {', '.join(triage_result.selected_prompts)}"
                )

            iterations.append(
                IterationResult(
                    iteration=iteration,
                    prompts_run=triage_result.selected_prompts,
                    changes_made=changes_made,
                    committed=commit_hash is not None,
                    commit_hash=commit_hash,
                    stage_results=iteration_stage_results,
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

        return RefinementResult(
            success=stages_failed == 0 and stages_completed > 0,
            profile_name="iterative",
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            plan_path=plan_path,
            stage_results=all_stage_results,
            iterations=iterations,
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

    def _run_triage(
        self, prd_content: str, code_context: str, history: RunHistory
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

        # Parse the triage response
        try:
            # Try to extract JSON from the response
            response_text = analysis_result.summary

            # Look for JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                return TriageResult(
                    success=True,
                    done=data.get("done", False),
                    assessment=data.get("assessment", ""),
                    priority_issues=data.get("priority_issues", []),
                    selected_prompts=data.get("selected_prompts", []),
                    reasoning=data.get("reasoning", ""),
                )
            else:
                # No JSON found, try to parse as plain text
                # If response contains "done" or similar, mark as done
                if "done" in response_text.lower() and "no further" in response_text.lower():
                    return TriageResult(
                        success=True,
                        done=True,
                        assessment=response_text,
                    )

                return TriageResult(
                    success=False,
                    error=f"Could not parse triage response: {response_text[:200]}",
                )

        except json.JSONDecodeError as e:
            return TriageResult(
                success=False,
                error=f"Failed to parse triage JSON: {e}",
            )

    def _implement_with_claude(self, stage_results: list[StageResult]) -> bool:
        """Implement changes using Claude Code.

        Args:
            stage_results: Results from the analysis stage.

        Returns:
            True if changes were made, False otherwise.
        """
        if not stage_results:
            return False

        # Build implementation prompt for Claude Code
        tasks = []
        for result in stage_results:
            for task in result.tasks:
                tasks.append(task)

        if not tasks:
            logger.info("No tasks to implement")
            return False

        # Create a prompt for Claude Code
        implementation_prompt = "Please implement the following improvements:\n\n"
        for i, task in enumerate(tasks, 1):
            if isinstance(task, dict):
                implementation_prompt += f"{i}. {task.get('description', task)}\n"
            else:
                implementation_prompt += f"{i}. {task}\n"

        # Write the prompt to a file for Claude Code to read
        prompt_file = self.config.repo_path / ".meta-agent-tasks.md"
        prompt_file.write_text(implementation_prompt, encoding="utf-8")

        logger.info(f"Implementation tasks written to: {prompt_file}")
        logger.info("Please run Claude Code to implement these changes.")

        # In a fully automated setup, we would call Claude Code here
        # For now, we just write the tasks and let the user run Claude Code
        # TODO: Integrate with Claude Code CLI when available

        return True

    def _commit_changes(self, message: str) -> Optional[str]:
        """Commit changes to git.

        Args:
            message: Commit message.

        Returns:
            Commit hash if successful, None otherwise.
        """
        try:
            # Check if there are changes to commit
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

            # Commit
            commit_message = f"meta-agent: {message}\n\n🤖 Generated with meta-agent"
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

            # Push to remote
            subprocess.run(
                ["git", "push"],
                cwd=self.config.repo_path,
                check=True,
            )
            logger.info("Pushed to remote")

            return commit_hash

        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e}")
            return None

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
