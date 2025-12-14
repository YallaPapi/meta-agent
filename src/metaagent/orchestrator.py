"""Orchestrator for the meta-agent refinement pipeline."""

from __future__ import annotations

import logging
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
class RefinementResult:
    """Result from a complete refinement run."""

    success: bool
    profile_name: str
    stages_completed: int
    stages_failed: int
    plan_path: Optional[Path] = None
    error: Optional[str] = None
    stage_results: list[StageResult] = field(default_factory=list)


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

    def refine(self, profile_id: str) -> RefinementResult:
        """Run the refinement pipeline for a profile.

        Args:
            profile_id: ID of the profile to use.

        Returns:
            RefinementResult with the outcome.
        """
        logger.info(f"Starting refinement with profile: {profile_id}")

        # Load PRD
        prd_content = self._load_prd()
        if prd_content is None:
            return RefinementResult(
                success=False,
                profile_name=profile_id,
                stages_completed=0,
                stages_failed=0,
                error=f"PRD file not found: {self.config.prd_path}",
            )

        # Get profile
        profile = self.prompt_library.get_profile(profile_id)
        if not profile:
            return RefinementResult(
                success=False,
                profile_name=profile_id,
                stages_completed=0,
                stages_failed=0,
                error=f"Profile not found: {profile_id}",
            )

        # Run codebase-digest for directory tree and metrics
        logger.info("Analyzing codebase structure with codebase-digest...")
        digest_result = self.digest_runner.analyze(self.config.repo_path)

        if digest_result.success:
            logger.info("Codebase structure analysis complete")
            structure_context = self._format_digest_output(digest_result)
        else:
            logger.warning(f"codebase-digest failed: {digest_result.error}")
            structure_context = ""

        # Run Repomix for full file contents
        logger.info("Packing codebase with Repomix...")
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

        # Combine both outputs for comprehensive code context
        code_context = self._build_code_context(structure_context, file_contents)

        # Run stages
        history = RunHistory()
        stage_results: list[StageResult] = []
        stages_completed = 0
        stages_failed = 0

        prompts = self.prompt_library.get_prompts_for_profile(profile_id)
        if not prompts:
            logger.warning(f"No prompts found for profile: {profile_id}")

        for prompt in prompts:
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
                stages_completed += 1
                history.add_entry(prompt.id, analysis_result.summary)

                stage_results.append(
                    StageResult(
                        stage_id=prompt.id,
                        stage_name=prompt.goal or prompt.id,
                        summary=analysis_result.summary,
                        recommendations=analysis_result.recommendations,
                        tasks=analysis_result.tasks,
                    )
                )
                logger.info(f"Stage {prompt.id} completed successfully")
            else:
                stages_failed += 1
                logger.error(f"Stage {prompt.id} failed: {analysis_result.error}")

                # Add partial result for failed stage
                stage_results.append(
                    StageResult(
                        stage_id=prompt.id,
                        stage_name=prompt.goal or prompt.id,
                        summary=f"Stage failed: {analysis_result.error}",
                        recommendations=[],
                        tasks=[],
                    )
                )

        # Write plan
        plan_path = None
        if stage_results:
            logger.info("Writing improvement plan...")
            plan_path = self.plan_writer.write_plan(
                prd_content=prd_content,
                profile_name=profile.name,
                stage_results=stage_results,
            )
            logger.info(f"Plan written to: {plan_path}")

        return RefinementResult(
            success=stages_failed == 0 and stages_completed > 0,
            profile_name=profile.name,
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            plan_path=plan_path,
            stage_results=stage_results,
        )

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
