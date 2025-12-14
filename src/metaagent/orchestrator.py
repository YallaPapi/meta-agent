"""Orchestrator for the meta-agent refinement pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .analysis import AnalysisEngine, AnalysisResult, create_analysis_engine
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
        analysis_engine: Optional[AnalysisEngine] = None,
        plan_writer: Optional[PlanWriter] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration settings.
            prompt_library: Optional PromptLibrary instance.
            repomix_runner: Optional RepomixRunner instance.
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

        # Run Repomix
        logger.info("Packing codebase with Repomix...")
        repomix_result = self.repomix_runner.pack(self.config.repo_path)

        if not repomix_result.success:
            logger.warning(f"Repomix failed: {repomix_result.error}")
            # Continue with empty code context - some analysis may still work
            code_context = f"[Repomix failed: {repomix_result.error}]"
        else:
            code_context = repomix_result.content
            if repomix_result.truncated:
                logger.warning(
                    f"Codebase was truncated from {repomix_result.original_size} chars"
                )

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
