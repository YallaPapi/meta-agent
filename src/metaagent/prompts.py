"""Prompt and profile loading for meta-agent."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Template, UndefinedError

logger = logging.getLogger(__name__)


# JSON response schema to append to Codebase Digest prompts that don't have their own schema
JSON_RESPONSE_SCHEMA = '''

---

## Required Response Format

You MUST respond with valid JSON in exactly this structure:
```json
{
  "summary": "2-4 sentence overview of your analysis findings",
  "recommendations": ["High-level recommendation 1", "High-level recommendation 2"],
  "tasks": [
    {
      "title": "Short task title",
      "description": "Detailed description of what needs to be done",
      "priority": "critical|high|medium|low",
      "file": "path/to/relevant/file.py"
    }
  ]
}
```

Do not include any text outside the JSON block.
'''


# Default mapping from conceptual stages to recommended Codebase Digest prompts
DEFAULT_STAGE_PROMPTS = {
    'alignment': ['meta_triage'],
    'architecture': [
        'architecture_layer_identification',
        'architecture_design_pattern_identification',
        'architecture_coupling_cohesion_analysis',
    ],
    'quality': [
        'quality_error_analysis',
        'quality_code_complexity_analysis',
    ],
    'hardening': [
        'quality_error_analysis',
        'quality_risk_assessment',
    ],
    'testing': [
        'testing_unit_test_generation',
    ],
    'security': [
        'security_vulnerability_analysis',
    ],
    'performance': [
        'performance_bottleneck_identification',
        'performance_scalability_analysis',
    ],
}


@dataclass
class Prompt:
    """A prompt template with metadata."""

    id: str
    goal: str
    template: str
    stage: str
    dependencies: list[str] = field(default_factory=list)
    when_to_use: Optional[str] = None
    category: Optional[str] = None
    source: str = 'yaml'  # 'yaml' or 'markdown'
    has_json_schema: bool = True  # Whether template includes JSON schema

    def render(
        self,
        prd: str = "",
        code_context: str = "",
        history: str = "",
        current_stage: str = "",
    ) -> str:
        """Render the prompt template with variables.

        For YAML prompts (source='yaml'):
            Uses Jinja2 to interpolate {{ prd }}, {{ code_context }}, etc.
            These prompts have their own structure with placeholders.

        For Markdown prompts (source='markdown'):
            Builds the prompt in the correct order:
            1. Context sections (PRD, codebase, history)
            2. The analysis prompt/instructions
            3. JSON schema (if prompt doesn't have built-in schema)

        Args:
            prd: The PRD content.
            code_context: The packed codebase content.
            history: Previous analysis summaries.
            current_stage: Current stage name.

        Returns:
            Rendered prompt string.
        """
        logger.debug(
            f"Rendering prompt '{self.id}' (source={self.source}, "
            f"has_json_schema={self.has_json_schema})"
        )

        # YAML prompts use Jinja2 template interpolation
        if self.source == 'yaml':
            return self._render_yaml_prompt(prd, code_context, history, current_stage)

        # Markdown prompts use structured context prepending
        return self._render_markdown_prompt(prd, code_context, history, current_stage)

    def _render_yaml_prompt(
        self,
        prd: str = "",
        code_context: str = "",
        history: str = "",
        current_stage: str = "",
    ) -> str:
        """Render a YAML prompt using Jinja2 interpolation.

        Args:
            prd: The PRD content.
            code_context: The packed codebase content.
            history: Previous analysis summaries.
            current_stage: Current stage name.

        Returns:
            Rendered prompt string with variables interpolated.
        """
        try:
            template = Template(self.template)
            return template.render(
                prd=prd,
                code_context=code_context,
                history=history,
                current_stage=current_stage,
            )
        except UndefinedError as e:
            logger.warning(f"Jinja2 template error in prompt '{self.id}': {e}")
            # Fallback to raw template if rendering fails
            return self.template

    def _render_markdown_prompt(
        self,
        prd: str = "",
        code_context: str = "",
        history: str = "",
        current_stage: str = "",
    ) -> str:
        """Render a markdown prompt by prepending context sections.

        Args:
            prd: The PRD content.
            code_context: The packed codebase content.
            history: Previous analysis summaries.
            current_stage: Current stage name.

        Returns:
            Rendered prompt string with context prepended.
        """
        # 1. Build context header (context comes FIRST)
        context_sections = []
        if prd:
            context_sections.append(f"## Product Requirements Document (PRD)\n\n{prd}")
        if code_context:
            context_sections.append(f"## Codebase\n\n{code_context}")
        if history:
            context_sections.append(f"## Previous Analysis\n\n{history}")

        context_block = "\n\n---\n\n".join(context_sections) if context_sections else ""

        # 2. The analysis prompt/instructions
        prompt_block = self.template

        # 3. Add JSON schema if this prompt doesn't have one
        json_schema = ""
        if not self.has_json_schema:
            logger.debug(f"Appending JSON schema to prompt '{self.id}'")
            json_schema = JSON_RESPONSE_SCHEMA

        # Build final prompt: context -> instructions -> schema
        parts = [p for p in [context_block, prompt_block, json_schema] if p]
        return "\n\n---\n\n".join(parts)


@dataclass
class Profile:
    """A profile defining which stages to run."""

    name: str
    description: str
    stages: list[str]


@dataclass
class StageConfig:
    """Configuration for a conceptual stage's candidate prompts."""

    candidates: list[str]
    max_prompts: int = 3


class PromptLibrary:
    """Manages loading and accessing prompts and profiles."""

    def __init__(
        self,
        prompts_path: Optional[Path] = None,
        profiles_path: Optional[Path] = None,
        prompt_library_path: Optional[Path] = None,
        stage_candidates_path: Optional[Path] = None,
    ):
        """Initialize the prompt library.

        Args:
            prompts_path: Path to prompts.yaml file (legacy, optional).
            profiles_path: Path to profiles.yaml file.
            prompt_library_path: Path to directory containing markdown prompts.
            stage_candidates_path: Path to stage_candidates.yaml file.
        """
        self.prompts_path = prompts_path
        self.profiles_path = profiles_path
        self.prompt_library_path = prompt_library_path
        self.stage_candidates_path = stage_candidates_path
        self._prompts: dict[str, Prompt] = {}
        self._profiles: dict[str, Profile] = {}
        self._stage_configs: dict[str, StageConfig] = {}
        self._loaded = False

    def load(self) -> None:
        """Load prompts and profiles from files."""
        if self._loaded:
            return

        logger.info("Loading prompt library...")
        self._load_prompts()
        self._load_profiles()
        self._load_stage_candidates()
        self._loaded = True

        # Log summary of loaded content
        prompt_count = len(self._prompts)
        profile_count = len(self._profiles)
        stage_config_count = len(self._stage_configs)
        logger.info(
            f"Prompt library loaded: {prompt_count} prompts, {profile_count} profiles, "
            f"{stage_config_count} stage configs"
        )

    def _load_prompts(self) -> None:
        """Load prompts from markdown files and optional YAML."""
        # Load from markdown prompt library (primary source)
        if self.prompt_library_path and self.prompt_library_path.exists():
            logger.debug(f"Loading markdown prompts from: {self.prompt_library_path}")
            self._load_markdown_prompts()
        else:
            logger.debug(f"No prompt library found at: {self.prompt_library_path}")

        # Also load from YAML if provided (for backwards compatibility)
        if self.prompts_path and self.prompts_path.exists():
            logger.debug(f"Loading YAML prompts from: {self.prompts_path}")
            self._load_yaml_prompts()
        else:
            logger.debug(f"No YAML prompts file at: {self.prompts_path}")

    def _load_markdown_prompts(self) -> None:
        """Load prompts from markdown files in prompt_library directory."""
        if not self.prompt_library_path:
            return

        loaded_count = 0
        failed_count = 0
        for md_file in self.prompt_library_path.glob("*.md"):
            prompt = self._parse_markdown_prompt(md_file)
            if prompt:
                self._prompts[prompt.id] = prompt
                loaded_count += 1
            else:
                failed_count += 1

        logger.debug(f"Loaded {loaded_count} markdown prompts ({failed_count} failed)")

    def _parse_markdown_prompt(self, file_path: Path) -> Optional[Prompt]:
        """Parse a markdown file into a Prompt object.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Prompt object or None if parsing fails.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            prompt_id = file_path.stem  # filename without extension

            # Extract title (first heading) as the goal
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            goal = title_match.group(1).strip() if title_match else prompt_id.replace("_", " ").title()

            # Extract category from filename prefix
            category = None
            if "_" in prompt_id:
                category = prompt_id.split("_")[0]

            # Determine stage from category
            stage_mapping = {
                "architecture": "architecture",
                "quality": "quality",
                "performance": "performance",
                "security": "security",
                "testing": "testing",
                "evolution": "evolution",
                "improvement": "improvement",
                "learning": "learning",
                "business": "business",
                "meta": "triage",
            }
            stage = stage_mapping.get(category, "analysis")

            # Check if prompt already includes JSON schema instructions
            # Look for the key JSON fields that indicate structured output is expected
            has_json_schema = bool(
                re.search(r'"summary".*"recommendations".*"tasks"', content, re.DOTALL)
                or re.search(r'"assessment".*"selected_prompts".*"done"', content, re.DOTALL)
            )

            logger.debug(
                f"Loaded markdown prompt '{prompt_id}': category={category}, "
                f"has_json_schema={has_json_schema}"
            )

            return Prompt(
                id=prompt_id,
                goal=goal,
                template=content,
                stage=stage,
                category=category,
                source='markdown',
                has_json_schema=has_json_schema,
            )
        except Exception as e:
            logger.warning(f"Failed to parse markdown prompt {file_path}: {e}")
            return None

    def _load_yaml_prompts(self) -> None:
        """Load prompts from YAML file (legacy support).

        YAML prompts are assumed to have built-in JSON schema.
        """
        if not self.prompts_path:
            return

        try:
            with open(self.prompts_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            prompts_data = data.get("prompts", {})
            for prompt_id, prompt_data in prompts_data.items():
                self._prompts[prompt_id] = Prompt(
                    id=prompt_id,
                    goal=prompt_data.get("goal", ""),
                    template=prompt_data.get("template", ""),
                    stage=prompt_data.get("stage", ""),
                    dependencies=prompt_data.get("dependencies", []),
                    when_to_use=prompt_data.get("when_to_use"),
                    source='yaml',
                    has_json_schema=True,  # YAML prompts have built-in schema
                )

            logger.debug(f"Loaded {len(prompts_data)} YAML prompts")
        except Exception as e:
            logger.warning(f"Failed to load YAML prompts from {self.prompts_path}: {e}")

    def _load_profiles(self) -> None:
        """Load profiles from YAML file."""
        if not self.profiles_path or not self.profiles_path.exists():
            logger.debug(f"No profiles file at: {self.profiles_path}")
            return

        try:
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            profiles_data = data.get("profiles", {})
            for profile_id, profile_data in profiles_data.items():
                self._profiles[profile_id] = Profile(
                    name=profile_data.get("name", profile_id),
                    description=profile_data.get("description", ""),
                    stages=profile_data.get("stages", []),
                )

            logger.debug(f"Loaded {len(profiles_data)} profiles")
        except Exception as e:
            logger.warning(f"Failed to load profiles from {self.profiles_path}: {e}")

    def _load_stage_candidates(self) -> None:
        """Load stage-to-candidate-prompts mapping from YAML.

        Falls back to DEFAULT_STAGE_PROMPTS if no file is provided.
        """
        if not self.stage_candidates_path or not self.stage_candidates_path.exists():
            logger.debug(
                f"No stage candidates file at: {self.stage_candidates_path}, "
                "using DEFAULT_STAGE_PROMPTS"
            )
            # Fall back to DEFAULT_STAGE_PROMPTS
            for stage, prompts in DEFAULT_STAGE_PROMPTS.items():
                self._stage_configs[stage] = StageConfig(candidates=prompts)
            return

        try:
            with open(self.stage_candidates_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            stage_candidates = data.get("stage_candidates", {})
            for stage, config in stage_candidates.items():
                if isinstance(config, list):
                    # Simple list format
                    self._stage_configs[stage] = StageConfig(candidates=config)
                elif isinstance(config, dict):
                    # Dict format with candidates and max_prompts
                    self._stage_configs[stage] = StageConfig(
                        candidates=config.get("candidates", []),
                        max_prompts=config.get("max_prompts", 3),
                    )
                else:
                    logger.warning(f"Invalid config format for stage '{stage}'")

            logger.debug(f"Loaded {len(stage_candidates)} stage candidate configs")
        except Exception as e:
            logger.warning(
                f"Failed to load stage candidates from {self.stage_candidates_path}: {e}"
            )
            # Fall back to DEFAULT_STAGE_PROMPTS on error
            for stage, prompts in DEFAULT_STAGE_PROMPTS.items():
                self._stage_configs[stage] = StageConfig(candidates=prompts)

    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Get a prompt by ID.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            Prompt instance or None if not found.
        """
        self.load()
        return self._prompts.get(prompt_id)

    def get_profile(self, profile_id: str) -> Optional[Profile]:
        """Get a profile by ID.

        Args:
            profile_id: The profile identifier.

        Returns:
            Profile instance or None if not found.
        """
        self.load()
        return self._profiles.get(profile_id)

    def list_profiles(self) -> list[Profile]:
        """Get all available profiles.

        Returns:
            List of Profile instances.
        """
        self.load()
        return list(self._profiles.values())

    def list_prompts(self) -> list[Prompt]:
        """Get all available prompts.

        Returns:
            List of Prompt instances.
        """
        self.load()
        return list(self._prompts.values())

    def list_prompts_by_category(self) -> dict[str, list[Prompt]]:
        """Get all prompts organized by category.

        Returns:
            Dict mapping category names to lists of Prompts.
        """
        self.load()
        by_category: dict[str, list[Prompt]] = {}
        for prompt in self._prompts.values():
            cat = prompt.category or "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(prompt)
        return by_category

    def get_prompts_for_profile(self, profile_id: str) -> list[Prompt]:
        """Get all prompts for a profile's stages in order.

        Args:
            profile_id: The profile identifier.

        Returns:
            Ordered list of Prompt instances for the profile's stages.
        """
        self.load()
        profile = self._profiles.get(profile_id)
        if not profile:
            return []

        prompts = []
        for stage in profile.stages:
            prompt = self._prompts.get(stage)
            if prompt:
                prompts.append(prompt)
        return prompts

    def get_prompts_for_stage(self, stage: str) -> list[Prompt]:
        """Get recommended prompts for a conceptual stage.

        Uses DEFAULT_STAGE_PROMPTS mapping to find appropriate prompts.

        Args:
            stage: The conceptual stage name (e.g., 'architecture', 'quality').

        Returns:
            List of Prompt instances for the stage.
        """
        self.load()
        prompt_ids = DEFAULT_STAGE_PROMPTS.get(stage, [])
        return [self.get_prompt(pid) for pid in prompt_ids if self.get_prompt(pid)]

    def validate_profile(self, profile_id: str) -> dict[str, bool]:
        """Validate that all prompts referenced in a profile exist.

        Args:
            profile_id: The profile identifier.

        Returns:
            Dict mapping stage/prompt names to existence (True if exists, False if missing).
        """
        self.load()
        profile = self._profiles.get(profile_id)
        if not profile:
            return {}

        results = {}
        for stage in profile.stages:
            results[stage] = self.get_prompt(stage) is not None
        return results

    def validate_all_profiles(self) -> dict[str, dict[str, bool]]:
        """Validate all profiles and their prompts.

        Returns:
            Dict mapping profile_id to validation results (stage -> exists).
        """
        self.load()
        results = {}
        for profile_id in self._profiles:
            results[profile_id] = self.validate_profile(profile_id)
        return results

    def get_stage_config(self, stage: str) -> Optional[StageConfig]:
        """Get configuration for a conceptual stage.

        Args:
            stage: The stage name (e.g., 'architecture', 'quality').

        Returns:
            StageConfig instance or None if not found.
        """
        self.load()
        return self._stage_configs.get(stage)

    def get_all_candidate_prompts_for_stage(self, stage: str) -> list[Prompt]:
        """Get all candidate prompts for a stage (for triage to select from).

        Args:
            stage: The conceptual stage name (e.g., 'architecture', 'quality').

        Returns:
            List of Prompt instances that are candidates for this stage.
        """
        self.load()
        config = self._stage_configs.get(stage)
        if not config:
            return []
        return [
            self.get_prompt(pid)
            for pid in config.candidates
            if self.get_prompt(pid)
        ]

    def list_stages(self) -> list[str]:
        """Get all available stage names.

        Returns:
            List of stage names from stage_candidates config.
        """
        self.load()
        return list(self._stage_configs.keys())
