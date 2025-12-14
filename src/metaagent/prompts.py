"""Prompt and profile loading for meta-agent."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

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

        Builds the prompt in the correct order:
        1. Context sections (PRD, codebase, history) - so LLM knows what it's analyzing
        2. The analysis prompt/instructions
        3. JSON schema (for markdown prompts without built-in schema)

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


class PromptLibrary:
    """Manages loading and accessing prompts and profiles."""

    def __init__(
        self,
        prompts_path: Optional[Path] = None,
        profiles_path: Optional[Path] = None,
        prompt_library_path: Optional[Path] = None,
    ):
        """Initialize the prompt library.

        Args:
            prompts_path: Path to prompts.yaml file (legacy, optional).
            profiles_path: Path to profiles.yaml file.
            prompt_library_path: Path to directory containing markdown prompts.
        """
        self.prompts_path = prompts_path
        self.profiles_path = profiles_path
        self.prompt_library_path = prompt_library_path
        self._prompts: dict[str, Prompt] = {}
        self._profiles: dict[str, Profile] = {}
        self._loaded = False

    def load(self) -> None:
        """Load prompts and profiles from files."""
        if self._loaded:
            return

        self._load_prompts()
        self._load_profiles()
        self._loaded = True

    def _load_prompts(self) -> None:
        """Load prompts from markdown files and optional YAML."""
        # Load from markdown prompt library (primary source)
        if self.prompt_library_path and self.prompt_library_path.exists():
            self._load_markdown_prompts()

        # Also load from YAML if provided (for backwards compatibility)
        if self.prompts_path and self.prompts_path.exists():
            self._load_yaml_prompts()

    def _load_markdown_prompts(self) -> None:
        """Load prompts from markdown files in prompt_library directory."""
        if not self.prompt_library_path:
            return

        for md_file in self.prompt_library_path.glob("*.md"):
            prompt = self._parse_markdown_prompt(md_file)
            if prompt:
                self._prompts[prompt.id] = prompt

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

    def _load_profiles(self) -> None:
        """Load profiles from YAML file."""
        if not self.profiles_path or not self.profiles_path.exists():
            return  # No profiles file, just skip

        with open(self.profiles_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        profiles_data = data.get("profiles", {})
        for profile_id, profile_data in profiles_data.items():
            self._profiles[profile_id] = Profile(
                name=profile_data.get("name", profile_id),
                description=profile_data.get("description", ""),
                stages=profile_data.get("stages", []),
            )

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
