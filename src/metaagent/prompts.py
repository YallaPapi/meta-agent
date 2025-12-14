"""Prompt and profile loading for meta-agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Template


@dataclass
class Prompt:
    """A prompt template with metadata."""

    id: str
    goal: str
    template: str
    stage: str
    dependencies: list[str] = field(default_factory=list)
    when_to_use: Optional[str] = None

    def render(
        self,
        prd: str = "",
        code_context: str = "",
        history: str = "",
        current_stage: str = "",
    ) -> str:
        """Render the prompt template with variables.

        Args:
            prd: The PRD content.
            code_context: The packed codebase content.
            history: Previous analysis summaries.
            current_stage: Current stage name.

        Returns:
            Rendered prompt string.
        """
        template = Template(self.template)
        return template.render(
            prd=prd,
            code_context=code_context,
            history=history,
            current_stage=current_stage or self.stage,
        )


@dataclass
class Profile:
    """A profile defining which stages to run."""

    name: str
    description: str
    stages: list[str]


class PromptLibrary:
    """Manages loading and accessing prompts and profiles."""

    def __init__(self, prompts_path: Path, profiles_path: Path):
        """Initialize the prompt library.

        Args:
            prompts_path: Path to prompts.yaml file.
            profiles_path: Path to profiles.yaml file.
        """
        self.prompts_path = prompts_path
        self.profiles_path = profiles_path
        self._prompts: dict[str, Prompt] = {}
        self._profiles: dict[str, Profile] = {}
        self._loaded = False

    def load(self) -> None:
        """Load prompts and profiles from YAML files."""
        if self._loaded:
            return

        self._load_prompts()
        self._load_profiles()
        self._loaded = True

    def _load_prompts(self) -> None:
        """Load prompts from YAML file."""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")

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
            )

    def _load_profiles(self) -> None:
        """Load profiles from YAML file."""
        if not self.profiles_path.exists():
            raise FileNotFoundError(f"Profiles file not found: {self.profiles_path}")

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
