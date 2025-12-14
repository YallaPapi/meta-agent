"""Tests for prompt and profile loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from metaagent.prompts import Prompt, Profile, PromptLibrary


class TestPrompt:
    """Tests for Prompt class."""

    def test_render_basic(self) -> None:
        """Test basic template rendering."""
        prompt = Prompt(
            id="test",
            goal="Test goal",
            template="PRD: {{ prd }}\nCode: {{ code_context }}",
            stage="testing",
        )

        result = prompt.render(prd="My PRD", code_context="My code")

        assert "PRD: My PRD" in result
        assert "Code: My code" in result

    def test_render_all_variables(self) -> None:
        """Test rendering with all variables."""
        prompt = Prompt(
            id="test",
            goal="Test goal",
            template="{{ prd }} | {{ code_context }} | {{ history }} | {{ current_stage }}",
            stage="testing",
        )

        result = prompt.render(
            prd="prd",
            code_context="code",
            history="history",
            current_stage="stage",
        )

        assert result == "prd | code | history | stage"

    def test_render_default_stage(self) -> None:
        """Test that current_stage defaults to prompt's stage."""
        prompt = Prompt(
            id="test",
            goal="Test",
            template="Stage: {{ current_stage }}",
            stage="my_stage",
        )

        result = prompt.render()

        assert result == "Stage: my_stage"


class TestPromptLibrary:
    """Tests for PromptLibrary class."""

    def test_load_prompts(self, sample_prompts_yaml: Path, sample_profiles_yaml: Path) -> None:
        """Test loading prompts from YAML."""
        library = PromptLibrary(sample_prompts_yaml, sample_profiles_yaml)
        library.load()

        prompt = library.get_prompt("test_prompt")
        assert prompt is not None
        assert prompt.goal == "Test prompt for unit testing"

    def test_load_profiles(self, sample_prompts_yaml: Path, sample_profiles_yaml: Path) -> None:
        """Test loading profiles from YAML."""
        library = PromptLibrary(sample_prompts_yaml, sample_profiles_yaml)
        library.load()

        profile = library.get_profile("test_profile")
        assert profile is not None
        assert profile.name == "Test Profile"
        assert "test_prompt" in profile.stages

    def test_get_prompts_for_profile(
        self, sample_prompts_yaml: Path, sample_profiles_yaml: Path
    ) -> None:
        """Test getting prompts for a profile."""
        library = PromptLibrary(sample_prompts_yaml, sample_profiles_yaml)
        library.load()

        prompts = library.get_prompts_for_profile("test_profile")
        assert len(prompts) == 1
        assert prompts[0].id == "test_prompt"

    def test_list_profiles(self, sample_prompts_yaml: Path, sample_profiles_yaml: Path) -> None:
        """Test listing all profiles."""
        library = PromptLibrary(sample_prompts_yaml, sample_profiles_yaml)
        library.load()

        profiles = library.list_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "Test Profile"

    def test_missing_prompts_file(self, tmp_path: Path) -> None:
        """Test error when prompts file is missing."""
        library = PromptLibrary(
            tmp_path / "nonexistent.yaml",
            tmp_path / "profiles.yaml",
        )

        with pytest.raises(FileNotFoundError):
            library.load()

    def test_get_nonexistent_prompt(
        self, sample_prompts_yaml: Path, sample_profiles_yaml: Path
    ) -> None:
        """Test getting a prompt that doesn't exist."""
        library = PromptLibrary(sample_prompts_yaml, sample_profiles_yaml)
        library.load()

        prompt = library.get_prompt("nonexistent")
        assert prompt is None

    def test_get_nonexistent_profile(
        self, sample_prompts_yaml: Path, sample_profiles_yaml: Path
    ) -> None:
        """Test getting a profile that doesn't exist."""
        library = PromptLibrary(sample_prompts_yaml, sample_profiles_yaml)
        library.load()

        profile = library.get_profile("nonexistent")
        assert profile is None
