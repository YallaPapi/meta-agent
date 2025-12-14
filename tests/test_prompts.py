"""Tests for prompt and profile loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from metaagent.prompts import Prompt, Profile, PromptLibrary


class TestPrompt:
    """Tests for Prompt class."""

    def test_render_basic(self) -> None:
        """Test basic template rendering with context appended."""
        prompt = Prompt(
            id="test",
            goal="Test goal",
            template="Analyze the codebase for issues.",
            stage="testing",
        )

        result = prompt.render(prd="My PRD", code_context="My code")

        # Template content should be present
        assert "Analyze the codebase for issues." in result
        # Context sections should be appended
        assert "## Product Requirements Document (PRD)" in result
        assert "My PRD" in result
        assert "## Codebase" in result
        assert "My code" in result

    def test_render_all_variables(self) -> None:
        """Test rendering with all context variables."""
        prompt = Prompt(
            id="test",
            goal="Test goal",
            template="Check this code.",
            stage="testing",
        )

        result = prompt.render(
            prd="prd_content",
            code_context="code_content",
            history="history_content",
            current_stage="stage",
        )

        # All context sections should be present
        assert "prd_content" in result
        assert "code_content" in result
        assert "history_content" in result
        assert "## Previous Analysis" in result

    def test_render_no_context(self) -> None:
        """Test rendering without context returns just template."""
        prompt = Prompt(
            id="test",
            goal="Test",
            template="Just the template.",
            stage="my_stage",
        )

        result = prompt.render()

        assert result == "Just the template."


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

    def test_missing_prompts_file_is_ok(self, tmp_path: Path) -> None:
        """Test that missing prompts file is okay (prompts can come from prompt_library)."""
        # Create a profiles file so it doesn't fail on that
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text("profiles:\n  test: {name: Test, stages: []}")

        library = PromptLibrary(
            prompts_path=tmp_path / "nonexistent.yaml",
            profiles_path=profiles_file,
        )

        # Should not raise - prompts are optional now
        library.load()
        assert library.list_prompts() == []

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


class TestMarkdownPromptLibrary:
    """Tests for loading prompts from markdown files."""

    def test_load_markdown_prompts(self, tmp_path: Path) -> None:
        """Test loading prompts from markdown files."""
        # Create a prompt library directory
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create a sample prompt file
        (prompt_lib / "quality_test_analysis.md").write_text(
            "# Test Quality Analysis\n\n"
            "**Objective:** Analyze test quality.\n\n"
            "**Instructions:**\n1. Check tests\n2. Report issues"
        )

        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            profiles_path=None,
        )
        library.load()

        # Should load the prompt
        prompts = library.list_prompts()
        assert len(prompts) == 1

        prompt = library.get_prompt("quality_test_analysis")
        assert prompt is not None
        assert prompt.goal == "Test Quality Analysis"
        assert prompt.category == "quality"
        assert "Analyze test quality" in prompt.template

    def test_list_prompts_by_category(self, tmp_path: Path) -> None:
        """Test organizing prompts by category."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create prompts in different categories
        (prompt_lib / "quality_error.md").write_text("# Error Analysis\nCheck errors.")
        (prompt_lib / "quality_style.md").write_text("# Style Analysis\nCheck style.")
        (prompt_lib / "architecture_layers.md").write_text("# Layer Analysis\nCheck layers.")

        library = PromptLibrary(prompt_library_path=prompt_lib)
        library.load()

        by_category = library.list_prompts_by_category()
        assert "quality" in by_category
        assert "architecture" in by_category
        assert len(by_category["quality"]) == 2
        assert len(by_category["architecture"]) == 1
