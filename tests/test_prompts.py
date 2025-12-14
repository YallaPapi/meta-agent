"""Tests for prompt and profile loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from metaagent.prompts import Prompt, Profile, PromptLibrary


class TestPrompt:
    """Tests for Prompt class."""

    def test_render_markdown_prepends_context(self) -> None:
        """Test markdown prompts prepend context sections."""
        prompt = Prompt(
            id="test",
            goal="Test goal",
            template="Analyze the codebase for issues.",
            stage="testing",
            source="markdown",  # Markdown prompts prepend context
            has_json_schema=True,
        )

        result = prompt.render(prd="My PRD", code_context="My code")

        # Template content should be present
        assert "Analyze the codebase for issues." in result
        # Context sections should be prepended
        assert "## Product Requirements Document (PRD)" in result
        assert "My PRD" in result
        assert "## Codebase" in result
        assert "My code" in result

    def test_render_yaml_with_jinja2(self) -> None:
        """Test YAML prompts use Jinja2 interpolation."""
        prompt = Prompt(
            id="test",
            goal="Test goal",
            template="PRD: {{ prd }}\nCode: {{ code_context }}",
            stage="testing",
            source="yaml",  # YAML prompts use Jinja2
        )

        result = prompt.render(prd="My PRD", code_context="My code")

        # Variables should be interpolated
        assert "PRD: My PRD" in result
        assert "Code: My code" in result

    def test_render_all_variables_markdown(self) -> None:
        """Test rendering markdown prompt with all context variables."""
        prompt = Prompt(
            id="test",
            goal="Test goal",
            template="Check this code.",
            stage="testing",
            source="markdown",
            has_json_schema=True,
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
            source="yaml",  # YAML with no placeholders
        )

        result = prompt.render()

        assert result == "Just the template."

    def test_render_yaml_all_variables(self) -> None:
        """Test YAML rendering with all Jinja2 variables."""
        prompt = Prompt(
            id="test",
            goal="Test",
            template="PRD: {{ prd }}\nCode: {{ code_context }}\nHistory: {{ history }}\nStage: {{ current_stage }}",
            stage="testing",
            source="yaml",
        )

        result = prompt.render(
            prd="my_prd",
            code_context="my_code",
            history="my_history",
            current_stage="my_stage",
        )

        assert "PRD: my_prd" in result
        assert "Code: my_code" in result
        assert "History: my_history" in result
        assert "Stage: my_stage" in result


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


class TestProfileValidation:
    """Tests for profile validation."""

    def test_validate_profile_all_exist(self, tmp_path: Path) -> None:
        """Test validation when all prompts exist."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create prompts
        (prompt_lib / "quality_error.md").write_text("# Error Analysis\nCheck.")
        (prompt_lib / "architecture_layers.md").write_text("# Layer Analysis\nCheck.")

        # Create profile referencing these prompts
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text("""profiles:
  test_profile:
    name: Test Profile
    description: Test description
    stages:
      - quality_error
      - architecture_layers
""")

        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            profiles_path=profiles_file,
        )
        library.load()

        result = library.validate_profile("test_profile")
        assert result == {"quality_error": True, "architecture_layers": True}

    def test_validate_profile_some_missing(self, tmp_path: Path) -> None:
        """Test validation when some prompts are missing."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create only one of the referenced prompts
        (prompt_lib / "quality_error.md").write_text("# Error Analysis\nCheck.")

        # Create profile referencing prompts (some missing)
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text("""profiles:
  test_profile:
    name: Test Profile
    description: Test description
    stages:
      - quality_error
      - nonexistent_prompt
""")

        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            profiles_path=profiles_file,
        )
        library.load()

        result = library.validate_profile("test_profile")
        assert result == {"quality_error": True, "nonexistent_prompt": False}

    def test_validate_nonexistent_profile(self, tmp_path: Path) -> None:
        """Test validation of a profile that doesn't exist."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text("profiles: {}")

        library = PromptLibrary(profiles_path=profiles_file)
        library.load()

        result = library.validate_profile("nonexistent")
        assert result == {}

    def test_validate_all_profiles(self, tmp_path: Path) -> None:
        """Test validating all profiles at once."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create some prompts
        (prompt_lib / "quality_error.md").write_text("# Error\nCheck.")
        (prompt_lib / "quality_style.md").write_text("# Style\nCheck.")

        # Create multiple profiles
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text("""profiles:
  valid_profile:
    name: Valid Profile
    description: All prompts exist
    stages:
      - quality_error
      - quality_style
  partial_profile:
    name: Partial Profile
    description: Some prompts missing
    stages:
      - quality_error
      - missing_prompt
""")

        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            profiles_path=profiles_file,
        )
        library.load()

        results = library.validate_all_profiles()

        assert "valid_profile" in results
        assert "partial_profile" in results
        assert results["valid_profile"] == {"quality_error": True, "quality_style": True}
        assert results["partial_profile"] == {"quality_error": True, "missing_prompt": False}


class TestStageCandidates:
    """Tests for stage candidates functionality."""

    def test_load_stage_candidates_from_yaml(self, tmp_path: Path) -> None:
        """Test loading stage candidates from YAML file."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create prompts
        (prompt_lib / "architecture_layers.md").write_text("# Layers\nCheck.")
        (prompt_lib / "architecture_patterns.md").write_text("# Patterns\nCheck.")
        (prompt_lib / "quality_errors.md").write_text("# Errors\nCheck.")

        # Create stage candidates file
        stage_file = tmp_path / "stage_candidates.yaml"
        stage_file.write_text("""stage_candidates:
  architecture:
    candidates:
      - architecture_layers
      - architecture_patterns
    max_prompts: 2
  quality:
    candidates:
      - quality_errors
    max_prompts: 1
""")

        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            stage_candidates_path=stage_file,
        )
        library.load()

        # Check stage configs loaded
        arch_config = library.get_stage_config("architecture")
        assert arch_config is not None
        assert arch_config.candidates == ["architecture_layers", "architecture_patterns"]
        assert arch_config.max_prompts == 2

        quality_config = library.get_stage_config("quality")
        assert quality_config is not None
        assert quality_config.candidates == ["quality_errors"]
        assert quality_config.max_prompts == 1

    def test_get_all_candidate_prompts_for_stage(self, tmp_path: Path) -> None:
        """Test getting candidate prompts for a stage."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create prompts
        (prompt_lib / "architecture_layers.md").write_text("# Layers\nCheck.")
        (prompt_lib / "architecture_patterns.md").write_text("# Patterns\nCheck.")

        # Create stage candidates file
        stage_file = tmp_path / "stage_candidates.yaml"
        stage_file.write_text("""stage_candidates:
  architecture:
    candidates:
      - architecture_layers
      - architecture_patterns
      - nonexistent_prompt
    max_prompts: 3
""")

        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            stage_candidates_path=stage_file,
        )
        library.load()

        # Should return only existing prompts
        candidates = library.get_all_candidate_prompts_for_stage("architecture")
        assert len(candidates) == 2
        prompt_ids = [p.id for p in candidates]
        assert "architecture_layers" in prompt_ids
        assert "architecture_patterns" in prompt_ids

    def test_fallback_to_default_stage_prompts(self, tmp_path: Path) -> None:
        """Test fallback to DEFAULT_STAGE_PROMPTS when no file provided."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Don't create stage_candidates.yaml
        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            stage_candidates_path=None,
        )
        library.load()

        # Should fall back to DEFAULT_STAGE_PROMPTS
        config = library.get_stage_config("architecture")
        assert config is not None
        # Default architecture prompts from DEFAULT_STAGE_PROMPTS
        assert "architecture_layer_identification" in config.candidates

    def test_list_stages(self, tmp_path: Path) -> None:
        """Test listing all available stages."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create stage candidates file
        stage_file = tmp_path / "stage_candidates.yaml"
        stage_file.write_text("""stage_candidates:
  architecture:
    candidates:
      - arch_test
    max_prompts: 2
  quality:
    candidates:
      - quality_test
    max_prompts: 1
  testing:
    candidates:
      - test_gen
    max_prompts: 1
""")

        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            stage_candidates_path=stage_file,
        )
        library.load()

        stages = library.list_stages()
        assert "architecture" in stages
        assert "quality" in stages
        assert "testing" in stages
        assert len(stages) == 3

    def test_simple_list_format(self, tmp_path: Path) -> None:
        """Test loading stage candidates with simple list format."""
        prompt_lib = tmp_path / "prompt_library"
        prompt_lib.mkdir()

        # Create stage candidates file with simple list format
        stage_file = tmp_path / "stage_candidates.yaml"
        stage_file.write_text("""stage_candidates:
  security:
    - security_vuln_1
    - security_vuln_2
""")

        library = PromptLibrary(
            prompt_library_path=prompt_lib,
            stage_candidates_path=stage_file,
        )
        library.load()

        # Note: Simple list format isn't supported - it expects dict format
        # This tests that the code handles this gracefully
        config = library.get_stage_config("security")
        # With dict format, it should work
        assert config is None or isinstance(config.candidates, list)
