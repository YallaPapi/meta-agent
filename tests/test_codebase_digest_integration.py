"""Integration tests for Codebase Digest prompt execution flow."""

import json
from pathlib import Path

import pytest

from metaagent.analysis import AnalysisResult, MockAnalysisEngine
from metaagent.prompts import JSON_RESPONSE_SCHEMA, Prompt, PromptLibrary


class TestCodebaseDigestPromptLoading:
    """Test loading and parsing of Codebase Digest markdown prompts."""

    def test_markdown_prompt_has_correct_source(self, prompt_library):
        """Verify markdown prompts are marked with source='markdown'."""
        prompt = prompt_library.get_prompt('quality_error_analysis')
        assert prompt is not None
        assert prompt.source == 'markdown'

    def test_markdown_prompt_without_json_schema_detected(self, prompt_library):
        """Verify prompts without JSON schema are detected."""
        # Most Codebase Digest prompts don't have built-in JSON schema
        prompt = prompt_library.get_prompt('quality_error_analysis')
        assert prompt is not None
        # This prompt should NOT have JSON schema (it has freeform Expected Output)
        assert prompt.has_json_schema is False

    def test_triage_prompt_has_json_schema(self, prompt_library):
        """Verify meta_triage prompt is detected as having JSON schema."""
        prompt = prompt_library.get_prompt('meta_triage')
        assert prompt is not None
        # meta_triage.md includes JSON output format
        assert prompt.has_json_schema is True

    def test_prompt_category_extracted(self, prompt_library):
        """Verify category is extracted from filename prefix."""
        prompt = prompt_library.get_prompt('architecture_layer_identification')
        assert prompt is not None
        assert prompt.category == 'architecture'

        prompt = prompt_library.get_prompt('quality_error_analysis')
        assert prompt is not None
        assert prompt.category == 'quality'


class TestPromptRendering:
    """Test prompt rendering with context and JSON schema."""

    def test_context_appears_before_instructions(self, prompt_library):
        """Verify context sections appear before the prompt instructions."""
        prompt = prompt_library.get_prompt('quality_error_analysis')
        assert prompt is not None

        rendered = prompt.render(
            prd='Test PRD content',
            code_context='def foo(): pass',
            history='No previous analysis',
        )

        # Context should appear BEFORE the prompt title
        prd_pos = rendered.find('Test PRD content')
        # The prompt title from the markdown file
        title_pos = rendered.find('Error Analysis')  # From the markdown file title

        assert prd_pos < title_pos, "PRD should appear before prompt instructions"

    def test_json_schema_appended_to_markdown_prompts(self, prompt_library):
        """Verify JSON schema is appended to prompts without built-in schema."""
        prompt = prompt_library.get_prompt('quality_error_analysis')
        assert prompt is not None
        assert prompt.has_json_schema is False

        rendered = prompt.render(
            prd='Test PRD',
            code_context='def foo(): pass',
        )

        # JSON schema should be appended
        assert '"summary"' in rendered
        assert '"recommendations"' in rendered
        assert '"tasks"' in rendered
        assert 'Required Response Format' in rendered

    def test_json_schema_not_duplicated_for_triage(self, prompt_library):
        """Verify JSON schema is NOT appended to prompts with built-in schema."""
        prompt = prompt_library.get_prompt('meta_triage')
        assert prompt is not None
        assert prompt.has_json_schema is True

        rendered = prompt.render(
            prd='Test PRD',
            code_context='def foo(): pass',
        )

        # Should NOT have the appended schema section
        assert 'Required Response Format' not in rendered
        # But should still have the built-in schema from the prompt itself
        assert '"selected_prompts"' in rendered

    def test_render_order_context_instructions_schema(self, prompt_library):
        """Verify render order: context -> instructions -> schema."""
        prompt = prompt_library.get_prompt('quality_error_analysis')
        assert prompt is not None

        rendered = prompt.render(
            prd='TEST_PRD_MARKER',
            code_context='TEST_CODE_MARKER',
            history='TEST_HISTORY_MARKER',
        )

        # Find positions
        prd_pos = rendered.find('TEST_PRD_MARKER')
        code_pos = rendered.find('TEST_CODE_MARKER')
        history_pos = rendered.find('TEST_HISTORY_MARKER')
        schema_pos = rendered.find('Required Response Format')

        # All should be found
        assert prd_pos >= 0
        assert code_pos >= 0
        assert history_pos >= 0
        assert schema_pos >= 0

        # Context before schema
        assert prd_pos < schema_pos
        assert code_pos < schema_pos
        assert history_pos < schema_pos


class TestEndToEndFlow:
    """Test the full flow from prompt loading to response parsing."""

    def test_markdown_prompt_produces_parseable_structure(self, prompt_library):
        """Verify rendered prompt would produce parseable JSON output."""
        prompt = prompt_library.get_prompt('quality_error_analysis')
        assert prompt is not None

        rendered = prompt.render(
            prd='Build a CLI tool for code analysis',
            code_context='class Analyzer:\n    def analyze(self): pass',
            history='No previous analysis.',
        )

        # The rendered prompt should contain JSON schema instructions
        assert 'summary' in rendered.lower()
        assert 'recommendations' in rendered.lower()
        assert 'tasks' in rendered.lower()

        # Verify it's a substantial prompt
        assert len(rendered) > 500

    def test_mock_engine_produces_valid_tasks(self):
        """Verify mock engine produces properly structured tasks."""
        engine = MockAnalysisEngine()
        result = engine.analyze("Test prompt")

        assert result.success is True
        assert len(result.tasks) > 0

        # Verify task structure
        task = result.tasks[0]
        assert 'title' in task
        assert 'description' in task
        assert 'priority' in task
        assert 'file' in task

    def test_profile_uses_codebase_digest_prompts(self, prompt_library):
        """Verify profiles reference actual Codebase Digest prompts."""
        profile = prompt_library.get_profile('automation_agent')

        if profile is None:
            pytest.skip("automation_agent profile not found")

        prompts = prompt_library.get_prompts_for_profile('automation_agent')

        # Should have found some prompts
        assert len(prompts) > 0

        # All prompts should be valid Prompt instances
        for prompt in prompts:
            assert isinstance(prompt, Prompt)
            assert prompt.id is not None
            assert prompt.template is not None


class TestStageMapping:
    """Test conceptual stage to prompt mapping."""

    def test_get_prompts_for_stage(self, prompt_library):
        """Verify get_prompts_for_stage returns appropriate prompts."""
        arch_prompts = prompt_library.get_prompts_for_stage('architecture')

        # Should find architecture prompts
        assert len(arch_prompts) > 0

        for prompt in arch_prompts:
            assert prompt.category == 'architecture'

    def test_quality_stage_prompts(self, prompt_library):
        """Verify quality stage includes error analysis."""
        quality_prompts = prompt_library.get_prompts_for_stage('quality')

        prompt_ids = [p.id for p in quality_prompts]
        assert 'quality_error_analysis' in prompt_ids


class TestJSONResponseSchema:
    """Test the JSON response schema constant."""

    def test_schema_has_required_fields(self):
        """Verify schema mentions all required fields."""
        assert '"summary"' in JSON_RESPONSE_SCHEMA
        assert '"recommendations"' in JSON_RESPONSE_SCHEMA
        assert '"tasks"' in JSON_RESPONSE_SCHEMA
        assert '"title"' in JSON_RESPONSE_SCHEMA
        assert '"description"' in JSON_RESPONSE_SCHEMA
        assert '"priority"' in JSON_RESPONSE_SCHEMA
        assert '"file"' in JSON_RESPONSE_SCHEMA

    def test_schema_instructs_json_only(self):
        """Verify schema instructs to output JSON only."""
        assert 'Do not include any text outside the JSON' in JSON_RESPONSE_SCHEMA


@pytest.fixture
def prompt_library(tmp_path):
    """Create a prompt library with test prompts."""
    # Create a test prompt library directory
    prompt_dir = tmp_path / "prompt_library"
    prompt_dir.mkdir()

    # Create a test quality prompt (without JSON schema)
    quality_prompt = prompt_dir / "quality_error_analysis.md"
    quality_prompt.write_text("""# Error Analysis

**Objective:** Identify errors and inconsistencies in the codebase.

**Instructions:**
1. Review the code for bugs
2. Check for logic errors
3. Identify potential runtime issues

**Expected Output:**
A detailed report of found errors with severity and location.
""")

    # Create a test architecture prompt
    arch_prompt = prompt_dir / "architecture_layer_identification.md"
    arch_prompt.write_text("""# Layer Identification

**Objective:** Identify architectural layers in the codebase.

**Instructions:**
1. Identify the presentation layer
2. Identify the business logic layer
3. Identify the data access layer

**Expected Output:**
A list of identified layers with their responsibilities.
""")

    # Create a triage prompt (with JSON schema)
    triage_prompt = prompt_dir / "meta_triage.md"
    triage_prompt.write_text("""# Codebase Triage and Prompt Selection

**Objective:** Analyze the codebase and select which prompts to run.

**Expected Output:**
```json
{
  "assessment": "Brief assessment",
  "priority_issues": [],
  "selected_prompts": [],
  "reasoning": "Why these prompts",
  "done": false
}
```
""")

    # Create a profiles.yaml
    profiles_path = tmp_path / "profiles.yaml"
    profiles_path.write_text("""profiles:
  automation_agent:
    name: "Automation Agent"
    description: "Build automation tools"
    stages:
      - quality_error_analysis
      - architecture_layer_identification
""")

    return PromptLibrary(
        prompts_path=None,
        profiles_path=profiles_path,
        prompt_library_path=prompt_dir,
    )
