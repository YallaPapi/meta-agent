"""Shared test fixtures for meta-agent tests."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from metaagent.config import Config
from metaagent.prompts import PromptLibrary


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository structure for testing."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "config").mkdir()

    # Create a sample PRD
    prd_content = """# Sample PRD

## Overview
This is a test PRD for unit testing.

## Requirements
- Requirement 1: Do something
- Requirement 2: Do something else
"""
    (tmp_path / "docs" / "prd.md").write_text(prd_content)

    # Create sample source file
    (tmp_path / "src" / "main.py").write_text('print("Hello, World!")\n')

    return tmp_path


@pytest.fixture
def sample_prompts_yaml(tmp_path: Path) -> Path:
    """Create a sample prompts.yaml file."""
    prompts_content = """prompts:
  test_prompt:
    id: test_prompt
    goal: "Test prompt for unit testing"
    stage: testing
    template: |
      PRD: {{ prd }}
      Code: {{ code_context }}
      History: {{ history }}
      Stage: {{ current_stage }}
  meta_triage:
    id: meta_triage
    goal: "Triage codebase and select prompts"
    stage: triage
    template: |
      Analyze this codebase and select prompts to run.
      PRD: {{ prd }}
      Code: {{ code_context }}
      History: {{ history }}
  quality_error_analysis:
    id: quality_error_analysis
    goal: "Find errors and inconsistencies"
    stage: quality
    template: |
      Analyze for errors.
      PRD: {{ prd }}
      Code: {{ code_context }}
  architecture_layer_identification:
    id: architecture_layer_identification
    goal: "Identify architectural layers"
    stage: architecture
    template: |
      Identify layers.
      PRD: {{ prd }}
      Code: {{ code_context }}
  quality_code_complexity_analysis:
    id: quality_code_complexity_analysis
    goal: "Analyze code complexity"
    stage: quality
    template: |
      Analyze complexity.
      PRD: {{ prd }}
      Code: {{ code_context }}
  testing_unit_test_generation:
    id: testing_unit_test_generation
    goal: "Generate unit tests"
    stage: testing
    template: |
      Generate tests.
      PRD: {{ prd }}
      Code: {{ code_context }}
  improvement_best_practice_analysis:
    id: improvement_best_practice_analysis
    goal: "Check best practices"
    stage: improvement
    template: |
      Check best practices.
      PRD: {{ prd }}
      Code: {{ code_context }}
"""
    prompts_path = tmp_path / "prompts.yaml"
    prompts_path.write_text(prompts_content)
    return prompts_path


@pytest.fixture
def sample_profiles_yaml(tmp_path: Path) -> Path:
    """Create a sample profiles.yaml file."""
    profiles_content = """profiles:
  test_profile:
    name: "Test Profile"
    description: "A test profile"
    stages:
      - test_prompt
"""
    profiles_path = tmp_path / "profiles.yaml"
    profiles_path.write_text(profiles_content)
    return profiles_path


@pytest.fixture
def prompt_library(sample_prompts_yaml: Path, sample_profiles_yaml: Path) -> PromptLibrary:
    """Create a PromptLibrary with sample data."""
    library = PromptLibrary(sample_prompts_yaml, sample_profiles_yaml)
    library.load()
    return library


@pytest.fixture
def mock_config(temp_repo: Path, sample_prompts_yaml: Path, sample_profiles_yaml: Path) -> Config:
    """Create a mock configuration for testing."""
    # Copy config files to temp repo
    config_dir = temp_repo / "config"
    (config_dir / "prompts.yaml").write_text(sample_prompts_yaml.read_text())
    (config_dir / "profiles.yaml").write_text(sample_profiles_yaml.read_text())

    return Config(
        perplexity_api_key="test-key",
        anthropic_api_key="test-key",
        repo_path=temp_repo,
        config_dir=config_dir,
        prd_path=temp_repo / "docs" / "prd.md",
        mock_mode=True,
    )
