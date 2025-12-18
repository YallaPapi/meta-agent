"""Tests for configuration management."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from metaagent.config import (
    Config,
    EvaluatorConfig,
    GrokSettings,
    LoopConfig,
    PerplexitySettings,
    get_evaluator_name,
)


class TestConfig:
    """Tests for Config class."""

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test Config.from_env with default values."""
        # Clear environment - including any loaded from .env
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("METAAGENT_TIMEOUT", raising=False)
        monkeypatch.delenv("METAAGENT_LOG_LEVEL", raising=False)
        monkeypatch.delenv("METAAGENT_MOCK_MODE", raising=False)

        # Also patch load_dotenv to prevent loading from .env file
        monkeypatch.setattr("metaagent.config.load_dotenv", lambda: None)

        config = Config.from_env(tmp_path)

        assert config.perplexity_api_key is None
        assert config.anthropic_api_key is None
        assert config.timeout == 120
        assert config.log_level == "INFO"
        assert config.mock_mode is False

    def test_from_env_with_values(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test Config.from_env with environment values."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-perplexity-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("METAAGENT_TIMEOUT", "60")
        monkeypatch.setenv("METAAGENT_MOCK_MODE", "true")

        config = Config.from_env(tmp_path)

        assert config.perplexity_api_key == "test-perplexity-key"
        assert config.anthropic_api_key == "test-anthropic-key"
        assert config.timeout == 60
        assert config.mock_mode is True

    def test_validate_mock_mode(self, tmp_path: Path) -> None:
        """Test validation passes in mock mode without API key."""
        (tmp_path / "config").mkdir()

        config = Config(
            repo_path=tmp_path,
            config_dir=tmp_path / "config",
            mock_mode=True,
        )

        errors = config.validate()
        assert not errors

    def test_validate_requires_api_key(self, tmp_path: Path) -> None:
        """Test validation fails without API key in non-mock mode."""
        (tmp_path / "config").mkdir()

        config = Config(
            repo_path=tmp_path,
            config_dir=tmp_path / "config",
            mock_mode=False,
        )

        errors = config.validate()
        assert any("PERPLEXITY_API_KEY" in e for e in errors)

    def test_validate_repo_path_exists(self) -> None:
        """Test validation fails for non-existent repo path."""
        config = Config(
            repo_path=Path("/nonexistent/path"),
            mock_mode=True,
        )

        errors = config.validate()
        assert any("Repository path does not exist" in e for e in errors)

    def test_prompts_file_property(self, tmp_path: Path) -> None:
        """Test prompts_file property returns correct path."""
        config = Config(
            config_dir=tmp_path / "config",
        )

        assert config.prompts_file == tmp_path / "config" / "prompts.yaml"

    def test_profiles_file_property(self, tmp_path: Path) -> None:
        """Test profiles_file property returns correct path."""
        config = Config(
            config_dir=tmp_path / "config",
        )

        assert config.profiles_file == tmp_path / "config" / "profiles.yaml"

    def test_from_env_retry_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test default retry configuration values."""
        # Prevent .env loading and clear env vars
        monkeypatch.setattr("metaagent.config.load_dotenv", lambda: None)
        monkeypatch.delenv("METAAGENT_RETRY_MAX_ATTEMPTS", raising=False)
        monkeypatch.delenv("METAAGENT_RETRY_BACKOFF_BASE", raising=False)
        monkeypatch.delenv("METAAGENT_RETRY_BACKOFF_MAX", raising=False)

        config = Config.from_env(tmp_path)

        assert config.retry_max_attempts == 3
        assert config.retry_backoff_base == 2.0
        assert config.retry_backoff_max == 60.0

    def test_from_env_retry_custom(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test custom retry configuration from environment."""
        monkeypatch.setenv("METAAGENT_RETRY_MAX_ATTEMPTS", "5")
        monkeypatch.setenv("METAAGENT_RETRY_BACKOFF_BASE", "1.5")
        monkeypatch.setenv("METAAGENT_RETRY_BACKOFF_MAX", "120.0")

        config = Config.from_env(tmp_path)

        assert config.retry_max_attempts == 5
        assert config.retry_backoff_base == 1.5
        assert config.retry_backoff_max == 120.0


class TestGrokSettings:
    """Tests for GrokSettings class."""

    def test_defaults(self) -> None:
        """Test default Grok settings."""
        settings = GrokSettings()
        assert settings.model == "grok-3-latest"
        assert settings.temperature == 0.3
        assert settings.max_tokens == 4096
        assert settings.timeout == 120

    def test_from_dict_with_values(self) -> None:
        """Test creating GrokSettings from dict."""
        data = {
            "model": "grok-2",
            "temperature": 0.5,
            "max_tokens": 8192,
            "timeout": 60,
        }
        settings = GrokSettings.from_dict(data)
        assert settings.model == "grok-2"
        assert settings.temperature == 0.5
        assert settings.max_tokens == 8192
        assert settings.timeout == 60

    def test_from_dict_with_defaults(self) -> None:
        """Test creating GrokSettings from empty dict uses defaults."""
        settings = GrokSettings.from_dict({})
        assert settings.model == "grok-3-latest"
        assert settings.temperature == 0.3


class TestPerplexitySettings:
    """Tests for PerplexitySettings class."""

    def test_defaults(self) -> None:
        """Test default Perplexity settings."""
        settings = PerplexitySettings()
        assert settings.model == "llama-3.1-sonar-large-128k-online"

    def test_from_dict(self) -> None:
        """Test creating PerplexitySettings from dict."""
        data = {"model": "custom-model"}
        settings = PerplexitySettings.from_dict(data)
        assert settings.model == "custom-model"


class TestEvaluatorConfig:
    """Tests for EvaluatorConfig class."""

    def test_defaults(self) -> None:
        """Test default evaluator config."""
        config = EvaluatorConfig()
        assert config.default == "grok"
        assert config.grok.model == "grok-3-latest"
        assert config.perplexity.model == "llama-3.1-sonar-large-128k-online"

    def test_from_dict(self) -> None:
        """Test creating EvaluatorConfig from dict."""
        data = {
            "evaluator": {
                "default": "perplexity",
                "grok": {"model": "grok-2", "temperature": 0.7},
                "perplexity": {"model": "custom-sonar"},
            }
        }
        config = EvaluatorConfig.from_dict(data)
        assert config.default == "perplexity"
        assert config.grok.model == "grok-2"
        assert config.grok.temperature == 0.7
        assert config.perplexity.model == "custom-sonar"


class TestGetEvaluatorName:
    """Tests for get_evaluator_name helper."""

    def test_returns_override_grok(self) -> None:
        """Test override for grok."""
        config = EvaluatorConfig(default="perplexity")
        result = get_evaluator_name(config, "grok")
        assert result == "grok"

    def test_returns_override_perplexity(self) -> None:
        """Test override for perplexity."""
        config = EvaluatorConfig(default="grok")
        result = get_evaluator_name(config, "perplexity")
        assert result == "perplexity"

    def test_returns_default_when_no_override(self) -> None:
        """Test returns default when no override provided."""
        config = EvaluatorConfig(default="grok")
        result = get_evaluator_name(config, None)
        assert result == "grok"

    def test_returns_default_when_invalid_override(self) -> None:
        """Test returns default when invalid override provided."""
        config = EvaluatorConfig(default="grok")
        result = get_evaluator_name(config, "invalid")
        assert result == "grok"


class TestLoopConfig:
    """Tests for LoopConfig class."""

    def test_defaults(self) -> None:
        """Test default loop config."""
        config = LoopConfig()
        assert config.enabled is False
        assert config.max_iterations == 15
        assert config.human_approve is True
        assert config.test_command == "pytest -q"
        assert config.branch_prefix == "meta-loop"
        assert config.evaluator.default == "grok"

    def test_from_dict(self) -> None:
        """Test creating LoopConfig from dict."""
        data = {
            "loop": {
                "enabled": True,
                "max_iterations": 20,
                "human_approve": False,
                "test_command": "npm test",
                "branch_prefix": "custom-prefix",
            },
            "evaluator": {"default": "perplexity"},
        }
        config = LoopConfig.from_dict(data)
        assert config.enabled is True
        assert config.max_iterations == 20
        assert config.human_approve is False
        assert config.test_command == "npm test"
        assert config.branch_prefix == "custom-prefix"
        assert config.evaluator.default == "perplexity"

    def test_load_from_file_not_exists(self, tmp_path: Path) -> None:
        """Test loading from non-existent file returns defaults."""
        config = LoopConfig.load_from_file(tmp_path)
        assert config.enabled is False
        assert config.max_iterations == 15

    def test_load_from_file_exists(self, tmp_path: Path) -> None:
        """Test loading from existing file."""
        config_content = """
loop:
  enabled: true
  max_iterations: 25
  test_command: "pytest -v"
evaluator:
  default: "perplexity"
"""
        config_file = tmp_path / "loop_config.yaml"
        config_file.write_text(config_content)

        config = LoopConfig.load_from_file(tmp_path)
        assert config.enabled is True
        assert config.max_iterations == 25
        assert config.test_command == "pytest -v"
        assert config.evaluator.default == "perplexity"
