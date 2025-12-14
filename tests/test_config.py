"""Tests for configuration management."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from metaagent.config import Config


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
