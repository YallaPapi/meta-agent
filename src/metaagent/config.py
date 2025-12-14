"""Configuration management for meta-agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration settings for the meta-agent."""

    # API Keys
    perplexity_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Paths
    repo_path: Path = field(default_factory=Path.cwd)
    config_dir: Path = field(default_factory=lambda: Path.cwd() / "config")
    prd_path: Optional[Path] = None

    # LLM Settings
    timeout: int = 120
    max_tokens: int = 100000

    # Runtime Settings
    log_level: str = "INFO"
    mock_mode: bool = False

    @classmethod
    def from_env(cls, repo_path: Optional[Path] = None) -> Config:
        """Load configuration from environment variables.

        Args:
            repo_path: Optional path to the repository. Defaults to CWD.

        Returns:
            Config instance populated from environment.
        """
        load_dotenv()

        repo = Path(repo_path) if repo_path else Path.cwd()

        return cls(
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            repo_path=repo,
            config_dir=repo / "config",
            prd_path=repo / "docs" / "prd.md",
            timeout=int(os.getenv("METAAGENT_TIMEOUT", "120")),
            max_tokens=int(os.getenv("METAAGENT_MAX_TOKENS", "100000")),
            log_level=os.getenv("METAAGENT_LOG_LEVEL", "INFO"),
            mock_mode=os.getenv("METAAGENT_MOCK_MODE", "").lower() in ("true", "1", "yes"),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        if not self.mock_mode and not self.perplexity_api_key:
            errors.append("PERPLEXITY_API_KEY is required when not in mock mode")

        if not self.repo_path.exists():
            errors.append(f"Repository path does not exist: {self.repo_path}")

        if not self.config_dir.exists():
            errors.append(f"Config directory does not exist: {self.config_dir}")

        return errors

    @property
    def prompts_file(self) -> Path:
        """Path to prompts.yaml file (legacy)."""
        return self.config_dir / "prompts.yaml"

    @property
    def profiles_file(self) -> Path:
        """Path to profiles.yaml file."""
        return self.config_dir / "profiles.yaml"

    @property
    def prompt_library_path(self) -> Path:
        """Path to prompt_library directory with markdown prompts."""
        return self.config_dir / "prompt_library"
