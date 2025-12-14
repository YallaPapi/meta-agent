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

    # Claude Code Settings
    claude_code_timeout: int = 600  # 10 minutes for implementation
    claude_code_model: str = "claude-sonnet-4-20250514"
    claude_code_max_turns: int = 50
    auto_implement: bool = False  # Whether to auto-invoke Claude Code

    # Git/Commit Settings
    auto_commit: bool = True  # Whether to auto-commit after implementation
    auto_push: bool = False  # Whether to auto-push after commit (disabled by default for safety)

    # Runtime Settings
    log_level: str = "INFO"
    mock_mode: bool = False
    dry_run: bool = False  # Preview prompts and token estimates without API calls

    # Retry Settings
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0  # Base seconds for exponential backoff
    retry_backoff_max: float = 60.0  # Maximum backoff in seconds

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
            claude_code_timeout=int(os.getenv("METAAGENT_CLAUDE_TIMEOUT", "600")),
            claude_code_model=os.getenv("METAAGENT_CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            claude_code_max_turns=int(os.getenv("METAAGENT_CLAUDE_MAX_TURNS", "50")),
            auto_implement=os.getenv("METAAGENT_AUTO_IMPLEMENT", "").lower() in ("true", "1", "yes"),
            auto_commit=os.getenv("METAAGENT_AUTO_COMMIT", "true").lower() in ("true", "1", "yes"),
            auto_push=os.getenv("METAAGENT_AUTO_PUSH", "").lower() in ("true", "1", "yes"),
            log_level=os.getenv("METAAGENT_LOG_LEVEL", "INFO"),
            mock_mode=os.getenv("METAAGENT_MOCK_MODE", "").lower() in ("true", "1", "yes"),
            retry_max_attempts=int(os.getenv("METAAGENT_RETRY_MAX_ATTEMPTS", "3")),
            retry_backoff_base=float(os.getenv("METAAGENT_RETRY_BACKOFF_BASE", "2.0")),
            retry_backoff_max=float(os.getenv("METAAGENT_RETRY_BACKOFF_MAX", "60.0")),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        # API key not required in mock or dry-run mode
        if not self.mock_mode and not self.dry_run and not self.perplexity_api_key:
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
