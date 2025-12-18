"""Configuration management for meta-agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv


# Type alias for evaluator names
EvaluatorName = Literal["grok", "perplexity"]


@dataclass
class GrokSettings:
    """Settings for Grok API evaluator."""

    model: str = "grok-3-latest"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120

    @classmethod
    def from_dict(cls, data: dict) -> GrokSettings:
        """Create GrokSettings from dictionary."""
        return cls(
            model=data.get("model", "grok-3-latest"),
            temperature=float(data.get("temperature", 0.3)),
            max_tokens=int(data.get("max_tokens", 4096)),
            timeout=int(data.get("timeout", 120)),
        )


@dataclass
class PerplexitySettings:
    """Settings for Perplexity API evaluator."""

    model: str = "llama-3.1-sonar-large-128k-online"

    @classmethod
    def from_dict(cls, data: dict) -> PerplexitySettings:
        """Create PerplexitySettings from dictionary."""
        return cls(
            model=data.get("model", "llama-3.1-sonar-large-128k-online"),
        )


@dataclass
class EvaluatorConfig:
    """Configuration for evaluator selection and settings."""

    default: EvaluatorName = "grok"
    grok: GrokSettings = field(default_factory=GrokSettings)
    perplexity: PerplexitySettings = field(default_factory=PerplexitySettings)

    @classmethod
    def from_dict(cls, data: dict) -> EvaluatorConfig:
        """Create EvaluatorConfig from dictionary."""
        evaluator_data = data.get("evaluator", {})
        return cls(
            default=evaluator_data.get("default", "grok"),
            grok=GrokSettings.from_dict(evaluator_data.get("grok", {})),
            perplexity=PerplexitySettings.from_dict(evaluator_data.get("perplexity", {})),
        )


def get_evaluator_name(config: EvaluatorConfig, override: Optional[str]) -> EvaluatorName:
    """Get the evaluator name, respecting CLI override.

    Args:
        config: The evaluator configuration.
        override: Optional CLI override for evaluator name.

    Returns:
        The evaluator name to use ("grok" or "perplexity").
    """
    if override in ("grok", "perplexity"):
        return override  # type: ignore
    return config.default


@dataclass
class LoopConfig:
    """Configuration for autonomous development loop mode."""

    enabled: bool = False
    max_iterations: int = 15
    human_approve: bool = True
    dry_run: bool = False
    test_command: str = "pytest -q"
    claude_model: str = "claude-sonnet-4-20250514"
    commit_per_task: bool = True
    branch_pattern: str = "meta-agent-loop/{timestamp}"
    branch_prefix: str = "meta-loop"
    create_branch: bool = True
    token_budget_per_iteration: int = 50000
    max_consecutive_failures: int = 3
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    @classmethod
    def from_dict(cls, data: dict) -> LoopConfig:
        """Create LoopConfig from dictionary."""
        loop_data = data.get("loop", {})
        return cls(
            enabled=loop_data.get("enabled", False),
            max_iterations=loop_data.get("max_iterations", 15),
            human_approve=loop_data.get("human_approve", True),
            dry_run=loop_data.get("dry_run", False),
            test_command=loop_data.get("test_command", "pytest -q"),
            claude_model=loop_data.get("claude_model", "claude-sonnet-4-20250514"),
            commit_per_task=loop_data.get("commit_per_task", True),
            branch_pattern=loop_data.get("branch_pattern", "meta-agent-loop/{timestamp}"),
            branch_prefix=loop_data.get("branch_prefix", "meta-loop"),
            create_branch=loop_data.get("create_branch", True),
            token_budget_per_iteration=loop_data.get("token_budget_per_iteration", 50000),
            max_consecutive_failures=loop_data.get("max_consecutive_failures", 3),
            evaluator=EvaluatorConfig.from_dict(data),
        )

    @classmethod
    def load_from_file(cls, config_dir: Path) -> LoopConfig:
        """Load loop config from YAML file."""
        loop_config_path = config_dir / "loop_config.yaml"
        if loop_config_path.exists():
            with open(loop_config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        return cls()  # Return defaults if file doesn't exist


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

    # Loop Settings (loaded from loop_config.yaml)
    loop: LoopConfig = field(default_factory=LoopConfig)

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
        config_dir = repo / "config"

        # Load loop config from file
        loop_config = LoopConfig.load_from_file(config_dir)

        return cls(
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            repo_path=repo,
            config_dir=config_dir,
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
            loop=loop_config,
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

    @property
    def stage_candidates_file(self) -> Path:
        """Path to stage_candidates.yaml file."""
        return self.config_dir / "stage_candidates.yaml"

    @property
    def loop_config_file(self) -> Path:
        """Path to loop_config.yaml file."""
        return self.config_dir / "loop_config.yaml"
