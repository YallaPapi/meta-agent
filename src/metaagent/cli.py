"""CLI entrypoint for meta-agent."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .config import Config
from .orchestrator import Orchestrator
from .prompts import PromptLibrary

# Initialize Typer app
app = typer.Typer(
    name="metaagent",
    help="Meta-agent for automated codebase refinement from v0 to MVP.",
    add_completion=False,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"metaagent version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Meta-agent for automated codebase refinement."""
    pass


@app.command()
def refine(
    profile: str = typer.Option(
        ...,
        "--profile",
        "-p",
        help="Profile to use for refinement (e.g., 'automation_agent').",
    ),
    repo: Path = typer.Option(
        Path.cwd(),
        "--repo",
        "-r",
        help="Path to the repository to refine.",
    ),
    prd: Optional[Path] = typer.Option(
        None,
        "--prd",
        help="Path to PRD file (default: docs/prd.md in repo).",
    ),
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config-dir",
        "-c",
        help="Path to config directory (default: config/ in repo).",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        "-m",
        help="Run in mock mode (no API calls).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Run refinement analysis on a repository.

    This command analyzes your codebase using the specified profile and
    generates an improvement plan at docs/mvp_improvement_plan.md.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Resolve paths
    repo_path = repo.resolve()
    if not repo_path.exists():
        console.print(f"[red]Error:[/red] Repository path does not exist: {repo_path}")
        raise typer.Exit(1)

    # Load configuration
    config = Config.from_env(repo_path)

    # Override with CLI options
    if prd:
        config.prd_path = prd.resolve()
    if config_dir:
        config.config_dir = config_dir.resolve()
    if mock:
        config.mock_mode = True

    # Validate configuration
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    # Check if profile exists
    try:
        prompt_library = PromptLibrary(config.prompts_file, config.profiles_file)
        prompt_library.load()

        if not prompt_library.get_profile(profile):
            available = [p.name for p in prompt_library.list_profiles()]
            console.print(f"[red]Error:[/red] Profile '{profile}' not found.")
            if available:
                console.print(f"Available profiles: {', '.join(available)}")
            raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Run refinement
    console.print(f"\n[bold]Starting refinement with profile:[/bold] {profile}")
    console.print(f"[dim]Repository:[/dim] {repo_path}")
    console.print(f"[dim]Mock mode:[/dim] {'enabled' if config.mock_mode else 'disabled'}\n")

    orchestrator = Orchestrator(config, prompt_library=prompt_library)
    result = orchestrator.refine(profile)

    # Display results
    if result.success:
        console.print("\n[green]Refinement completed successfully![/green]\n")
    else:
        console.print("\n[yellow]Refinement completed with issues.[/yellow]\n")

    console.print(f"Stages completed: {result.stages_completed}")
    console.print(f"Stages failed: {result.stages_failed}")

    if result.plan_path:
        console.print(f"\n[bold]Improvement plan written to:[/bold] {result.plan_path}")
        console.print("\nNext steps:")
        console.print("  1. Review the improvement plan")
        console.print("  2. Open Claude Code in your repository")
        console.print("  3. Ask Claude Code to implement the plan")

    if result.error:
        console.print(f"\n[red]Error:[/red] {result.error}")
        raise typer.Exit(1)


@app.command("list-profiles")
def list_profiles(
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config-dir",
        "-c",
        help="Path to config directory.",
    ),
) -> None:
    """List available refinement profiles."""
    # Determine config directory
    cfg_dir = config_dir.resolve() if config_dir else Path.cwd() / "config"

    if not cfg_dir.exists():
        console.print(f"[red]Error:[/red] Config directory not found: {cfg_dir}")
        raise typer.Exit(1)

    try:
        prompt_library = PromptLibrary(
            prompts_path=cfg_dir / "prompts.yaml",
            profiles_path=cfg_dir / "profiles.yaml",
        )
        prompt_library.load()
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    profiles = prompt_library.list_profiles()

    if not profiles:
        console.print("[yellow]No profiles found.[/yellow]")
        return

    table = Table(title="Available Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Stages", style="dim")

    for profile in profiles:
        stages = ", ".join(profile.stages)
        table.add_row(profile.name, profile.description, stages)

    console.print(table)


if __name__ == "__main__":
    app()
