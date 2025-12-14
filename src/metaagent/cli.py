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
        prompt_library = PromptLibrary(
            prompts_path=config.prompts_file,
            profiles_path=config.profiles_file,
            prompt_library_path=config.prompt_library_path,
        )
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
            prompt_library_path=cfg_dir / "prompt_library",
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
        stages = ", ".join(profile.stages[:3])
        if len(profile.stages) > 3:
            stages += f" (+{len(profile.stages) - 3} more)"
        table.add_row(profile.name, profile.description, stages)

    console.print(table)


@app.command("refine-auto")
def refine_auto(
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
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-n",
        help="Maximum number of refinement iterations.",
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
    """Run iterative refinement with AI-driven prompt selection.

    This mode uses a triage step where the AI analyzes your codebase and
    decides which prompts to run. After each iteration, changes are committed
    and the codebase is re-analyzed.
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
    if mock:
        config.mock_mode = True

    # Validate configuration
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    # Run iterative refinement
    console.print("\n[bold]Starting iterative refinement[/bold]")
    console.print(f"[dim]Repository:[/dim] {repo_path}")
    console.print(f"[dim]Max iterations:[/dim] {max_iterations}")
    console.print(f"[dim]Mock mode:[/dim] {'enabled' if config.mock_mode else 'disabled'}\n")

    orchestrator = Orchestrator(config)
    result = orchestrator.refine_iterative(max_iterations=max_iterations)

    # Display results
    if result.success:
        console.print("\n[green]Iterative refinement completed successfully![/green]\n")
    else:
        console.print("\n[yellow]Refinement completed with issues.[/yellow]\n")

    console.print(f"Iterations completed: {len(result.iterations)}")
    console.print(f"Stages completed: {result.stages_completed}")
    console.print(f"Stages failed: {result.stages_failed}")

    if result.iterations:
        console.print("\n[bold]Iteration Summary:[/bold]")
        for it in result.iterations:
            status = "[green]committed[/green]" if it.committed else "[yellow]no changes[/yellow]"
            console.print(f"  {it.iteration}. {', '.join(it.prompts_run)} - {status}")

    if result.plan_path:
        console.print(f"\n[bold]Final plan written to:[/bold] {result.plan_path}")

    if result.error:
        console.print(f"\n[red]Error:[/red] {result.error}")
        raise typer.Exit(1)


@app.command("list-prompts")
def list_prompts(
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config-dir",
        "-c",
        help="Path to config directory.",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help="Filter by category (e.g., 'quality', 'architecture').",
    ),
) -> None:
    """List available analysis prompts from codebase-digest."""
    # Determine config directory
    cfg_dir = config_dir.resolve() if config_dir else Path.cwd() / "config"

    if not cfg_dir.exists():
        console.print(f"[red]Error:[/red] Config directory not found: {cfg_dir}")
        raise typer.Exit(1)

    prompt_library = PromptLibrary(
        prompts_path=cfg_dir / "prompts.yaml",
        profiles_path=cfg_dir / "profiles.yaml",
        prompt_library_path=cfg_dir / "prompt_library",
    )
    prompt_library.load()

    by_category = prompt_library.list_prompts_by_category()

    if not by_category:
        console.print("[yellow]No prompts found.[/yellow]")
        return

    # Filter by category if specified
    if category:
        if category not in by_category:
            console.print(f"[red]Error:[/red] Category '{category}' not found.")
            console.print(f"Available categories: {', '.join(sorted(by_category.keys()))}")
            raise typer.Exit(1)
        by_category = {category: by_category[category]}

    total = sum(len(prompts) for prompts in by_category.values())
    console.print(f"\n[bold]Available Prompts ({total} total)[/bold]\n")

    for cat_name in sorted(by_category.keys()):
        prompts = by_category[cat_name]
        table = Table(title=f"{cat_name.title()} ({len(prompts)} prompts)")
        table.add_column("ID", style="cyan")
        table.add_column("Goal")

        for prompt in sorted(prompts, key=lambda p: p.id):
            goal = prompt.goal[:60] + "..." if len(prompt.goal) > 60 else prompt.goal
            table.add_row(prompt.id, goal)

        console.print(table)
        console.print()


if __name__ == "__main__":
    app()
