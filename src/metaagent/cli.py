"""CLI entrypoint for meta-agent.

This CLI operates on two conceptually distinct directories:
1. "meta-agent repo" - Where the CLI code and config/ directory live (prompt library, profiles)
2. "target repo" - The codebase being refined (where Repomix/codebase-digest run, where PRD lives)

When dogfooding (running on itself), these are the same directory.
When refining other repos, they are different.
"""

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

# Default config directory (where meta-agent's prompts/profiles live)
# This is relative to where the package is installed
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


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


def get_config_dir(config_dir: Optional[Path]) -> Path:
    """Resolve the config directory (where prompts/profiles live).

    Args:
        config_dir: User-provided config dir, or None for default.

    Returns:
        Resolved config directory path.
    """
    if config_dir:
        return config_dir.resolve()

    # Try default locations in order:
    # 1. Package's config/ directory
    # 2. Current working directory's config/
    if DEFAULT_CONFIG_DIR.exists():
        return DEFAULT_CONFIG_DIR

    cwd_config = Path.cwd() / "config"
    if cwd_config.exists():
        return cwd_config

    # Fall back to package config even if it doesn't exist (will error later)
    return DEFAULT_CONFIG_DIR


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
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Profile to use for refinement (e.g., 'automation_agent', 'quick_review').",
    ),
    repo: Path = typer.Option(
        Path.cwd(),
        "--repo",
        "-r",
        help="Path to the target repository to refine.",
    ),
    prd: Optional[Path] = typer.Option(
        None,
        "--prd",
        help="Path to PRD file (default: docs/prd.md in target repo).",
    ),
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config-dir",
        "-c",
        help="Path to meta-agent config directory containing prompts/profiles.",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        "-m",
        help="Run in mock mode (no API calls).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview prompts and token estimates without making API calls.",
    ),
    smart: bool = typer.Option(
        False,
        "--smart",
        "-s",
        help="Use Ollama for intelligent triage (requires Ollama installed). "
             "Analyzes full codebase locally for FREE, sends only relevant files to Perplexity.",
    ),
    focus: Optional[str] = typer.Option(
        None,
        "--focus",
        "-f",
        help="Custom focus for analysis (used with --smart). "
             "E.g., 'electron frontend with gamified UX' or 'API performance and security'.",
    ),
    loop: bool = typer.Option(
        False,
        "--loop",
        "-l",
        help="Run in iterative loop mode (used with --smart --focus). "
             "Continuously analyzes, implements, and re-analyzes until feature is complete.",
    ),
    autodev: bool = typer.Option(
        False,
        "--autodev",
        help="Run full autonomous development loop: analyze -> implement -> test -> fix -> repeat. "
             "Automatically implements tasks using Claude API, runs tests, and fixes failures.",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-n",
        help="Maximum iterations for loop mode (default: 10).",
    ),
    human_approve: bool = typer.Option(
        None,
        "--human-approve/--no-human-approve",
        help="Require human approval after each iteration (default from config).",
    ),
    auto_implement: bool = typer.Option(
        False,
        "--auto-implement",
        "-a",
        help="Automatically run Claude Code to implement changes.",
    ),
    test_command: Optional[str] = typer.Option(
        None,
        "--test-command",
        "-t",
        help="Test command to run after each implementation (default: pytest -q).",
    ),
    create_branch: bool = typer.Option(
        None,
        "--create-branch/--no-create-branch",
        help="Create a new branch for loop work (default from config).",
    ),
    local_loop: Optional[Path] = typer.Option(
        None,
        "--local-loop",
        help="Run Grok-powered local development loop with PRD file path. "
             "Analyzes, implements, tests, and fixes using Grok for error diagnosis.",
    ),
    evaluator: Optional[str] = typer.Option(
        None,
        "--evaluator",
        "-e",
        help="Evaluator to use for error diagnosis: 'grok' (default) or 'perplexity'.",
    ),
    branch_prefix: Optional[str] = typer.Option(
        None,
        "--branch-prefix",
        help="Custom branch prefix for loop work (default: meta-loop).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Run refinement analysis on a target repository.

    This command analyzes the target codebase using the specified profile and
    generates an improvement plan at docs/mvp_improvement_plan.md in the target repo.

    Examples:
        # Dogfooding (run on meta-agent itself):
        metaagent refine --profile automation_agent --mock

        # Refine another repository:
        metaagent refine --profile automation_agent --repo /path/to/other/repo

        # With explicit config directory:
        metaagent refine --profile automation_agent --repo /path/to/repo --config-dir /path/to/meta-agent/config

        # Grok-powered local development loop:
        metaagent refine --local-loop docs/prd.md

        # Local loop with options:
        metaagent refine --local-loop docs/prd.md --max-iterations 15 --human-approve
        metaagent refine --local-loop docs/prd.md --evaluator perplexity --dry-run
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Resolve target repo path
    repo_path = repo.resolve()
    if not repo_path.exists():
        console.print(f"[red]Error:[/red] Target repository does not exist: {repo_path}")
        raise typer.Exit(1)

    # Resolve config directory (where prompts/profiles live)
    cfg_dir = get_config_dir(config_dir)
    if not cfg_dir.exists():
        console.print(f"[red]Error:[/red] Config directory not found: {cfg_dir}")
        console.print("[dim]Hint: Use --config-dir to specify the meta-agent config location[/dim]")
        raise typer.Exit(1)

    # Load configuration
    config = Config.from_env(repo_path)
    config.config_dir = cfg_dir  # Override with resolved config dir

    # Override PRD path if provided
    if prd:
        config.prd_path = prd.resolve()

    if mock:
        config.mock_mode = True

    if dry_run:
        config.dry_run = True
        config.mock_mode = True  # Dry-run implies mock mode

    if auto_implement:
        config.auto_implement = True

    # Apply loop config overrides from CLI
    if human_approve is not None:
        config.loop.human_approve = human_approve
    if test_command is not None:
        config.loop.test_command = test_command
    if create_branch is not None:
        config.loop.create_branch = create_branch
    if max_iterations:
        config.loop.max_iterations = max_iterations
    if dry_run:
        config.loop.dry_run = True
    if branch_prefix is not None:
        config.loop.branch_prefix = branch_prefix

    # Validate configuration
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    # Handle --local-loop mode (Grok-powered local development loop)
    if local_loop is not None:
        # Validate PRD file exists
        prd_path = local_loop.resolve()
        if not prd_path.exists():
            console.print(f"[red]Error:[/red] PRD file not found: {prd_path}")
            raise typer.Exit(1)

        # Validate evaluator option
        if evaluator is not None and evaluator not in ("grok", "perplexity"):
            console.print(f"[red]Error:[/red] Invalid evaluator '{evaluator}'. Use 'grok' or 'perplexity'.")
            raise typer.Exit(1)

        # Display local loop configuration
        console.print(f"\n[bold]Starting Grok-powered local development loop[/bold]")
        console.print(f"[dim]PRD file:[/dim] {prd_path}")
        console.print(f"[dim]Target repo:[/dim] {repo_path}")
        console.print(f"[dim]Config dir:[/dim] {cfg_dir}")
        console.print(f"[dim]Evaluator:[/dim] {evaluator or config.loop.evaluator.default}")
        console.print(f"[dim]Max iterations:[/dim] {config.loop.max_iterations}")
        console.print(f"[dim]Test command:[/dim] {config.loop.test_command}")
        console.print(f"[dim]Human approval:[/dim] {'enabled' if config.loop.human_approve else 'disabled'}")
        console.print(f"[dim]Branch prefix:[/dim] {config.loop.branch_prefix}")
        console.print(f"[dim]Dry run:[/dim] {'yes' if config.loop.dry_run else 'no'}")
        console.print()

        # Load prompt library for local loop
        try:
            prompt_library = PromptLibrary(
                prompts_path=config.prompts_file,
                profiles_path=config.profiles_file,
                prompt_library_path=config.prompt_library_path,
            )
            prompt_library.load()
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print(f"[dim]Config directory: {cfg_dir}[/dim]")
            raise typer.Exit(1)

        # Run local loop
        orchestrator = Orchestrator(config, prompt_library=prompt_library)
        result = orchestrator.run_local_loop(
            prd_path=str(prd_path),
            evaluator_override=evaluator,
        )
        _display_local_loop_results(result)

        if result.error:
            console.print(f"\n[red]Error:[/red] {result.error}")
            raise typer.Exit(1)
        return

    # Validate that either --profile or --smart is specified (for non-local-loop modes)
    if not smart and not profile:
        console.print("[red]Error:[/red] Either --profile, --smart, or --local-loop is required.")
        console.print("[dim]Use --profile for predefined analysis stages, --smart for Ollama triage, or --local-loop for Grok-powered local dev.[/dim]")
        raise typer.Exit(1)

    # Load and validate prompt library
    try:
        prompt_library = PromptLibrary(
            prompts_path=config.prompts_file,
            profiles_path=config.profiles_file,
            prompt_library_path=config.prompt_library_path,
        )
        prompt_library.load()

        # Only validate profile if not using smart mode
        if not smart and not prompt_library.get_profile(profile):
            available = [p.name for p in prompt_library.list_profiles()]
            console.print(f"[red]Error:[/red] Profile '{profile}' not found.")
            if available:
                console.print(f"Available profiles: {', '.join(available)}")
            console.print(f"[dim]Config directory: {cfg_dir}[/dim]")
            raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(f"[dim]Config directory: {cfg_dir}[/dim]")
        raise typer.Exit(1)

    # Display configuration
    if dry_run:
        console.print("\n[bold yellow]*** DRY RUN: No external API calls will be made. ***[/bold yellow]")
        console.print("[dim]This is a preview of planned analysis steps.[/dim]\n")

    mode_label = "Smart (Ollama)" if smart else profile
    console.print(f"[bold]{'[DRY-RUN] ' if dry_run else ''}Starting refinement[/bold]")
    console.print(f"[dim]Mode:[/dim] {mode_label}")
    console.print(f"[dim]Target repo:[/dim] {repo_path}")
    console.print(f"[dim]Config dir:[/dim] {cfg_dir}")
    console.print(f"[dim]PRD:[/dim] {config.prd_path}")
    if not dry_run:
        console.print(f"[dim]Mock mode:[/dim] {'enabled' if config.mock_mode else 'disabled'}")
    if smart:
        console.print("[dim]Ollama:[/dim] Will analyze full codebase locally (free), send only relevant files to Perplexity")
        if focus:
            console.print(f"[dim]Focus:[/dim] {focus}")
            if autodev:
                console.print(f"[dim]Mode:[/dim] Autonomous development loop (max {config.loop.max_iterations} iterations)")
                console.print(f"[dim]Test command:[/dim] {config.loop.test_command}")
                console.print(f"[dim]Human approval:[/dim] {'enabled' if config.loop.human_approve else 'disabled'}")
                console.print(f"[dim]Create branch:[/dim] {'yes' if config.loop.create_branch else 'no'}")
            elif loop:
                console.print(f"[dim]Mode:[/dim] Iterative feature-focused (loop until complete, max {max_iterations} iterations)")
            else:
                console.print("[dim]Mode:[/dim] Feature-focused (Perplexity selects and rewrites prompts)")
    console.print()

    # Run refinement (or dry-run preview)
    orchestrator = Orchestrator(config, prompt_library=prompt_library)

    if dry_run:
        if smart:
            console.print("[yellow]Note: Dry-run not supported for smart mode yet.[/yellow]")
            raise typer.Exit(0)
        result = orchestrator.refine_dry_run(profile)
        _display_dry_run_results(result)
    elif autodev:
        # Autonomous development loop: analyze -> implement -> test -> fix -> repeat
        if not smart or not focus:
            console.print("[red]Error:[/red] --autodev requires --smart and --focus flags.")
            console.print("[dim]Example: metaagent refine --smart --focus 'add feature' --autodev[/dim]")
            raise typer.Exit(1)
        result = orchestrator.run_autonomous_loop(feature_request=focus)
        _display_autonomous_loop_results(result)
    elif smart and focus and loop:
        # Iterative feature-focused mode: Loop until feature is complete
        result = orchestrator.refine_with_feature_focus_iterative(
            feature_request=focus,
            max_iterations=max_iterations,
        )
        _display_refinement_results(result)
        _display_iteration_summary(result)
    elif smart and focus:
        # Feature-focused mode: Ollama does file selection, Perplexity does heavy lifting
        # (selects and rewrites codebase-digest prompts for this specific feature)
        result = orchestrator.refine_with_feature_focus(feature_request=focus)
        _display_refinement_results(result)
    elif smart:
        # Smart mode without focus: standard Ollama triage
        result = orchestrator.refine_with_ollama_triage(focus=focus)
        _display_refinement_results(result)
    else:
        result = orchestrator.refine(profile)
        _display_refinement_results(result)

    if result.error:
        console.print(f"\n[red]Error:[/red] {result.error}")
        raise typer.Exit(1)


def _display_dry_run_results(result) -> None:
    """Display dry-run results with planned calls and token estimates.

    Args:
        result: RefinementResult from dry-run.
    """
    from .tokens import format_token_count

    if not result.planned_calls:
        console.print("[yellow]No stages would be executed for this profile.[/yellow]")
        return

    console.print(f"\n[bold]Planned Analysis Steps ({len(result.planned_calls)} stages)[/bold]\n")

    total_tokens = 0

    for i, call in enumerate(result.planned_calls, 1):
        total_tokens += call.estimated_tokens

        console.print(f"[bold cyan]{i}. {call.prompt_id}[/bold cyan]")
        console.print(f"   Profile: {call.profile} | Stage: {call.stage}")
        console.print(f"   Estimated tokens: [yellow]{format_token_count(call.estimated_tokens)}[/yellow]")

        # Show prompt preview (first 5 lines)
        lines = call.rendered_prompt.split("\n")[:5]
        preview = "\n".join(lines)
        if len(call.rendered_prompt.split("\n")) > 5:
            preview += "\n   ..."
        console.print(f"   [dim]--- Prompt Preview ---[/dim]")
        for line in preview.split("\n"):
            console.print(f"   [dim]{line[:80]}{'...' if len(line) > 80 else ''}[/dim]")
        console.print()

    console.print("[bold]Summary[/bold]")
    console.print(f"  Total stages: {len(result.planned_calls)}")
    console.print(f"  Total estimated tokens: [yellow]{format_token_count(total_tokens)}[/yellow]")
    console.print()
    console.print("[dim]Note: Token estimates are approximate. Actual usage may vary by 10-20%.[/dim]")


def _display_refinement_results(result) -> None:
    """Display standard refinement results.

    Includes the structured ImplementationReport to the terminal so the
    current Claude Code session can see and act on the tasks.

    Args:
        result: RefinementResult from refinement.
    """
    if result.success:
        console.print("\n[green]Refinement completed successfully![/green]\n")
    else:
        console.print("\n[yellow]Refinement completed with issues.[/yellow]\n")

    console.print(f"Stages completed: {result.stages_completed}")
    console.print(f"Stages failed: {result.stages_failed}")

    if result.plan_path:
        console.print(f"\n[bold]Improvement plan written to:[/bold] {result.plan_path}")

    # Display the implementation report for the current Claude Code session
    if result.implementation_report and result.implementation_report.tasks:
        _display_implementation_report(result.implementation_report)

    if not result.implementation_report or not result.implementation_report.tasks:
        console.print("\nNext steps:")
        console.print("  1. Review the improvement plan")
        console.print("  2. Open Claude Code in your repository")
        console.print("  3. Ask Claude Code to implement the plan")


def _display_iteration_summary(result) -> None:
    """Display a summary of iterations for iterative mode.

    Args:
        result: RefinementResult from iterative refinement.
    """
    from rich.panel import Panel
    from rich.table import Table

    if not result.iterations:
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]ITERATION SUMMARY[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    # Get the latest layer status from the last iteration
    latest_layer_status = None
    for it in reversed(result.iterations):
        if it.layer_status:
            latest_layer_status = it.layer_status
            break

    # Show layer progress bar
    layer_names = ["Scaffold", "Core", "Integration", "Polish"]

    if latest_layer_status:
        # Use actual layer status from the response
        layer_complete = [
            latest_layer_status.scaffold_complete,
            latest_layer_status.core_complete,
            latest_layer_status.integration_complete,
            latest_layer_status.polish_complete,
        ]

        layer_display = ""
        for i, (name, complete) in enumerate(zip(layer_names, layer_complete), 1):
            if complete:
                layer_display += f"[green][+][/green] {name}  "
            elif i == latest_layer_status.current_layer:
                layer_display += f"[yellow][>][/yellow] {name}  "
            else:
                layer_display += f"[dim][ ] {name}[/dim]  "

        console.print(f"  Layers: {layer_display}")
        if latest_layer_status.layer_progress:
            # Truncate long progress messages
            progress = latest_layer_status.layer_progress
            if len(progress) > 80:
                progress = progress[:80] + "..."
            console.print(f"  [dim]{progress}[/dim]")
    else:
        # Fallback: determine from stage names if no layer_status
        completed_layers = 0
        for stage in result.stage_results:
            stage_name = stage.stage_name.lower()
            if "layer 1" in stage_name or "scaffold" in stage_name:
                completed_layers = max(completed_layers, 1)
            elif "layer 2" in stage_name or "core" in stage_name:
                completed_layers = max(completed_layers, 2)
            elif "layer 3" in stage_name or "integration" in stage_name:
                completed_layers = max(completed_layers, 3)
            elif "layer 4" in stage_name or "polish" in stage_name:
                completed_layers = max(completed_layers, 4)

        layer_display = ""
        for i, name in enumerate(layer_names, 1):
            if i <= completed_layers:
                layer_display += f"[green][+][/green] {name}  "
            else:
                layer_display += f"[dim][ ] {name}[/dim]  "

        console.print(f"  Layers: {layer_display}")

    console.print()

    # Show iteration table
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("#", style="dim", width=3)
    table.add_column("Layer", width=15)
    table.add_column("Tasks", width=6)
    table.add_column("Status", width=10)

    for it in result.iterations:
        if it.prompts_run:
            # Use layer_status if available
            if it.layer_status:
                layer_info = it.layer_status.layer_name.capitalize()
            else:
                # Fallback: extract layer from stage name
                layer_info = "Unknown"
                if it.stage_results:
                    stage_name = it.stage_results[0].stage_name
                    if "Layer 1" in stage_name:
                        layer_info = "Scaffold"
                    elif "Layer 2" in stage_name:
                        layer_info = "Core"
                    elif "Layer 3" in stage_name:
                        layer_info = "Integration"
                    elif "Layer 4" in stage_name:
                        layer_info = "Polish"

            status = "[green]Done[/green]" if it.changes_made else "[yellow]Pending[/yellow]"
            task_count = str(len(it.stage_results[0].tasks)) if it.stage_results else "0"
            table.add_row(str(it.iteration), layer_info, task_count, status)
        else:
            # Feature complete - show final status
            if it.layer_status:
                layer_info = "[green]Complete[/green]"
            else:
                layer_info = "[green]Complete[/green]"
            table.add_row(str(it.iteration), layer_info, "-", "[green]Done[/green]")

    console.print(table)
    console.print()


def _display_autonomous_loop_results(result) -> None:
    """Display results from the autonomous development loop.

    Args:
        result: AutonomousLoopResult from the autonomous loop.
    """
    from rich.panel import Panel
    from rich.table import Table

    if result.success:
        console.print("\n[green]Autonomous development loop completed successfully![/green]\n")
    else:
        console.print("\n[yellow]Autonomous development loop completed with issues.[/yellow]\n")

    console.print(Panel.fit(
        "[bold cyan]AUTONOMOUS LOOP SUMMARY[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    # Summary stats
    console.print(f"[bold]Iterations:[/bold] {result.iterations_completed}/{result.max_iterations}")
    console.print(f"[bold]Tasks implemented:[/bold] {result.tasks_completed}")
    console.print(f"[bold]Tests passed:[/bold] {result.tests_passed}")
    console.print(f"[bold]Fixes applied:[/bold] {result.fixes_applied}")
    if result.tokens_used > 0:
        console.print(f"[bold]Tokens used:[/bold] {result.tokens_used:,}")
    console.print()

    # Show iteration details
    if result.iteration_details:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("#", style="dim", width=3)
        table.add_column("Task", width=40)
        table.add_column("Tests", width=8)
        table.add_column("Fixes", width=6)
        table.add_column("Status", width=10)

        for detail in result.iteration_details:
            task_name = detail.get("task", "Unknown")[:40]
            tests = "[green]Pass[/green]" if detail.get("tests_passed") else "[red]Fail[/red]"
            fixes = str(detail.get("fixes_applied", 0))
            status = "[green]Done[/green]" if detail.get("success") else "[red]Failed[/red]"
            table.add_row(str(detail.get("iteration", "?")), task_name, tests, fixes, status)

        console.print(table)
        console.print()

    # Show branch info if created
    if result.branch_name:
        console.print(f"[bold]Work branch:[/bold] {result.branch_name}")
        console.print(f"[bold]Commits:[/bold] {result.commits_made}")
        console.print()

    # Show final evaluation if available
    if result.final_evaluation:
        console.print(Panel(
            f"[bold]Final PRD Evaluation:[/bold]\n{result.final_evaluation}",
            border_style="green" if result.prd_aligned else "yellow",
        ))
        console.print()

    # Show error if any
    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        console.print()

    # Next steps
    if result.success:
        console.print(Panel(
            "[bold green]SUCCESS[/bold green]\n\n"
            "The autonomous loop has completed successfully.\n"
            "All tasks have been implemented and tests are passing.\n\n"
            "Next steps:\n"
            "1. Review the changes in your working branch\n"
            "2. Run full test suite\n"
            "3. Create a pull request if satisfied",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[bold yellow]INCOMPLETE[/bold yellow]\n\n"
            "The autonomous loop stopped before completion.\n\n"
            "Review the iteration details above to understand what happened.\n"
            "You can re-run with --autodev to continue from where it left off.",
            border_style="yellow",
        ))


def _display_local_loop_results(result) -> None:
    """Display results from the Grok-powered local development loop.

    Args:
        result: LocalLoopResult from the local loop.
    """
    from rich.panel import Panel
    from rich.table import Table

    if result.success:
        console.print("\n[green]Local development loop completed successfully![/green]\n")
    else:
        console.print("\n[yellow]Local development loop completed with issues.[/yellow]\n")

    console.print(Panel.fit(
        "[bold cyan]LOCAL LOOP SUMMARY[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    # Summary stats
    console.print(f"[bold]Iterations:[/bold] {result.iterations_completed}/{result.max_iterations}")
    console.print(f"[bold]Tasks implemented:[/bold] {result.tasks_completed}")
    console.print(f"[bold]Tests passed:[/bold] {result.tests_passed}")
    console.print(f"[bold]Fixes applied:[/bold] {result.fixes_applied}")
    console.print(f"[bold]Evaluator:[/bold] {result.evaluator_used}")
    if result.tokens_used > 0:
        console.print(f"[bold]Tokens used:[/bold] {result.tokens_used:,}")
    console.print()

    # Show iteration details
    if result.iteration_details:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("#", style="dim", width=3)
        table.add_column("Task", width=40)
        table.add_column("Tests", width=8)
        table.add_column("Fixes", width=6)
        table.add_column("Status", width=10)

        for detail in result.iteration_details:
            task_name = detail.get("task", "Unknown")[:40]
            tests = "[green]Pass[/green]" if detail.get("tests_passed") else "[red]Fail[/red]"
            fixes = str(detail.get("fixes_applied", 0))
            status = "[green]Done[/green]" if detail.get("success") else "[red]Failed[/red]"
            table.add_row(str(detail.get("iteration", "?")), task_name, tests, fixes, status)

        console.print(table)
        console.print()

    # Show branch info if created
    if result.branch_name:
        console.print(f"[bold]Work branch:[/bold] {result.branch_name}")
        console.print(f"[bold]Commits:[/bold] {result.commits_made}")
        console.print()

    # Show final evaluation if available
    if result.final_evaluation:
        console.print(Panel(
            f"[bold]Final PRD Evaluation (Grok):[/bold]\n{result.final_evaluation}",
            border_style="green" if result.prd_aligned else "yellow",
        ))
        console.print()

    # Show error if any
    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        console.print()

    # Next steps
    if result.success:
        console.print(Panel(
            "[bold green]SUCCESS[/bold green]\n\n"
            "The Grok-powered local development loop has completed successfully.\n"
            "All tasks have been implemented and tests are passing.\n\n"
            "Next steps:\n"
            "1. Review the changes in your working branch\n"
            "2. Run full test suite\n"
            "3. Create a pull request if satisfied",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[bold yellow]INCOMPLETE[/bold yellow]\n\n"
            "The local development loop stopped before completion.\n\n"
            "Review the iteration details above to understand what happened.\n"
            "You can re-run with --local-loop to continue from where it left off.",
            border_style="yellow",
        ))


def _display_implementation_report(report) -> None:
    """Display the structured implementation report for the current session.

    This outputs the report in a format that Claude Code can read and act upon,
    enabling the current Claude Code session to implement the tasks directly.

    Args:
        report: ImplementationReport with tasks and recommendations.
    """
    from rich.markdown import Markdown
    from rich.panel import Panel

    console.print()
    console.print(Panel.fit(
        "[bold cyan]IMPLEMENTATION REPORT[/bold cyan]\n"
        "[dim]Tasks for the current Claude Code session to implement[/dim]",
        border_style="cyan",
    ))
    console.print()

    # Summary
    console.print(f"[bold]Total Tasks:[/bold] {len(report.tasks)}")
    if report.total_estimated_effort:
        console.print(f"[bold]Estimated Effort:[/bold] {report.total_estimated_effort}")
    console.print()

    # Priority breakdown
    priority_counts = {}
    for task in report.tasks:
        priority = task.priority.lower()
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    if priority_counts:
        console.print("[bold]Tasks by Priority:[/bold]")
        # Use ASCII-safe markers that work on all terminals
        priority_markers = {
            "critical": "[red][!][/red]",
            "high": "[yellow][*][/yellow]",
            "medium": "[blue][-][/blue]",
            "low": "[green][.][/green]",
        }
        for priority in ["critical", "high", "medium", "low"]:
            count = priority_counts.get(priority, 0)
            if count > 0:
                marker = priority_markers.get(priority, "[ ]")
                console.print(f"  {marker} {priority.title()}: {count}")
        console.print()

    # Display each task
    console.print("[bold]Tasks to Implement:[/bold]")
    console.print()

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_tasks = sorted(report.tasks, key=lambda t: priority_order.get(t.priority.lower(), 2))

    for i, task in enumerate(sorted_tasks, 1):
        priority = task.priority.lower()
        # Use ASCII-safe markers that work on all terminals
        marker = {
            "critical": "[red][!][/red]",
            "high": "[yellow][*][/yellow]",
            "medium": "[blue][-][/blue]",
            "low": "[green][.][/green]",
        }.get(priority, "[ ]")

        console.print(f"[bold]{i}. {marker} {task.title}[/bold]")
        console.print(f"   [dim]Priority:[/dim] {priority.title()}", end="")
        if task.estimated_complexity:
            console.print(f" | [dim]Complexity:[/dim] {task.estimated_complexity}", end="")
        console.print()

        if task.description and task.description != task.title:
            # Truncate long descriptions
            desc = task.description
            if len(desc) > 200:
                desc = desc[:200] + "..."
            console.print(f"   {desc}")

        if task.affected_files:
            files = ", ".join(task.affected_files[:3])
            if len(task.affected_files) > 3:
                files += f" (+{len(task.affected_files) - 3} more)"
            console.print(f"   [dim]Files:[/dim] {files}")

        console.print()

    # Display as markdown for easy copy-paste
    console.print()
    console.print(Panel.fit(
        "[bold]Markdown Summary[/bold]\n"
        "[dim]Copy this to share or track progress[/dim]",
        border_style="dim",
    ))
    console.print()
    console.print(Markdown(report.to_markdown()))
    console.print()

    # Instructions for current session
    console.print(Panel(
        "[bold cyan]NEXT STEPS FOR CURRENT SESSION[/bold cyan]\n\n"
        "1. Work through tasks in priority order (critical → low)\n"
        "2. For each task:\n"
        "   - Read the task description and affected files\n"
        "   - Make the required code changes\n"
        "   - Run tests to verify\n"
        "3. After completing related tasks, commit your changes\n"
        "4. Run meta-agent again to check for remaining issues",
        border_style="green",
    ))


@app.command("refine-iterative")
def refine_iterative(
    repo: Path = typer.Option(
        Path.cwd(),
        "--repo",
        "-r",
        help="Path to the target repository to refine.",
    ),
    prd: Optional[Path] = typer.Option(
        None,
        "--prd",
        help="Path to PRD file (default: docs/prd.md in target repo).",
    ),
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config-dir",
        "-c",
        help="Path to meta-agent config directory containing prompts/profiles.",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-n",
        help="Maximum number of triage iterations to run.",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        "-m",
        help="Run in mock mode (no API calls).",
    ),
    auto_implement: bool = typer.Option(
        False,
        "--auto-implement",
        "-a",
        help="Automatically run Claude Code to implement changes.",
    ),
    no_commit: bool = typer.Option(
        False,
        "--no-commit",
        help="Skip auto-committing changes after implementation.",
    ),
    auto_push: bool = typer.Option(
        False,
        "--auto-push",
        help="Automatically push commits to remote (default: disabled for safety).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Run iterative refinement with AI-driven triage.

    Instead of running a fixed profile, this mode lets the AI decide which
    prompts to run based on analyzing the codebase against the PRD. The loop:

    1. Pack codebase with Repomix
    2. Triage (AI decides which prompts to run)
    3. Run selected analysis prompts
    4. Implement changes with Claude Code
    5. Commit changes
    6. Repeat until done or max iterations reached

    Examples:
        # Run iterative refinement on current repo:
        metaagent refine-iterative --mock

        # Refine another repository with max 5 iterations:
        metaagent refine-iterative --repo /path/to/repo --max-iterations 5
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Resolve target repo path
    repo_path = repo.resolve()
    if not repo_path.exists():
        console.print(f"[red]Error:[/red] Target repository does not exist: {repo_path}")
        raise typer.Exit(1)

    # Resolve config directory
    cfg_dir = get_config_dir(config_dir)
    if not cfg_dir.exists():
        console.print(f"[red]Error:[/red] Config directory not found: {cfg_dir}")
        console.print("[dim]Hint: Use --config-dir to specify the meta-agent config location[/dim]")
        raise typer.Exit(1)

    # Load configuration
    config = Config.from_env(repo_path)
    config.config_dir = cfg_dir

    if prd:
        config.prd_path = prd.resolve()

    if mock:
        config.mock_mode = True

    if auto_implement:
        config.auto_implement = True

    if no_commit:
        config.auto_commit = False

    if auto_push:
        config.auto_push = True

    # Validate configuration
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    # Load prompt library
    try:
        prompt_library = PromptLibrary(
            prompts_path=config.prompts_file,
            profiles_path=config.profiles_file,
            prompt_library_path=config.prompt_library_path,
        )
        prompt_library.load()

        # Verify triage prompt exists
        if not prompt_library.get_prompt("meta_triage"):
            console.print("[red]Error:[/red] Triage prompt (meta_triage) not found in prompt library.")
            console.print(f"[dim]Config directory: {cfg_dir}[/dim]")
            raise typer.Exit(1)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(f"[dim]Config directory: {cfg_dir}[/dim]")
        raise typer.Exit(1)

    # Display configuration
    console.print(f"\n[bold]Starting iterative refinement[/bold]")
    console.print(f"[dim]Target repo:[/dim] {repo_path}")
    console.print(f"[dim]Config dir:[/dim] {cfg_dir}")
    console.print(f"[dim]PRD:[/dim] {config.prd_path}")
    console.print(f"[dim]Max iterations:[/dim] {max_iterations}")
    console.print(f"[dim]Mock mode:[/dim] {'enabled' if config.mock_mode else 'disabled'}\n")

    # Run iterative refinement
    orchestrator = Orchestrator(config, prompt_library=prompt_library)
    result = orchestrator.refine_iterative(max_iterations=max_iterations)

    # Display results
    if result.success:
        console.print("\n[green]Iterative refinement completed successfully![/green]\n")
    else:
        console.print("\n[yellow]Iterative refinement completed with issues.[/yellow]\n")

    console.print(f"Iterations run: {len(result.iterations)}")
    console.print(f"Stages completed: {result.stages_completed}")
    console.print(f"Stages failed: {result.stages_failed}")

    # Show iteration summary
    if result.iterations:
        console.print("\n[bold]Iteration Summary:[/bold]")
        for iteration in result.iterations:
            prompts = ", ".join(iteration.prompts_run) if iteration.prompts_run else "none"
            status = "[green]+[/green]" if iteration.changes_made else "[dim]-[/dim]"
            console.print(f"  {status} Iteration {iteration.iteration}: {prompts}")

    if result.plan_path:
        console.print(f"\n[bold]Improvement plan written to:[/bold] {result.plan_path}")

    # Display the implementation report for the current Claude Code session
    if result.implementation_report and result.implementation_report.tasks:
        _display_implementation_report(result.implementation_report)

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
    validate: bool = typer.Option(
        False,
        "--validate",
        "-V",
        help="Validate that all referenced prompts exist.",
    ),
) -> None:
    """List available refinement profiles.

    Use --validate to check if all prompts referenced by each profile exist.
    """
    cfg_dir = get_config_dir(config_dir)

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

    if validate:
        # Detailed validation view
        _show_profiles_with_validation(prompt_library, profiles)
    else:
        # Simple list view
        table = Table(title="Available Profiles")
        table.add_column("ID", style="cyan")
        table.add_column("Description")
        table.add_column("Stages", style="dim")

        for profile in profiles:
            stages = ", ".join(profile.stages[:3])
            if len(profile.stages) > 3:
                stages += f" (+{len(profile.stages) - 3} more)"
            table.add_row(profile.name, profile.description, stages)

        console.print(table)


def _show_profiles_with_validation(
    prompt_library: PromptLibrary, profiles: list
) -> None:
    """Display profiles with validation status for each stage.

    Args:
        prompt_library: The loaded prompt library.
        profiles: List of Profile instances.
    """
    validation_results = prompt_library.validate_all_profiles()
    has_errors = False

    for profile in profiles:
        # Get validation for this profile
        profile_key = None
        for key, val in validation_results.items():
            if prompt_library.get_profile(key) == profile:
                profile_key = key
                break

        if profile_key is None:
            continue

        stage_validation = validation_results.get(profile_key, {})
        valid_count = sum(1 for v in stage_validation.values() if v)
        total_count = len(stage_validation)

        # Profile header
        if valid_count == total_count:
            status = "[green]OK[/green]"
        else:
            status = f"[red]{total_count - valid_count} missing[/red]"
            has_errors = True

        console.print(f"\n[bold cyan]{profile.name}[/bold cyan] ({status})")
        console.print(f"  [dim]{profile.description}[/dim]")
        console.print("  [bold]Stages:[/bold]")

        for stage, exists in stage_validation.items():
            if exists:
                icon = "[green]+[/green]"
            else:
                icon = "[red]X[/red]"
            console.print(f"    {icon} {stage}")

    console.print()

    if has_errors:
        console.print("[yellow]Warning:[/yellow] Some profiles reference missing prompts.")
        console.print("[dim]Run 'metaagent list-prompts' to see available prompts.[/dim]")


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
    cfg_dir = get_config_dir(config_dir)

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


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(
        8765,
        "--port",
        "-p",
        help="Port to run the dashboard server on.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind the dashboard server to.",
    ),
) -> None:
    """Launch the real-time observability dashboard.

    The dashboard provides a live view of meta-agent refinement progress,
    including layer tracking, task status, and logs.

    After starting the dashboard, run a refinement in another terminal
    and watch the progress in real-time at http://localhost:8765

    Examples:
        # Start dashboard on default port:
        metaagent dashboard

        # Start on custom port:
        metaagent dashboard --port 9000

        # Then in another terminal, run refinement:
        metaagent refine --smart --focus "add feature" --loop
    """
    try:
        from .dashboard import run_server
    except ImportError:
        console.print("[red]Error:[/red] Dashboard dependencies not installed.")
        console.print("[dim]Install with: pip install fastapi uvicorn websockets[/dim]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Meta-Agent Dashboard[/bold cyan]")
    console.print(f"[dim]Starting server at http://{host}:{port}[/dim]")
    console.print()
    console.print("Open the URL in your browser to view the dashboard.")
    console.print("Run a refinement in another terminal to see live progress.")
    console.print()
    console.print("[dim]Press Ctrl+C to stop the server.[/dim]")
    console.print()

    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard server stopped.[/dim]")


if __name__ == "__main__":
    app()
