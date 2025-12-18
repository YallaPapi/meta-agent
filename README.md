# Meta-Agent

A Python CLI tool for automated codebase refinement from v0 to MVP.

Meta-agent analyzes your codebase against its PRD (Product Requirements Document) using LLM-powered analysis and generates a prioritized improvement plan. It can also autonomously implement tasks, run tests, and fix errors.

## Quick Start (30 seconds)

```bash
# Install
pip install -e .

# Run autonomous task loop (free tier: 5 iterations)
metaagent loop --prd docs/prd.md

# Or preview without making changes
metaagent loop --prd docs/prd.md --dry-run
```

That's it! Meta-agent will:
1. Parse your PRD into tasks
2. Show each task clearly for implementation
3. Run tests after each change
4. Get AI help to fix any failures
5. Generate a completion report

## Features

- **Autonomous Task Loop**: Parse PRD -> implement -> test -> fix -> repeat
- **PRD Alignment Analysis**: Identifies gaps between your implementation and requirements
- **Architecture Review**: Checks code organization and best practices
- **Grok Integration**: Uses Grok for PRD parsing and error diagnosis (free with xAI API)
- **Claude Implementation**: Uses Claude for code changes
- **Test Integration**: Automatically runs tests after each change
- **Multiple Profiles**: Different analysis profiles for different project types
- **Freemium Model**: 5 iterations free, pro key unlocks unlimited
- **Windows-First**: Full Windows compatibility (also works on macOS/Linux)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd meta-agent

# Install in development mode
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Required for real analysis (not needed in mock mode)
PERPLEXITY_API_KEY=your_perplexity_api_key

# Optional
ANTHROPIC_API_KEY=your_anthropic_api_key
METAAGENT_TIMEOUT=120
METAAGENT_MAX_TOKENS=100000
METAAGENT_LOG_LEVEL=INFO
```

### Prerequisites

- **Python 3.10+**
- **Repomix** (for codebase packing): `npm install -g repomix`
- **codebase-digest** (for directory tree & metrics): `pip install codebase-digest` (installed automatically)
- **xAI/Grok API key** (for task loop - set `GROK_API_KEY` in .env)
- **Anthropic API key** (for implementation - set `ANTHROPIC_API_KEY` in .env)
- **Perplexity API key** (for analysis mode - set `PERPLEXITY_API_KEY` in .env)

## Autonomous Task Loop

The new `loop` command runs an autonomous development loop:

```bash
# Basic usage (free tier: 5 iterations max)
metaagent loop --prd docs/prd.md

# With pro key for unlimited iterations
metaagent loop --prd docs/prd.md --pro-key YOUR_KEY

# Skip human approval for fully autonomous
metaagent loop --prd docs/prd.md --no-human-approve

# Preview tasks without implementing
metaagent loop --prd docs/prd.md --dry-run

# Custom test command
metaagent loop --prd docs/prd.md --test-command "npm test"
```

### How It Works

1. **Parse PRD**: Grok reads your PRD and creates prioritized tasks
2. **Display Task**: Shows what needs to be implemented
3. **Human Approval**: (Optional) Confirm before each task
4. **Implement**: Claude makes the code changes
5. **Test**: Runs your test suite
6. **Diagnose**: If tests fail, Grok analyzes errors and suggests fixes
7. **Commit**: Changes are committed to git
8. **Repeat**: Until all tasks complete or limit reached

### Using in Claude Code Sessions

When you run `metaagent loop` in a Claude Code session, each task is displayed clearly:

```
================================================================================
IMPLEMENT THIS TASK (Claude Code)
================================================================================

Task ID: 1
Title: Add user authentication
Priority: HIGH

Description:
----------------------------------------
Implement JWT-based authentication for the API endpoints.
----------------------------------------

================================================================================
Make your changes now. When finished, type 'y' and press Enter.
================================================================================
```

Just make your changes in Claude Code, then press Enter to continue.

### Freemium Model

- **Free Tier**: 5 iterations per PRD
- **Pro Tier**: Unlimited iterations

Get your pro key at: https://yoursite.gumroad.com/l/meta-agent-pro

Set via environment or CLI:
```bash
export METAAGENT_PRO_KEY=your-key
# or
metaagent loop --prd docs/prd.md --pro-key your-key
```

### Codebase Analysis Tools

Meta-agent uses two complementary tools for comprehensive codebase analysis:

| Tool | Purpose | Output |
|------|---------|--------|
| **codebase-digest** | Directory structure & metrics | Tree view, file counts, token estimates |
| **Repomix** | Full file contents | All source code packed into one file |

Both tools run automatically during refinement, providing the LLM with both high-level structure and detailed code content.

## Usage

### Basic Usage

```bash
# Run refinement analysis on current directory
metaagent refine --profile automation_agent

# Run on a specific repository
metaagent refine --profile automation_agent --repo /path/to/repo

# Run in mock mode (no API calls)
metaagent refine --profile automation_agent --mock

# Preview prompts and token estimates without API calls
metaagent refine --profile automation_agent --dry-run
```

### Available Commands

```bash
# Show version
metaagent --version

# Show help
metaagent --help

# List available profiles
metaagent list-profiles

# List profiles with validation (check if prompts exist)
metaagent list-profiles --validate

# Run refinement with verbose output
metaagent refine --profile automation_agent --verbose
```

### Available Profiles

| Profile | Description |
|---------|-------------|
| `automation_agent` | For CLI tools and automation agents |
| `backend_service` | For API backends (includes security review) |
| `internal_tool` | Lighter profile for internal tools |
| `quick_review` | Fast PRD alignment check only |
| `full_review` | Comprehensive analysis with all stages |

### Output

After running, meta-agent generates `docs/mvp_improvement_plan.md` containing:

1. **PRD Summary**: Brief recap of requirements
2. **Stage Summaries**: Results from each analysis stage
3. **Task List**: Prioritized checklist of improvements
4. **Instructions**: How to use the plan with Claude Code

## Project Structure

```
meta-agent/
├── src/metaagent/
│   ├── __init__.py
│   ├── cli.py           # CLI entrypoint
│   ├── config.py        # Configuration management
│   ├── orchestrator.py  # Main pipeline orchestration
│   ├── prompts.py       # Prompt/profile loading
│   ├── repomix.py       # Repomix integration
│   ├── analysis.py      # LLM analysis engine
│   └── plan_writer.py   # Plan file generation
├── config/
│   ├── prompts.yaml     # Prompt templates
│   └── profiles.yaml    # Profile definitions
├── tests/
│   ├── conftest.py      # Test fixtures
│   ├── test_*.py        # Test modules
└── docs/
    └── prd.md           # Project PRD
```

## Adding Custom Prompts

Add new prompts to `config/prompts.yaml`:

```yaml
prompts:
  my_custom_prompt:
    id: my_custom_prompt
    goal: "Description of what this prompt analyzes"
    stage: custom
    template: |
      Your prompt template here.
      Use {{ prd }}, {{ code_context }}, {{ history }}, {{ current_stage }}.

      Respond with JSON:
      {
        "summary": "...",
        "recommendations": [...],
        "tasks": [...]
      }
```

## Adding Custom Profiles

Add new profiles to `config/profiles.yaml`:

```yaml
profiles:
  my_profile:
    name: "My Custom Profile"
    description: "Description of when to use this profile"
    stages:
      - alignment_with_prd
      - my_custom_prompt
```

## Analysis Response Contract

All analysis prompts must return JSON in this structure:

```json
{
  "summary": "Brief 2-3 sentence overview of findings",
  "recommendations": [
    "High-level recommendation 1",
    "High-level recommendation 2"
  ],
  "tasks": [
    {
      "title": "Short, actionable task title",
      "description": "Detailed description of what needs to be done",
      "priority": "critical|high|medium|low",
      "file": "path/to/relevant/file.py"
    }
  ]
}
```

### Task Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | Yes | Short, actionable description |
| `description` | string | No | Detailed implementation guidance |
| `priority` | string | No | One of: `critical`, `high`, `medium`, `low`. Default: `medium` |
| `file` | string | No | Path to the relevant source file |

### Codebase Digest Prompts

When using prompts from the `config/prompt_library/` directory (Codebase Digest format), the system automatically appends JSON response instructions. No modification to the original prompt files is needed.

The system:
1. Detects if a prompt already has JSON schema instructions
2. Appends the standard JSON format suffix if not present
3. Parses responses using multiple strategies (code blocks, bare JSON, fallback)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=metaagent

# Type checking
mypy src/metaagent

# Linting
ruff check src tests
```

## Workflow

### Analysis Mode (Manual)

1. **Build v0**: Use Claude Code to build initial implementation from PRD
2. **Run Meta-Agent**: `metaagent refine --profile automation_agent`
3. **Review Plan**: Check `docs/mvp_improvement_plan.md`
4. **Implement**: Feed the plan to Claude Code to implement improvements
5. **Iterate**: Re-run meta-agent to find remaining gaps

### Task Loop Mode (Autonomous)

1. **Write PRD**: Create `docs/prd.md` with your requirements
2. **Run Loop**: `metaagent loop --prd docs/prd.md`
3. **Approve Tasks**: Review each task, approve with 'y', skip with 'skip'
4. **Review Report**: Check `meta-agent-report.md` for summary
5. **Continue**: Re-run to pick up where you left off

## Building Standalone Executable

Create a single-file executable for distribution:

```bash
# Install PyInstaller
pip install pyinstaller

# Build using the spec file
pyinstaller meta-agent.spec

# Or use the build script
python scripts/build.py

# Output at: dist/meta-agent(.exe)
```

## Output Files

Meta-agent generates these files:

| File | Description |
|------|-------------|
| `docs/mvp_improvement_plan.md` | Analysis mode output (tasks, recommendations) |
| `meta-agent-report.md` | Task loop completion report |
| `meta_agent_tasks.json` | Task state for resumability |

## License

MIT
