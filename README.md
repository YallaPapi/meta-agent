# Meta-Agent

A Python CLI tool for automated codebase refinement from v0 to MVP.

Meta-agent analyzes your codebase against its PRD (Product Requirements Document) using LLM-powered analysis and generates a prioritized improvement plan that can be executed by Claude Code or similar AI coding assistants.

## Features

- **PRD Alignment Analysis**: Identifies gaps between your implementation and requirements
- **Architecture Review**: Checks code organization and best practices
- **Robustness Hardening**: Finds opportunities to improve error handling and edge cases
- **Test Coverage Analysis**: Identifies missing tests for MVP quality
- **Multiple Profiles**: Different analysis profiles for different project types
- **Mock Mode**: Test the workflow without API calls

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
- **Perplexity API key** (for analysis, or use `--mock` mode)

## Usage

### Basic Usage

```bash
# Run refinement analysis on current directory
metaagent refine --profile automation_agent

# Run on a specific repository
metaagent refine --profile automation_agent --repo /path/to/repo

# Run in mock mode (no API calls)
metaagent refine --profile automation_agent --mock
```

### Available Commands

```bash
# Show version
metaagent --version

# Show help
metaagent --help

# List available profiles
metaagent list-profiles

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

1. **Build v0**: Use Claude Code to build initial implementation from PRD
2. **Run Meta-Agent**: `metaagent refine --profile automation_agent`
3. **Review Plan**: Check `docs/mvp_improvement_plan.md`
4. **Implement**: Feed the plan to Claude Code to implement improvements
5. **Iterate**: Re-run meta-agent to find remaining gaps

## License

MIT
