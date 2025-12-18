# Claude Code Instructions

## IMPORTANT: Always Use Taskmaster
**You MUST use Taskmaster for ALL task management in this project. This is MANDATORY.**

### Task Management (Always Use)
- `task-master list` - See all current tasks with status
- `task-master next` - Get the next available task to work on
- `task-master show <id>` - View detailed task information
- `task-master set-status --id=<id> --status=<status>` - Update task status (pending, in-progress, done, deferred, cancelled, blocked)

### Research & Analysis (Use for ALL Research)
- `task-master add-task --prompt="description" --research` - Add new task WITH research
- `task-master expand --id=<id> --research --force` - Break task into subtasks WITH research
- `task-master update-task --id=<id> --prompt="changes" --research` - Update task WITH research
- `task-master update --from=<id> --prompt="changes" --research` - Update multiple tasks WITH research
- `task-master analyze-complexity --research` - Analyze task complexity with AI
- **ALWAYS use `--research` flag when you need to gather information or make informed decisions**

### Progress Tracking
- `task-master update-subtask --id=<id> --prompt="notes"` - Log implementation progress and notes
- `task-master complexity-report` - View complexity analysis report

### Task Organization
- `task-master add-dependency --id=<id> --depends-on=<id>` - Add task dependencies
- `task-master move --from=<id> --to=<id>` - Reorganize task hierarchy
- `task-master validate-dependencies` - Check for dependency issues
- `task-master expand --all --research` - Expand all eligible tasks with research

### Configuration
- `task-master models --setup` - Configure AI models interactively
- `task-master models` - View current model configuration

### Key Rules
1. **NEVER manage tasks manually** - always use Taskmaster commands or MCP tools
2. **ALWAYS use `--research` flag** when doing research, analysis, or making decisions
3. **Mark tasks in-progress** before starting work
4. **Mark tasks complete immediately** after finishing
5. **Use `update-subtask`** to log implementation notes during development

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

---

## Meta-Agent: Automated Codebase Refinement

Meta-agent is a Python CLI tool that analyzes codebases against their PRD and generates prioritized improvement plans. It uses a multi-LLM pipeline for cost-effective, high-quality analysis.

### Pipeline Architecture

Meta-agent uses a **two-LLM architecture** for intelligent context management:

```
┌─────────────────────────────────────────────────────────────────┐
│                     META-AGENT PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│  1. REPOMIX (Pack codebase)                                     │
│     └── Combines all source files into single XML file          │
│                                                                 │
│  2. OLLAMA (Local Triage - FREE)                                │
│     ├── Reads FULL codebase + PRD + all available prompts       │
│     ├── Analyzes what's relevant for the task                   │
│     └── Returns: selected files + selected prompts              │
│                                                                 │
│  3. PERPLEXITY (Paid Analysis - Reduced Context)                │
│     ├── Receives ONLY relevant files (not full codebase)        │
│     ├── Receives ONLY selected prompts                          │
│     ├── Can rewrite/customize prompts for the specific task     │
│     └── Returns: detailed analysis + task recommendations       │
│                                                                 │
│  4. PLAN WRITER                                                 │
│     └── Generates mvp_improvement_plan.md                       │
│                                                                 │
│  5. IMPLEMENTATION REPORT                                       │
│     └── Structured JSON for Claude Code to implement            │
└─────────────────────────────────────────────────────────────────┘
```

**Why Two LLMs?**
- **Ollama (local)**: Analyzes the FULL codebase for FREE - no token costs
- **Perplexity (paid)**: Only receives 10-30% of the context - reduced costs, better focus
- **Result**: High-quality analysis at a fraction of the cost of sending everything to a paid API

### Refinement Modes

#### 1. Profile-Based Mode (Simple)
```bash
metaagent refine --profile automation_agent
```
Runs predefined analysis stages from `config/profiles.yaml`. Deterministic, no triage.

#### 2. Smart Mode with Ollama Triage
```bash
metaagent refine --smart
```
- Ollama analyzes full codebase against PRD
- Selects which prompts to run
- Sends only relevant files to Perplexity

#### 3. Feature-Focused Mode (Recommended)
```bash
metaagent refine --smart --focus "add user authentication"
```
- Ollama selects relevant FILES for the feature
- Perplexity receives reduced context + prompt library
- Perplexity selects AND rewrites prompts for your specific feature
- Best for adding specific features

#### 4. Iterative Loop Mode (Full Automation)
```bash
metaagent refine --smart --focus "add user authentication" --loop
```
- Runs feature-focused mode in a loop
- After each iteration: analyze → implement → commit → repeat
- Continues until feature is complete or max iterations reached
- Use `--max-iterations 10` to limit

#### 5. Auto-Implement Mode
```bash
metaagent refine --smart --focus "feature" --auto-implement
```
- Automatically runs Claude Code to implement recommended changes
- Commits changes after implementation

#### 6. Grok-Powered Local Loop (NEW)
```bash
metaagent refine --local-loop docs/prd.md
```
- Uses **Grok (xAI)** as the primary evaluator instead of Perplexity
- Works directly in current repository (local-first, no workspace isolation)
- Uses **GitPython** for safer git operations
- Cycle: Analyze → Implement (Claude) → Test → Diagnose errors (Grok) → Fix → Commit
- Final Grok evaluation for PRD alignment

```bash
# Basic local loop
metaagent refine --local-loop docs/prd.md

# With options
metaagent refine --local-loop docs/prd.md --max-iterations 15 --human-approve

# Use Perplexity instead of Grok
metaagent refine --local-loop docs/prd.md --evaluator perplexity

# Dry-run mode (no commits)
metaagent refine --local-loop docs/prd.md --dry-run
```

### Essential Commands

```bash
# Basic usage (profile-based)
metaagent refine --profile automation_agent

# Smart mode with Ollama triage (recommended)
metaagent refine --smart

# Feature-focused (most useful)
metaagent refine --smart --focus "your feature description"

# Iterative until complete
metaagent refine --smart --focus "feature" --loop --max-iterations 10

# With auto-implementation
metaagent refine --smart --focus "feature" --auto-implement

# Preview without API calls
metaagent refine --profile automation_agent --dry-run

# Mock mode for testing
metaagent refine --smart --mock

# Grok-powered local loop (NEW)
metaagent refine --local-loop docs/prd.md
metaagent refine --local-loop docs/prd.md --evaluator perplexity

# List available profiles
metaagent list-profiles

# List available prompts
metaagent list-prompts

# Launch real-time dashboard
metaagent dashboard
```

### Configuration

#### Environment Variables (`.env`)
```bash
PERPLEXITY_API_KEY=pplx-xxx   # Required for analysis (or use --mock)
ANTHROPIC_API_KEY=sk-ant-xxx  # Optional - for Claude integration
GROK_API_KEY=xai-xxx          # Required for --local-loop mode (xAI)
METAAGENT_TIMEOUT=120         # Request timeout
METAAGENT_MAX_TOKENS=100000   # Max tokens per request
```

#### Prerequisites
- Python 3.10+
- Repomix: `npm install -g repomix`
- Ollama: Must be installed and running for `--smart` mode
- Perplexity API key: Required for paid analysis

### Project Structure

```
meta-agent/
├── src/metaagent/
│   ├── cli.py           # CLI entrypoint
│   ├── orchestrator.py  # Pipeline orchestration (smart mode, triage, local loop)
│   ├── ollama_engine.py # Ollama integration for local triage
│   ├── analysis.py      # Perplexity/LLM analysis engine
│   ├── grok_client.py   # Grok API client for error diagnosis (NEW)
│   ├── local_manager.py # GitPython-based repo manager (NEW)
│   ├── repomix.py       # Codebase packing
│   ├── plan_writer.py   # Plan file generation
│   └── claude_runner.py # Claude Code integration
├── config/
│   ├── prompts.yaml     # Custom prompt templates
│   ├── profiles.yaml    # Profile definitions (stages)
│   ├── loop_config.yaml # Loop mode configuration
│   └── prompt_library/  # Codebase-digest prompts
└── docs/
    └── prd.md           # Project PRD
```

### Output

After running, meta-agent generates:
- `docs/mvp_improvement_plan.md`: Human-readable improvement plan
- Console output: Implementation report with prioritized tasks
- The current Claude Code session receives structured tasks to implement

### Workflow Integration

1. **Write PRD**: Create `docs/prd.md` describing your feature/project
2. **Run Meta-Agent**: `metaagent refine --smart --focus "feature"`
3. **Review Plan**: Check generated `docs/mvp_improvement_plan.md`
4. **Implement**: Work through tasks (or use `--auto-implement`)
5. **Iterate**: Re-run to find remaining gaps

### Available Profiles

| Profile | Description |
|---------|-------------|
| `automation_agent` | For CLI tools and automation |
| `backend_service` | APIs with security review |
| `internal_tool` | Lighter analysis |
| `quick_review` | PRD alignment only |
| `full_review` | Comprehensive analysis |

### Tips

- **Use `--smart --focus` for new features**: Most effective mode
- **Use `--loop` for complex features**: Automated iteration
- **Use `--local-loop` for Grok-powered development**: Works directly in your repo
- **Always have a PRD**: Meta-agent works best with clear requirements
- **Check Ollama status**: Run `ollama list` to ensure models are available
- **Start with `--mock` or `--dry-run`**: Test your workflow without API costs
