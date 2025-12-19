## Meta-Agent: Automated Codebase Refinement

Meta-agent is a Python CLI tool that analyzes codebases against their PRD and generates prioritized improvement plans. It uses a multi-LLM pipeline for cost-effective, high-quality analysis.

### Pipeline Architecture

Meta-agent has **two main modes** with different architectures:

#### Loop Mode (Primary - `metaagent loop`)
```
┌─────────────────────────────────────────────────────────────────┐
│                     LOOP MODE PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  1. GROK (xAI) - PRD Parsing                                    │
│     └── Reads PRD and generates prioritized task list           │
│                                                                 │
│  2. CLAUDE (Anthropic) - Implementation                         │
│     └── Implements each task, writes code changes               │
│                                                                 │
│  3. TEST RUNNER                                                 │
│     └── Runs pytest (or custom test command) after each change  │
│                                                                 │
│  4. GROK - Error Diagnosis (if tests fail)                      │
│     └── Analyzes failures, suggests fixes                       │
│                                                                 │
│  5. GROK - Final Evaluation                                     │
│     └── Checks PRD alignment, generates completion report       │
└─────────────────────────────────────────────────────────────────┘
```

#### Refine Mode (Analysis - `metaagent refine --smart`)
```
┌─────────────────────────────────────────────────────────────────┐
│                     REFINE MODE PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  1. REPOMIX - Pack codebase into single file                    │
│  2. OLLAMA (local) - Triage: select relevant files/prompts      │
│  3. PERPLEXITY - Analysis with reduced context                  │
│  4. PLAN WRITER - Generate mvp_improvement_plan.md              │
└─────────────────────────────────────────────────────────────────┘
```

**Which to use?**
- **`loop`**: For autonomous implementation (PRD → working code)
- **`refine --smart`**: For analysis only (generates improvement plan)

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
# PRIMARY: Autonomous task loop (Grok + Claude)
metaagent loop --prd docs/prd.md                    # Run with human approval
metaagent loop --prd docs/prd.md --no-human-approve # Fully autonomous
metaagent loop --prd docs/prd.md --dry-run          # Preview without changes

# Launch GUI (no terminal needed)
metaagent --gui

# Analysis only (no implementation)
metaagent refine --profile automation_agent         # Profile-based
metaagent refine --smart                            # Ollama triage mode
metaagent refine --smart --focus "your feature"     # Feature-focused

# List available profiles/prompts
metaagent list-profiles
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

