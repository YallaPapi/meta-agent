# PRD: Grok-Powered Local Development Loop

## Overview

Upgrade the existing autonomous development loop to use **Grok as the primary evaluator** and implement a **local-first approach** using GitPython for safer, more Pythonic git operations. This enhancement transforms the agent to work directly in the current repository without workspace isolation.

## Background

The current implementation (from previous PR) uses:
- Perplexity for error diagnosis and evaluation
- Subprocess-based git operations in `workspace_manager.py`
- `--autodev` flag for loop mode
- Workspace isolation approach

This upgrade changes to:
- **Grok API** as the default evaluator (configurable)
- **GitPython** library for safer git operations
- `--loop "path/to/prd.txt"` as the primary entry point
- Local-first approach (work directly in current repo)

## Core Requirements

### 1. Grok API Client (`grok_client.py`)

Create a new Grok API client module:

```python
# Required functionality:
- query_grok(messages, model="grok-4", temperature=0.3, max_tokens=4096)
- Support for GROK_API_KEY environment variable
- Timeout handling (120s default)
- Error handling with clear messages
- Rate limiting awareness
```

**API Endpoint:** `https://api.x.ai/v1/chat/completions`

**Authentication:** Bearer token via `GROK_API_KEY`

### 2. Configuration Updates

Update `config/loop_config.yaml`:

```yaml
evaluator:
  default: "grok"  # grok or perplexity
  grok:
    model: "grok-4"
    temperature: 0.3
    max_tokens: 4096
    timeout: 120
  perplexity:
    model: "llama-3.1-sonar-large-128k-online"

loop:
  max_iterations: 15
  human_approve: true
  test_command: "pytest -q"
  branch_prefix: "meta-loop"
  commit_per_task: true
  dry_run: false
```

Update `src/metaagent/config.py`:
- Add `EvaluatorConfig` dataclass
- Support for evaluator selection (grok/perplexity)
- Grok-specific settings

### 3. Local Repository Manager (`local_manager.py`)

Create a new GitPython-based local manager:

```python
class LocalRepoManager:
    """Manages local git operations using GitPython."""

    def __init__(self, config: LoopConfig):
        # Use git.Repo(os.getcwd()) - current directory
        # Create branch: meta-loop-YYYYMMDD-HHMM

    def create_branch(self) -> str:
        # Create and checkout new branch
        # Never touch main/master

    def commit_changes(self, message: str) -> Optional[str]:
        # git add -A
        # git commit if there are changes
        # Return commit hash

    def run_tests(self, command: str = "pytest -q") -> TestResult:
        # Run test command
        # Capture stdout/stderr
        # Return structured result

    def get_current_branch(self) -> str:
        # Return current branch name

    def has_uncommitted_changes(self) -> bool:
        # Check for staged/unstaged changes
```

**Dependencies:** Add `GitPython` to `pyproject.toml`

### 4. CLI Enhancement

Add new `--loop` flag to CLI:

```bash
# Primary usage
metaagent refine --loop "docs/prd.md"

# With options
metaagent refine --loop "docs/prd.md" --max-iterations 15
metaagent refine --loop "docs/prd.md" --human-approve
metaagent refine --loop "docs/prd.md" --dry-run
metaagent refine --loop "docs/prd.md" --evaluator grok
metaagent refine --loop "docs/prd.md" --evaluator perplexity
```

**Flags:**
- `--loop PATH`: Path to PRD file (triggers local loop mode)
- `--max-iterations N`: Maximum loop iterations (default: 15)
- `--human-approve`: Require human approval between tasks
- `--dry-run`: Preview without making changes
- `--evaluator [grok|perplexity]`: Choose evaluator (default: grok)
- `--branch-prefix PREFIX`: Custom branch prefix (default: meta-loop)

### 5. Orchestrator: `run_local_loop()` Method

Add new method to `Orchestrator`:

```python
def run_local_loop(self, prd_path: str) -> AutonomousLoopResult:
    """
    Run autonomous development loop on local repository.

    Flow:
    1. Read PRD from file
    2. Initialize LocalRepoManager (creates branch)
    3. Pack current repo with Repomix
    4. Generate task plan (using existing analysis)
    5. For each task:
       a. Implement task (Claude)
       b. Commit changes
       c. Run tests
       d. If tests fail:
          - Pack repo + errors
          - Send to Grok for diagnosis
          - Get fix prompt
          - Apply fix (Claude)
          - Repeat until pass or max retries
    6. Final Grok evaluation of PRD alignment
    7. Return summary result
    """
```

### 6. Grok Integration Points

Replace Perplexity with Grok for:

1. **Error Diagnosis:**
   ```python
   def diagnose_with_grok(repo_xml: str, task: dict, errors: str) -> str:
       """Send failure context to Grok, get fix prompt."""
       prompt = f"""
       Task: {task['description']}
       Errors: {errors}
       Repository Context: {repo_xml[:50000]}  # Truncate if needed

       Diagnose the root cause and write a precise, actionable prompt
       for Claude to fix this issue. Be specific about file paths and
       exact changes needed.
       """
       return query_grok([{"role": "user", "content": prompt}])
   ```

2. **Final PRD Evaluation:**
   ```python
   def evaluate_completion_with_grok(repo_xml: str, prd_text: str) -> dict:
       """Final evaluation of PRD alignment."""
       prompt = f"""
       PRD Requirements:
       {prd_text}

       Current Implementation:
       {repo_xml[:50000]}

       Evaluate:
       1. Are all PRD requirements implemented?
       2. Code quality assessment
       3. Any remaining gaps?
       4. Overall completion percentage

       Return JSON: {{"approved": bool, "completion_pct": int, "gaps": [], "assessment": str}}
       """
       return query_grok([{"role": "user", "content": prompt}])
   ```

### 7. Windows Compatibility

Ensure all code is Windows-compatible:
- Use `os.path.join()` for paths
- Use `pathlib.Path` where appropriate
- Subprocess commands as lists, not strings
- Handle CRLF line endings
- Use `python-dotenv` for env vars

### 8. New Prompts

Create `config/prompt_library/meta_grok_diagnosis.md`:
- Template for error diagnosis requests to Grok

Create `config/prompt_library/meta_grok_evaluation.md`:
- Template for final PRD evaluation requests to Grok

### 9. Testing Requirements

**Unit Tests:**
- `tests/test_grok_client.py`: Mock API tests
- `tests/test_local_manager.py`: GitPython operations

**Integration Tests:**
- `tests/test_local_loop_integration.py`: Full loop with mocks

### 10. Environment Variables

Update `.env.example`:
```bash
# Required for Grok evaluator
GROK_API_KEY=xai-xxx

# Existing
ANTHROPIC_API_KEY=sk-ant-xxx
PERPLEXITY_API_KEY=pplx-xxx
```

## Implementation Order

1. Add GitPython dependency to pyproject.toml
2. Create `grok_client.py`
3. Update config with evaluator settings
4. Create `local_manager.py` with GitPython
5. Add `--loop` CLI flag and options
6. Implement `run_local_loop()` in orchestrator
7. Create Grok prompt templates
8. Add unit tests
9. Add integration tests
10. Update documentation

## Success Criteria

- [ ] `metaagent refine --loop docs/prd.md` works end-to-end
- [ ] Grok API integration functional
- [ ] GitPython-based git operations work on Windows
- [ ] Branch creation/commits work correctly
- [ ] Test execution and error capture works
- [ ] Error diagnosis via Grok produces useful fix prompts
- [ ] Final evaluation provides completion assessment
- [ ] All new tests pass
- [ ] Backward compatible with existing `--autodev` mode

## Non-Goals

- Replacing the existing `--autodev` mode (keep both)
- Remote repository operations (local only)
- Multi-repo support
- CI/CD integration (future enhancement)
