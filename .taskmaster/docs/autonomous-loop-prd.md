# PRD: Autonomous Development Loop for Meta-Agent

**Project:** Upgrade Meta-Agent to Full Autonomous Development Loop
**Date:** 2025-12-18
**Status:** New Feature

---

## Executive Summary

Transform the current meta-agent from a one-shot analysis + planning tool into a **fully autonomous, closed-loop software development agent** that:

1. Takes a repo + PRD
2. Analyzes it deeply (current strength)
3. Generates a prioritized task plan
4. **Autonomously implements tasks via Claude**
5. Runs tests
6. Captures errors/output
7. Packs updated repo with Repomix
8. Sends to Perplexity for evaluation + custom fix/optimization prompts
9. Feeds those back to Claude to apply fixes
10. **Loops until complete or max iterations reached**

---

## Core Principles

- Build on existing strengths: Keep current analysis stages, orchestrator, Repomix integration, Perplexity profiles.
- Add implementation loop without breaking current one-shot mode.
- Safety first: Human approval gates, max iterations, git branching, dry-run support.
- No infinite loops or unchecked API calls.

---

## New Features to Implement

### 1. New CLI Mode: `--loop` (or `--autodev`)

- Runs full autonomous cycle
- Optional flags: `--max-iterations 10`, `--human-approve`, `--dry-run`
- Backward compatible with existing one-shot mode

### 2. Git Workspace Management

- Clone repo to temporary workspace (or use existing)
- Create branch: `meta-agent-loop/{timestamp}`
- Commit after each successful Claude implementation step
- Support for working on existing repo or cloned workspace

### 3. Claude Implementation Stage

After plan generation, break tasks into individual work items. For each task:
- Send to Claude (Anthropic API) with full context (plan, current files via Repomix XML if needed)
- Request code changes + file paths
- Apply changes to workspace (write files, respect deletions)
- Git commit with meaningful message

### 4. Automated Testing

After changes:
- Run test command (default `pytest`, configurable in config.yaml)
- Capture stdout/stderr + exit code
- On failure: Extract error traceback

### 5. Error → Perplexity → Custom Prompt Loop

On test failure (or manual trigger):
- Repack workspace with Repomix
- Send to Perplexity with new profile: "error_analysis" or "fix_generator"
  - Input: Repomix XML + test errors + failing task
  - Output: Diagnosis + customized prompt for Claude to fix
- Feed Perplexity's suggested prompt directly to Claude as next message

### 6. Success Criteria & Termination

- Success: All tests pass + final Perplexity evaluation confirms PRD alignment
- Stop conditions: Max iterations, all tasks done, human abort

---

## Implementation Requirements

### Step 1: Add Config Extensions

Create loop configuration in `config.yaml` or new `loop_config.yaml`:

```yaml
loop:
  enabled: false
  max_iterations: 10
  human_approve: true
  dry_run: false
  test_command: "pytest -q"
  claude_model: "claude-sonnet-4-20250514"
  commit_per_task: true
```

### Step 2: Extend Orchestrator

Modify `orchestrator.py`:
- Add `--loop` CLI flag
- New method `run_autonomous_loop()` that:
  1. Generates plan (existing)
  2. Breaks into tasks
  3. Loops:
     - Pick next task
     - Send to Claude impl agent
     - Apply changes
     - Git commit
     - Run tests
     - If fail → Perplexity error analysis → custom prompt → back to Claude
     - If pass → next task
  4. Final evaluation stage

### Step 3: Add Claude Implementation Client

New file `claude_impl.py`:
- Function `apply_task_to_repo(task_description: str, workspace_path: str, repomix_xml: str = None) -> dict`
- Build prompt with task + context
- Call Anthropic API
- Parse suggested file changes (path + content)
- Write files
- Return success + commit message

### Step 4: Add Perplexity Error Fix Profile

Add to prompts/profiles:
```yaml
error_fix:
  system: "You are a senior debugging engineer..."
  user_template: |
    Repo XML: {repomix}
    Failing task: {task}
    Test errors: {errors}

    Diagnose the root cause and write a precise prompt for Claude to fix it.
```

### Step 5: Add Git + Test Utilities

New `workspace_manager.py`:
- `clone_repo(source: str, target: str) -> str`
- `create_branch(workspace: str, branch_name: str) -> None`
- `apply_changes(workspace: str, changes: list) -> None`
- `commit(workspace: str, message: str) -> str`
- `run_tests(workspace: str, command: str) -> tuple[bool, str]` - returns (passed, output)

### Step 6: Safety Controls

- After each cycle: Print summary + wait for Enter if `human_approve=true`
- Log all API inputs/outputs
- Token tracking and budget limits
- Maximum iteration enforcement
- Graceful termination on errors

### Step 7: Testing Strategy

- Add integration tests: Mock Claude/Perplexity responses
- Test full loop on tiny sample repo (e.g., "hello world" → add function)
- Verify git history, commits, test runs
- Test error recovery and retry logic

---

## Backward Compatibility

- Existing `--analyze` mode unchanged
- Existing `--smart` and `--focus` modes unchanged
- New loop mode is optional and additive

---

## Success Criteria

After implementation, run on a small test repo + simple PRD and demonstrate:
1. Plan generated
2. First task implemented
3. Test fails → Perplexity diagnoses → Claude fixes → test passes
4. Git history shows proper commits
5. Final evaluation confirms PRD alignment

---

## Technical Notes

### File Structure After Implementation

```
meta-agent/
├── src/metaagent/
│   ├── cli.py                 # Extended with --loop flags
│   ├── orchestrator.py        # Extended with run_autonomous_loop()
│   ├── claude_impl.py         # NEW: Claude implementation client
│   ├── workspace_manager.py   # NEW: Git and test utilities
│   └── ...
├── config/
│   ├── loop_config.yaml       # NEW: Loop-specific configuration
│   └── ...
└── tests/
    ├── test_claude_impl.py    # NEW
    ├── test_workspace_manager.py  # NEW
    └── test_loop_integration.py   # NEW
```

### API Dependencies

- Anthropic API (Claude) - Required for implementation
- Perplexity API - Required for error analysis (existing)
- Git - Required for workspace management
