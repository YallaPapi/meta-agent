# Architecture Layer Identification

# Architectural Analysis: Meta-Agent CLI Tool

## 1. Actual vs. Intended Architecture

### 1.1 PRD-Defined Architecture Layers
The PRD clearly defines 6 architectural layers:
1. **CLI Entrypoint** (`cli.py`) - Argument parsing, orchestrator initialization
2. **Orchestrator** (`orchestrator.py`) - Main refinement loop coordination
3. **Repomix Integration** (`repomix.py`) - Codebase packing subprocess
4. **Prompt/Profile Loading** (`prompts.py`) - Template rendering and configuration
5. **Analysis Engine** (`analysis.py`) - LLM API integration (Perplexity)
6. **Plan Writer** (`plan_writer.py`) - Improvement plan generation

### 1.2 Actual Implementation Layers
The codebase implements the intended layers but with significant additional components:
- ✅ `cli.py` - CLI entrypoint layer (implemented)
- ✅ `orchestrator.py` - Main orchestration layer (implemented)
- ✅ `repomix.py` - Codebase packing layer (implemented)
- ✅ `prompts.py` - Prompt/profile layer (implemented)
- ✅ `analysis.py` - Analysis engine layer (implemented)
- ✅ `plan_writer.py` - Plan generation layer (implemented)
- ❌ **Extra:** `claude_runner.py` - **NOT in PRD** - Claude Code subprocess integration
- ❌ **Extra:** `codebase_digest.py` - **NOT in PRD** - Alternative to Repomix
- ❌ **Extra:** `tokens.py` - **NOT in PRD** - Token estimation utilities
- ❌ **Extra:** `config.py` - Configuration management (partially implied in PRD)

## 2. Data Flow Analysis

### 2.1 Intended Data Flow (Per PRD)
```
CLI → Orchestrator → {Repomix, Prompts, Analysis} → Plan Writer → OUTPUT FILE
                                                                         ↓
                                                             Hand-off to Claude Code
```

### 2.2 Actual Data Flow (From Code)
```
CLI → Orchestrator → {Repomix/CodebaseDigest, Prompts, Analysis} → Plan Writer → FILE
                                                                                    ↓
                                                              ClaudeCodeRunner.implement()
                                                                                    ↓
                                                                        SEPARATE SUBPROCESS
```

**CRITICAL DEVIATION:** The actual implementation spawns a **separate Claude Code subprocess** instead of outputting a plan file for human handoff to the current Claude session.

## 3. Architectural Inconsistencies

### 3.1 CORE PROBLEM: Subprocess Spawning
**Location:** `claude_runner.py:141-179`
```python
def implement(self, repo_path: Path, prompt: str, plan_file: Optional[Path] = None) -> ClaudeCodeResult:
    # Build the implementation prompt
    full_prompt = self._build_prompt(prompt, plan_file)
    
    # Run Claude Code in non-interactive mode with the prompt
    result = subprocess.run([
        "claude",
        "--print",  # Non-interactive mode, output only
        "--model", self.model,
        "--max-turns", str(self.max_turns),
        "--dangerously-skip-permissions",  # Auto-approve for automation
        "-p", full_prompt,  # Pass prompt directly
    ], ...)
```

**Problem:** This creates a **separate Claude Code session** instead of returning analysis to the **current** Claude session that invoked the meta-agent.

### 3.2 Intended vs. Actual Handoff Pattern

**PRD Intention (Section 4.2, Step 8):**
> "User opens Claude Code on the repo and feeds in `mvp_improvement_plan.md` with instructions to execute tasks in order"

**Actual Implementation:**
- Meta-agent automatically spawns `claude` subprocess
- No human review opportunity
- Bypasses the current Claude session entirely

### 3.3 Configuration Layer Violations

**PRD Violation:** The `config.py` introduces settings not mentioned in PRD:
```python
# Claude Code Settings - NOT IN PRD
claude_code_timeout: int = 600
claude_code_model: str = "claude-sonnet-4-20250514"
auto_implement: bool = False  # Whether to auto-invoke Claude Code
auto_commit: bool = True      # Whether to auto-commit after implementation
```

These settings enable automation patterns **not specified in the PRD**.

### 3.4 Dual Codebase Analysis Tools

**Inconsistency:** Two competing codebase analysis tools:
- `repomix.py` - **PRD-specified** integration
- `codebase_digest.py` - **Additional** tool, not in PRD

This violates the clean architectural boundaries defined in the PRD.

## 4. Adherence to PRD Architecture

### 4.1 ✅ Compliant Areas
- **Layer separation:** Core layers are properly separated
- **Configuration management:** YAML-based prompts and profiles work as intended
- **Plan generation:** `plan_writer.py` generates proper markdown plans
- **Analysis engine:** Perplexity integration with retry logic is well-implemented
- **Mock testing:** Proper mock implementations for testing

### 4.2 ❌ PRD Violations

#### Major Violations:
1. **Automatic subprocess spawning** instead of human handoff
2. **Additional architecture layers** not specified in PRD
3. **Auto-implementation settings** that bypass human review
4. **Dual tool integration** (Repomix + Codebase Digest)

#### Minor Violations:
1. **Token estimation utilities** not specified in PRD
2. **Extended configuration** beyond PRD requirements
3. **Auto-commit/push functionality** not in original design

## 5. Specific Code Examples

### 5.1 Subprocess Spawning Issue
**File:** `src/metaagent/claude_runner.py:141-179`
```python
# THIS BREAKS THE INTENDED ARCHITECTURE
result = subprocess.run([
    "claude",
    "--print",  # Creates NEW session instead of returning to current
    "--model", self.model,
    "--max-turns", str(self.max_turns),
    "--dangerously-skip-permissions",  # Auto-approve bypasses human review
    "-p", full_prompt,
], ...)
```

### 5.2 Correct PRD Implementation Should Be:
**File:** `src/metaagent/plan_writer.py:156-172` (This is CORRECT)
```python
def _generate_instructions(self) -> str:
    """Generate instructions for using the plan with Claude Code."""
    return """---

## Instructions for Claude Code

To implement this plan, open Claude Code in the repository and use the following prompt:

```
Read docs/mvp_improvement_plan.md and implement the tasks in order of priority.
```
"""
```

### 5.3 Configuration Overreach
**File:** `src/metaagent/config.py:33-36`
```python
# THESE SETTINGS ENABLE PRD VIOLATIONS
auto_implement: bool = False  # Whether to auto-invoke Claude Code
auto_commit: bool = True      # Whether to auto-commit after implementation  
auto_push: bool = False       # Whether to auto-push after commit
```

## 6. Recommendations

### 6.1 Immediate Fixes
1. **Remove `claude_runner.py`** - This violates the core PRD architecture
2. **Disable auto-implementation** - Keep human review in the loop
3. **Stick to Repomix only** - Remove `codebase_digest.py` or make it a clear alternative
4. **Simplify configuration** - Remove settings that enable PRD violations

### 6.2 Architecture Alignment
1. **Enforce plan-file handoff pattern** as specified in PRD Section 4.2
2. **Return to human-in-the-loop** design for implementation phase
3. **Clean separation** between analysis (meta-agent) and implementation (human + Claude)

### 6.3 Preserve Valid Extensions
1. **Keep `tokens.py`** - Useful utility that doesn't break architecture
2. **Keep mock implementations** - Essential for testing
3. **Keep enhanced configuration** - But remove auto-implementation settings

The core architectural problem is that the system **bypasses the intended human review and handoff process** by automatically spawning separate Claude Code subprocesses, violating the PRD's explicit design for human oversight in the implementation phase.