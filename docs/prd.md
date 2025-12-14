# Product Requirements Document (PRD)

**Project:** Meta-Agent for Automated Codebase Refinement
**Owner:** Developer
**Date:** 2025-12-14

---

## Executive Summary

This document describes a Python CLI "meta-agent" that refines an existing codebase from v0 to MVP. The system integrates with:
- A codebase packer (Repomix) to generate a single-file representation of the repo
- An analysis/planning LLM (like Perplexity) to run prompt-library driven analyses
- A coding assistant (like Claude Code) to apply code changes and run tests

The system uses a prompt library and profile system that define stages such as:
- `alignment_with_prd`
- `architecture_sanity`
- `core_flow_hardening`
- `test_suite_mvp`

---

## 1. Problem Statement

Building an initial v0 from a PRD with Claude Code is now relatively fast, but turning that v0 into a robust, production-ready MVP still requires a lot of manual review, planning, and refactoring. Human time is spent on repetitive tasks: scanning the codebase, deciding which Codebase-Digest-style prompts to use, interpreting results, and translating them into concrete implementation work.

The goal is to create a **meta-agent system** that can automatically analyze a codebase, choose appropriate prompts from a prompt library (inspired by Codebase Digest), and orchestrate Perplexity + Claude/Claude Code to iteratively refine a project from "v0 that runs" into "viable MVP," with minimal human intervention.

---

## 2. Goals & Non-Goals

### 2.1 Goals

- **G1 – Automate post-v0 refinement:**
  Given a repository and its PRD, the system should automatically run analysis cycles, generate improvement plans, and invoke Claude Code to apply changes.

- **G2 – Prompt-driven analysis engine:**
  Maintain a configurable prompt library (similar to Codebase Digest's) and allow the system to select and apply prompts in stages (alignment, architecture, hardening, tests).

- **G3 – Tool integration:**
  Integrate at least these tools into a coherent pipeline:
  - Repomix (codebase packing)
  - Perplexity (analysis/planning)
  - Claude (review/summary)
  - Claude Code (implementation)

- **G4 – Project profiles:**
  Support multiple "profiles" (e.g., backend service, automation/agent, internal tool) that define which stages and prompts to run.

- **G5 – Transparent plans and diffs:**
  Every cycle should produce:
  - A human-readable analysis summary.
  - A prioritized task list / implementation plan.
  - Links or references to actual code changes (diffs) created by Claude Code.

### 2.2 Non-Goals

- Not trying to fully replace human review for production-critical code; human sign-off is still expected.
- Not building a generic LLM platform; this is a **developer-centric pipeline** optimized for your own workflows.
- Not designing a UI beyond a basic CLI or minimal web dashboard in v1.

---

## 3. System Architecture

### 3.1 Project Structure

```
meta-agent/
├── pyproject.toml                 # Project configuration (uv/pip compatible)
├── README.md                      # Setup and usage documentation
├── .env.example                   # Environment variable template
├── src/
│   └── metaagent/
│       ├── __init__.py
│       ├── cli.py                 # CLI entrypoint (Click/Typer)
│       ├── orchestrator.py        # Main refinement orchestration logic
│       ├── repomix.py             # Repomix subprocess integration
│       ├── prompts.py             # Prompt/profile loading and rendering
│       ├── analysis.py            # Analysis LLM integration (Perplexity)
│       ├── plan_writer.py         # Plan file generation
│       └── config.py              # Configuration management
├── config/
│   ├── prompts.yaml               # Prompt templates library
│   └── profiles.yaml              # Profile definitions
├── docs/
│   └── prd.md                     # Project PRD (this document)
└── tests/
    ├── __init__.py
    ├── test_cli.py
    ├── test_orchestrator.py
    ├── test_repomix.py
    └── test_prompts.py
```

### 3.2 Key Components

#### CLI Entrypoint (`cli.py`)
- Provides `metaagent refine --profile <profile> --repo <path>` command
- Handles argument parsing and validation
- Initializes and invokes the orchestrator

#### Orchestrator (`orchestrator.py`)
- Main refinement loop coordinator
- Loads PRD, prompts, and profile configuration
- Executes stages in order defined by profile
- Aggregates results and produces improvement plan

#### Repomix Integration (`repomix.py`)
- Runs Repomix CLI via subprocess
- Reads and returns packed codebase content
- Handles errors and timeouts

#### Prompt/Profile Loading (`prompts.py`)
- Loads `config/prompts.yaml` and `config/profiles.yaml`
- Renders prompt templates with variables:
  - `{{prd}}` - PRD content
  - `{{code_context}}` - Packed codebase
  - `{{history}}` - Previous analysis summaries
  - `{{current_stage}}` - Current stage name

#### Analysis Engine (`analysis.py`)
- Wraps Perplexity API calls
- Sends rendered prompts
- Parses structured responses (summary, recommendations, tasks)
- Provides mock mode for testing

#### Plan Writer (`plan_writer.py`)
- Generates `docs/mvp_improvement_plan.md`
- Includes PRD recap, stage summaries, and prioritized task list
- Formats tasks with checkboxes for Claude Code consumption

---

## 4. High-Level User Flows

### 4.1 Flow A: PRD to v0 (baseline - outside this system)

1. User writes a PRD for a new project and saves it as `docs/prd.md` in an empty repo.
2. User opens the repo in Claude Code.
3. User instructs Claude Code to implement v0.
4. Claude Code builds the initial implementation and passes tests.
5. User commits v0 to version control.

*(This phase is manual and outside this system, but assumed as a prerequisite.)*

### 4.2 Flow B: Meta-Agent MVP Refinement

1. User runs the meta-agent CLI:
   ```bash
   metaagent refine --profile automation_agent --repo /path/to/repo
   ```

2. Meta-agent:
   - Reads `docs/prd.md`
   - Runs Repomix on the repo to produce a packed codebase file
   - Loads the configured profile and its stages

3. Stage 1 – **PRD Alignment Analysis**:
   - Selects the `alignment_with_prd` prompt
   - Calls Perplexity with PRD + Repomix output + prompt template
   - Receives: summary of gaps vs PRD, task list to close those gaps

4. Stage 2 – **Architecture / Best Practices**:
   - Selects `architecture_sanity` and/or `best_practices_analysis` prompts
   - Calls Perplexity again
   - Receives: architecture issues, refactor suggestions, and tasks

5. Stage 3 – **Feature-specific Hardening**:
   - Runs `core_flow_hardening` (retry logic, error handling)
   - Calls Perplexity with the packed code + PRD + current history
   - Receives: detailed implementation plan for robustness

6. Stage 4 – **Test Suite MVP**:
   - Runs a testing prompt (e.g., `test_suite_mvp`)
   - Receives: list of missing tests (file names, test cases)

7. Meta-agent merges all tasks into a single `mvp_improvement_plan.md`

8. User opens Claude Code on the repo and feeds in `mvp_improvement_plan.md` with instructions to execute tasks in order

9. After implementation:
   - Meta-agent can re-run to confirm improvements and suggest final tweaks

---

## 5. Functional Requirements

### 5.1 CLI / Orchestrator

- **FR1:** Provide a CLI command `metaagent refine --profile <profile> --repo <path>`
- **FR2:** Detect and load:
  - PRD file (default `docs/prd.md`)
  - Prompt library configuration (YAML)
  - Profile configuration (mapping stages to prompts)
- **FR3:** Run Repomix on the repo to produce a packed code representation
- **FR4:** Maintain a simple "history log" for each run

### 5.2 Prompt Library & Profiles

- **FR5:** Store prompt templates in `config/prompts.yaml` including:
  - `id`
  - `goal`
  - `template`
  - `stage` or `category`
  - Optional `dependencies` or `when_to_use` hints
- **FR6:** Store profiles in `config/profiles.yaml` mapping:
  - Profile name → ordered list of stages
- **FR7:** Allow selection of prompts per stage based on profile

### 5.3 Analysis Engine (Perplexity)

- **FR8:** Construct Perplexity prompts including:
  - PRD text
  - Truncated Repomix output (within context budget)
  - Run history summary
  - Stage's prompt template
- **FR9:** Expect structured responses:
  - `summary` (what was found)
  - `recommendations`
  - `tasks` (actionable items with file references)
- **FR10:** Aggregate tasks from all stages into ordered improvement plan

### 5.4 Plan & Handoff to Claude Code

- **FR11:** Write aggregated plan to `docs/mvp_improvement_plan.md`:
  - Short recap of PRD
  - Stage summaries
  - Prioritized task list with checkboxes
- **FR12:** Provide standard instruction block for Claude Code
- **FR13:** Optionally provide separate prompt for Claude to create polished review docs

### 5.5 Iteration / Re-analysis

- **FR14:** Support re-running refinement after code changes
- **FR15:** Track whether "Must-fix" tasks are resolved

---

## 6. Non-Functional Requirements

- **NFR1:** Implementation language: Python 3.10+
- **NFR2:** All orchestration via CLI; no GUI required for v1
- **NFR3:** Configurable timeouts and max token sizes for LLM calls
- **NFR4:** Keep secrets (API keys) in environment variables
- **NFR5:** Easy to extend prompt library and profiles without changing Python code

---

## 7. Configuration Layer Design

### 7.1 config/prompts.yaml Format

```yaml
prompts:
  alignment_with_prd:
    id: alignment_with_prd
    goal: "Identify gaps between current implementation and PRD requirements"
    stage: alignment
    template: |
      You are analyzing a codebase against its PRD.

      ## PRD:
      {{prd}}

      ## Current Codebase:
      {{code_context}}

      ## Previous Analysis (if any):
      {{history}}

      Current Stage: {{current_stage}}

      Please analyze and provide:
      1. Summary of alignment gaps
      2. Missing features or incomplete implementations
      3. Prioritized task list to close gaps

      Format your response as JSON with keys: summary, recommendations, tasks

  architecture_sanity:
    id: architecture_sanity
    goal: "Review architecture for best practices and maintainability"
    stage: architecture
    template: |
      Review this codebase for architectural quality.

      ## PRD Context:
      {{prd}}

      ## Codebase:
      {{code_context}}

      Analyze:
      1. Code organization and modularity
      2. Separation of concerns
      3. Error handling patterns
      4. Dependency management

      Format your response as JSON with keys: summary, recommendations, tasks

  core_flow_hardening:
    id: core_flow_hardening
    goal: "Identify robustness improvements for core flows"
    stage: hardening
    template: |
      Analyze core flows for robustness.

      ## PRD:
      {{prd}}

      ## Codebase:
      {{code_context}}

      ## Analysis History:
      {{history}}

      Focus on:
      1. Error handling and recovery
      2. Retry logic for external calls
      3. Input validation
      4. Edge cases

      Format your response as JSON with keys: summary, recommendations, tasks

  test_suite_mvp:
    id: test_suite_mvp
    goal: "Identify critical tests needed for MVP quality"
    stage: testing
    template: |
      Review test coverage for this codebase.

      ## PRD:
      {{prd}}

      ## Codebase:
      {{code_context}}

      Identify:
      1. Missing unit tests for core functions
      2. Missing integration tests for main flows
      3. Edge cases without test coverage

      Format your response as JSON with keys: summary, recommendations, tasks
```

### 7.2 config/profiles.yaml Format

```yaml
profiles:
  automation_agent:
    name: "Automation Agent"
    description: "Profile for CLI tools and automation agents"
    stages:
      - alignment_with_prd
      - architecture_sanity
      - core_flow_hardening
      - test_suite_mvp

  backend_service:
    name: "Backend Service"
    description: "Profile for API backends and services"
    stages:
      - alignment_with_prd
      - architecture_sanity
      - core_flow_hardening
      - test_suite_mvp

  internal_tool:
    name: "Internal Tool"
    description: "Profile for internal developer tools"
    stages:
      - alignment_with_prd
      - core_flow_hardening
```

---

## 8. Milestones

### M1 – Minimal Orchestrator (MVP)
- CLI command implementation
- Repomix integration
- Single profile with 2 stages: `alignment_with_prd`, `core_flow_hardening`
- Generates basic `mvp_improvement_plan.md`
- Mock analysis function with clear interface for future LLM integration

### M2 – Full Profile + Prompt Library
- Add `architecture_sanity` and `test_suite_mvp` stages
- Config-driven prompts and profiles
- Basic run history logging
- Full Perplexity API integration

### M3 – Iteration Support
- Re-run refinement after changes
- Detect remaining gaps
- Simple rule-based logic for skipping/repeating stages

### M4 – Optional Review Generation
- Claude integration for polished review documents
- Diff tracking and reporting

---

## 9. Integrations & Dependencies

| Tool | Purpose | Integration Method |
|------|---------|-------------------|
| Repomix | Codebase packing | CLI subprocess |
| Perplexity API | Analysis/planning | HTTP API |
| Claude API | Review generation (optional) | HTTP API |
| Claude Code | Implementation execution | Plan file handoff |

---

## 10. Extension Points

The following functions should be clearly marked as extension points:

```python
def run_analysis(prompt: str) -> dict:
    """
    Extension point for LLM analysis calls.

    Args:
        prompt: Rendered prompt template

    Returns:
        dict with keys: summary, recommendations, tasks
    """
    pass

def generate_review_document(analysis_results: list, prd: str) -> str:
    """
    Extension point for generating polished review documents.

    Args:
        analysis_results: List of analysis results from all stages
        prd: Original PRD content

    Returns:
        Markdown formatted review document
    """
    pass
```

---

## 11. Environment Variables

```
PERPLEXITY_API_KEY=<your-perplexity-api-key>
ANTHROPIC_API_KEY=<your-anthropic-api-key>  # Optional, for Claude integration
METAAGENT_LOG_LEVEL=INFO
METAAGENT_TIMEOUT=120
METAAGENT_MAX_TOKENS=100000
```

---

## 12. Success Criteria

1. Running `metaagent refine --profile automation_agent --repo .` produces a valid `mvp_improvement_plan.md`
2. The plan contains actionable tasks derived from PRD alignment analysis
3. The system can be extended with new prompts/profiles via YAML configuration only
4. All core functionality works with mock analysis (no API keys required for testing)
