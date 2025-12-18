# MVP Improvement Plan

**Generated:** 2025-12-15 14:44:47
**Profile:** Ollama Intelligent Triage
**Status:** Ready for implementation

---

## PRD Summary

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

*[PRD truncated for brevity]*

## Analysis Stages

### Performance Bottleneck Identification

No actual codebase files are available for performance analysis, limiting direct profiling or bottleneck identification. The PRD outlines a Python CLI meta-agent system with multiple components including orchestrator, LLM integrations, and subprocess calls, which may introduce performance risks in areas like file I/O, subprocess execution, and API request handling. Potential bottlenecks stem from sequential stage processing, large codebase packing with Repomix, and multiple LLM calls without caching or parallelism.

**Recommendations:**
- Implement profiling infrastructure and async processing for LLM calls and subprocesses to identify runtime bottlenecks early.
- Add caching mechanisms for Repomix outputs and LLM responses to avoid redundant expensive operations during iterations.
- Optimize codebase packing by implementing file filtering and compression to reduce token/context limits for analysis.
- Introduce configurable parallelism and timeout controls for all external tool integrations.


### Security Vulnerability Analysis of Codebase

No application code files were provided, so no concrete vulnerabilities in implementation logic, data handling, or request processing can be identified. However, the PRD reveals several high‑risk areas for future security issues, especially around API key handling, LLM prompt/response handling, subprocess execution (Repomix), and plan files that will be fed into external tools. A proactive security design and hardening plan is required before and during implementation.

**Recommendations:**
- Define and enforce secure patterns for handling secrets, subprocess calls, and external API interactions (Perplexity, Claude/Claude Code) as part of the initial architecture.
- Implement strict validation and sanitization of all LLM inputs/outputs and Repomix outputs, and add security-focused tests and checks (linting, type checking, dependency scanning) into the MVP pipeline.


## Implementation Tasks

### [CRITICAL] Critical Priority

- [ ] **Implement Repomix output caching** (`src/metaagent/repomix.py`)
  - Cache repomix output with SHA256 hash of repo contents in repomix.py. Skip repomix execution if cache is valid (within 5 minutes). Add cache invalidation on git changes. Store cache in .metaagent/cache/ directory.
- [ ] **Establish secure secrets and configuration handling** (`src/metaagent/config.py`)
  - Design and implement a configuration layer that reads PERPLEXITY_API_KEY, ANTHROPIC_API_KEY, and other sensitive settings strictly from environment variables, never logs their values, and avoids writing them to plan or history files. Add input validation for configuration (timeouts, max tokens) and ensure config defaults are safe (e.g., conservative token limits, non-verbose logging by default).
- [ ] **Harden Repomix subprocess execution** (`src/metaagent/repomix.py`)
  - Implement secure subprocess invocation for Repomix: avoid shell=True, sanitize and validate the --repo path, enforce timeouts, and robustly handle errors. Log only necessary metadata (exit codes, high-level messages) and ensure that no sensitive content (e.g., packed secrets) is accidentally logged. Consider a denylist/allowlist for directories and file types included in packs to avoid leaking secrets.
- [ ] **Validate and sanitize LLM prompts and responses** (`src/metaagent/analysis.py`)
  - Define a strict schema for analysis engine responses (summary, recommendations, tasks) and implement robust JSON parsing with validation and error handling. Reject or sanitize responses that include suspicious content (e.g., shell commands, code that modifies environment, sensitive file paths) before writing them into mvp_improvement_plan.md. Add guardrails to prevent prompt injection from PRD or codebase content from causing unsafe instructions to be passed downstream to Claude Code.

### [HIGH] High Priority

- [ ] **Add performance profiling to orchestrator** (`src/metaagent/orchestrator.py`)
  - Integrate cProfile or py-spy to measure execution time of each stage in orchestrator.py. Log top 5 slowest functions and total runtime per refinement cycle. Add profiling results to mvp_improvement_plan.md.
- [ ] **Add async LLM calls with timeout** (`src/metaagent/analysis.py`)
  - Replace synchronous Perplexity API calls in analysis.py with asyncio-based httpx client. Add configurable timeout (default 120s) and max_concurrent=3. Include circuit breaker pattern for repeated failures.
- [ ] **Implement stage-level parallelism** (`src/metaagent/orchestrator.py`)
  - Modify orchestrator.py to run independent stages concurrently using asyncio.gather(). Independent stages: architecture_sanity and test_suite_mvp can run after alignment_with_prd completes.
- [ ] **Secure plan writer and Claude Code handoff** (`src/metaagent/plan_writer.py`)
  - Design the mvp_improvement_plan.md format to avoid executable code or shell snippets where possible, and clearly scope Claude Code instructions to only modify the target repo. Add sanitization for all interpolated content (PRD excerpts, analysis summaries, tasks) to avoid embedding malicious content that could trick downstream tools. Document safe operational procedures for developers when using the generated plan.
- [ ] **Implement secure logging and history management** (`src/metaagent/orchestrator.py`)
  - Create a history log mechanism that stores high-level analysis context without secrets, raw API keys, or other sensitive payloads. Redact or hash any potentially sensitive strings before logging. Ensure log rotation or size limits are in place and document recommended log levels for development vs. production.

### [MEDIUM] Medium Priority

- [ ] **Optimize prompt template rendering** (`src/metaagent/prompts.py`)
  - Profile prompts.py template rendering with large {{code_context}}. Implement token truncation with warning when exceeding METAAGENT_MAX_TOKENS. Add file filtering config to exclude tests/docs from repomix input.
- [ ] **Add YAML parsing performance test** (`src/metaagent/config.py`)
  - Benchmark config/prompts.yaml and profiles.yaml loading in config.py. Cache parsed configs with file modification watchers. Test with 100+ prompts to ensure <100ms load time.
- [ ] **Add dependency and supply-chain security checks** (`pyproject.toml`)
  - Specify explicit versions and constraints for critical dependencies (HTTP clients, YAML parser, CLI libraries) in pyproject.toml. Integrate automated tools (e.g., pip-audit, safety, or similar) into the development workflow to detect known vulnerabilities. Avoid unsafe YAML loading (use safe loaders) and review third-party libraries used for CLI and LLM interaction for security posture.
- [ ] **Introduce security-focused tests and type checking** (`tests/test_orchestrator.py`)
  - Add tests that cover error paths and edge cases for config loading, Repomix execution, and analysis response parsing, ensuring they behave safely under malformed input or hostile content. Integrate static analysis (e.g., bandit), type checking (mypy/pyright), and linting into the test suite to catch common Python security and robustness issues early.

### [LOW] Low Priority

- [ ] **Document secure usage and threat model** (`README.md`)
  - Extend the PRD and README with a concise threat model for the meta-agent, outlining assumptions (e.g., trusted vs. untrusted repos/PRDs), identified risks (prompt injection, secret leakage, dangerous code suggestions), and mitigations. Provide operational guidance for developers on safely running the tool on third-party codebases and handling generated plans.


---

## Instructions for Claude Code

To implement this plan, open Claude Code in the repository and use the following prompt:

```
Read docs/mvp_improvement_plan.md and implement the tasks in order of priority.
For each task:
1. Understand the requirement
2. Make the necessary code changes
3. Run relevant tests
4. Mark the checkbox as complete when done

Start with the highest priority tasks first.
```

### Implementation Notes

- Work through tasks systematically, starting with Critical/High priority
- Run tests after each significant change
- Commit changes incrementally with descriptive messages
- If a task is unclear, review the relevant stage summary above for context
