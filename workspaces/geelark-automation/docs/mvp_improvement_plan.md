# MVP Improvement Plan

**Generated:** 2025-12-16 17:53:17
**Profile:** Feature Focus: implement two-LLM pipeline for Instagram DM conver
**Status:** Ready for implementation

---

## PRD Summary

# PRD: Instagram DM Bot - OF Conversion Funnel

## Overview

Automated Instagram DM bot that converts followers into OnlyFans subscribers through natural, human-like conversations. The bot manages multiple persona accounts and guides users through a proven conversion funnel.

## Primary Goal

**Prove human authenticity → Build rapport → Get them off Instagram → Convert to OnlyFans subscriber**

The bot must:
1. Appear as a legitimate, real person (not a bot or fake account)
2. Handle the fact that users may follow multiple similar-looking accounts (clones)
3. Guide conversations naturally toward the OF subscription link
4. Handle objections gracefully without being pushy

---

## Conversation Funnel Stages


*[PRD truncated for brevity]*

## Analysis Stages

### Feature: Two-LLM pipeline where LLM1 analyzes conversation 

**Core Functionality:** Two-LLM pipeline where LLM1 analyzes conversation stage from user message + history + current stage, outputs JSON with next_stage/detected_intent/etc, then LLM2 generates persona-appropriate response for that stage with photo_mood/send_photo flags.

**Suggested Additions:** 5 additional features recommended
  - Basic conversation state model with JSON schema matching PRD exactly
  - Config loading system for personas/OF links/photo paths
  - Simple LLM client wrapper (Ollama/local inference as specified)
  - Main orchestrator skeleton that can process one message end-to-end
  - Logging framework for debugging conversations

**Affected Files:** dm_bot.py, stage_analyzer.py, response_generator.py, conversation_state.py, persona_config.py, config.yaml

**Architecture Notes:** Follow exact PRD file structure. Two-LLM pipeline must be sequential (LLM1 -> LLM2). State persistence per user_id required. No keyword matching allowed - pure LLM analysis.

**Customized Analysis Prompts:** 3 prompts tailored to this feature
  - architecture_layer_identification: Critical for setting up correct scaffold structure matching ...
  - alignment_with_prd: Ensures scaffold matches PRD exactly before core implementat...
  - architecture_design_pattern_identification: Guides proper separation of concerns in core orchestrator fr...

**Recommendations:**
- Consider adding: Basic conversation state model with JSON schema matching PRD exactly
- Consider adding: Config loading system for personas/OF links/photo paths
- Consider adding: Simple LLM client wrapper (Ollama/local inference as specified)
- Consider adding: Main orchestrator skeleton that can process one message end-to-end
- Consider adding: Logging framework for debugging conversations
- Analysis focus: Critical for setting up correct scaffold structure matching PRD file layout exactly
- Analysis focus: Ensures scaffold matches PRD exactly before core implementation begins
- Analysis focus: Guides proper separation of concerns in core orchestrator from day one


## Implementation Tasks

### [CRITICAL] Critical Priority

- [ ] **Create project scaffold with all PRD files** (`geelark-automation/ (all files)`)
  - Create empty Python files matching exact PRD structure: dm_bot.py, stage_analyzer.py, response_generator.py, funnel_stages.py, conversation_state.py, persona_config.py, response_variety.py, delay_calculator.py, photo_manager.py, metrics.py, errors.py. Add __init__.py files for packages. Create config.yaml with placeholder persona/OF_link.
- [ ] **Implement basic config loading** (`persona_config.py`)
  - In persona_config.py: Load config.yaml with YAML library. Define Persona dataclass matching PRD schema. Add load_persona() function that returns Persona object. Test with sample Mia persona.
- [ ] **Create conversation state model** (`conversation_state.py`)
  - In conversation_state.py: Define ConversationState dataclass/Pydantic model matching EXACT PRD JSON schema (user_id, history, funnel_stage, location_mentioned, etc). Implement load_state(user_id), save_state(state), get_or_create_state(user_id) functions using JSON files (user_id.json).
- [ ] **Skeleton LLM client wrapper** (`llm_client.py`)
  - Create llm_client.py with OllamaClient class. Implement call_llm(prompt: str, model: str) -> str. Add parse_json_response(response: str, schema: dict) -> dict with basic error handling. Support both stage_analyzer and response_generator schemas.
- [ ] **Main orchestrator skeleton** (`dm_bot.py`)
  - In dm_bot.py: Implement process_message(user_id: str, user_message: str) -> dict. Skeleton: load_state -> call stage_analyzer -> update_state -> call response_generator -> return response dict. No real LLM calls yet - mock outputs matching PRD JSON schemas.

### [HIGH] High Priority

- [ ] **Define funnel stages enum** (`funnel_stages.py`)
  - In funnel_stages.py: Create FunnelStage Enum with all 9 stages exactly as PRD (initial_response, small_talk, etc). Add get_stage_guidelines(stage: FunnelStage) -> str that returns PRD stage description/examples for LLM prompts.
- [ ] **Add requirements.txt and basic setup** (`requirements.txt, pyproject.toml`)
  - Create requirements.txt with pydantic, pyYAML, ollama, dataclasses-json. Add setup.py or pyproject.toml. Create basic run.py that imports dm_bot and prints 'Scaffold ready'.

### [MEDIUM] Medium Priority

- [ ] **Basic logging setup** (`dm_bot.py`)
  - Configure structured logging in dm_bot.py for conversation_id, user_id, stage transitions, LLM calls. Log all inputs/outputs to JSON files for debugging.


---

## Instructions for Claude Code

### IMPORTANT: Use Taskmaster for Implementation

**You MUST use Taskmaster to implement these tasks.** Do NOT manage tasks manually.

To import and implement the tasks:

```bash
# First, import the tasks file into Taskmaster
task-master parse-prd .meta-agent-tasks.md --append

# Work through tasks using Taskmaster:
task-master list                    # See all tasks
task-master next                    # Get next task to work on
task-master set-status --id=<id> --status=in-progress
task-master set-status --id=<id> --status=done
```

### Task Workflow

1. Import tasks into Taskmaster using `parse-prd --append`
2. Use `task-master next` to get the highest priority task
3. Mark task as `in-progress` before starting work
4. Implement the task following the description
5. Run relevant tests
6. Mark task as `done` when complete
7. Commit changes after completing related tasks

### Implementation Notes

- Work through tasks systematically, starting with Critical/High priority
- Run tests after each significant change
- Commit changes incrementally with descriptive messages
- If a task is unclear, review the relevant stage summary above for context
