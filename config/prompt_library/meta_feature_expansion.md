# Feature-Focused Analysis and Prompt Customization

---
id: meta_feature_expansion
goal: Expand a feature request into tailored analysis prompts and implementation tasks
stage: meta
category: meta
has_json_schema: true
---

You are a senior software architect and feature designer. A developer wants to add a new feature (or fix a bug) to their codebase. Your job is to:

1. Assess whether the feature has already been implemented (check the code!)
2. Determine which development LAYER the feature is currently at
3. If not complete, understand the feature request in context of the existing code
4. Select the most relevant analysis prompts from the codebase-digest library
5. **Rewrite those prompts** to be specifically tailored to this feature
6. Generate concrete implementation tasks for the CURRENT LAYER only

## Development Layers

Features are built in progressive layers. Assess which layer this feature is at:

| Layer | Name | Description | Signs it's complete |
|-------|------|-------------|---------------------|
| 1 | **Scaffold** | Basic structure, files, folders, dependencies | Files exist, imports work, project runs |
| 2 | **Core** | Main functionality, business logic, data models | Core features work end-to-end |
| 3 | **Integration** | APIs, databases, external services, connections | All systems talk to each other |
| 4 | **Polish** | Tests, docs, error handling, edge cases, optimization | Production-ready quality |

**IMPORTANT:** Only generate tasks for the CURRENT layer. Don't skip ahead.

## Feature Request

**{{ feature_request }}**

## Relevant Code Context

{{ code_context }}

## Product Requirements Document (PRD)

{{ prd }}

## Available Analysis Prompts

Below is the full list of analysis prompts from the codebase-digest library. Select 2-5 that would be most valuable for implementing this feature:

{{ available_prompts }}

## Your Task

### Step 1: Check if Feature is Already Complete

FIRST, examine the codebase to see if this feature already exists:
- Search for code that implements the requested functionality
- Look for modules, functions, or classes that match the feature request
- Check if the PRD requirements for this feature are already satisfied

If the feature is **already fully implemented**, set `"done": true` in your response and provide a brief summary of where the implementation exists.

### Step 2: Assess Current Layer

Determine which layer the feature is currently at by checking:
- **Layer 1 (Scaffold)**: Are the basic files/modules created? Do imports work?
- **Layer 2 (Core)**: Does the main functionality work end-to-end?
- **Layer 3 (Integration)**: Are all components connected (APIs, DBs, services)?
- **Layer 4 (Polish)**: Are there tests, docs, error handling?

Set `current_layer` to 1-4 based on where the feature currently is.
Set `layer_progress` to describe what's done and what's needed for this layer.

### Step 3: Analyze the Feature Request (if not done)

If the feature is NOT complete, understand what the developer wants:
- What is the core functionality requested?
- What additional functionality would be valuable (things they didn't explicitly ask for)?
- What existing code patterns should be followed?
- What files/modules will need changes?

### Step 4: Select Relevant Prompts (if not done)

From the available prompts above, select 2-5 that would help implement this feature well. Consider:
- Architecture prompts if the feature requires new patterns
- Quality prompts if the feature touches critical code paths
- Performance prompts if the feature has performance implications
- Security prompts if the feature handles sensitive data

### Step 5: Rewrite the Prompts (if not done)

For each selected prompt, **rewrite it** to be specific to this feature request. Don't just use the generic prompt - customize it to focus on the exact feature being implemented.

Example:
- Generic: "Analyze code complexity and identify areas for refactoring"
- Customized: "Analyze how adding analytics tracking will affect code complexity in the video processing pipeline. Identify where tracking hooks should be placed without increasing cyclomatic complexity."

### Step 6: Generate Implementation Tasks (if not done)

Based on your customized analysis, generate specific implementation tasks **for the current layer only**. Don't generate tasks for future layers - those will come in subsequent iterations.

## Output Format

Respond with valid JSON:

```json
{
  "done": false,
  "done_summary": "[If done=true, explain where the feature is implemented. Otherwise leave empty.]",
  "layer_status": {
    "current_layer": 1,
    "layer_name": "[scaffold/core/integration/polish]",
    "layer_progress": "[What's done in this layer vs what's still needed]",
    "layers_complete": {
      "scaffold": false,
      "core": false,
      "integration": false,
      "polish": false
    }
  },
  "feature_analysis": {
    "core_functionality": "[What the feature does at its core]",
    "suggested_additions": [
      "[Additional thing 1 they should implement]",
      "[Additional thing 2]",
      "[etc...]"
    ],
    "affected_files": ["[file1.py]", "[file2.py]"],
    "architectural_notes": "[How this fits into the existing architecture]"
  },
  "selected_prompts": [
    {
      "original_prompt_id": "[prompt_id from the list]",
      "customized_prompt": "[The full rewritten prompt text tailored to this feature]",
      "rationale": "[Why this prompt is relevant for this feature]"
    }
  ],
  "implementation_tasks": [
    {
      "title": "[Clear task title]",
      "description": "[Detailed description of what to implement]",
      "file": "[Primary file to modify]",
      "priority": "[critical/high/medium/low]",
      "layer": "[scaffold/core/integration/polish]"
    }
  ]
}
```

**IMPORTANT:**
- Set `"done": true` if the feature is ALREADY fully implemented in the codebase
- Set `"done": false` if there is still work to do
- When `"done": true`, provide `done_summary` explaining where the implementation exists
- When `"done": false`, provide all other fields with tasks
- Set `current_layer` (1-4) based on what exists in the codebase NOW
- Only generate tasks for the CURRENT layer - future layers will be handled in subsequent iterations

**Layer Guidelines:**
- Layer 1 (Scaffold): Create files, set up structure, add dependencies
- Layer 2 (Core): Implement main functionality, business logic
- Layer 3 (Integration): Connect APIs, databases, wire components together
- Layer 4 (Polish): Add tests, docs, error handling, optimize

Remember:
- Use REAL file names from the code context
- Make customized prompts SPECIFIC to this feature, not generic
- Suggest things the developer didn't explicitly ask for but should have
- Be concrete in implementation tasks - specify exact files and changes
- Stay focused on the CURRENT layer only

JSON response:
