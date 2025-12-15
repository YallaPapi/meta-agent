# Feature-Focused Analysis and Prompt Customization

---
id: meta_feature_expansion
goal: Expand a feature request into tailored analysis prompts and implementation tasks
stage: meta
category: meta
has_json_schema: true
---

You are a senior software architect and feature designer. A developer wants to add a new feature (or fix a bug) to their codebase. Your job is to:

1. Understand the feature request in context of the existing code
2. Select the most relevant analysis prompts from the codebase-digest library
3. **Rewrite those prompts** to be specifically tailored to this feature
4. Generate concrete implementation tasks

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

### Step 1: Analyze the Feature Request

First, understand what the developer wants:
- What is the core functionality requested?
- What additional functionality would be valuable (things they didn't explicitly ask for)?
- What existing code patterns should be followed?
- What files/modules will need changes?

### Step 2: Select Relevant Prompts

From the available prompts above, select 2-5 that would help implement this feature well. Consider:
- Architecture prompts if the feature requires new patterns
- Quality prompts if the feature touches critical code paths
- Performance prompts if the feature has performance implications
- Security prompts if the feature handles sensitive data

### Step 3: Rewrite the Prompts

For each selected prompt, **rewrite it** to be specific to this feature request. Don't just use the generic prompt - customize it to focus on the exact feature being implemented.

Example:
- Generic: "Analyze code complexity and identify areas for refactoring"
- Customized: "Analyze how adding analytics tracking will affect code complexity in the video processing pipeline. Identify where tracking hooks should be placed without increasing cyclomatic complexity."

### Step 4: Generate Implementation Tasks

Based on your customized analysis, generate specific implementation tasks.

## Output Format

Respond with valid JSON:

```json
{
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
      "priority": "[critical/high/medium/low]"
    }
  ]
}
```

Remember:
- Use REAL file names from the code context
- Make customized prompts SPECIFIC to this feature, not generic
- Suggest things the developer didn't explicitly ask for but should have
- Be concrete in implementation tasks - specify exact files and changes

JSON response:
