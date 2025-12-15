# Codebase Analysis and Prompt Selection

**Objective:** Analyze the codebase against its PRD and select the most relevant analysis prompts to run.

**Context:** You are a senior software architect reviewing a codebase. Your job is to:
1. Understand the current state of the codebase
2. Compare it against the Product Requirements Document (PRD)
3. Identify the most pressing issues or gaps
4. Select which analysis prompt(s) would be most valuable to run next

## Available Analysis Prompts

Select from ANY of the following prompts based on what the codebase needs:

{{ available_prompts }}

## Instructions

1. Review the codebase structure and contents provided
2. Review the PRD requirements
3. Identify the most critical gaps, issues, or areas needing analysis
4. Select 1-5 prompts that would provide the most value for THIS specific codebase
5. If the codebase is complete and meets all PRD requirements, respond with "done": true

**Selection Criteria:**
- Choose prompts that address the ACTUAL issues you see in the code
- Don't just pick generic prompts - be specific to this codebase's needs
- Consider what stage of development the project is in
- Prioritize prompts that will find actionable improvements

## Expected Output

Respond with JSON in this exact format:

```json
{
  "assessment": "Brief 2-3 sentence assessment of current codebase state and main issues",
  "priority_issues": ["Most critical issue 1", "Issue 2", "Issue 3"],
  "selected_prompts": ["prompt_id_1", "prompt_id_2", "prompt_id_3"],
  "reasoning": "Why these specific prompts were selected for this codebase",
  "done": false
}
```

If the codebase is complete and meets all requirements:
```json
{
  "assessment": "The codebase meets PRD requirements and is production-ready",
  "priority_issues": [],
  "selected_prompts": [],
  "reasoning": "No further analysis needed - codebase is complete",
  "done": true
}
```
