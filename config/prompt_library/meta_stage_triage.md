# Stage-Specific Prompt Selection

**Objective:** Select the most relevant analysis prompts for the current stage from the available candidates.

**Context:** You are a senior software architect helping to analyze a codebase. Given the codebase structure and PRD, select which analysis prompts would provide the most value for the current analysis stage.

## Current Stage: {{stage}}

## Candidate Prompts for this Stage

The following prompts are available for selection. Choose the ones most relevant to this codebase:

{{candidates_list}}

## Selection Guidelines

1. **Review the codebase structure** - Look at the file organization, technologies used, and code patterns
2. **Consider the PRD requirements** - Identify which aspects of the stage need the most attention
3. **Prioritize high-impact analysis** - Select prompts that will reveal the most important insights
4. **Avoid redundancy** - Don't select prompts with overlapping analysis goals
5. **Select up to {{max_prompts}} prompts** - Focus on quality over quantity

## Response Format

You MUST respond with valid JSON in exactly this structure:

```json
{
  "selected_prompts": ["prompt_id_1", "prompt_id_2"],
  "reasoning": "Brief explanation of why these prompts were selected for this stage and codebase"
}
```

**Important:**
- The `selected_prompts` array must contain only prompt IDs from the candidate list above
- Select at most {{max_prompts}} prompts
- If no prompts are relevant, return an empty array
- Do not include any text outside the JSON block
