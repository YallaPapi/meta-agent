# Prompt Processing Architecture

This document explains how meta-agent loads, processes, and renders prompts for LLM analysis.

## Prompt Sources

Meta-agent loads prompts from two sources:

### 1. Codebase Digest Prompts (Primary)

**Location:** `config/prompt_library/*.md`

These are comprehensive analysis prompts from the [codebase-digest](https://github.com/kamilstanuch/codebase-digest) project. They provide thorough, specialized analysis for different aspects of code review.

**Characteristics:**
- Markdown format with detailed instructions
- Categorized by prefix: `architecture_*`, `quality_*`, `security_*`, etc.
- Most do NOT include JSON output schema (free-form expected output)
- Loaded with `source='markdown'` and `has_json_schema=False`

**Examples:**
- `quality_error_analysis.md` - Find errors and inconsistencies
- `architecture_layer_identification.md` - Identify architectural layers
- `security_vulnerability_analysis.md` - Security vulnerability detection

### 2. YAML Prompts (Legacy)

**Location:** `config/prompts.yaml`

These are lightweight custom prompts with built-in JSON schema. Kept for backwards compatibility.

**Characteristics:**
- YAML format with Jinja2 template syntax
- Include JSON response schema in the template
- Loaded with `source='yaml'` and `has_json_schema=True`

**Examples:**
- `alignment_with_prd` - Quick PRD alignment check
- `architecture_sanity` - Basic architecture review

## Prompt Rendering Flow

When a prompt is rendered via `Prompt.render()`, the following order is used:

```
┌─────────────────────────────────────────────────────────────┐
│  1. CONTEXT SECTIONS (what the LLM is analyzing)            │
│     ├── ## Product Requirements Document (PRD)              │
│     ├── ## Codebase                                         │
│     └── ## Previous Analysis                                │
├─────────────────────────────────────────────────────────────┤
│  2. PROMPT INSTRUCTIONS (what analysis to perform)          │
│     └── The actual prompt template content                  │
├─────────────────────────────────────────────────────────────┤
│  3. JSON SCHEMA (for markdown prompts only)                 │
│     └── ## Required Response Format                         │
│         └── {summary, recommendations, tasks}               │
└─────────────────────────────────────────────────────────────┘
```

### Why This Order?

1. **Context First**: The LLM needs to understand what it's analyzing before receiving instructions
2. **Instructions Second**: Analysis instructions reference the context provided above
3. **Schema Last**: Output format is specified after the LLM understands the task

## JSON Schema Wrapping

### Automatic Detection

When loading markdown prompts, `_parse_markdown_prompt()` checks if the prompt already includes JSON schema:

```python
has_json_schema = bool(
    re.search(r'"summary".*"recommendations".*"tasks"', content, re.DOTALL)
    or re.search(r'"assessment".*"selected_prompts".*"done"', content, re.DOTALL)
)
```

### Schema Appending

For prompts without built-in schema (`has_json_schema=False`), the standard JSON response schema is automatically appended:

```json
{
  "summary": "2-4 sentence overview of your analysis findings",
  "recommendations": ["High-level recommendation 1", "High-level recommendation 2"],
  "tasks": [
    {
      "title": "Short task title",
      "description": "Detailed description of what needs to be done",
      "priority": "critical|high|medium|low",
      "file": "path/to/relevant/file.py"
    }
  ]
}
```

### Special Cases

- **`meta_triage.md`**: Has its own JSON schema (assessment, selected_prompts, done) - NOT wrapped
- **YAML prompts**: Always have built-in schema - NOT wrapped

## Response Parsing

The `PerplexityAnalysisEngine._parse_response()` method uses multiple strategies:

1. **Code Block Extraction**: Look for ```json ... ``` blocks
2. **Brace Matching**: Find raw JSON objects `{ ... }`
3. **Fallback Parsing**: Extract bullet points and paragraphs as tasks

### Task Normalization

All tasks are normalized to ensure required fields:

```python
{
    'title': str,       # Required, max 80 chars
    'description': str, # Required
    'priority': str,    # 'critical', 'high', 'medium', 'low'
    'file': str,        # Optional file path
}
```

## Profiles and Stage Mapping

### Profiles (`config/profiles.yaml`)

Define sequences of prompts for specific use cases:

```yaml
profiles:
  prd_alignment:
    stages:
      - meta_triage
      - quality_error_analysis
      - improvement_best_practice_analysis
```

### Stage Helpers

`PromptLibrary.get_prompts_for_stage()` maps conceptual stages to prompts:

```python
DEFAULT_STAGE_PROMPTS = {
    'architecture': ['architecture_layer_identification', ...],
    'quality': ['quality_error_analysis', ...],
    'security': ['security_vulnerability_analysis'],
    ...
}
```

## Adding New Prompts

### Adding a Codebase Digest Prompt

1. Create `config/prompt_library/category_name.md`
2. Include a title as the first `# Heading`
3. The prompt will auto-load with JSON schema wrapping

### Adding a Custom YAML Prompt

1. Add to `config/prompts.yaml`
2. Include JSON schema in the template
3. The prompt loads with `has_json_schema=True`

## Debugging

Enable debug logging to see prompt rendering decisions:

```python
import logging
logging.getLogger('metaagent.prompts').setLevel(logging.DEBUG)
```

Output example:
```
DEBUG: Rendering prompt 'quality_error_analysis' (source=markdown, has_json_schema=False)
DEBUG: Appending JSON schema to prompt 'quality_error_analysis'
```
