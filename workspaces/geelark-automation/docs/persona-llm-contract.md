# Persona LLM & JSON Contract Specification

## Phase 3 Review: Prompt Design & Output Validation

This document specifies the JSON contract for persona LLM responses and documents the improved prompt structure.

---

## JSON Output Contract

### Schema

```json
{
  "text": "string",
  "send_photo": boolean,
  "photo_mood": "enum",
  "goal_status": "enum"
}
```

### Field Specifications

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | N/A | The conversational reply (1-2 sentences max) |
| `send_photo` | boolean | No | `false` | Whether to attach a photo |
| `photo_mood` | enum | No | `"casual"` | Mood category for photo selection |
| `goal_status` | enum | No | `"chatting"` | Current conversation goal state |

### Enum Values

**photo_mood:**
```
happy | sad | flirty | casual | morning | night | excited | selfie
```

**goal_status:**
```
chatting | asking_location | got_location | sending_link | sent_link | rejected
```

---

## Prompt Architecture

The system prompt is now built by `prompt_builder.py` with clear separation:

### Section 1: Identity (WHO you are)
- Persona name, age, occupation
- Personality traits
- Ultimate goal

### Section 2: Style (HOW to respond)
- Texting style rules
- Message length constraints
- Tone guidelines

### Section 3: JSON Contract (WHAT to output)
- Exact schema with field types
- Enum value constraints
- Output format rules

### Section 4: Goal Progression (WHEN to do what)
- State machine for conversation goals
- Transition triggers
- Examples for each state

### Section 5: Photo Guidelines
- When to send photos (~30%)
- Mood selection criteria
- Context-based mood matching

### Section 6: Edge Cases
- Spam handling
- Rude users
- Bot accusations
- Empty messages
- Stalled conversations

---

## Validation Layer

### Location: `persona_llm.py:_parse_response()`

### Validations Performed

1. **JSON Parsing**
   - Extracts JSON from response (handles extra text)
   - Falls back to defaults on parse error

2. **Mood Validation**
   - Checks against `VALID_MOODS` enum
   - Defaults to `"casual"` if invalid

3. **Goal Validation**
   - Checks against `VALID_GOALS` enum
   - Defaults to `"chatting"` if invalid

4. **Text Validation**
   - Ensures non-empty response
   - Falls back to `"haha nice!"` if empty

### Default Response (on error)

```python
LLMResponse(
    text="haha nice! wbu?",
    send_photo=False,
    photo_mood="casual",
    goal_status="chatting",
)
```

---

## Files Modified

1. **`prompt_builder.py`** (NEW)
   - `PersonaDefinition` - Pure data class for persona
   - `PersonaPromptBuilder` - Builds prompts from persona
   - `VALID_MOODS`, `VALID_GOALS` - Centralized enum constants
   - `JSON_SCHEMA` - Schema definition string

2. **`persona_llm.py`** (UPDATED)
   - Uses `PersonaPromptBuilder` instead of inline prompt
   - Imports validation constants from `prompt_builder`
   - Added empty text validation

3. **`config.py`** (UNCHANGED)
   - `PersonaConfig` kept for backward compatibility
   - `to_system_prompt()` still exists but now deprecated

---

## Migration Path

### Current (Deprecated)
```python
from config import PersonaConfig
persona = PersonaConfig(name="Mia", ...)
prompt = persona.to_system_prompt()
```

### New (Recommended)
```python
from prompt_builder import PersonaDefinition, build_prompt_for_persona
persona = PersonaDefinition(name="Mia", ...)
prompt = build_prompt_for_persona(persona)
```

---

## Goal State Machine

```
                    ┌──────────────┐
                    │   chatting   │ (default)
                    └──────┬───────┘
                           │ ask location
                           ▼
                    ┌──────────────────┐
                    │ asking_location  │
                    └──────┬───────────┘
                           │ user mentions city
                           ▼
                    ┌──────────────┐
                    │ got_location │
                    └──────┬───────┘
                           │ transition to share
                           ▼
                    ┌──────────────┐
                    │ sending_link │
                    └──────┬───────┘
                           │ link sent
                           ▼
                    ┌──────────────┐
                    │  sent_link   │
                    └──────────────┘

At any stage:
  └──► rejected (if user declines/asks to stop)
```

---

## Error Recovery Behavior

| Scenario | LLM Instruction | Validation Fallback |
|----------|-----------------|---------------------|
| Spam input | Respond "wait what lol" | N/A |
| Invalid mood | N/A | Default to "casual" |
| Invalid goal | N/A | Keep current goal |
| Empty text | N/A | Default "haha nice!" |
| Malformed JSON | N/A | Full default response |
| Bot accusation | Deflect playfully | N/A |

---

*Generated as part of Persona LLM Contract Review (Phase 3)*
*Review prompt: `reviews/03-persona-llm-prompt-json-contract.md`*
