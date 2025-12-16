# Prompt 3 - Persona LLM Prompt + JSON Contract Review

## Objective
Critically review the persona prompt and JSON output schema used in persona_llm.py and config.py:PersonaConfig.to_system_prompt() to make sure it's robust for automation.

## Instructions

1. Inspect the system and user prompts used for persona replies, the JSON schema (fields: text, send_photo, photo_mood, goal_status, maybe city), and how parsing is done.

2. Evaluate:
   - Is the JSON schema explicit enough to avoid malformed outputs?
   - Are persona instructions clear but not bloated?
   - Is goal progression (chatting -> asking_location -> got_location -> sending_link) fully encoded in prompt logic or partially outside?

3. Suggest improvements, such as:
   - adding goal_status enum description
   - separating "how to talk" vs "what JSON to output" into distinct prompt sections
   - instructions for error recovery (e.g., what to do if input is spam)

4. Recommend a validation layer for parsed responses (e.g., default mood if photo_mood missing).

## Expected Output
A refined prompt/JSON contract spec with concrete edits you should apply to PersonaConfig and persona_llm.py before running at scale.
