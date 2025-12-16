# Prompt 2 - Coupling & Cohesion for DM Modules

## Objective
Check that responsibilities are cleanly separated between persona logic, state, photos, and UI controller, and identify any tangling that will make the bot fragile or hard to extend.

## Instructions

1. Review dm_bot.py and see which tasks it performs directly vs delegating:
   - UI navigation (Appium)
   - persona prompt construction / parsing
   - reading/writing state
   - photo selection
   - timing/rate limiting

2. For each supporting module (persona_llm.py, conversation_state.py, photo_manager.py, appium_ui_controller.py):
   - Summarize its responsibilities
   - List which other modules it talks to

3. Identify cases of high coupling or low cohesion, e.g.:
   - state logic leaking into persona_llm
   - DM-specific UI assumptions hard-coded in multiple places
   - photo selection scattered across modules

4. Propose 2-3 small refactors to improve separation (e.g., keep dm_bot as orchestrator, move any reusable DM UI helpers into the controller, keep persona and state independent).

## Expected Output
A report listing where responsibilities are mixed and concrete suggestions to keep each piece focused and easier to reason about.
