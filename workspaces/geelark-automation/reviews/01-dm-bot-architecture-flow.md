# Prompt 1 - DM Bot Architecture & Flow Diagram

## Objective
Map the DM bot architecture and message flow end-to-end so it's clear how persona LLM, state, photos, and Appium UI fit together.

## Instructions

1. From geelark-automation.xml, focus on:
   - appium_ui_controller.py (DM methods)
   - dm_bot.py
   - persona_llm.py
   - photo_manager.py
   - conversation_state.py
   - config.py, main.py

2. Draw a diagram (Mermaid/ASCII) showing:
   - DM inbox polling -> unread selection -> conversation state load -> persona LLM call -> photo selection -> send via Appium -> state update

3. Show how persona goals (goal_status) and extracted city travel between persona_llm, conversation_state, and dm_bot.

4. Highlight current boundaries: which module owns state, which owns UI, which owns business logic.

## Expected Output
A diagram plus short explanation that makes the DM pipeline obvious, so you can see where to add rate limiting, error handling, and new personas.
