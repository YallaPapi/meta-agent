# Prompt 5 - DM UI Automation Robustness & Rate-Limiting

## Objective
Review the DM-specific Appium navigation and sending logic for robustness across devices and for basic spam/rate-limit protection.

## Instructions

1. Inspect appium_ui_controller.py DM methods (open_dm_inbox, get_unread_dms, open_conversation, send_dm_message, attach_photo_to_dm, etc.) and how they rely on element IDs vs coordinate fallbacks.

2. Assess whether the coordinate strategy will survive:
   - different resolutions / aspect ratios
   - minor UI layout changes

3. Identify where and how to add:
   - human-like delays (already partially present)
   - per-conversation and per-hour message caps
   - randomization in timing/photo usage to reduce bot-like footprints

4. Suggest any refactors to keep DM UI logic centralized so adding new actions (e.g., react with emoji, send link) doesn't spread locators across the codebase.

## Expected Output
A short set of recommendations and, ideally, a small checklist for "safe DM automation" (delays, caps, randomization, UI locator strategy).
