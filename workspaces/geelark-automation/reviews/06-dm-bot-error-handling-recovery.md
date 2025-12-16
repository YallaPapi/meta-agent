# Prompt 6 - DM Bot Error Handling & Recovery Plan

## Objective
Ensure dm_bot.py:run_loop() can run for hours/days without crashing on transient failures and with clear logging.

## Instructions

1. Analyze run_loop() and any helper methods that handle:
   - network/device/Appium failures
   - JSON parse errors from the LLM
   - UI element not found / mismatched state
   - corrupted state files

2. Identify places where exceptions would bubble up and kill the loop.

3. Propose a recovery strategy, e.g.:
   - catch specific exceptions and: retry, skip user, or restart driver / reconnect device
   - structured logs with user, platform, error type

4. Suggest a minimal set of metrics/stats (e.g., successful replies, failed replies, goal conversions) and how --stats should compute them.

## Expected Output
A recovery plan with specific exception handling patterns and a metrics schema for operational visibility.
