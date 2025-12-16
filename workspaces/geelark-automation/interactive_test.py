"""Interactive Test Mode - Chat with the DM bot in real-time.

Run this script to test the two-LLM pipeline interactively.
You type messages as the "fan" and see how the bot responds.

Usage:
    python interactive_test.py              # Default (uses qwen2.5:7b)
    python interactive_test.py --model qwen2.5:14b  # Use larger model
    python interactive_test.py --verbose    # Show detailed analysis
    python interactive_test.py --reset      # Clear conversation history first

Commands while chatting:
    /debug    - Show detailed stage analysis
    /state    - Show current conversation state
    /reset    - Reset conversation and start fresh
    /stages   - Show all funnel stages
    /quit     - Exit
"""

import argparse
import logging
import sys
from datetime import datetime

from funnel_stages import FunnelStage, STAGE_GUIDELINES
from two_llm_pipeline import create_pipeline, PipelineResult
from response_generator import PersonaConfig


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 60)
    print("  DM BOT INTERACTIVE TEST")
    print("  Two-LLM Pipeline for OF Conversion Funnel")
    print("=" * 60)
    print("\nYou are the FAN. Type messages to test the bot's responses.")
    print("Commands: /debug /state /reset /stages /quit")
    print("-" * 60 + "\n")


def print_stage_info(stage: FunnelStage):
    """Print info about a stage."""
    guidelines = STAGE_GUIDELINES[stage]
    print(f"\n  Stage: {stage.value}")
    print(f"  Goal: {guidelines.goal}")


def print_analysis(result: PipelineResult, verbose: bool = False):
    """Print analysis details."""
    print(f"\n  [Analysis]")
    print(f"  Intent: {result.detected_intent}")
    print(f"  Stage: {result.stage_before.value} -> {result.stage_after.value}")
    if result.stage_transitioned:
        print(f"  ** STAGE TRANSITION **")
    if result.location_mentioned:
        print(f"  Location detected: {result.location_mentioned}")
    if result.objection_detected:
        print(f"  Objection detected!")
    if result.subscription_claimed:
        print(f"  User claims subscribed!")
    print(f"  Latency: {result.total_latency_ms:.0f}ms (analysis: {result.stage_analysis_ms:.0f}ms, response: {result.response_generation_ms:.0f}ms)")


def print_state(pipeline, user_id: str):
    """Print current conversation state."""
    summary = pipeline.get_conversation_summary(user_id)
    print("\n" + "-" * 40)
    print("CONVERSATION STATE")
    print("-" * 40)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("-" * 40)


def print_all_stages():
    """Print all funnel stages."""
    print("\n" + "-" * 40)
    print("FUNNEL STAGES")
    print("-" * 40)
    for i, stage in enumerate(FunnelStage, 1):
        guidelines = STAGE_GUIDELINES[stage]
        print(f"\n{i}. {stage.value}")
        print(f"   Trigger: {guidelines.trigger}")
        print(f"   Goal: {guidelines.goal}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Interactive DM bot test")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed analysis")
    parser.add_argument("--reset", action="store_true", help="Reset conversation on start")
    parser.add_argument("--user", default="test_fan", help="User ID for conversation")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    print_header()
    print(f"Using model: {args.model}")
    print(f"Ollama host: {args.host}")
    print(f"User ID: {args.user}")

    # Create persona
    persona = PersonaConfig(
        name="Mia",
        age=22,
        occupation="content creator",
        personality="friendly, flirty, casual",
        texting_style="short messages, lowercase, occasional emojis, asks questions back",
        of_link="onlyfans.com/miaxxxx",
    )

    # Create pipeline (real mode, not mock)
    try:
        pipeline = create_pipeline(
            mock_mode=False,
            model=args.model,
            ollama_host=args.host,
            state_dir="./test_state",
            persona=persona,
        )
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"\nError initializing pipeline: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"And the model is available: ollama pull {args.model}")
        sys.exit(1)

    # Reset if requested
    if args.reset:
        pipeline.reset_conversation(args.user)
        print("Conversation reset!")

    # Show initial state
    print_state(pipeline, args.user)

    last_result = None
    show_debug = args.verbose

    print("\n" + "=" * 60)
    print("START CHATTING (type your message as the fan)")
    print("=" * 60 + "\n")

    while True:
        try:
            # Get user input
            user_input = input("\n[YOU (fan)]: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()

                if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                    print("\nGoodbye!")
                    break

                elif cmd == "/debug":
                    show_debug = not show_debug
                    print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
                    if last_result and show_debug:
                        print_analysis(last_result, verbose=True)
                    continue

                elif cmd == "/state":
                    print_state(pipeline, args.user)
                    continue

                elif cmd == "/reset":
                    pipeline.reset_conversation(args.user)
                    print("Conversation reset! Starting fresh.")
                    continue

                elif cmd == "/stages":
                    print_all_stages()
                    continue

                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /debug  - Toggle detailed analysis")
                    print("  /state  - Show conversation state")
                    print("  /reset  - Reset conversation")
                    print("  /stages - Show all funnel stages")
                    print("  /quit   - Exit")
                    continue

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands")
                    continue

            # Process message through pipeline
            print("\n  [Processing...]")

            try:
                result = pipeline.process_message(
                    user_id=args.user,
                    user_message=user_input,
                )
                last_result = result

                # Show response
                print(f"\n[MIA (bot)]: {result.text}")

                if result.send_photo:
                    print(f"  [Would send {result.photo_mood} photo]")

                # Show stage transition
                if result.stage_transitioned:
                    print(f"\n  >> Stage changed: {result.stage_before.value} -> {result.stage_after.value}")
                    print_stage_info(result.stage_after)
                else:
                    print(f"\n  [Stage: {result.stage_after.value}]")

                # Show debug info if enabled
                if show_debug:
                    print_analysis(result, verbose=True)

                # Check for terminal states
                if result.stage_after == FunnelStage.CONVERTED:
                    print("\n" + "=" * 40)
                    print("  SUCCESS! Conversation converted!")
                    print("=" * 40)
                elif result.stage_after == FunnelStage.DEAD_LEAD:
                    print("\n" + "=" * 40)
                    print("  Conversation ended (dead lead)")
                    print("=" * 40)

            except Exception as e:
                print(f"\n  [Error processing message: {e}]")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break

    # Show final state
    print("\n" + "=" * 60)
    print("FINAL CONVERSATION STATE")
    print_state(pipeline, args.user)


if __name__ == "__main__":
    main()
