#!/usr/bin/env python3
"""DM Bot CLI - Main entrypoint for persona DM automation.

Usage:
    python main.py --simulate              # Dry run without device
    python main.py --user-id test123       # Single conversation test
    python main.py --run                   # Run bot loop on device
    python main.py --init-config           # Create default config.yaml
"""

import argparse
import logging
import sys

from config import load_config, save_default_config, DMBotConfig
from conversation_state import ConversationStateManager
from dm_bot import DMBot
from persona_llm import MockPersonaLLM, PersonaLLM
from photo_manager import PhotoManager

# Optional Appium imports
try:
    from appium import webdriver
    from appium.options.android import UiAutomator2Options
    from appium_ui_controller import AppiumUIController
    HAS_APPIUM = True
except ImportError:
    HAS_APPIUM = False


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_simulation(config: DMBotConfig, user_id: str = "test_user"):
    """Run a simulated conversation without device.

    Args:
        config: Bot configuration
        user_id: Simulated user ID
    """
    print("\n=== DM Bot Simulation Mode ===\n")

    # Use mock LLM
    llm = MockPersonaLLM(config.persona)
    state_manager = ConversationStateManager(config.state_dir)
    photo_manager = PhotoManager(config.photo_bucket_path, config.persona_name)

    # Simulate conversation
    test_messages = [
        "hey! saw your profile, you seem cool",
        "im good! just got off work lol. you?",
        "oh nice! im from new york",
        "haha sure ill check it out",
    ]

    state = state_manager.load_state(user_id)

    for msg in test_messages:
        print(f"User: {msg}")

        # Add to history
        state.add_message("user", msg)

        # Generate response
        response = llm.generate(
            conversation_history=state.history,
            goal_status=state.goal_status,
        )

        # Update goal
        state.goal_status = response.goal_status

        # Select photo
        photo = None
        if response.send_photo:
            photo = photo_manager.select_photo(response.photo_mood, force_send=True)

        # Add response to history
        state.add_message("assistant", response.text)

        print(f"Bot: {response.text}")
        if photo:
            print(f"     [Photo: {photo}]")
        print(f"     [Goal: {response.goal_status}]")
        print()

    # Save state
    state_manager.save_state(user_id, state)
    print(f"State saved to {config.state_dir}/")


def run_single_test(config: DMBotConfig, user_id: str, message: str):
    """Test a single message/response cycle.

    Args:
        config: Bot configuration
        user_id: User ID to test with
        message: Message to process
    """
    print(f"\n=== Single Message Test ===\n")
    print(f"User ID: {user_id}")
    print(f"Message: {message}\n")
    print(f"LLM: {config.llm_provider} / {config.llm_model}\n")

    bot = DMBot(config)

    # Use mock LLM if no API key (not needed for Ollama)
    if config.llm_provider not in ("ollama",) and not config.llm_api_key:
        print("[Using MockLLM - set ANTHROPIC_API_KEY for real responses]\n")
        bot.llm = MockPersonaLLM(config.persona)

    response = bot.process_conversation(user_id, message)

    # Handle emoji in console output (Windows encoding issues)
    text = response['text'].encode('ascii', 'replace').decode('ascii')
    print(f"Response: {text}")
    if response.get('photo_path'):
        print(f"Photo: {response['photo_path']}")
    print(f"Goal Status: {response['goal_status']}")


def show_stats(config: DMBotConfig):
    """Show conversation and photo statistics."""
    print("\n=== DM Bot Statistics ===\n")

    # Conversation stats
    state_manager = ConversationStateManager(config.state_dir)
    conv_stats = state_manager.get_stats()
    print("Conversation Goals:")
    for status, count in conv_stats.items():
        print(f"  {status}: {count}")

    # Photo stats
    photo_manager = PhotoManager(config.photo_bucket_path, config.persona_name)
    photo_stats = photo_manager.get_stats()
    print("\nPhoto Buckets:")
    for mood, count in photo_stats.items():
        print(f"  {mood}: {count} photos")


def connect_appium(config: DMBotConfig):
    """Connect to Appium server and return UI controller.

    Args:
        config: Bot configuration with Appium settings

    Returns:
        AppiumUIController instance
    """
    if not HAS_APPIUM:
        print("ERROR: Appium not installed. Run: pip install Appium-Python-Client")
        sys.exit(1)

    appium_url = f"http://{config.appium_host}:{config.appium_port}"
    print(f"Connecting to Appium at {appium_url}...")

    options = UiAutomator2Options()
    options.platform_name = "Android"
    options.automation_name = "UiAutomator2"
    options.no_reset = True  # Don't reset app state

    if config.device_id:
        options.udid = config.device_id

    try:
        driver = webdriver.Remote(appium_url, options=options)
        print("Connected to device")
        return AppiumUIController(driver)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)


def run_bot(config: DMBotConfig):
    """Run the DM bot loop on a connected device."""
    print("\n=== Starting DM Bot ===\n")

    # Connect to device
    ui_controller = connect_appium(config)

    # Create bot
    bot = DMBot(config)
    bot.set_ui_controller(ui_controller)

    # Use mock LLM if no API key (not needed for Ollama)
    if config.llm_provider not in ("ollama",) and not config.llm_api_key:
        print("[Using MockLLM - set ANTHROPIC_API_KEY for real responses]\n")
        bot.llm = MockPersonaLLM(config.persona)

    print(f"Persona: {config.persona.name}, {config.persona.age}")
    print(f"LLM: {config.llm_provider} / {config.llm_model}")
    print(f"Poll interval: {config.poll_interval}s")
    print(f"Photo probability: {config.photo_probability * 100:.0f}%")
    print("\nStarting main loop... (Ctrl+C to stop)\n")

    try:
        bot.run_loop(poll_interval=config.poll_interval)
    except KeyboardInterrupt:
        print("\n\nStopping bot...")
    finally:
        if ui_controller and ui_controller.driver:
            ui_controller.driver.quit()
            print("Disconnected from device")


def run_single_dm(config: DMBotConfig, username: str):
    """Process a single DM conversation on device.

    Args:
        config: Bot configuration
        username: Instagram username to respond to
    """
    print(f"\n=== Single DM Test: {username} ===\n")
    print(f"LLM: {config.llm_provider} / {config.llm_model}\n")

    # Connect to device
    ui_controller = connect_appium(config)

    # Create bot
    bot = DMBot(config)
    bot.set_ui_controller(ui_controller)

    # Use mock LLM if no API key (not needed for Ollama)
    if config.llm_provider not in ("ollama",) and not config.llm_api_key:
        print("[Using MockLLM - set ANTHROPIC_API_KEY for real responses]\n")
        bot.llm = MockPersonaLLM(config.persona)

    try:
        response = bot.process_single_conversation(username)
        if response:
            print(f"\nResponse sent: {response['text']}")
            if response.get('photo_path'):
                print(f"Photo: {response['photo_path']}")
            print(f"Goal Status: {response['goal_status']}")
        else:
            print("No response generated")
    finally:
        if ui_controller and ui_controller.driver:
            ui_controller.driver.quit()
            print("Disconnected from device")


def main():
    parser = argparse.ArgumentParser(
        description="DM Bot - Persona-based Instagram DM automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --init-config           Create default config.yaml
  python main.py --simulate              Run full conversation simulation
  python main.py --user-id abc --message "hey!"   Test single response (no device)
  python main.py --run                   Run bot loop on device
  python main.py --dm username           Reply to single DM on device
  python main.py --stats                 Show conversation statistics
        """,
    )

    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create default config.yaml and exit",
    )
    parser.add_argument(
        "--simulate", "-s",
        action="store_true",
        help="Run simulation without actual device",
    )
    parser.add_argument(
        "--user-id", "-u",
        help="User ID for single test (offline, no device)",
    )
    parser.add_argument(
        "--message", "-m",
        help="Message to process (with --user-id)",
    )
    parser.add_argument(
        "--run", "-r",
        action="store_true",
        help="Run the DM bot loop on connected device",
    )
    parser.add_argument(
        "--dm",
        metavar="USERNAME",
        help="Reply to a single DM from USERNAME on device",
    )
    parser.add_argument(
        "--device-id", "-d",
        help="Android device ID for Appium connection",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Init config
    if args.init_config:
        save_default_config(args.config)
        sys.exit(0)

    # Load config
    config = load_config(args.config)

    # Override device_id from CLI
    if args.device_id:
        config.device_id = args.device_id

    # Stats
    if args.stats:
        show_stats(config)
        sys.exit(0)

    # Simulation
    if args.simulate:
        run_simulation(config)
        sys.exit(0)

    # Single offline test
    if args.user_id and args.message:
        run_single_test(config, args.user_id, args.message)
        sys.exit(0)

    # Run bot loop on device
    if args.run:
        run_bot(config)
        sys.exit(0)

    # Single DM on device
    if args.dm:
        run_single_dm(config, args.dm)
        sys.exit(0)

    # No action specified
    parser.print_help()
    print("\nError: Specify --simulate, --run, --dm, --stats, or --user-id with --message")
    sys.exit(1)


if __name__ == "__main__":
    main()
