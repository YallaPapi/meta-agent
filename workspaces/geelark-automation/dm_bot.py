"""DM Bot - Main orchestrator for persona-based Instagram DM automation.

This module handles the core conversation loop:
1. Poll for new DMs via Appium
2. Load conversation state
3. Generate response via structured LLM
4. Select photo based on mood
5. Send reply via Appium
6. Update state

Includes comprehensive error tracking, metrics, and goal state validation.
"""

import logging
import random
import time
import uuid
from typing import Optional

from config import DMBotConfig, load_config
from conversation_state import ConversationStateManager
from persona_llm import PersonaLLM, LLMResponse, RetryConfig
from photo_manager import PhotoManager
from prompt_builder import VALID_GOALS
from errors import (
    DMBotError,
    InfrastructureError,
    PersonaContractError,
    PolicyViolationError,
    UIAutomationError,
    GoalStateError,
    get_recovery_strategy,
    RecoveryAction,
)
from metrics import BotMetrics, MetricsManager

# Optional: Import UI controller for type hints
try:
    from appium_ui_controller import AppiumUIController
except ImportError:
    AppiumUIController = None

logger = logging.getLogger(__name__)


# Valid goal state transitions
GOAL_TRANSITIONS = {
    "chatting": ["chatting", "asking_location", "rejected"],
    "asking_location": ["asking_location", "got_location", "chatting", "rejected"],
    "got_location": ["got_location", "sending_link", "chatting", "rejected"],
    "sending_link": ["sending_link", "sent_link", "rejected"],
    "sent_link": ["sent_link", "rejected"],
    "rejected": ["rejected", "chatting"],  # Can retry
}


class DMBot:
    """Main DM automation bot with comprehensive error tracking."""

    def __init__(self, config: DMBotConfig):
        self.config = config
        self.state_manager = ConversationStateManager(config.state_dir)
        self.photo_manager = PhotoManager(config.photo_bucket_path, config.persona_name)
        self.metrics = MetricsManager(config.state_dir + "/metrics")

        # Create LLM with metrics callback
        self.llm = PersonaLLM(
            persona=config.persona,
            api_key=config.llm_api_key,
            provider=config.llm_provider,
            model=config.llm_model,
            ollama_host=config.ollama_host,
            retry_config=RetryConfig(max_retries=3),
            metrics_callback=self._llm_metrics_callback,
        )

        self.ui_controller = None  # Injected when running on device
        self._retry_counts: dict = {}  # Track retries per user
        self._goal_repeat_counts: dict = {}  # Track stuck goals per user

    def _llm_metrics_callback(self, event_type: str, **kwargs):
        """Callback for LLM metrics tracking."""
        if event_type == "llm_call":
            self.metrics.current.record_llm_call(
                success=kwargs.get("success", False),
                latency_ms=kwargs.get("latency_ms", 0.0),
            )
        elif event_type == "llm_retry":
            self.metrics.current.record_llm_retry(kwargs.get("reason", ""))
        elif event_type == "llm_empty_response":
            self.metrics.current.record_llm_empty_response()
        elif event_type == "llm_json_error":
            self.metrics.current.record_llm_json_error()

    def set_ui_controller(self, controller):
        """Inject the Appium UI controller for device operations."""
        self.ui_controller = controller

    def _validate_goal_transition(
        self,
        user_id: str,
        current_goal: str,
        new_goal: str,
        trace_id: str,
    ) -> str:
        """Validate and potentially correct goal state transition.

        Args:
            user_id: User ID for tracking
            current_goal: Current goal state
            new_goal: Proposed new goal state
            trace_id: Trace ID for logging

        Returns:
            Valid goal state (may be corrected)

        Raises:
            GoalStateError: If stuck in invalid state too long
        """
        # Check if transition is valid
        valid_next = GOAL_TRANSITIONS.get(current_goal, VALID_GOALS)

        if new_goal not in valid_next:
            logger.warning(
                f"[{trace_id}] Invalid goal transition: {current_goal} -> {new_goal}, "
                f"valid: {valid_next}"
            )
            # Keep current goal instead of invalid transition
            new_goal = current_goal

        # Track repeated goals (potential stuck state)
        key = f"{user_id}:{new_goal}"
        repeat_count = self._goal_repeat_counts.get(key, 0)

        if new_goal == current_goal:
            repeat_count += 1
            self._goal_repeat_counts[key] = repeat_count

            if repeat_count > 5:
                # Stuck in same state too long
                self.metrics.current.record_goal_stuck()
                logger.warning(
                    f"[{trace_id}] Goal stuck: {new_goal} repeated {repeat_count} times"
                )

                # Force progression or rejection
                if new_goal in ("chatting", "asking_location"):
                    # Likely not progressing - might need to abort
                    if repeat_count > 10:
                        raise GoalStateError(
                            message="Goal stuck - conversation not progressing",
                            current_state=current_goal,
                            attempted_state=new_goal,
                            loop_count=repeat_count,
                            user_id=user_id,
                            trace_id=trace_id,
                        )
        else:
            # Goal changed - reset counter
            self._goal_repeat_counts[key] = 0
            self.metrics.current.record_goal_transition(current_goal, new_goal)

        return new_goal

    def process_conversation(
        self,
        user_id: str,
        latest_message: str,
        trace_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Process a single conversation turn.

        Args:
            user_id: Instagram user ID
            latest_message: The latest message from the user
            trace_id: Optional trace ID for debugging

        Returns:
            Dict with response text and optional photo path, or None if error
        """
        trace_id = trace_id or str(uuid.uuid4())[:8]
        logger.info(f"[{trace_id}] Processing conversation for {user_id}")

        # Load conversation state
        state = self.state_manager.load_state(user_id)
        previous_goal = state.goal_status

        # Add new message to history
        state.add_message(role="user", content=latest_message)

        # Generate LLM response with trace ID
        response: LLMResponse = self.llm.generate(
            conversation_history=state.history,
            goal_status=state.goal_status,
            user_id=user_id,
            trace_id=trace_id,
        )

        # Validate and update goal status
        validated_goal = self._validate_goal_transition(
            user_id=user_id,
            current_goal=previous_goal,
            new_goal=response.goal_status,
            trace_id=trace_id,
        )
        state.goal_status = validated_goal

        # Check if location was mentioned
        if validated_goal == "got_location" and not state.city:
            state.city = self._extract_city(latest_message)

        # Select photo if needed
        photo_path = None
        if response.send_photo:
            photo_path = self.photo_manager.select_photo(
                mood=response.photo_mood,
                time_override=True,
            )

        # Add bot response to history
        state.add_message(role="assistant", content=response.text)

        # Save state
        self.state_manager.save_state(user_id, state)

        logger.info(
            f"[{trace_id}] Response generated: goal={validated_goal}, "
            f"photo={photo_path is not None}, latency={response.latency_ms:.0f}ms"
        )

        return {
            "text": response.text,
            "photo_path": photo_path,
            "goal_status": validated_goal,
            "trace_id": trace_id,
            "latency_ms": response.latency_ms,
        }

    def _extract_city(self, message: str) -> Optional[str]:
        """Extract city name from message (simple heuristic for now)."""
        # TODO: Use NLP or LLM for better extraction
        # For now, just store the raw message
        return message

    def send_reply(self, username: str, text: str, photo_path: Optional[str] = None) -> bool:
        """Send reply via UI controller.

        Args:
            username: Instagram username
            text: Message text
            photo_path: Optional path to photo to attach

        Returns:
            True if reply was sent successfully
        """
        if not self.ui_controller:
            logger.warning("No UI controller set - dry run mode")
            logger.info(f"Would send to {username}: {text}")
            if photo_path:
                logger.info(f"Would attach photo: {photo_path}")
            return True

        # Ensure we're in the conversation
        if not self.ui_controller.is_in_conversation():
            if not self.ui_controller.open_conversation(username):
                logger.error(f"Could not open conversation with {username}")
                return False

        # Send message with optional photo
        if photo_path:
            return self.ui_controller.send_dm_with_photo(text, photo_path)
        else:
            return self.ui_controller.send_dm_message(text)

    def run_loop(self, poll_interval: float = 30.0):
        """Main loop - poll for new DMs and respond.

        Args:
            poll_interval: Seconds between inbox checks
        """
        if not self.ui_controller:
            logger.error("No UI controller set - cannot run loop without device")
            return

        logger.info(f"Starting DM bot loop (poll every {poll_interval}s)")
        self.metrics.start_session()
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            try:
                # Navigate to inbox
                if not self.ui_controller.is_in_dm_inbox():
                    if not self.ui_controller.open_dm_inbox():
                        raise UIAutomationError("Failed to open DM inbox")
                    time.sleep(2)

                # Get unread conversations
                try:
                    unread = self.ui_controller.get_unread_dms()
                except Exception as e:
                    raise UIAutomationError(f"Failed to get unread DMs: {e}")

                logger.debug(f"Found {len(unread)} unread conversations")

                for convo in unread:
                    username = convo.get("username")
                    if not username:
                        continue

                    trace_id = str(uuid.uuid4())[:8]

                    try:
                        self._process_single_dm(username, trace_id)
                        consecutive_errors = 0  # Reset on success
                        self._retry_counts.pop(username, None)  # Clear retry count
                    except DMBotError as e:
                        self._handle_dm_error(e, username)
                    except Exception as e:
                        # Wrap unknown errors
                        wrapped = UIAutomationError(str(e), user_id=username)
                        self._handle_dm_error(wrapped, username)

                # Save metrics periodically
                if self.metrics.current.conversations_processed % 10 == 0:
                    self.metrics.save()

            except InfrastructureError as e:
                e.log()
                consecutive_errors += 1
                self.metrics.current.record_error("infrastructure")

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive errors ({consecutive_errors}), stopping")
                    self.metrics.save()
                    raise

                # Cooldown before retry
                logger.warning(f"Infrastructure error, cooling down 30s (error {consecutive_errors}/{max_consecutive_errors})")
                time.sleep(30)

            except DMBotError as e:
                e.log()
                self.metrics.current.record_error(e.category.value)
                consecutive_errors += 1

            except Exception as e:
                logger.error(f"Unexpected error in DM loop: {e}", exc_info=True)
                self.metrics.current.record_error("infrastructure")
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("Too many consecutive errors, stopping")
                    self.metrics.save()
                    raise

            time.sleep(poll_interval)

    def _process_single_dm(self, username: str, trace_id: str):
        """Process a single DM conversation with error handling.

        Args:
            username: Instagram username
            trace_id: Trace ID for debugging

        Raises:
            DMBotError: On recoverable errors
        """
        logger.info(f"[{trace_id}] Processing conversation with {username}")

        # Check rate limit
        state = self.state_manager.load_state(username)
        if not state.can_send_message():
            raise PolicyViolationError("Rate limit exceeded", user_id=username)

        # Open the conversation
        if not self.ui_controller.open_conversation(username):
            raise UIAutomationError("Could not open conversation", element="conversation", user_id=username)

        time.sleep(1)

        try:
            # Read latest message
            latest = self.ui_controller.get_latest_message()
            if not latest or latest.get("role") != "user":
                logger.debug(f"[{trace_id}] No new user message from {username}")
                self.metrics.current.record_conversation(processed=False)
                return

            message = latest.get("content", "")
            logger.info(f"[{trace_id}] New DM from {username}: {message[:50]}...")
            self.metrics.current.record_message_received()

            # Process and generate response
            response = self.process_conversation(username, message, trace_id)
            if response:
                # Record goal
                self.metrics.current.record_goal(response.get("goal_status", "chatting"))

                # Add human-like delay
                delay = self._calculate_response_delay(response["text"])
                logger.debug(f"[{trace_id}] Waiting {delay:.1f}s before responding...")
                time.sleep(delay)

                # Send reply
                has_photo = response.get("photo_path") is not None
                if self.send_reply(
                    username=username,
                    text=response["text"],
                    photo_path=response.get("photo_path"),
                ):
                    self.metrics.current.record_message_sent(with_photo=has_photo)
                    self.metrics.current.record_conversation(processed=True)
                    logger.info(f"[{trace_id}] Reply sent successfully")
                else:
                    raise UIAutomationError("Failed to send reply", user_id=username)

        finally:
            # Always try to return to inbox
            self.ui_controller.go_back()
            time.sleep(1)

    def _handle_dm_error(self, error: DMBotError, username: str):
        """Handle a DM processing error.

        Args:
            error: The error that occurred
            username: User being processed
        """
        error.log()
        self.metrics.current.record_error(error.category.value)

        # Track retries
        retry_count = self._retry_counts.get(username, 0)
        action = get_recovery_strategy(error, retry_count)

        if action == RecoveryAction.RETRY:
            self._retry_counts[username] = retry_count + 1
            logger.info(f"Will retry {username} (attempt {retry_count + 1})")
        elif action == RecoveryAction.SKIP_USER:
            self._retry_counts.pop(username, None)
            self.metrics.current.record_conversation(processed=False)
            logger.info(f"Skipping {username}")
        elif action == RecoveryAction.COOLDOWN:
            time.sleep(10)

        # Try to return to safe state
        try:
            self.ui_controller.go_back()
            time.sleep(1)
        except Exception:
            pass

    def process_single_conversation(self, username: str) -> Optional[dict]:
        """Process a single conversation (for manual/test use).

        Args:
            username: Instagram username to process

        Returns:
            Response dict or None
        """
        if not self.ui_controller:
            logger.error("No UI controller set")
            return None

        trace_id = str(uuid.uuid4())[:8]

        # Open conversation
        if not self.ui_controller.is_in_dm_inbox():
            self.ui_controller.open_dm_inbox()
            time.sleep(2)

        if not self.ui_controller.open_conversation(username):
            logger.error(f"Could not open conversation with {username}")
            return None

        # Get latest message
        latest = self.ui_controller.get_latest_message()
        if not latest:
            logger.warning("No messages in conversation")
            return None

        message = latest.get("content", "")
        logger.info(f"Latest message: {message}")

        # Process
        response = self.process_conversation(username, message, trace_id)

        if response:
            # Send reply
            self.send_reply(
                username=username,
                text=response["text"],
                photo_path=response.get("photo_path"),
            )

        return response

    def _calculate_response_delay(self, text: str) -> float:
        """Calculate realistic typing delay based on message length."""
        # ~3-5 seconds per sentence, with some randomness
        base_delay = len(text) / 20  # ~20 chars per second typing
        jitter = random.uniform(
            self.config.min_response_delay,
            self.config.max_response_delay
        )
        return base_delay + jitter

    def show_metrics(self) -> str:
        """Get current metrics summary."""
        return self.metrics.current.summary()


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = load_config()
    bot = DMBot(config)

    # Simulate a conversation
    response = bot.process_conversation(
        user_id="test_user",
        latest_message="hey! how are you?",
    )
    print(f"Response: {response}")
    print(f"\nMetrics:\n{bot.show_metrics()}")
