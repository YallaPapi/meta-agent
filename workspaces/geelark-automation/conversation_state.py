"""Conversation State Manager - Tracks per-user conversation state.

Stores:
- Message history
- Funnel stage (initial_response, small_talk, location_exchange, etc.)
- Location information (mentioned location, whether matched)
- Objection tracking
- Subscription/conversion status
- Response cache (for variety)
- Last interaction timestamp

Safety features:
- Atomic writes (temp file + rename)
- Max history length enforcement
- Corrupted file recovery
- Rate limiting counters
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set

from funnel_stages import FunnelStage

logger = logging.getLogger(__name__)

# Safety constants
MAX_HISTORY_LENGTH = 100  # Max messages per conversation
MAX_MESSAGE_LENGTH = 2000  # Max chars per message (truncate if longer)
MAX_RESPONSE_CACHE = 50  # Max cached responses for variety tracking


@dataclass
class Message:
    """A single message in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationState:
    """State for a single user conversation.

    Tracks the full funnel state as defined in the PRD.
    """
    user_id: str
    history: List[Dict] = field(default_factory=list)

    # Funnel stage tracking (new 9-stage funnel)
    funnel_stage: str = "initial_response"

    # Location tracking
    location_mentioned: Optional[str] = None
    location_matched: bool = False

    # Objection tracking
    objections_count: int = 0

    # Conversion tracking
    link_sent: bool = False
    subscription_claimed: bool = False
    converted: bool = False

    # Response variety tracking
    response_cache: List[str] = field(default_factory=list)

    # Timestamps
    last_interaction: str = field(default_factory=lambda: datetime.now().isoformat())

    # Rate limiting counters
    messages_sent_today: int = 0
    messages_sent_hour: int = 0
    last_message_date: str = ""  # YYYY-MM-DD
    last_message_hour: str = ""  # YYYY-MM-DD-HH

    # Legacy field for backward compatibility
    goal_status: str = "chatting"
    city: Optional[str] = None

    def get_funnel_stage(self) -> FunnelStage:
        """Get current funnel stage as enum.

        Returns:
            FunnelStage enum value
        """
        try:
            return FunnelStage(self.funnel_stage)
        except ValueError:
            return FunnelStage.INITIAL_RESPONSE

    def set_funnel_stage(self, stage: FunnelStage):
        """Set funnel stage from enum.

        Args:
            stage: FunnelStage enum value
        """
        self.funnel_stage = stage.value
        # Update legacy field for backward compatibility
        self._sync_legacy_goal()

    def _sync_legacy_goal(self):
        """Sync legacy goal_status field with new funnel_stage."""
        # Map new stages to legacy goal statuses
        stage_to_goal = {
            FunnelStage.INITIAL_RESPONSE: "chatting",
            FunnelStage.SMALL_TALK: "chatting",
            FunnelStage.LOCATION_EXCHANGE: "asking_location",
            FunnelStage.MEETUP_TEASE: "got_location",
            FunnelStage.PLATFORM_REDIRECT: "sending_link",
            FunnelStage.OBJECTION_HANDLING: "sending_link",
            FunnelStage.VERIFICATION: "sent_link",
            FunnelStage.CONVERTED: "sent_link",
            FunnelStage.DEAD_LEAD: "rejected",
        }
        stage = self.get_funnel_stage()
        self.goal_status = stage_to_goal.get(stage, "chatting")

    def get_used_phrases(self) -> Set[str]:
        """Get set of used phrases for variety tracking.

        Returns:
            Set of phrases already used in this conversation
        """
        return set(self.response_cache)

    def add_used_phrase(self, phrase: str):
        """Add a phrase to the used cache.

        Args:
            phrase: Response text used
        """
        if phrase and phrase not in self.response_cache:
            self.response_cache.append(phrase)
            # Keep cache bounded
            if len(self.response_cache) > MAX_RESPONSE_CACHE:
                self.response_cache = self.response_cache[-MAX_RESPONSE_CACHE:]

    def add_message(self, role: str, content: str):
        """Add a message to history with safety enforcement."""
        # Truncate long messages
        if len(content) > MAX_MESSAGE_LENGTH:
            content = content[:MAX_MESSAGE_LENGTH] + "..."
            logger.warning(f"Truncated message for {self.user_id} (was {len(content)} chars)")

        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.last_interaction = datetime.now().isoformat()

        # Enforce max history length (keep most recent)
        if len(self.history) > MAX_HISTORY_LENGTH:
            removed = len(self.history) - MAX_HISTORY_LENGTH
            self.history = self.history[-MAX_HISTORY_LENGTH:]
            logger.debug(f"Trimmed {removed} old messages for {self.user_id}")

        # Update rate limiting counters for assistant messages
        if role == "assistant":
            self._update_rate_counters()

    def _update_rate_counters(self):
        """Update rate limiting counters."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        this_hour = now.strftime("%Y-%m-%d-%H")

        # Reset daily counter if new day
        if self.last_message_date != today:
            self.messages_sent_today = 0
            self.last_message_date = today

        # Reset hourly counter if new hour
        if self.last_message_hour != this_hour:
            self.messages_sent_hour = 0
            self.last_message_hour = this_hour

        self.messages_sent_today += 1
        self.messages_sent_hour += 1

    def can_send_message(self, max_per_hour: int = 10, max_per_day: int = 50) -> bool:
        """Check if rate limits allow sending a message.

        Args:
            max_per_hour: Maximum messages per hour
            max_per_day: Maximum messages per day

        Returns:
            True if within rate limits
        """
        # Refresh counters first
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        this_hour = now.strftime("%Y-%m-%d-%H")

        if self.last_message_date != today:
            self.messages_sent_today = 0
        if self.last_message_hour != this_hour:
            self.messages_sent_hour = 0

        if self.messages_sent_hour >= max_per_hour:
            logger.warning(f"Rate limit: {self.user_id} hit hourly limit ({max_per_hour})")
            return False
        if self.messages_sent_today >= max_per_day:
            logger.warning(f"Rate limit: {self.user_id} hit daily limit ({max_per_day})")
            return False
        return True

    def get_recent_history(self, max_messages: int = 20) -> List[Dict]:
        """Get recent message history for LLM context.

        Args:
            max_messages: Maximum messages to return

        Returns:
            List of recent messages
        """
        return self.history[-max_messages:]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        """Create from dictionary."""
        return cls(**data)


class ConversationStateManager:
    """Manages conversation states with JSON file persistence."""

    def __init__(self, state_dir: str = "./state"):
        """Initialize state manager.

        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, ConversationState] = {}

    def _get_state_path(self, user_id: str) -> Path:
        """Get path to state file for user."""
        # Sanitize user_id for filename
        safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
        return self.state_dir / f"{safe_id}.json"

    def _backup_corrupted(self, state_path: Path):
        """Backup a corrupted state file for debugging.

        Args:
            state_path: Path to corrupted file
        """
        if not state_path.exists():
            return

        backup_path = state_path.with_suffix(
            f".corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            state_path.rename(backup_path)
            logger.info(f"Backed up corrupted state to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup corrupted state: {e}")
            # Try to just delete it
            try:
                state_path.unlink()
            except Exception:
                pass

    def load_state(self, user_id: str) -> ConversationState:
        """Load or create state for user.

        Args:
            user_id: Instagram user ID

        Returns:
            ConversationState for user
        """
        # Check cache first
        if user_id in self._cache:
            return self._cache[user_id]

        state_path = self._get_state_path(user_id)

        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    data = json.load(f)
                state = ConversationState.from_dict(data)
                logger.debug(f"Loaded state for {user_id}: {state.goal_status}")
            except json.JSONDecodeError as e:
                # Corrupted JSON - backup and start fresh
                logger.error(f"Corrupted state file for {user_id}: {e}")
                self._backup_corrupted(state_path)
                state = ConversationState(user_id=user_id)
            except Exception as e:
                logger.error(f"Error loading state for {user_id}: {e}")
                state = ConversationState(user_id=user_id)
        else:
            state = ConversationState(user_id=user_id)
            logger.debug(f"Created new state for {user_id}")

        self._cache[user_id] = state
        return state

    def save_state(self, user_id: str, state: ConversationState):
        """Save state for user with atomic write.

        Uses temp file + rename pattern to prevent corrupted state on crash.

        Args:
            user_id: Instagram user ID
            state: ConversationState to save
        """
        state_path = self._get_state_path(user_id)

        try:
            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                prefix=f"state_{user_id}_",
                dir=self.state_dir
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state.to_dict(), f, indent=2)

                # Atomic rename (works on POSIX, mostly works on Windows)
                # On Windows, need to remove target first
                if os.path.exists(state_path):
                    os.replace(temp_path, state_path)
                else:
                    os.rename(temp_path, state_path)

            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

            self._cache[user_id] = state
            logger.debug(f"Saved state for {user_id}")

        except Exception as e:
            logger.error(f"Error saving state for {user_id}: {e}")

    def delete_state(self, user_id: str):
        """Delete state for user.

        Args:
            user_id: Instagram user ID
        """
        state_path = self._get_state_path(user_id)

        if state_path.exists():
            state_path.unlink()

        if user_id in self._cache:
            del self._cache[user_id]

        logger.info(f"Deleted state for {user_id}")

    def list_users(self) -> List[str]:
        """List all users with saved state.

        Returns:
            List of user IDs
        """
        users = []
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                users.append(data.get("user_id", state_file.stem))
            except Exception:
                pass
        return users

    def get_stats(self) -> dict:
        """Get statistics about conversations.

        Returns:
            Dict with counts by goal_status
        """
        stats = {status: 0 for status in GOAL_STATUSES}

        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                status = data.get("goal_status", "chatting")
                if status in stats:
                    stats[status] += 1
            except Exception:
                pass

        return stats


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.DEBUG)

    manager = ConversationStateManager("./test_state")

    # Create a conversation
    state = manager.load_state("test_user_123")
    state.add_message("user", "hey! how are you?")
    state.add_message("assistant", "hii! im good, just studying lol. wbu?")
    state.goal_status = "chatting"
    manager.save_state("test_user_123", state)

    # Reload and verify
    state2 = manager.load_state("test_user_123")
    print(f"History: {len(state2.history)} messages")
    print(f"Goal: {state2.goal_status}")
    print(f"Stats: {manager.get_stats()}")
