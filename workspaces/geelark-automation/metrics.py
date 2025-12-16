"""DM Bot Metrics - Operational statistics and counters.

Tracks:
- Messages sent/received
- Errors by category
- Goal conversions
- Session runtime
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BotMetrics:
    """Runtime metrics for the DM bot."""

    # Session info
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = ""

    # Message counts
    messages_received: int = 0
    messages_sent: int = 0
    photos_sent: int = 0

    # Error counts by category
    errors_infrastructure: int = 0
    errors_persona: int = 0
    errors_state: int = 0
    errors_policy: int = 0
    errors_ui: int = 0

    # LLM-specific error tracking
    llm_calls_total: int = 0
    llm_calls_success: int = 0
    llm_calls_failed: int = 0
    llm_retries_total: int = 0
    llm_empty_responses: int = 0
    llm_json_parse_errors: int = 0
    llm_latency_sum_ms: float = 0.0  # Sum for average calculation

    # Goal conversions
    goals_chatting: int = 0
    goals_asking_location: int = 0
    goals_got_location: int = 0
    goals_sending_link: int = 0
    goals_sent_link: int = 0
    goals_rejected: int = 0

    # Goal state tracking
    goal_transitions: int = 0
    goal_stuck_count: int = 0

    # Conversation stats
    conversations_processed: int = 0
    conversations_skipped: int = 0

    def record_message_received(self):
        """Record an incoming message."""
        self.messages_received += 1

    def record_message_sent(self, with_photo: bool = False):
        """Record an outgoing message."""
        self.messages_sent += 1
        if with_photo:
            self.photos_sent += 1

    def record_error(self, category: str):
        """Record an error by category."""
        attr = f"errors_{category}"
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + 1)

    def record_goal(self, goal_status: str):
        """Record a goal status."""
        attr = f"goals_{goal_status}"
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + 1)

    def record_conversation(self, processed: bool = True):
        """Record a conversation attempt."""
        if processed:
            self.conversations_processed += 1
        else:
            self.conversations_skipped += 1

    def record_llm_call(self, success: bool, latency_ms: float = 0.0):
        """Record an LLM call attempt."""
        self.llm_calls_total += 1
        if success:
            self.llm_calls_success += 1
        else:
            self.llm_calls_failed += 1
        if latency_ms > 0:
            self.llm_latency_sum_ms += latency_ms

    def record_llm_retry(self, reason: str = ""):
        """Record an LLM retry attempt."""
        self.llm_retries_total += 1
        logger.debug(f"LLM retry recorded: {reason}")

    def record_llm_empty_response(self):
        """Record an empty LLM response."""
        self.llm_empty_responses += 1

    def record_llm_json_error(self):
        """Record a JSON parse error from LLM."""
        self.llm_json_parse_errors += 1

    def record_goal_transition(self, from_state: str, to_state: str):
        """Record a goal state transition."""
        self.goal_transitions += 1
        logger.debug(f"Goal transition: {from_state} -> {to_state}")

    def record_goal_stuck(self):
        """Record a stuck goal state."""
        self.goal_stuck_count += 1

    @property
    def llm_success_rate(self) -> float:
        """Calculate LLM call success rate."""
        if self.llm_calls_total == 0:
            return 1.0
        return self.llm_calls_success / self.llm_calls_total

    @property
    def llm_avg_latency_ms(self) -> float:
        """Calculate average LLM latency."""
        if self.llm_calls_success == 0:
            return 0.0
        return self.llm_latency_sum_ms / self.llm_calls_success

    @property
    def success_rate(self) -> float:
        """Calculate message success rate."""
        total = self.messages_sent + self.total_errors
        return self.messages_sent / total if total > 0 else 1.0

    @property
    def total_errors(self) -> int:
        """Total errors across all categories."""
        return (
            self.errors_infrastructure
            + self.errors_persona
            + self.errors_state
            + self.errors_policy
            + self.errors_ui
        )

    @property
    def goal_conversion_rate(self) -> float:
        """Calculate rate of conversations reaching sent_link."""
        total_goals = (
            self.goals_chatting
            + self.goals_asking_location
            + self.goals_got_location
            + self.goals_sending_link
            + self.goals_sent_link
            + self.goals_rejected
        )
        return self.goals_sent_link / total_goals if total_goals > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary."""
        runtime = datetime.now() - datetime.fromisoformat(self.session_start)
        return f"""
=== DM Bot Metrics ===
Session: {self.session_id or 'default'}
Runtime: {runtime}

Messages:
  Received: {self.messages_received}
  Sent: {self.messages_sent}
  Photos: {self.photos_sent}
  Success Rate: {self.success_rate:.1%}

LLM Performance:
  Total Calls: {self.llm_calls_total}
  Successful: {self.llm_calls_success}
  Failed: {self.llm_calls_failed}
  Retries: {self.llm_retries_total}
  Empty Responses: {self.llm_empty_responses}
  JSON Parse Errors: {self.llm_json_parse_errors}
  Success Rate: {self.llm_success_rate:.1%}
  Avg Latency: {self.llm_avg_latency_ms:.0f}ms

Conversations:
  Processed: {self.conversations_processed}
  Skipped: {self.conversations_skipped}

Goal Progression:
  chatting: {self.goals_chatting}
  asking_location: {self.goals_asking_location}
  got_location: {self.goals_got_location}
  sending_link: {self.goals_sending_link}
  sent_link: {self.goals_sent_link}
  rejected: {self.goals_rejected}
  Transitions: {self.goal_transitions}
  Stuck States: {self.goal_stuck_count}
  Conversion Rate: {self.goal_conversion_rate:.1%}

Errors:
  Infrastructure: {self.errors_infrastructure}
  Persona/LLM: {self.errors_persona}
  State: {self.errors_state}
  Policy: {self.errors_policy}
  UI: {self.errors_ui}
  Total: {self.total_errors}
"""


class MetricsManager:
    """Manages metrics persistence and aggregation."""

    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current = BotMetrics()

    def start_session(self, session_id: str = ""):
        """Start a new metrics session."""
        self.current = BotMetrics(session_id=session_id)
        logger.info(f"Started metrics session: {session_id or 'default'}")

    def save(self):
        """Save current metrics to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.metrics_dir / f"metrics_{timestamp}.json"

        try:
            with open(filepath, "w") as f:
                json.dump(self.current.to_dict(), f, indent=2)
            logger.debug(f"Saved metrics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def load_aggregate(self) -> Dict[str, int]:
        """Load and aggregate all metrics files.

        Returns:
            Aggregated counts across all sessions.
        """
        aggregate = {
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_photos_sent": 0,
            "total_errors": 0,
            "total_goals_sent_link": 0,
            "total_conversations": 0,
            "sessions": 0,
        }

        for filepath in self.metrics_dir.glob("metrics_*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                aggregate["total_messages_sent"] += data.get("messages_sent", 0)
                aggregate["total_messages_received"] += data.get("messages_received", 0)
                aggregate["total_photos_sent"] += data.get("photos_sent", 0)
                aggregate["total_errors"] += (
                    data.get("errors_infrastructure", 0)
                    + data.get("errors_persona", 0)
                    + data.get("errors_state", 0)
                    + data.get("errors_policy", 0)
                    + data.get("errors_ui", 0)
                )
                aggregate["total_goals_sent_link"] += data.get("goals_sent_link", 0)
                aggregate["total_conversations"] += data.get("conversations_processed", 0)
                aggregate["sessions"] += 1
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        return aggregate


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.DEBUG)

    manager = MetricsManager("./test_metrics")
    manager.start_session("test_session_001")

    # Simulate some activity
    manager.current.record_message_received()
    manager.current.record_message_sent(with_photo=True)
    manager.current.record_goal("chatting")
    manager.current.record_goal("asking_location")
    manager.current.record_error("ui")
    manager.current.record_conversation(processed=True)

    print(manager.current.summary())
    manager.save()

    # Show aggregate
    print("\n=== Aggregate Stats ===")
    print(manager.load_aggregate())
