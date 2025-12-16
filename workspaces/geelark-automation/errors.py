"""DM Bot Error Taxonomy and Recovery Strategies.

Defines typed exceptions for different failure modes and
recovery strategies for the bot loop.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Top-level error categories."""
    INFRASTRUCTURE = "infrastructure"  # Retryable: network, device, Appium
    PERSONA = "persona"  # Retryable: LLM errors, JSON parse
    STATE = "state"  # Recoverable: corrupted state, missing data
    POLICY = "policy"  # Not retryable: rate limits, blocked users
    UI = "ui"  # Retryable: element not found, stale UI


class RecoveryAction(Enum):
    """Actions to take on error."""
    RETRY = "retry"  # Retry the same operation
    SKIP_USER = "skip_user"  # Skip this conversation, continue with others
    RECONNECT = "reconnect"  # Reconnect to device/driver
    COOLDOWN = "cooldown"  # Wait before continuing
    ABORT = "abort"  # Stop the bot loop


@dataclass
class DMBotError(Exception):
    """Base exception for DM Bot errors."""
    message: str
    category: ErrorCategory
    recovery: RecoveryAction
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self):
        return f"[{self.category.value}] {self.message}"

    def log(self):
        """Log the error with structured context."""
        extra = {
            "error_category": self.category.value,
            "recovery_action": self.recovery.value,
            "user_id": self.user_id,
            **self.context,
        }
        logger.error(f"{self.category.value}: {self.message}", extra=extra)


class InfrastructureError(DMBotError):
    """Network, device, or Appium connection errors."""

    def __init__(self, message: str, user_id: Optional[str] = None, **context):
        super().__init__(
            message=message,
            category=ErrorCategory.INFRASTRUCTURE,
            recovery=RecoveryAction.RECONNECT,
            user_id=user_id,
            context=context,
        )


class PersonaContractError(DMBotError):
    """LLM response parsing or validation errors."""

    def __init__(self, message: str, raw_response: str = "", user_id: Optional[str] = None, **context):
        super().__init__(
            message=message,
            category=ErrorCategory.PERSONA,
            recovery=RecoveryAction.RETRY,
            user_id=user_id,
            context={"raw_response": raw_response[:500], **context},
        )


class StateConsistencyError(DMBotError):
    """State file corruption or missing data."""

    def __init__(self, message: str, user_id: Optional[str] = None, **context):
        super().__init__(
            message=message,
            category=ErrorCategory.STATE,
            recovery=RecoveryAction.SKIP_USER,
            user_id=user_id,
            context=context,
        )


class PolicyViolationError(DMBotError):
    """Rate limits, blocked users, or policy violations."""

    def __init__(self, message: str, user_id: Optional[str] = None, **context):
        super().__init__(
            message=message,
            category=ErrorCategory.POLICY,
            recovery=RecoveryAction.SKIP_USER,
            user_id=user_id,
            context=context,
        )


class UIAutomationError(DMBotError):
    """Element not found, stale UI, or automation failures."""

    def __init__(self, message: str, element: str = "", user_id: Optional[str] = None, **context):
        super().__init__(
            message=message,
            category=ErrorCategory.UI,
            recovery=RecoveryAction.RETRY,
            user_id=user_id,
            context={"element": element, **context},
        )


class LLMCallError(DMBotError):
    """Network, timeout, or API errors when calling the LLM."""

    def __init__(
        self,
        message: str,
        model: str = "",
        attempt: int = 0,
        max_attempts: int = 0,
        status_code: Optional[int] = None,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **context,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PERSONA,
            recovery=RecoveryAction.RETRY,
            user_id=user_id,
            context={
                "model": model,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "status_code": status_code,
                "trace_id": trace_id,
                **context,
            },
        )


class LLMEmptyResponseError(DMBotError):
    """LLM returned empty or whitespace-only response."""

    def __init__(
        self,
        message: str = "LLM returned empty response",
        model: str = "",
        attempt: int = 0,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **context,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PERSONA,
            recovery=RecoveryAction.RETRY,
            user_id=user_id,
            context={
                "model": model,
                "attempt": attempt,
                "trace_id": trace_id,
                **context,
            },
        )


class LLMInvalidJSONError(DMBotError):
    """LLM response failed JSON parsing or schema validation."""

    def __init__(
        self,
        message: str,
        raw_response: str = "",
        model: str = "",
        missing_fields: Optional[list] = None,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **context,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PERSONA,
            recovery=RecoveryAction.RETRY,
            user_id=user_id,
            context={
                "raw_response": raw_response[:500] if raw_response else "",
                "model": model,
                "missing_fields": missing_fields or [],
                "trace_id": trace_id,
                **context,
            },
        )


class GoalStateError(DMBotError):
    """Invalid goal state transition or stuck state detected."""

    def __init__(
        self,
        message: str,
        current_state: str = "",
        attempted_state: str = "",
        loop_count: int = 0,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **context,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.STATE,
            recovery=RecoveryAction.SKIP_USER,
            user_id=user_id,
            context={
                "current_state": current_state,
                "attempted_state": attempted_state,
                "loop_count": loop_count,
                "trace_id": trace_id,
                **context,
            },
        )


# Recovery configuration
RECOVERY_CONFIG = {
    ErrorCategory.INFRASTRUCTURE: {
        "max_retries": 3,
        "cooldown_seconds": 30,
        "escalate_to": RecoveryAction.ABORT,
    },
    ErrorCategory.PERSONA: {
        "max_retries": 2,
        "cooldown_seconds": 5,
        "escalate_to": RecoveryAction.SKIP_USER,
    },
    ErrorCategory.STATE: {
        "max_retries": 1,
        "cooldown_seconds": 0,
        "escalate_to": RecoveryAction.SKIP_USER,
    },
    ErrorCategory.POLICY: {
        "max_retries": 0,
        "cooldown_seconds": 0,
        "escalate_to": RecoveryAction.SKIP_USER,
    },
    ErrorCategory.UI: {
        "max_retries": 3,
        "cooldown_seconds": 2,
        "escalate_to": RecoveryAction.SKIP_USER,
    },
}


def get_recovery_strategy(error: DMBotError, retry_count: int = 0) -> RecoveryAction:
    """Determine recovery action based on error type and retry count.

    Args:
        error: The error that occurred
        retry_count: Number of retries already attempted

    Returns:
        RecoveryAction to take
    """
    config = RECOVERY_CONFIG.get(error.category, {})
    max_retries = config.get("max_retries", 0)

    if retry_count >= max_retries:
        return config.get("escalate_to", RecoveryAction.SKIP_USER)

    return error.recovery


if __name__ == "__main__":
    # Test error creation
    logging.basicConfig(level=logging.DEBUG)

    err = InfrastructureError(
        "Appium connection lost",
        user_id="test123",
        driver_status="disconnected",
    )
    print(err)
    err.log()

    err2 = PersonaContractError(
        "Invalid JSON from LLM",
        raw_response='{"broken": json',
        user_id="test123",
    )
    print(err2)
    print(f"Recovery: {get_recovery_strategy(err2, retry_count=0)}")
    print(f"After 2 retries: {get_recovery_strategy(err2, retry_count=2)}")
