"""Real-time dashboard for meta-agent refinement progress."""

from .events import (
    Event,
    EventEmitter,
    EventType,
    SessionState,
    emit,
    get_emitter,
)
from .server import app, run_server

__all__ = [
    "Event",
    "EventEmitter",
    "EventType",
    "SessionState",
    "app",
    "emit",
    "get_emitter",
    "run_server",
]
