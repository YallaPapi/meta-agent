"""Event system for real-time dashboard updates.

This module provides a simple event emitter that broadcasts events
to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events emitted during refinement."""

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Phase/iteration tracking
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"

    # Layer tracking
    LAYER_UPDATE = "layer_update"

    # Task tracking
    TASK_LIST = "task_list"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"

    # File operations
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"

    # Progress updates
    PROGRESS = "progress"
    LOG = "log"
    ERROR = "error"

    # Analysis steps
    OLLAMA_START = "ollama_start"
    OLLAMA_COMPLETE = "ollama_complete"
    PERPLEXITY_START = "perplexity_start"
    PERPLEXITY_COMPLETE = "perplexity_complete"


@dataclass
class Event:
    """A single event to broadcast to clients."""

    type: EventType
    data: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        })


@dataclass
class SessionState:
    """Current state of a refinement session."""

    session_id: str = ""
    phase: str = ""
    phase_number: int = 0
    total_phases: int = 5
    current_layer: int = 1
    layer_name: str = "scaffold"
    layers_complete: dict = field(default_factory=lambda: {
        "scaffold": False,
        "core": False,
        "integration": False,
        "polish": False,
    })
    iteration: int = 0
    max_iterations: int = 10
    tasks: list = field(default_factory=list)
    current_task: Optional[str] = None
    current_task_index: int = 0
    total_tasks: int = 0
    tasks_completed: int = 0
    files_modified: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    started_at: str = ""
    feature_request: str = ""

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return asdict(self)


class EventEmitter:
    """Singleton event emitter for broadcasting to WebSocket clients."""

    _instance: Optional[EventEmitter] = None
    _lock = asyncio.Lock() if asyncio.get_event_loop_policy() else None

    def __new__(cls) -> EventEmitter:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._subscribers: list[Callable[[Event], None]] = []
        self._async_subscribers: list[Callable[[Event], Any]] = []
        self._state = SessionState()
        self._event_queue: asyncio.Queue = None
        self._running = False

    @property
    def state(self) -> SessionState:
        """Get current session state."""
        return self._state

    def reset_state(self) -> None:
        """Reset session state for a new session."""
        self._state = SessionState()

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to events with a synchronous callback."""
        self._subscribers.append(callback)

    def subscribe_async(self, callback: Callable[[Event], Any]) -> None:
        """Subscribe to events with an async callback."""
        self._async_subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Remove a subscriber."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
        if callback in self._async_subscribers:
            self._async_subscribers.remove(callback)

    def emit(self, event_type: EventType, data: Optional[dict] = None) -> None:
        """Emit an event to all subscribers.

        This method is synchronous and can be called from the orchestrator.
        """
        event = Event(type=event_type, data=data or {})

        # Update internal state based on event
        self._update_state(event)

        # Add to logs
        if event_type == EventType.LOG:
            self._state.logs.append({
                "timestamp": event.timestamp,
                "message": data.get("message", "") if data else "",
                "level": data.get("level", "info") if data else "info",
            })
            # Keep only last 100 logs
            if len(self._state.logs) > 100:
                self._state.logs = self._state.logs[-100:]

        # Call sync subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")

        # Queue for async subscribers
        if self._event_queue is not None:
            try:
                self._event_queue.put_nowait(event)
            except Exception:
                pass  # Queue might be full or not running

    def _update_state(self, event: Event) -> None:
        """Update session state based on event."""
        data = event.data

        if event.type == EventType.SESSION_START:
            self._state.session_id = data.get("session_id", "")
            self._state.feature_request = data.get("feature_request", "")
            self._state.started_at = event.timestamp
            self._state.max_iterations = data.get("max_iterations", 10)

        elif event.type == EventType.PHASE_START:
            self._state.phase = data.get("phase", "")
            self._state.phase_number = data.get("phase_number", 0)
            self._state.total_phases = data.get("total_phases", 5)

        elif event.type == EventType.ITERATION_START:
            self._state.iteration = data.get("iteration", 0)

        elif event.type == EventType.LAYER_UPDATE:
            self._state.current_layer = data.get("current_layer", 1)
            self._state.layer_name = data.get("layer_name", "scaffold")
            if "layers_complete" in data:
                self._state.layers_complete = data["layers_complete"]

        elif event.type == EventType.TASK_LIST:
            self._state.tasks = data.get("tasks", [])
            self._state.total_tasks = len(self._state.tasks)
            self._state.tasks_completed = 0
            self._state.current_task_index = 0

        elif event.type == EventType.TASK_START:
            self._state.current_task = data.get("title", "")
            self._state.current_task_index = data.get("index", 0)

        elif event.type == EventType.TASK_COMPLETE:
            self._state.tasks_completed += 1
            self._state.current_task = None

        elif event.type == EventType.FILE_CREATED or event.type == EventType.FILE_MODIFIED:
            filepath = data.get("file", "")
            if filepath and filepath not in self._state.files_modified:
                self._state.files_modified.append(filepath)

        elif event.type == EventType.ERROR:
            self._state.errors.append({
                "timestamp": event.timestamp,
                "message": data.get("message", ""),
            })

    async def run_async_dispatch(self) -> None:
        """Run async event dispatch loop."""
        self._event_queue = asyncio.Queue()
        self._running = True

        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                for callback in self._async_subscribers:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Error in async event subscriber: {e}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in async dispatch: {e}")

    def stop_async_dispatch(self) -> None:
        """Stop the async dispatch loop."""
        self._running = False


# Global emitter instance
_emitter: Optional[EventEmitter] = None


def get_emitter() -> EventEmitter:
    """Get the global event emitter instance."""
    global _emitter
    if _emitter is None:
        _emitter = EventEmitter()
    return _emitter


def emit(event_type: EventType, data: Optional[dict] = None) -> None:
    """Convenience function to emit an event."""
    get_emitter().emit(event_type, data)
