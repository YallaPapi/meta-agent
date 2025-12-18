"""Custom task management system for meta-agent autonomous loop.

This module provides a lightweight task management system that:
- Parses PRDs into structured tasks using Grok/LLM
- Tracks task status (pending, in_progress, done, failed)
- Manages subtasks and error notes
- Persists state to JSON for resumability

Designed to be license-free and Windows-compatible.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class Subtask:
    """A subtask within a parent task."""

    id: str
    title: str
    description: str = ""
    status: str = "pending"  # pending, in_progress, done, failed
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Subtask:
        return cls(**data)


@dataclass
class Task:
    """A task parsed from a PRD."""

    id: str
    title: str
    description: str = ""
    priority: str = "medium"  # critical, high, medium, low
    status: str = "pending"  # pending, in_progress, done, failed, blocked
    subtasks: List[Subtask] = field(default_factory=list)
    notes: str = ""
    error_log: List[str] = field(default_factory=list)
    fix_attempts: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['subtasks'] = [s.to_dict() if isinstance(s, Subtask) else s for s in self.subtasks]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        subtasks_data = data.pop('subtasks', [])
        task = cls(**data)
        task.subtasks = [
            Subtask.from_dict(s) if isinstance(s, dict) else s
            for s in subtasks_data
        ]
        return task

    def add_error(self, error: str) -> None:
        """Log an error for this task."""
        timestamp = datetime.now().isoformat()
        self.error_log.append(f"[{timestamp}] {error}")
        self.fix_attempts += 1


@dataclass
class TaskManagerState:
    """Persistent state for the task manager."""

    prd_path: str
    prd_hash: str
    tasks: List[Task]
    created_at: str
    last_updated: str
    iteration_count: int = 0
    total_fixes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prd_path': self.prd_path,
            'prd_hash': self.prd_hash,
            'tasks': [t.to_dict() for t in self.tasks],
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'iteration_count': self.iteration_count,
            'total_fixes': self.total_fixes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskManagerState:
        tasks = [Task.from_dict(t) for t in data.get('tasks', [])]
        return cls(
            prd_path=data['prd_path'],
            prd_hash=data['prd_hash'],
            tasks=tasks,
            created_at=data['created_at'],
            last_updated=data['last_updated'],
            iteration_count=data.get('iteration_count', 0),
            total_fixes=data.get('total_fixes', 0),
        )


class TaskManagerError(Exception):
    """Exception raised for task manager errors."""
    pass


class TaskManager:
    """Manages tasks parsed from a PRD for autonomous development.

    Features:
    - Parse PRD into structured tasks using LLM
    - Track task/subtask status
    - Persist state for resumability
    - Windows-compatible path handling
    """

    DEFAULT_TASKS_FILE = "meta_agent_tasks.json"

    def __init__(
        self,
        prd_path: Optional[str] = None,
        prd_text: Optional[str] = None,
        tasks_file: Optional[str] = None,
        query_fn: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the task manager.

        Args:
            prd_path: Path to PRD file.
            prd_text: PRD text content (alternative to prd_path).
            tasks_file: Path to persist tasks. Defaults to meta_agent_tasks.json.
            query_fn: Function to query LLM for parsing. Signature: (prompt) -> response.
        """
        self.tasks_file = Path(tasks_file or self.DEFAULT_TASKS_FILE)
        self.query_fn = query_fn
        self.state: Optional[TaskManagerState] = None

        # Load existing state or parse new PRD
        if self.tasks_file.exists():
            self.load()
            logger.info(f"Loaded existing task state from {self.tasks_file}")
        elif prd_path or prd_text:
            self._initialize_from_prd(prd_path, prd_text)
        else:
            # Empty state - will need to call parse_prd later
            self.state = None

    def _initialize_from_prd(self, prd_path: Optional[str], prd_text: Optional[str]) -> None:
        """Initialize task manager from PRD."""
        if prd_path:
            prd_file = Path(prd_path)
            if not prd_file.exists():
                raise TaskManagerError(f"PRD file not found: {prd_path}")
            prd_text = prd_file.read_text(encoding='utf-8')
            prd_path_str = str(prd_file.absolute())
        else:
            prd_path_str = "inline"

        # Hash PRD for change detection
        import hashlib
        prd_hash = hashlib.md5(prd_text.encode()).hexdigest()[:12]

        now = datetime.now().isoformat()
        self.state = TaskManagerState(
            prd_path=prd_path_str,
            prd_hash=prd_hash,
            tasks=[],
            created_at=now,
            last_updated=now,
        )

        # Parse PRD into tasks
        if self.query_fn:
            self.parse_prd(prd_text)
        else:
            logger.warning("No query function provided - tasks list is empty")

    def parse_prd(self, prd_text: str) -> List[Task]:
        """Parse PRD text into structured tasks using LLM.

        Args:
            prd_text: The PRD content to parse.

        Returns:
            List of parsed tasks.
        """
        if not self.query_fn:
            raise TaskManagerError("No query function configured for PRD parsing")

        prompt = self._build_parse_prompt(prd_text)

        try:
            response = self.query_fn(prompt)
            tasks = self._parse_llm_response(response)

            if self.state:
                self.state.tasks = tasks
                self.state.last_updated = datetime.now().isoformat()
                self.save()

            logger.info(f"Parsed {len(tasks)} tasks from PRD")
            return tasks

        except Exception as e:
            logger.error(f"Failed to parse PRD: {e}")
            raise TaskManagerError(f"PRD parsing failed: {e}") from e

    def _build_parse_prompt(self, prd_text: str) -> str:
        """Build the prompt for PRD parsing."""
        return f"""Analyze this PRD and break it into implementation tasks.

PRD:
{prd_text}

Instructions:
1. Create 5-15 prioritized tasks based on the PRD requirements
2. Each task should be a concrete implementation step
3. Order by dependency (earlier tasks should not depend on later ones)
4. Include subtasks for complex items

Output ONLY a JSON array with this structure (no other text):
[
  {{
    "id": "1",
    "title": "Task title",
    "description": "Detailed description of what to implement",
    "priority": "high|medium|low",
    "subtasks": [
      {{"id": "1.1", "title": "Subtask", "description": "Details"}}
    ]
  }}
]

Focus on actionable implementation tasks, not documentation or planning."""

    def _parse_llm_response(self, response: str) -> List[Task]:
        """Parse LLM response into Task objects."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON array
            json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response[:500]}")
            raise TaskManagerError(f"Invalid JSON in LLM response: {e}")

        tasks = []
        for item in data:
            subtasks = []
            for st in item.get('subtasks', []):
                subtasks.append(Subtask(
                    id=str(st.get('id', '')),
                    title=st.get('title', ''),
                    description=st.get('description', ''),
                ))

            task = Task(
                id=str(item.get('id', len(tasks) + 1)),
                title=item.get('title', ''),
                description=item.get('description', ''),
                priority=item.get('priority', 'medium'),
                subtasks=subtasks,
            )
            tasks.append(task)

        return tasks

    def get_tasks(self) -> List[Task]:
        """Get all tasks."""
        return self.state.tasks if self.state else []

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID."""
        for task in self.get_tasks():
            if task.id == task_id:
                return task
        return None

    def get_next_task(self) -> Optional[Task]:
        """Get the next pending task to work on.

        Returns tasks in order, prioritizing:
        1. In-progress tasks (resume)
        2. Pending tasks by priority
        """
        tasks = self.get_tasks()

        # First, check for in-progress tasks
        for task in tasks:
            if task.status == 'in_progress':
                return task

        # Then get first pending task
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        pending = [t for t in tasks if t.status == 'pending']
        pending.sort(key=lambda t: priority_order.get(t.priority, 2))

        return pending[0] if pending else None

    def set_task_status(
        self,
        task_id: str,
        status: str,
        notes: str = "",
    ) -> None:
        """Update a task's status.

        Args:
            task_id: Task ID to update.
            status: New status (pending, in_progress, done, failed, blocked).
            notes: Optional notes about the status change.
        """
        task = self.get_task(task_id)
        if not task:
            raise TaskManagerError(f"Task not found: {task_id}")

        task.status = status
        if notes:
            task.notes = notes

        if status == 'done':
            task.completed_at = datetime.now().isoformat()

        if self.state:
            self.state.last_updated = datetime.now().isoformat()
            self.save()

        logger.info(f"Task {task_id} status -> {status}")

    def add_task_error(self, task_id: str, error: str) -> None:
        """Add an error to a task's error log.

        Args:
            task_id: Task ID.
            error: Error message to log.
        """
        task = self.get_task(task_id)
        if not task:
            raise TaskManagerError(f"Task not found: {task_id}")

        task.add_error(error)

        if self.state:
            self.state.total_fixes += 1
            self.state.last_updated = datetime.now().isoformat()
            self.save()

    def increment_iteration(self) -> int:
        """Increment and return the iteration count."""
        if self.state:
            self.state.iteration_count += 1
            self.state.last_updated = datetime.now().isoformat()
            self.save()
            return self.state.iteration_count
        return 0

    def get_progress(self) -> Dict[str, Any]:
        """Get progress summary.

        Returns:
            Dict with counts by status and completion percentage.
        """
        tasks = self.get_tasks()
        total = len(tasks)

        if total == 0:
            return {'total': 0, 'completed': 0, 'percentage': 0}

        by_status = {}
        for task in tasks:
            by_status[task.status] = by_status.get(task.status, 0) + 1

        done = by_status.get('done', 0)

        return {
            'total': total,
            'completed': done,
            'percentage': round(done / total * 100, 1),
            'by_status': by_status,
            'iterations': self.state.iteration_count if self.state else 0,
            'total_fixes': self.state.total_fixes if self.state else 0,
        }

    def is_complete(self) -> bool:
        """Check if all tasks are done."""
        tasks = self.get_tasks()
        return len(tasks) > 0 and all(t.status == 'done' for t in tasks)

    def save(self) -> None:
        """Persist state to JSON file."""
        if not self.state:
            return

        try:
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            logger.debug(f"Saved task state to {self.tasks_file}")
        except Exception as e:
            logger.error(f"Failed to save task state: {e}")

    def load(self) -> None:
        """Load state from JSON file."""
        try:
            with open(self.tasks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.state = TaskManagerState.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load task state: {e}")
            raise TaskManagerError(f"Failed to load tasks: {e}") from e

    def reset(self) -> None:
        """Reset all tasks to pending status."""
        for task in self.get_tasks():
            task.status = 'pending'
            task.notes = ''
            task.error_log = []
            task.fix_attempts = 0
            task.completed_at = None
            for subtask in task.subtasks:
                subtask.status = 'pending'
                subtask.notes = ''

        if self.state:
            self.state.iteration_count = 0
            self.state.total_fixes = 0
            self.state.last_updated = datetime.now().isoformat()
            self.save()

    def to_markdown(self) -> str:
        """Export tasks as markdown."""
        lines = ["# Tasks\n"]

        progress = self.get_progress()
        lines.append(f"**Progress:** {progress['completed']}/{progress['total']} ({progress['percentage']}%)\n")

        for task in self.get_tasks():
            status_icon = {
                'pending': '[ ]',
                'in_progress': '[~]',
                'done': '[x]',
                'failed': '[!]',
                'blocked': '[B]',
            }.get(task.status, '[ ]')

            lines.append(f"\n## {status_icon} {task.id}. {task.title}")
            lines.append(f"**Priority:** {task.priority} | **Status:** {task.status}")
            if task.description:
                lines.append(f"\n{task.description}")

            if task.subtasks:
                lines.append("\n**Subtasks:**")
                for st in task.subtasks:
                    st_icon = '[x]' if st.status == 'done' else '[ ]'
                    lines.append(f"- {st_icon} {st.id}. {st.title}")

            if task.error_log:
                lines.append("\n**Errors:**")
                for err in task.error_log[-3:]:  # Last 3 errors
                    lines.append(f"- {err}")

        return '\n'.join(lines)


class MockTaskManager(TaskManager):
    """Mock task manager for testing."""

    def __init__(self, tasks: Optional[List[Task]] = None):
        """Initialize with optional predefined tasks."""
        self.tasks_file = Path("mock_tasks.json")
        self.query_fn = None

        now = datetime.now().isoformat()
        self.state = TaskManagerState(
            prd_path="mock",
            prd_hash="mock123",
            tasks=tasks or [],
            created_at=now,
            last_updated=now,
        )

    def save(self) -> None:
        """No-op for mock."""
        pass

    def load(self) -> None:
        """No-op for mock."""
        pass
