"""Tests for the custom task management system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metaagent.task_manager import (
    Task,
    Subtask,
    TaskManager,
    TaskManagerState,
    TaskManagerError,
    MockTaskManager,
)


class TestSubtask:
    """Tests for Subtask dataclass."""

    def test_create_subtask(self):
        """Test creating a subtask."""
        st = Subtask(id="1.1", title="Test subtask", description="Do something")
        assert st.id == "1.1"
        assert st.title == "Test subtask"
        assert st.status == "pending"

    def test_to_dict(self):
        """Test converting to dict."""
        st = Subtask(id="1.1", title="Test", status="done")
        d = st.to_dict()
        assert d['id'] == "1.1"
        assert d['status'] == "done"

    def test_from_dict(self):
        """Test creating from dict."""
        d = {'id': '2.1', 'title': 'From dict', 'description': 'Desc', 'status': 'in_progress', 'notes': '', 'created_at': '2024-01-01'}
        st = Subtask.from_dict(d)
        assert st.id == '2.1'
        assert st.status == 'in_progress'


class TestTask:
    """Tests for Task dataclass."""

    def test_create_task(self):
        """Test creating a task."""
        task = Task(id="1", title="Test task", priority="high")
        assert task.id == "1"
        assert task.title == "Test task"
        assert task.priority == "high"
        assert task.status == "pending"
        assert task.subtasks == []

    def test_task_with_subtasks(self):
        """Test task with subtasks."""
        subtasks = [
            Subtask(id="1.1", title="Sub 1"),
            Subtask(id="1.2", title="Sub 2"),
        ]
        task = Task(id="1", title="Main", subtasks=subtasks)
        assert len(task.subtasks) == 2
        assert task.subtasks[0].id == "1.1"

    def test_add_error(self):
        """Test adding error to task."""
        task = Task(id="1", title="Test")
        task.add_error("Something went wrong")
        assert len(task.error_log) == 1
        assert "Something went wrong" in task.error_log[0]
        assert task.fix_attempts == 1

    def test_to_dict_and_from_dict(self):
        """Test round-trip conversion."""
        original = Task(
            id="1",
            title="Test",
            description="Desc",
            priority="critical",
            status="in_progress",
            subtasks=[Subtask(id="1.1", title="Sub")],
        )
        d = original.to_dict()
        restored = Task.from_dict(d)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.priority == original.priority
        assert len(restored.subtasks) == 1
        assert restored.subtasks[0].id == "1.1"


class TestTaskManagerState:
    """Tests for TaskManagerState."""

    def test_create_state(self):
        """Test creating state."""
        tasks = [Task(id="1", title="Task 1")]
        state = TaskManagerState(
            prd_path="/path/to/prd.md",
            prd_hash="abc123",
            tasks=tasks,
            created_at="2024-01-01",
            last_updated="2024-01-01",
        )
        assert state.prd_path == "/path/to/prd.md"
        assert len(state.tasks) == 1

    def test_to_dict_and_from_dict(self):
        """Test round-trip conversion."""
        original = TaskManagerState(
            prd_path="/prd.md",
            prd_hash="xyz",
            tasks=[Task(id="1", title="T1"), Task(id="2", title="T2")],
            created_at="2024-01-01",
            last_updated="2024-01-02",
            iteration_count=5,
            total_fixes=3,
        )
        d = original.to_dict()
        restored = TaskManagerState.from_dict(d)

        assert restored.prd_path == original.prd_path
        assert len(restored.tasks) == 2
        assert restored.iteration_count == 5
        assert restored.total_fixes == 3


class TestTaskManager:
    """Tests for TaskManager class."""

    def test_init_empty(self):
        """Test initializing without PRD."""
        tm = TaskManager()
        assert tm.state is None
        assert tm.get_tasks() == []

    def test_init_with_prd_text(self, tmp_path):
        """Test initializing with PRD text and mock query."""
        mock_response = json.dumps([
            {"id": "1", "title": "Task 1", "description": "Do thing 1", "priority": "high", "subtasks": []},
            {"id": "2", "title": "Task 2", "description": "Do thing 2", "priority": "medium", "subtasks": []},
        ])

        def mock_query(prompt):
            return mock_response

        tm = TaskManager(
            prd_text="Build a feature",
            tasks_file=str(tmp_path / "tasks.json"),
            query_fn=mock_query,
        )

        assert len(tm.get_tasks()) == 2
        assert tm.get_task("1").title == "Task 1"

    def test_init_with_prd_file(self, tmp_path):
        """Test initializing with PRD file."""
        prd_file = tmp_path / "prd.md"
        prd_file.write_text("# PRD\nBuild something cool")

        mock_response = json.dumps([
            {"id": "1", "title": "Cool task", "priority": "high", "subtasks": []}
        ])

        tm = TaskManager(
            prd_path=str(prd_file),
            tasks_file=str(tmp_path / "tasks.json"),
            query_fn=lambda p: mock_response,
        )

        assert len(tm.get_tasks()) == 1

    def test_get_next_task(self):
        """Test getting next task."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="First", status="done"),
            Task(id="2", title="Second", status="pending"),
            Task(id="3", title="Third", status="pending"),
        ])

        next_task = tm.get_next_task()
        assert next_task.id == "2"

    def test_get_next_task_prioritizes_in_progress(self):
        """Test that in-progress tasks are returned first."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="Pending", status="pending", priority="high"),
            Task(id="2", title="In Progress", status="in_progress", priority="low"),
        ])

        next_task = tm.get_next_task()
        assert next_task.id == "2"  # In-progress first

    def test_get_next_task_by_priority(self):
        """Test that pending tasks are sorted by priority."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="Low", status="pending", priority="low"),
            Task(id="2", title="Critical", status="pending", priority="critical"),
            Task(id="3", title="Medium", status="pending", priority="medium"),
        ])

        next_task = tm.get_next_task()
        assert next_task.id == "2"  # Critical first

    def test_set_task_status(self):
        """Test updating task status."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="Task", status="pending"),
        ])

        tm.set_task_status("1", "in_progress")
        assert tm.get_task("1").status == "in_progress"

        tm.set_task_status("1", "done", notes="Completed successfully")
        task = tm.get_task("1")
        assert task.status == "done"
        assert task.notes == "Completed successfully"
        assert task.completed_at is not None

    def test_set_task_status_not_found(self):
        """Test updating non-existent task."""
        tm = MockTaskManager(tasks=[])

        with pytest.raises(TaskManagerError) as exc_info:
            tm.set_task_status("999", "done")
        assert "not found" in str(exc_info.value)

    def test_add_task_error(self):
        """Test adding error to task."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="Task"),
        ])

        tm.add_task_error("1", "Test failed: assertion error")
        task = tm.get_task("1")

        assert len(task.error_log) == 1
        assert "Test failed" in task.error_log[0]
        assert task.fix_attempts == 1

    def test_increment_iteration(self):
        """Test incrementing iteration count."""
        tm = MockTaskManager(tasks=[])
        tm.state.iteration_count = 0

        count = tm.increment_iteration()
        assert count == 1

        count = tm.increment_iteration()
        assert count == 2

    def test_get_progress(self):
        """Test getting progress summary."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="T1", status="done"),
            Task(id="2", title="T2", status="done"),
            Task(id="3", title="T3", status="pending"),
            Task(id="4", title="T4", status="in_progress"),
        ])
        tm.state.iteration_count = 5
        tm.state.total_fixes = 2

        progress = tm.get_progress()

        assert progress['total'] == 4
        assert progress['completed'] == 2
        assert progress['percentage'] == 50.0
        assert progress['by_status']['done'] == 2
        assert progress['by_status']['pending'] == 1
        assert progress['iterations'] == 5
        assert progress['total_fixes'] == 2

    def test_is_complete(self):
        """Test checking if all tasks complete."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="T1", status="done"),
            Task(id="2", title="T2", status="pending"),
        ])
        assert not tm.is_complete()

        tm.set_task_status("2", "done")
        assert tm.is_complete()

    def test_is_complete_empty(self):
        """Test is_complete with no tasks."""
        tm = MockTaskManager(tasks=[])
        assert not tm.is_complete()  # Empty is not complete

    def test_reset(self):
        """Test resetting all tasks."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="T1", status="done", notes="Done", fix_attempts=2),
            Task(id="2", title="T2", status="failed", error_log=["Error"]),
        ])
        tm.state.iteration_count = 10
        tm.state.total_fixes = 5

        tm.reset()

        for task in tm.get_tasks():
            assert task.status == "pending"
            assert task.notes == ""
            assert task.error_log == []
            assert task.fix_attempts == 0

        assert tm.state.iteration_count == 0
        assert tm.state.total_fixes == 0

    def test_save_and_load(self, tmp_path):
        """Test persisting and loading state."""
        tasks_file = tmp_path / "tasks.json"

        # Create and save
        tm1 = MockTaskManager(tasks=[
            Task(id="1", title="Persist me", priority="high"),
        ])
        tm1.tasks_file = tasks_file
        tm1.state.iteration_count = 7
        # Override save to actually write
        with open(tasks_file, 'w') as f:
            json.dump(tm1.state.to_dict(), f)

        # Load in new instance
        tm2 = TaskManager(tasks_file=str(tasks_file))

        assert len(tm2.get_tasks()) == 1
        assert tm2.get_task("1").title == "Persist me"
        assert tm2.state.iteration_count == 7

    def test_to_markdown(self):
        """Test markdown export."""
        tm = MockTaskManager(tasks=[
            Task(id="1", title="Done task", status="done", priority="high"),
            Task(
                id="2",
                title="Pending task",
                status="pending",
                description="Do this thing",
                subtasks=[Subtask(id="2.1", title="Sub 1")],
            ),
        ])

        md = tm.to_markdown()

        assert "# Tasks" in md
        assert "[x] 1. Done task" in md
        assert "[ ] 2. Pending task" in md
        assert "Do this thing" in md
        assert "2.1. Sub 1" in md

    def test_parse_llm_response_with_code_block(self, tmp_path):
        """Test parsing JSON from markdown code block."""
        response = '''Here are the tasks:

```json
[
  {"id": "1", "title": "Task 1", "description": "Do 1", "priority": "high", "subtasks": []},
  {"id": "2", "title": "Task 2", "description": "Do 2", "priority": "medium", "subtasks": []}
]
```

Let me know if you need more.'''

        tm = TaskManager(tasks_file=str(tmp_path / "tasks.json"))
        tasks = tm._parse_llm_response(response)

        assert len(tasks) == 2
        assert tasks[0].title == "Task 1"
        assert tasks[1].priority == "medium"

    def test_parse_llm_response_raw_json(self, tmp_path):
        """Test parsing raw JSON response."""
        response = '[{"id": "1", "title": "Single", "priority": "low", "subtasks": []}]'

        tm = TaskManager(tasks_file=str(tmp_path / "tasks.json"))
        tasks = tm._parse_llm_response(response)

        assert len(tasks) == 1
        assert tasks[0].title == "Single"

    def test_init_nonexistent_prd_file(self, tmp_path):
        """Test error when PRD file doesn't exist."""
        with pytest.raises(TaskManagerError) as exc_info:
            TaskManager(prd_path="/nonexistent/prd.md")
        assert "not found" in str(exc_info.value)


class TestMockTaskManager:
    """Tests for MockTaskManager."""

    def test_mock_no_persistence(self):
        """Test that mock doesn't persist."""
        tm = MockTaskManager(tasks=[Task(id="1", title="Test")])
        tm.save()  # Should not raise
        tm.load()  # Should not raise
