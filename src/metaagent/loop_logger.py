"""Logging utilities for the autonomous development loop.

This module provides structured logging for the autonomous loop,
including file logging, token tracking, and API call logging.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class APICallLog:
    """Log entry for an API call."""

    timestamp: str
    api: str  # "anthropic", "perplexity", "ollama"
    purpose: str
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class LoopStats:
    """Statistics for the autonomous loop."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    iterations: int = 0
    tasks_attempted: int = 0
    tasks_completed: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    fixes_attempted: int = 0
    fixes_successful: int = 0
    commits_made: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    api_calls: list[APICallLog] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "iterations": self.iterations,
            "tasks": {
                "attempted": self.tasks_attempted,
                "completed": self.tasks_completed,
            },
            "tests": {
                "run": self.tests_run,
                "passed": self.tests_passed,
            },
            "fixes": {
                "attempted": self.fixes_attempted,
                "successful": self.fixes_successful,
            },
            "commits": self.commits_made,
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_tokens,
            },
            "api_calls": [
                {
                    "timestamp": call.timestamp,
                    "api": call.api,
                    "purpose": call.purpose,
                    "tokens": call.input_tokens + call.output_tokens,
                    "success": call.success,
                }
                for call in self.api_calls
            ],
        }


class LoopLogger:
    """Logger for the autonomous development loop."""

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        feature_name: str = "loop",
    ):
        """Initialize the loop logger.

        Args:
            log_dir: Directory for log files. Defaults to logs/autonomous_loop.
            feature_name: Name of the feature being implemented.
        """
        self.log_dir = log_dir or Path("logs/autonomous_loop")
        self.feature_name = feature_name
        self.stats = LoopStats()

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_feature = "".join(c if c.isalnum() else "_" for c in feature_name[:30])
        self.log_file = self.log_dir / f"{timestamp}_{safe_feature}.json"

        # Initialize log data
        self.log_data = {
            "session": {
                "id": timestamp,
                "feature": feature_name,
                "start_time": datetime.now().isoformat(),
            },
            "iterations": [],
            "api_calls": [],
            "errors": [],
        }

        logger.info(f"Loop logger initialized: {self.log_file}")

    def log_iteration_start(self, iteration: int, max_iterations: int) -> None:
        """Log the start of an iteration."""
        self.stats.iterations = iteration
        self.log_data["iterations"].append({
            "number": iteration,
            "max": max_iterations,
            "start_time": datetime.now().isoformat(),
            "tasks": [],
        })
        logger.info(f"Iteration {iteration}/{max_iterations} started")

    def log_iteration_end(self, iteration: int, tasks_completed: int) -> None:
        """Log the end of an iteration."""
        if self.log_data["iterations"]:
            self.log_data["iterations"][-1]["end_time"] = datetime.now().isoformat()
            self.log_data["iterations"][-1]["tasks_completed"] = tasks_completed

    def log_task_start(self, task: dict) -> None:
        """Log the start of a task implementation."""
        self.stats.tasks_attempted += 1
        task_log = {
            "title": task.get("title", "Unknown"),
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
        }
        if self.log_data["iterations"]:
            self.log_data["iterations"][-1]["tasks"].append(task_log)

    def log_task_end(
        self,
        task: dict,
        success: bool,
        tests_passed: bool,
        fix_applied: bool = False,
    ) -> None:
        """Log the end of a task implementation."""
        if success:
            self.stats.tasks_completed += 1
        self.stats.tests_run += 1
        if tests_passed:
            self.stats.tests_passed += 1
        if fix_applied:
            self.stats.fixes_attempted += 1
            self.stats.fixes_successful += 1

        if self.log_data["iterations"] and self.log_data["iterations"][-1]["tasks"]:
            task_log = self.log_data["iterations"][-1]["tasks"][-1]
            task_log["end_time"] = datetime.now().isoformat()
            task_log["status"] = "completed" if success else "failed"
            task_log["tests_passed"] = tests_passed
            task_log["fix_applied"] = fix_applied

    def log_api_call(
        self,
        api: str,
        purpose: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: int = 0,
    ) -> None:
        """Log an API call."""
        call = APICallLog(
            timestamp=datetime.now().isoformat(),
            api=api,
            purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            error=error,
            duration_ms=duration_ms,
        )
        self.stats.api_calls.append(call)
        self.stats.total_input_tokens += input_tokens
        self.stats.total_output_tokens += output_tokens

        self.log_data["api_calls"].append({
            "timestamp": call.timestamp,
            "api": api,
            "purpose": purpose,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "success": success,
            "error": error,
            "duration_ms": duration_ms,
        })

    def log_commit(self, commit_hash: str, message: str) -> None:
        """Log a git commit."""
        self.stats.commits_made += 1
        if self.log_data["iterations"]:
            if "commits" not in self.log_data["iterations"][-1]:
                self.log_data["iterations"][-1]["commits"] = []
            self.log_data["iterations"][-1]["commits"].append({
                "hash": commit_hash,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            })

    def log_error(self, error: str, context: Optional[dict] = None) -> None:
        """Log an error."""
        self.log_data["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {},
        })
        logger.error(f"Loop error: {error}")

    def finalize(self, success: bool, final_evaluation: Optional[str] = None) -> None:
        """Finalize the log and write to file."""
        self.stats.end_time = datetime.now()

        self.log_data["session"]["end_time"] = datetime.now().isoformat()
        self.log_data["session"]["success"] = success
        self.log_data["session"]["final_evaluation"] = final_evaluation
        self.log_data["stats"] = self.stats.to_dict()

        # Write log file
        try:
            with open(self.log_file, "w") as f:
                json.dump(self.log_data, f, indent=2)
            logger.info(f"Loop log written to: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to write loop log: {e}")

    def print_summary(self) -> None:
        """Print a summary of the loop execution."""
        print("\n" + "=" * 60)
        print("AUTONOMOUS LOOP SUMMARY")
        print("=" * 60)
        print(f"Feature: {self.feature_name}")
        print(f"Duration: {self.stats.duration_seconds:.1f}s")
        print(f"Iterations: {self.stats.iterations}")
        print(f"Tasks: {self.stats.tasks_completed}/{self.stats.tasks_attempted} completed")
        print(f"Tests: {self.stats.tests_passed}/{self.stats.tests_run} passed")
        print(f"Fixes: {self.stats.fixes_successful}/{self.stats.fixes_attempted} successful")
        print(f"Commits: {self.stats.commits_made}")
        print(f"Tokens: {self.stats.total_tokens:,} ({self.stats.total_input_tokens:,} in, {self.stats.total_output_tokens:,} out)")
        print(f"API calls: {len(self.stats.api_calls)}")
        print(f"Log file: {self.log_file}")
        print("=" * 60 + "\n")
