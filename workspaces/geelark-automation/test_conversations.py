"""Conversation Test Harness for DM Bot.

Simulates multi-turn conversations to test:
- Persona voice and consistency
- Goal state progression
- JSON output quality
- Error handling and recovery

Usage:
    python test_conversations.py                    # Run all tests
    python test_conversations.py --scenario eager   # Run specific scenario
    python test_conversations.py --count 5          # Run 5 conversations
    python test_conversations.py --verbose          # Show full transcripts
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from config import load_config
from dm_bot import DMBot
from prompt_builder import VALID_GOALS

logger = logging.getLogger(__name__)


# =============================================================================
# Test Scenarios - Simulated User Behaviors
# =============================================================================

@dataclass
class TestScenario:
    """A test scenario with simulated user messages."""
    name: str
    description: str
    messages: List[str]
    expected_goal_progression: List[str]  # Expected goal at each turn
    tags: List[str] = field(default_factory=list)


# Different user personas/behaviors to test
TEST_SCENARIOS = [
    TestScenario(
        name="eager_responder",
        description="User who responds quickly and shares location easily",
        messages=[
            "hey! how's it going?",
            "im good! just chilling at home. you?",
            "haha nice. im from miami btw, you?",
            "oh cool! ive always wanted to go to LA",
            "yeah for sure! whats your ig?",
        ],
        expected_goal_progression=[
            "chatting",
            "chatting",
            "got_location",  # After they mention miami
            "got_location",
            "sending_link",
        ],
        tags=["happy_path", "quick"],
    ),
    TestScenario(
        name="slow_responder",
        description="User who gives short responses, needs more engagement",
        messages=[
            "hey",
            "good u",
            "nm",
            "yeah",
            "idk",
            "chicago",
        ],
        expected_goal_progression=[
            "chatting",
            "chatting",
            "asking_location",
            "asking_location",
            "asking_location",
            "got_location",
        ],
        tags=["slow", "challenge"],
    ),
    TestScenario(
        name="flirty_user",
        description="User who is flirty and engaged",
        messages=[
            "heyyy cutie 😊",
            "you're so pretty! where are you from?",
            "im in nyc, we should meet up sometime",
            "yeah id love that! whats your number?",
        ],
        expected_goal_progression=[
            "chatting",
            "asking_location",
            "got_location",
            "sending_link",
        ],
        tags=["flirty", "engaged"],
    ),
    TestScenario(
        name="suspicious_user",
        description="User who asks probing questions",
        messages=[
            "are you real?",
            "you seem like a bot",
            "prove youre not a bot",
            "ok fine. im from seattle",
        ],
        expected_goal_progression=[
            "chatting",
            "chatting",
            "chatting",
            "got_location",
        ],
        tags=["suspicious", "challenge"],
    ),
    TestScenario(
        name="location_first",
        description="User who mentions location immediately",
        messages=[
            "hey! im john from denver, nice to meet you",
            "so what do you do for fun?",
            "thats cool! we should hang out",
        ],
        expected_goal_progression=[
            "got_location",  # Immediate location
            "got_location",
            "sending_link",
        ],
        tags=["fast_progression"],
    ),
    TestScenario(
        name="topic_changer",
        description="User who changes topics frequently",
        messages=[
            "hey whats up",
            "do you like music?",
            "whats your favorite food?",
            "have you traveled anywhere cool?",
            "im from austin tx btw",
        ],
        expected_goal_progression=[
            "chatting",
            "chatting",
            "chatting",
            "asking_location",
            "got_location",
        ],
        tags=["random", "challenge"],
    ),
    TestScenario(
        name="rejection_path",
        description="User who seems uninterested",
        messages=[
            "hey",
            "not interested",
            "stop messaging me",
        ],
        expected_goal_progression=[
            "chatting",
            "rejected",
            "rejected",
        ],
        tags=["rejection", "edge_case"],
    ),
    TestScenario(
        name="long_conversation",
        description="Extended conversation to test state management",
        messages=[
            "hey there!",
            "how are you doing today?",
            "thats nice, what are you up to?",
            "sounds fun! do you go to school?",
            "cool what do you study?",
            "psychology is interesting! where do you go?",
            "oh LA! im from boston, its so different there",
            "yeah the weather is way better there haha",
            "we should totally meet up sometime!",
            "yeah that would be fun!",
        ],
        expected_goal_progression=[
            "chatting",
            "chatting",
            "chatting",
            "chatting",
            "chatting",
            "asking_location",
            "got_location",
            "got_location",
            "sending_link",
            "sending_link",
        ],
        tags=["long", "natural"],
    ),
]


# =============================================================================
# Test Results Tracking
# =============================================================================

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_number: int
    user_message: str
    bot_response: str
    goal_status: str
    expected_goal: str
    goal_correct: bool
    latency_ms: float
    trace_id: str
    error: Optional[str] = None


@dataclass
class ConversationResult:
    """Result of a single conversation test."""
    scenario_name: str
    user_id: str
    start_time: str
    end_time: str
    turns: List[ConversationTurn]
    success: bool
    goal_accuracy: float  # % of turns with correct goal
    total_latency_ms: float
    avg_latency_ms: float
    errors: List[str]
    final_goal: str


@dataclass
class TestReport:
    """Overall test report."""
    run_id: str
    start_time: str
    end_time: str
    total_scenarios: int
    successful_scenarios: int
    failed_scenarios: int
    total_turns: int
    goal_accuracy: float
    avg_latency_ms: float
    error_count: int
    results: List[ConversationResult]

    def summary(self) -> str:
        """Generate human-readable summary."""
        duration = datetime.fromisoformat(self.end_time) - datetime.fromisoformat(self.start_time)
        return f"""
================================================================================
                         DM BOT CONVERSATION TEST REPORT
================================================================================
Run ID: {self.run_id}
Duration: {duration}

OVERALL RESULTS
---------------
Scenarios Run:     {self.total_scenarios}
Successful:        {self.successful_scenarios} ({self.successful_scenarios/max(1,self.total_scenarios)*100:.0f}%)
Failed:            {self.failed_scenarios}
Total Turns:       {self.total_turns}
Goal Accuracy:     {self.goal_accuracy:.1f}%
Avg Latency:       {self.avg_latency_ms:.0f}ms
Errors:            {self.error_count}

SCENARIO BREAKDOWN
------------------
{"".join(self._scenario_summary(r) for r in self.results)}
================================================================================
"""

    def _scenario_summary(self, r: ConversationResult) -> str:
        status = "✓" if r.success else "✗"
        return f"{status} {r.scenario_name:20s} | Goals: {r.goal_accuracy:5.1f}% | Latency: {r.avg_latency_ms:5.0f}ms | Final: {r.final_goal}\n"


# =============================================================================
# Conversation Tester
# =============================================================================

class ConversationTester:
    """Test harness for running conversation simulations."""

    def __init__(self, config=None, output_dir: str = "./test_results"):
        self.config = config or load_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bot = DMBot(self.config)

    def run_scenario(self, scenario: TestScenario, verbose: bool = False) -> ConversationResult:
        """Run a single test scenario.

        Args:
            scenario: Test scenario to run
            verbose: Print detailed output

        Returns:
            ConversationResult with all turn data
        """
        user_id = f"test_{scenario.name}_{int(time.time())}"
        start_time = datetime.now().isoformat()
        turns = []
        errors = []
        total_latency = 0.0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario.name}")
            print(f"Description: {scenario.description}")
            print(f"{'='*60}")

        # Clear any existing state for this test user
        state_file = Path(self.config.state_dir) / f"{user_id}.json"
        if state_file.exists():
            state_file.unlink()

        for i, message in enumerate(scenario.messages):
            expected_goal = scenario.expected_goal_progression[i] if i < len(scenario.expected_goal_progression) else "chatting"

            try:
                response = self.bot.process_conversation(user_id, message)

                if response:
                    actual_goal = response.get("goal_status", "unknown")
                    latency = response.get("latency_ms", 0)
                    trace_id = response.get("trace_id", "")
                    bot_text = response.get("text", "")

                    goal_correct = actual_goal == expected_goal
                    total_latency += latency

                    turn = ConversationTurn(
                        turn_number=i + 1,
                        user_message=message,
                        bot_response=bot_text,
                        goal_status=actual_goal,
                        expected_goal=expected_goal,
                        goal_correct=goal_correct,
                        latency_ms=latency,
                        trace_id=trace_id,
                    )
                    turns.append(turn)

                    if verbose:
                        status = "✓" if goal_correct else "✗"
                        print(f"\n[Turn {i+1}] {status}")
                        print(f"  User: {message}")
                        print(f"  Bot:  {bot_text}")
                        print(f"  Goal: {actual_goal} (expected: {expected_goal})")
                        print(f"  Latency: {latency:.0f}ms")
                else:
                    turn = ConversationTurn(
                        turn_number=i + 1,
                        user_message=message,
                        bot_response="[NO RESPONSE]",
                        goal_status="error",
                        expected_goal=expected_goal,
                        goal_correct=False,
                        latency_ms=0,
                        trace_id="",
                        error="No response returned",
                    )
                    turns.append(turn)
                    errors.append(f"Turn {i+1}: No response")

            except Exception as e:
                error_msg = str(e)
                turn = ConversationTurn(
                    turn_number=i + 1,
                    user_message=message,
                    bot_response="[ERROR]",
                    goal_status="error",
                    expected_goal=expected_goal,
                    goal_correct=False,
                    latency_ms=0,
                    trace_id="",
                    error=error_msg,
                )
                turns.append(turn)
                errors.append(f"Turn {i+1}: {error_msg}")

                if verbose:
                    print(f"\n[Turn {i+1}] ERROR: {error_msg}")

        # Calculate metrics
        end_time = datetime.now().isoformat()
        correct_goals = sum(1 for t in turns if t.goal_correct)
        goal_accuracy = (correct_goals / len(turns) * 100) if turns else 0
        avg_latency = total_latency / len(turns) if turns else 0
        final_goal = turns[-1].goal_status if turns else "unknown"

        # Consider success if >60% goals correct and no critical errors
        success = goal_accuracy >= 60 and len(errors) == 0

        result = ConversationResult(
            scenario_name=scenario.name,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            turns=turns,
            success=success,
            goal_accuracy=goal_accuracy,
            total_latency_ms=total_latency,
            avg_latency_ms=avg_latency,
            errors=errors,
            final_goal=final_goal,
        )

        if verbose:
            print(f"\n--- Result: {'PASS' if success else 'FAIL'} ---")
            print(f"Goal Accuracy: {goal_accuracy:.1f}%")
            print(f"Avg Latency: {avg_latency:.0f}ms")

        return result

    def run_all_scenarios(
        self,
        scenarios: Optional[List[TestScenario]] = None,
        verbose: bool = False,
    ) -> TestReport:
        """Run all test scenarios.

        Args:
            scenarios: List of scenarios to run (default: all)
            verbose: Print detailed output

        Returns:
            TestReport with all results
        """
        scenarios = scenarios or TEST_SCENARIOS
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = datetime.now().isoformat()
        results = []

        print(f"\n{'='*60}")
        print(f"Starting Conversation Test Run: {run_id}")
        print(f"Scenarios to run: {len(scenarios)}")
        print(f"{'='*60}\n")

        for i, scenario in enumerate(scenarios):
            print(f"[{i+1}/{len(scenarios)}] Running: {scenario.name}...", end=" ", flush=True)
            result = self.run_scenario(scenario, verbose=verbose)
            results.append(result)

            status = "PASS" if result.success else "FAIL"
            print(f"{status} (goals: {result.goal_accuracy:.0f}%, latency: {result.avg_latency_ms:.0f}ms)")

        end_time = datetime.now().isoformat()

        # Calculate overall metrics
        total_turns = sum(len(r.turns) for r in results)
        total_correct = sum(sum(1 for t in r.turns if t.goal_correct) for r in results)
        total_latency = sum(r.total_latency_ms for r in results)
        total_errors = sum(len(r.errors) for r in results)

        report = TestReport(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            total_scenarios=len(scenarios),
            successful_scenarios=sum(1 for r in results if r.success),
            failed_scenarios=sum(1 for r in results if not r.success),
            total_turns=total_turns,
            goal_accuracy=(total_correct / total_turns * 100) if total_turns else 0,
            avg_latency_ms=(total_latency / total_turns) if total_turns else 0,
            error_count=total_errors,
            results=results,
        )

        # Save report
        self._save_report(report)

        return report

    def _save_report(self, report: TestReport):
        """Save test report to file."""
        # Save JSON report
        json_path = self.output_dir / f"report_{report.run_id}.json"
        with open(json_path, "w") as f:
            # Convert dataclasses to dicts
            data = asdict(report)
            json.dump(data, f, indent=2, default=str)

        # Save human-readable summary
        txt_path = self.output_dir / f"report_{report.run_id}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report.summary())

        # Save transcripts
        transcripts_dir = self.output_dir / f"transcripts_{report.run_id}"
        transcripts_dir.mkdir(exist_ok=True)

        for result in report.results:
            transcript_path = transcripts_dir / f"{result.scenario_name}.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(f"Scenario: {result.scenario_name}\n")
                f.write(f"User ID: {result.user_id}\n")
                f.write(f"Success: {result.success}\n")
                f.write(f"Goal Accuracy: {result.goal_accuracy:.1f}%\n")
                f.write(f"Avg Latency: {result.avg_latency_ms:.0f}ms\n")
                f.write(f"\n{'='*40}\n\n")

                for turn in result.turns:
                    status = "✓" if turn.goal_correct else "✗"
                    f.write(f"[Turn {turn.turn_number}] {status} [{turn.trace_id}]\n")
                    f.write(f"User: {turn.user_message}\n")
                    f.write(f"Bot: {turn.bot_response}\n")
                    f.write(f"Goal: {turn.goal_status} (expected: {turn.expected_goal})\n")
                    f.write(f"Latency: {turn.latency_ms:.0f}ms\n")
                    if turn.error:
                        f.write(f"ERROR: {turn.error}\n")
                    f.write("\n")

        print(f"\nReports saved to: {self.output_dir}")
        print(f"  - {json_path.name}")
        print(f"  - {txt_path.name}")
        print(f"  - {transcripts_dir.name}/")


def main():
    parser = argparse.ArgumentParser(description="DM Bot Conversation Test Harness")
    parser.add_argument(
        "--scenario",
        type=str,
        help="Run specific scenario by name",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of scenarios to run (0 = all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each turn",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Run scenarios with specific tag",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # List scenarios
    if args.list:
        print("\nAvailable Test Scenarios:")
        print("-" * 60)
        for s in TEST_SCENARIOS:
            tags = ", ".join(s.tags) if s.tags else "none"
            print(f"  {s.name:20s} | {len(s.messages)} turns | tags: {tags}")
        print()
        return

    # Select scenarios
    scenarios = TEST_SCENARIOS

    if args.scenario:
        scenarios = [s for s in TEST_SCENARIOS if s.name == args.scenario]
        if not scenarios:
            print(f"Unknown scenario: {args.scenario}")
            print("Use --list to see available scenarios")
            return

    if args.tag:
        scenarios = [s for s in scenarios if args.tag in s.tags]
        if not scenarios:
            print(f"No scenarios with tag: {args.tag}")
            return

    if args.count > 0:
        scenarios = scenarios[:args.count]

    # Run tests
    tester = ConversationTester()
    report = tester.run_all_scenarios(scenarios, verbose=args.verbose)

    # Print summary (handle Windows encoding)
    try:
        print(report.summary())
    except UnicodeEncodeError:
        # Replace unicode symbols for Windows console
        summary = report.summary()
        summary = summary.replace("✓", "[PASS]").replace("✗", "[FAIL]")
        print(summary)


if __name__ == "__main__":
    main()
