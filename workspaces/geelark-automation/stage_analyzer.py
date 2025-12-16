"""Stage Analyzer - LLM 1 for analyzing conversation stage.

This module handles the first LLM call in the two-LLM pipeline:
- Analyzes incoming user message
- Considers conversation history
- Determines the correct funnel stage
- Extracts relevant entities (location, objections, subscription claims)

IMPORTANT: Does NOT use keyword matching. Every message is fully analyzed by the LLM.
"""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from funnel_stages import (
    FunnelStage,
    STAGE_TRANSITIONS,
    STAGE_GUIDELINES,
    is_valid_transition,
    format_stage_context_for_llm,
)

logger = logging.getLogger(__name__)


# JSON Schema for Stage Analyzer output
STAGE_ANALYSIS_SCHEMA = """{
  "current_stage": "string - one of the valid funnel stages",
  "detected_intent": "string - brief description of what the user is trying to communicate",
  "should_transition": "boolean - true if we should transition to a different stage",
  "next_stage": "string - the stage to transition to (same as current if no transition)",
  "location_mentioned": "string | null - city/location if user mentioned one",
  "platform_requested": "string | null - platform name if user asked for contact (snap, whatsapp, number, etc)",
  "objection_detected": "boolean - true if user is hesitating or refusing",
  "objection_type": "string | null - type of objection (no_of, dont_pay, weird, not_interested, etc)",
  "subscription_claimed": "boolean - true if user claims to have subscribed",
  "sentiment": "string - positive, neutral, or negative",
  "confidence": "number - 0.0 to 1.0 confidence in analysis"
}"""


@dataclass
class StageAnalysis:
    """Result from stage analysis."""

    current_stage: FunnelStage
    detected_intent: str
    should_transition: bool
    next_stage: FunnelStage
    location_mentioned: Optional[str] = None
    platform_requested: Optional[str] = None
    objection_detected: bool = False
    objection_type: Optional[str] = None
    subscription_claimed: bool = False
    sentiment: str = "neutral"
    confidence: float = 0.5
    raw_response: Optional[str] = None
    trace_id: Optional[str] = None
    latency_ms: float = 0.0


def build_stage_analyzer_prompt(
    current_stage: FunnelStage,
    conversation_history: List[Dict],
    user_message: str,
) -> str:
    """Build the system prompt for stage analysis.

    Args:
        current_stage: Current funnel stage
        conversation_history: List of conversation messages
        user_message: Latest message from user

    Returns:
        Complete system prompt for the LLM
    """
    # Get valid transitions
    valid_transitions = STAGE_TRANSITIONS.get(current_stage, list(FunnelStage))
    valid_stage_names = [s.value for s in valid_transitions]

    # Get guidelines for all stages (for context)
    all_stages_context = []
    for stage in FunnelStage:
        guidelines = STAGE_GUIDELINES[stage]
        all_stages_context.append(f"**{stage.value}**")
        all_stages_context.append(f"  Trigger: {guidelines.trigger}")
        all_stages_context.append(f"  Goal: {guidelines.goal}")
        all_stages_context.append("")

    # Format recent history
    history_text = []
    for msg in conversation_history[-10:]:  # Last 10 messages
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:200]  # Truncate long messages
        history_text.append(f"[{role}]: {content}")

    return f"""You are a conversation stage analyzer for an Instagram DM bot.
Your job is to analyze the user's latest message and determine the current funnel stage.

## Funnel Stages Overview
{chr(10).join(all_stages_context)}

## Current Stage: {current_stage.value}
{format_stage_context_for_llm(current_stage)}

## Valid Transitions
From {current_stage.value}, you can transition to: {', '.join(valid_stage_names)}

## Conversation History
{chr(10).join(history_text) if history_text else "(No prior messages)"}

## Latest User Message
"{user_message}"

## Your Task
Analyze the user's message and conversation context. Determine:
1. What is the user's intent?
2. Should we transition to a different stage?
3. What stage should we be in?
4. Did they mention a location, request another platform, object, or claim to have subscribed?

## CRITICAL RULES
- Do NOT use keyword matching. Analyze the FULL context.
- Consider the conversation history, not just the latest message.
- Only transition if there's a clear reason to do so.
- Stay in current stage if unsure.
- Be conservative with stage transitions.

## IMPORTANT: What is NOT a dead_lead
- Sexual/forward messages like "can we fuck", "send nudes", "you're hot" - these show INTEREST, not rejection
  → Stay in current stage or move to small_talk, deflect playfully ("lol slow down, we just met 😂")
- One-word responses or short messages - user is still engaged
- User being flirty or suggestive - this is GOOD, redirect to building rapport
- User asking personal questions - they're interested

## What IS a dead_lead (ONLY these):
- User explicitly says "stop messaging me", "leave me alone", "not interested"
- User blocks or reports
- User is persistently hostile/rude after multiple attempts
- User explicitly refuses to subscribe MULTIPLE times with finality ("never", "absolutely not")
- Conversation has been completely cold for days (not applicable in single session)

## Output Format
Respond with ONLY valid JSON. No other text, no explanation.

{STAGE_ANALYSIS_SCHEMA}

## Example Output
{{"current_stage": "{current_stage.value}", "detected_intent": "user asking casual question", "should_transition": false, "next_stage": "{current_stage.value}", "location_mentioned": null, "platform_requested": null, "objection_detected": false, "objection_type": null, "subscription_claimed": false, "sentiment": "neutral", "confidence": 0.8}}
"""


class StageAnalyzer:
    """LLM-based stage analyzer for conversation funnel.

    This is LLM 1 in the two-LLM pipeline.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        ollama_host: str = "http://localhost:11434",
        metrics_callback: Optional[Callable] = None,
    ):
        """Initialize stage analyzer.

        Args:
            model: Ollama model name
            ollama_host: Ollama server URL
            metrics_callback: Optional callback for metrics
        """
        if not HAS_REQUESTS:
            raise ImportError("requests package not installed")

        self.model = model
        self.ollama_host = ollama_host
        self.metrics_callback = metrics_callback

    def analyze(
        self,
        current_stage: FunnelStage,
        conversation_history: List[Dict],
        user_message: str,
        trace_id: Optional[str] = None,
    ) -> StageAnalysis:
        """Analyze user message and determine funnel stage.

        Args:
            current_stage: Current funnel stage
            conversation_history: Conversation history
            user_message: Latest user message
            trace_id: Optional trace ID for debugging

        Returns:
            StageAnalysis with detected stage and entities
        """
        trace_id = trace_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(f"[{trace_id}] Analyzing stage: current={current_stage.value}")

        # Build prompt
        prompt = build_stage_analyzer_prompt(
            current_stage=current_stage,
            conversation_history=conversation_history,
            user_message=user_message,
        )

        # Call Ollama
        try:
            raw_response = self._call_ollama(prompt, trace_id)
            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            analysis = self._parse_response(raw_response, current_stage, trace_id)
            analysis.latency_ms = latency_ms
            analysis.trace_id = trace_id
            analysis.raw_response = raw_response

            # Validate transition
            if analysis.should_transition:
                if not is_valid_transition(current_stage, analysis.next_stage):
                    logger.warning(
                        f"[{trace_id}] Invalid transition: {current_stage.value} -> {analysis.next_stage.value}, "
                        f"keeping current stage"
                    )
                    analysis.should_transition = False
                    analysis.next_stage = current_stage

            if self.metrics_callback:
                self.metrics_callback("stage_analysis", success=True, latency_ms=latency_ms)

            logger.info(
                f"[{trace_id}] Stage analysis complete: "
                f"stage={analysis.next_stage.value}, transition={analysis.should_transition}, "
                f"latency={latency_ms:.0f}ms"
            )

            return analysis

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"[{trace_id}] Stage analysis failed: {e}")

            if self.metrics_callback:
                self.metrics_callback("stage_analysis", success=False, latency_ms=latency_ms)

            # Return safe fallback - stay in current stage
            return StageAnalysis(
                current_stage=current_stage,
                detected_intent="analysis failed",
                should_transition=False,
                next_stage=current_stage,
                confidence=0.0,
                trace_id=trace_id,
                latency_ms=latency_ms,
            )

    def _call_ollama(self, prompt: str, trace_id: str) -> str:
        """Call Ollama API.

        Args:
            prompt: System prompt
            trace_id: Trace ID for logging

        Returns:
            Raw response text
        """
        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 500,
                "temperature": 0.3,  # Lower temperature for more consistent analysis
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            logger.error(f"[{trace_id}] Could not connect to Ollama at {self.ollama_host}")
            raise
        except requests.exceptions.Timeout:
            logger.error(f"[{trace_id}] Ollama request timed out")
            raise

    def _parse_response(
        self,
        raw: str,
        current_stage: FunnelStage,
        trace_id: str,
    ) -> StageAnalysis:
        """Parse LLM response into StageAnalysis.

        Args:
            raw: Raw LLM response
            current_stage: Current stage (for fallback)
            trace_id: Trace ID for logging

        Returns:
            StageAnalysis object
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)

            # Parse stage values
            current = self._parse_stage(data.get("current_stage", current_stage.value))
            next_stage = self._parse_stage(data.get("next_stage", current_stage.value))

            return StageAnalysis(
                current_stage=current,
                detected_intent=data.get("detected_intent", "unknown"),
                should_transition=bool(data.get("should_transition", False)),
                next_stage=next_stage,
                location_mentioned=data.get("location_mentioned"),
                platform_requested=data.get("platform_requested"),
                objection_detected=bool(data.get("objection_detected", False)),
                objection_type=data.get("objection_type"),
                subscription_claimed=bool(data.get("subscription_claimed", False)),
                sentiment=data.get("sentiment", "neutral"),
                confidence=float(data.get("confidence", 0.5)),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[{trace_id}] Failed to parse stage analysis: {e}")
            # Return fallback
            return StageAnalysis(
                current_stage=current_stage,
                detected_intent="parse_error",
                should_transition=False,
                next_stage=current_stage,
                confidence=0.0,
            )

    def _parse_stage(self, stage_str: str) -> FunnelStage:
        """Parse stage string to FunnelStage enum.

        Args:
            stage_str: Stage name string

        Returns:
            FunnelStage enum value
        """
        try:
            return FunnelStage(stage_str.lower())
        except ValueError:
            # Try matching by name
            for stage in FunnelStage:
                if stage.value.lower() == stage_str.lower():
                    return stage
            # Default to small_talk if unknown
            return FunnelStage.SMALL_TALK


class MockStageAnalyzer:
    """Mock stage analyzer for testing without LLM calls."""

    def __init__(self, **kwargs):
        self.call_count = 0

    def analyze(
        self,
        current_stage: FunnelStage,
        conversation_history: List[Dict],
        user_message: str,
        trace_id: Optional[str] = None,
    ) -> StageAnalysis:
        """Return mock analysis based on simple heuristics."""
        self.call_count += 1
        trace_id = trace_id or str(uuid.uuid4())[:8]

        # Simple heuristic-based mock
        message_lower = user_message.lower()

        # Detect location mentions
        location = None
        location_keywords = ["from", "in", "live", "city", "la", "nyc", "miami", "chicago"]
        for word in location_keywords:
            if word in message_lower:
                # Extract potential city (very basic)
                words = message_lower.split()
                for i, w in enumerate(words):
                    if w in ["la", "nyc", "miami", "chicago", "seattle", "denver"]:
                        location = w.upper()
                        break
                break

        # Detect platform requests
        platform = None
        platform_keywords = ["snap", "snapchat", "whatsapp", "number", "phone", "telegram", "insta"]
        for p in platform_keywords:
            if p in message_lower:
                platform = p
                break

        # Detect objection
        objection = False
        objection_type = None
        objection_keywords = ["no", "nah", "don't", "cant", "won't", "weird", "not interested"]
        for obj in objection_keywords:
            if obj in message_lower:
                objection = True
                objection_type = "general_hesitation"
                break

        # Detect subscription claim
        subscribed = any(word in message_lower for word in ["done", "subscribed", "subbed", "followed"])

        # Determine stage transition
        should_transition = False
        next_stage = current_stage

        if location and current_stage in [FunnelStage.SMALL_TALK, FunnelStage.LOCATION_EXCHANGE]:
            next_stage = FunnelStage.LOCATION_EXCHANGE
            should_transition = True
        elif platform and current_stage in [FunnelStage.MEETUP_TEASE, FunnelStage.PLATFORM_REDIRECT]:
            next_stage = FunnelStage.PLATFORM_REDIRECT
            should_transition = True
        elif objection and current_stage == FunnelStage.PLATFORM_REDIRECT:
            next_stage = FunnelStage.OBJECTION_HANDLING
            should_transition = True
        elif subscribed and current_stage in [FunnelStage.PLATFORM_REDIRECT, FunnelStage.OBJECTION_HANDLING]:
            next_stage = FunnelStage.VERIFICATION
            should_transition = True

        return StageAnalysis(
            current_stage=current_stage,
            detected_intent=f"mock analysis of: {user_message[:50]}",
            should_transition=should_transition,
            next_stage=next_stage,
            location_mentioned=location,
            platform_requested=platform,
            objection_detected=objection,
            objection_type=objection_type,
            subscription_claimed=subscribed,
            sentiment="neutral",
            confidence=0.7,
            trace_id=trace_id,
        )


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.DEBUG)

    # Test mock analyzer
    analyzer = MockStageAnalyzer()

    test_messages = [
        ("hey whats up", FunnelStage.INITIAL_RESPONSE),
        ("I'm from LA", FunnelStage.SMALL_TALK),
        ("whats your snap?", FunnelStage.MEETUP_TEASE),
        ("nah I don't have OF", FunnelStage.PLATFORM_REDIRECT),
        ("ok done I subscribed", FunnelStage.OBJECTION_HANDLING),
    ]

    for message, stage in test_messages:
        result = analyzer.analyze(
            current_stage=stage,
            conversation_history=[],
            user_message=message,
        )
        print(f"\nMessage: '{message}'")
        print(f"Current Stage: {stage.value}")
        print(f"Detected: intent='{result.detected_intent}'")
        print(f"Transition: {result.should_transition} -> {result.next_stage.value}")
        print(f"Location: {result.location_mentioned}, Platform: {result.platform_requested}")
        print(f"Objection: {result.objection_detected}, Subscribed: {result.subscription_claimed}")
