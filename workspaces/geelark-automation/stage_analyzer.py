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

import os

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

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

    # Count messages in history to determine progression
    msg_count = len(conversation_history)

    return f"""You are a conversation stage analyzer for an Instagram DM bot converting users to OnlyFans.

## THE GOAL
Progress the conversation toward OnlyFans subscription. MOVE FORWARD through stages, don't get stuck.

## FUNNEL STAGES (in order)
1. initial_response - ONLY for the very first reply to their first message. ONE exchange max.
2. small_talk - Chat briefly, then ask "what city u in". 2-3 exchanges max.
3. location_exchange - They said a city. Match it ("im in X too" or "visiting X soon").
4. vibing - Keep chatting naturally. DO NOT suggest meetups. Wait for THEM to ask for contact.
5. platform_redirect - THEY asked for snap/number/to meet. Redirect to OF (it's how you verify they're real).
6. objection_handling - They pushed back. Stay chill, frame it as trust/verification.
7. verification - They claim they subscribed. Ask username.
8. converted - Confirmed. Move convo to OF.
9. dead_lead - ONLY if they explicitly refuse multiple times or say "stop messaging me".

## CURRENT STATE
- Current stage: {current_stage.value}
- Messages in conversation: {msg_count}

## VALID NEXT STAGES (YOU MUST PICK FROM THIS LIST ONLY)
{', '.join(valid_stage_names)}

CRITICAL: Your next_stage output MUST be one of: {', '.join(valid_stage_names)}
Any other value is INVALID and will cause an error.

## CONVERSATION HISTORY
{chr(10).join(history_text) if history_text else "(No prior messages)"}

## LATEST MESSAGE FROM USER
"{user_message}"

## STAGE PROGRESSION RULES

SIMPLE RULE FOR PLATFORM_REDIRECT:
- If message contains snap/number/insta/whatsapp -> STAY in platform_redirect
- Only go to objection_handling if message is JUST "no"/"nah" without any platform word

- initial_response → small_talk: After ANY reply
- small_talk → location_exchange: When THEY mention a city
- location_exchange → vibing: After location match
- vibing → platform_redirect: When they ask for snap/number/insta
- platform_redirect → objection_handling: ONLY "no"/"nah"/"don't use OF" with NO platform word
- objection_handling → platform_redirect: If they ask for contact again
- verification: If they claim subscribed

## CRITICAL: WHEN TO STAY IN PLATFORM_REDIRECT
- If their message contains "snap", "number", "insta", "whatsapp", "telegram" = STAY in platform_redirect
- "come on just give me your number" = STAY (contains "number")
- "nice, whats your insta" = STAY (contains "insta")
- "ok let me check it out" or "fine" = STAY (waiting)
- ONLY move to objection_handling if they say: "no", "nah", "I don't use OF", "I don't pay", "not interested"
- ONLY if they refuse WITHOUT mentioning any contact platform

## CRITICAL: VIBING STAGE - READ CAREFULLY
- In vibing stage, just chat naturally. STAY HERE unless they ask for contact.
- These are VIBING (stay in vibing):
  * "what do you do" = asking about job/life, NOT contact
  * "thats cool" = just chatting
  * "haha nice" = just chatting
  * "we should hang" = flirty chat, NOT asking for contact yet
- These are PLATFORM_REDIRECT (move to platform_redirect):
  * "whats ur snap" = explicitly asking for contact
  * "can i get ur number" = explicitly asking for contact
  * "give me ur insta" = explicitly asking for contact
  * "do you have whatsapp" = explicitly asking for contact
- If they don't explicitly mention snap/number/insta/whatsapp/telegram, STAY IN VIBING

## CRITICAL: SUBSCRIPTION DETECTION
- If user says "i subscribed", "done subscribed", "just subbed", "ok i followed", "ok i subscribed":
  - subscription_claimed = TRUE
  - should_transition = TRUE
  - next_stage = "verification"
- THIS OVERRIDES EVERYTHING - if they claim subscription, ALWAYS go to verification
- Example: "ok i subscribed" -> subscription_claimed: true, next_stage: "verification"

## CRITICAL: WHEN TO TRANSITION
- Message count is {msg_count}. If > 1 and still in initial_response, MUST transition to small_talk.
- If in same stage for 3+ exchanges, SHOULD transition forward.
- NEVER go backwards (e.g., small_talk back to initial_response).
- Forward momentum is key - don't get stuck chatting forever.

## WHAT IS NOT A DEAD_LEAD (STAY IN OBJECTION_HANDLING)
- "i dont use onlyfans" = objection, NOT dead
- "i dont pay for that" = objection, NOT dead
- "thats weird" = objection, NOT dead
- "nah" or "no thanks" = objection, NOT dead
- Short responses = still engaged
- Any response that isn't hostile = NOT dead

## WHAT IS A DEAD_LEAD (ONLY THESE - BE VERY CONSERVATIVE)
- "stop messaging me", "leave me alone", "fuck off"
- Hostile/rude AND refuses multiple times
- Explicitly says "NEVER" or "STOP"
- Blocks you or reports you

IMPORTANT: If unsure, stay in objection_handling. Only go to dead_lead if they're clearly done.

## OUTPUT (JSON only, no other text)
{STAGE_ANALYSIS_SCHEMA}
"""


class StageAnalyzer:
    """LLM-based stage analyzer for conversation funnel.

    This is LLM 1 in the two-LLM pipeline.
    Supports OpenAI (gpt-5-mini), Anthropic (Haiku), and Ollama (local).
    """

    def __init__(
        self,
        provider: str = "openai",  # "openai", "anthropic", or "ollama"
        model: str = "gpt-5-mini",
        ollama_host: str = "http://localhost:11434",
        metrics_callback: Optional[Callable] = None,
    ):
        """Initialize stage analyzer.

        Args:
            provider: "openai" (gpt-5-mini), "anthropic" (Haiku), or "ollama" (local)
            model: Model name
            ollama_host: Ollama server URL (only for ollama provider)
            metrics_callback: Optional callback for metrics
        """
        self.provider = provider
        self.model = model
        self.ollama_host = ollama_host
        self.metrics_callback = metrics_callback

        if provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed. Run: pip install openai")
            self.openai_client = openai.OpenAI()
        elif provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self.anthropic_client = anthropic.Anthropic()
        elif provider == "ollama":
            if not HAS_REQUESTS:
                raise ImportError("requests package not installed")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'ollama'")

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

        # Call LLM (OpenAI, Anthropic, or Ollama)
        try:
            if self.provider == "openai":
                raw_response = self._call_openai(prompt, trace_id)
            elif self.provider == "anthropic":
                raw_response = self._call_anthropic(prompt, trace_id)
            else:
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

    def _call_openai(self, prompt: str, trace_id: str) -> str:
        """Call OpenAI API (gpt-5-mini).

        Args:
            prompt: System prompt
            trace_id: Trace ID for logging

        Returns:
            Raw response text (JSON)
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\n\nRespond with ONLY the JSON object, no other text."
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[{trace_id}] OpenAI API error: {e}")
            raise

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

    def _call_anthropic(self, prompt: str, trace_id: str) -> str:
        """Call Anthropic API (Haiku).

        Args:
            prompt: System prompt
            trace_id: Trace ID for logging

        Returns:
            Raw response text (JSON)
        """
        try:
            message = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\n\nRespond with ONLY the JSON object, no other text."
                    }
                ],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"[{trace_id}] Anthropic API error: {e}")
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
        elif platform and current_stage in [FunnelStage.VIBING, FunnelStage.PLATFORM_REDIRECT]:
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
        ("whats your snap?", FunnelStage.VIBING),
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
