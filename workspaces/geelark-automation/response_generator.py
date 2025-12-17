"""Response Generator - LLM 2 for generating persona responses.

This module handles the second LLM call in the two-LLM pipeline:
- Takes stage analysis from LLM 1
- Generates natural, varied responses in persona voice
- Handles photo selection decisions
- Ensures response variety (never repeats exact messages)

Response style: Short, casual texting style with occasional typos and emojis.
"""

import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Set

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

from funnel_stages import (
    FunnelStage,
    STAGE_GUIDELINES,
    format_stage_context_for_llm,
)
from stage_analyzer import StageAnalysis

logger = logging.getLogger(__name__)


# Valid photo moods
VALID_MOODS = ["happy", "sad", "flirty", "casual", "morning", "night", "excited", "selfie"]

# JSON Schema for Response Generator output
RESPONSE_SCHEMA = """{
  "text": "string - your conversational reply (1-2 sentences max)",
  "send_photo": "boolean - true to attach a photo, false otherwise",
  "photo_mood": "string - one of: happy, sad, flirty, casual, morning, night, excited, selfie",
  "internal_notes": "string - brief reasoning for this response (not shown to user)"
}"""


@dataclass
class PersonaConfig:
    """Persona configuration for response generation."""

    name: str = "Zen"
    age: int = 48
    occupation: str = "content creator, former massage therapist"
    personality: str = "free-spirited, sensual, grounded, worldly, confident"
    texting_style: str = "complete sentences, warm but not eager, mature flirtation, uses darling/lovely, ellipses for flirty pauses"
    of_link: str = "onlyfans.com/lionessuntamed"


@dataclass
class GeneratedResponse:
    """Response from the response generator."""

    text: str
    send_photo: bool
    photo_mood: str
    internal_notes: str = ""
    raw_response: Optional[str] = None
    trace_id: Optional[str] = None
    latency_ms: float = 0.0


def build_response_generator_prompt(
    stage_analysis: StageAnalysis,
    conversation_history: List[Dict],
    persona: PersonaConfig,
    used_phrases: Set[str],
    location_to_match: Optional[str] = None,
) -> str:
    """Build the system prompt for response generation.

    Args:
        stage_analysis: Analysis from stage analyzer (LLM 1)
        conversation_history: List of conversation messages
        persona: Persona configuration
        used_phrases: Set of phrases already used (for variety)
        location_to_match: Location to match if in location_exchange stage

    Returns:
        Complete system prompt for the LLM
    """
    # Get stage guidelines
    stage = stage_analysis.next_stage
    guidelines = STAGE_GUIDELINES[stage]

    # Format recent history
    history_text = []
    for msg in conversation_history[-10:]:  # Last 10 messages
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:200]
        history_text.append(f"[{role}]: {content}")

    # Build used phrases list for variety
    used_list = list(used_phrases)[:20]  # Limit to recent 20
    used_text = "\n".join(f"- {phrase}" for phrase in used_list) if used_list else "(none yet)"

    # Extract the ACTUAL last user message for grounding
    last_user_message = ""
    for msg in reversed(conversation_history):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break

    # Build location context if needed
    location_context = ""
    if stage in (FunnelStage.LOCATION_EXCHANGE, FunnelStage.VIBING):
        if location_to_match:
            location_context = f"""
## Location Context
User's city: {location_to_match}
MATCH their city: say you're in {location_to_match} too or visiting soon.
Example: "oh nice i'm in {location_to_match} too"
"""
        else:
            location_context = """
## Location Context
You don't know their city yet. ASK them: "what city are you in?"
"""

    # Build objection context if needed
    objection_context = ""
    if stage == FunnelStage.OBJECTION_HANDLING and stage_analysis.objection_type:
        objection_context = f"""
## Objection Handling
User objection type: {stage_analysis.objection_type}
Be playful, not pushy. Frame it as trust/verification - you need to know they're real.
Do NOT say it's free - OF costs money.
"""

    # Build stage-specific goal
    stage_goals = {
        FunnelStage.INITIAL_RESPONSE: "Reply warmly but not eagerly. Acknowledge their message, maybe a light flirt.",
        FunnelStage.SMALL_TALK: "Chat naturally, build rapport. Eventually ask what city they're in.",
        FunnelStage.LOCATION_EXCHANGE: "Match their city - say you're there too or visiting soon.",
        FunnelStage.VIBING: "Build connection, be intriguing. DO NOT suggest meetups. Wait for THEM to ask for contact.",
        FunnelStage.PLATFORM_REDIRECT: "They want contact or to meet? Warmly redirect to your page - that's where you connect with people properly.",
        FunnelStage.OBJECTION_HANDLING: "Accept gracefully, no pressure. 'no problem, if you're not ready I won't push you'",
        FunnelStage.VERIFICATION: "Ask for their username so you can find them on your page.",
        FunnelStage.CONVERTED: "Direct them to message you on your page.",
        FunnelStage.DEAD_LEAD: "End gracefully and warmly. No hard feelings.",
    }
    stage_goal = stage_goals.get(stage, "Keep it natural")

    # Build stage-specific example
    stage_examples = {
        FunnelStage.INITIAL_RESPONSE: '"hey thanks" or "well hello"',
        FunnelStage.SMALL_TALK: '"what city you in" or "just chilling. you?"',
        FunnelStage.LOCATION_EXCHANGE: f'"oh nice im in {location_to_match or "[city]"} too"' if location_to_match else '"what city you in"',
        FunnelStage.VIBING: '"haha nice" or "you seem fun"',
        FunnelStage.PLATFORM_REDIRECT: '"sub to my page first" or "gotta sub first"',
        FunnelStage.OBJECTION_HANDLING: '"no pressure" or "fair enough"',
        FunnelStage.VERIFICATION: '"whats your username"',
        FunnelStage.CONVERTED: '"message me there"',
        FunnelStage.DEAD_LEAD: '"take care"',
    }
    example = stage_examples.get(stage, '"hey"')

    return f"""Generate JSON for a DM reply. The "text" MUST be 5 words or less, lowercase.

Stage: {stage.value}
Goal: {stage_goal}
Example: {example}

User said: "{last_user_message}"

{location_context}

Output format:
{{"text": "your 5-word-max reply", "send_photo": false, "photo_mood": "casual"}}
"""


class ResponseGenerator:
    """LLM-based response generator for persona messages.

    This is LLM 2 in the two-LLM pipeline.
    Supports both Ollama (local/free) and Anthropic Haiku (fast/cheap).
    """

    def __init__(
        self,
        persona: Optional[PersonaConfig] = None,
        provider: str = "anthropic",  # "anthropic" or "ollama"
        model: str = "claude-3-haiku-20240307",
        ollama_host: str = "http://localhost:11434",
        metrics_callback: Optional[Callable] = None,
    ):
        """Initialize response generator.

        Args:
            persona: Persona configuration
            provider: "anthropic" (Haiku) or "ollama" (local)
            model: Model name (haiku or ollama model)
            ollama_host: Ollama server URL (only for ollama provider)
            metrics_callback: Optional callback for metrics
        """
        self.provider = provider
        self.persona = persona or PersonaConfig()
        self.model = model
        self.ollama_host = ollama_host
        self.metrics_callback = metrics_callback

        if provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self.anthropic_client = anthropic.Anthropic()
        elif provider == "ollama":
            if not HAS_REQUESTS:
                raise ImportError("requests package not installed")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'ollama'")

    def generate(
        self,
        stage_analysis: StageAnalysis,
        conversation_history: List[Dict],
        used_phrases: Optional[Set[str]] = None,
        location_to_match: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> GeneratedResponse:
        """Generate a response based on stage analysis.

        Args:
            stage_analysis: Analysis from stage analyzer (LLM 1)
            conversation_history: Conversation history
            used_phrases: Set of phrases already used
            location_to_match: Location to match in response
            trace_id: Optional trace ID for debugging

        Returns:
            GeneratedResponse with message and photo decision
        """
        trace_id = trace_id or str(uuid.uuid4())[:8]
        used_phrases = used_phrases or set()
        start_time = time.time()

        logger.info(f"[{trace_id}] Generating response for stage: {stage_analysis.next_stage.value}")

        # Build prompt
        prompt = build_response_generator_prompt(
            stage_analysis=stage_analysis,
            conversation_history=conversation_history,
            persona=self.persona,
            used_phrases=used_phrases,
            location_to_match=location_to_match,
        )

        # Call LLM (Anthropic or Ollama)
        try:
            if self.provider == "anthropic":
                raw_response = self._call_anthropic(prompt, trace_id)
            else:
                raw_response = self._call_ollama(prompt, trace_id)
            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            response = self._parse_response(raw_response, trace_id)
            response.latency_ms = latency_ms
            response.trace_id = trace_id
            response.raw_response = raw_response

            if self.metrics_callback:
                self.metrics_callback("response_generation", success=True, latency_ms=latency_ms)

            logger.info(
                f"[{trace_id}] Response generated: "
                f"text='{response.text[:50]}...', photo={response.send_photo}, "
                f"latency={latency_ms:.0f}ms"
            )

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"[{trace_id}] Response generation failed: {e}")

            if self.metrics_callback:
                self.metrics_callback("response_generation", success=False, latency_ms=latency_ms)

            # Return safe fallback - ultra short
            return GeneratedResponse(
                text="lol",
                send_photo=False,
                photo_mood="casual",
                internal_notes="fallback response due to error",
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
                "num_predict": 300,
                "temperature": 0.8,  # Higher temperature for more variety
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
                max_tokens=200,  # Enough for JSON
                temperature=0.7,  # Allow variety
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            raw = message.content[0].text
            logger.debug(f"[{trace_id}] Raw Haiku response: {raw}")
            return raw
        except anthropic.APIConnectionError as e:
            logger.error(f"[{trace_id}] Anthropic connection error: {e}")
            raise
        except anthropic.RateLimitError as e:
            logger.error(f"[{trace_id}] Anthropic rate limit: {e}")
            raise
        except anthropic.APIStatusError as e:
            logger.error(f"[{trace_id}] Anthropic API error: {e}")
            raise

    def _validate_word_count(self, text: str, max_words: int = 5) -> bool:
        """Check if response is within word limit.

        Args:
            text: Response text to validate
            max_words: Maximum allowed words (default 12)

        Returns:
            True if valid, False if too long
        """
        words = text.strip().split()
        return len(words) <= max_words

    def _truncate_response(self, text: str, max_words: int = 5) -> str:
        """Truncate response to max words if too long.

        Args:
            text: Response text
            max_words: Maximum words

        Returns:
            Truncated text
        """
        words = text.strip().split()
        if len(words) <= max_words:
            return text
        # Truncate and return
        return " ".join(words[:max_words])

    def _parse_response(self, raw: str, trace_id: str) -> GeneratedResponse:
        """Parse LLM response into GeneratedResponse.

        Args:
            raw: Raw LLM response
            trace_id: Trace ID for logging

        Returns:
            GeneratedResponse object
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)

            # Extract fields
            text = data.get("text", "")
            if not text or not text.strip():
                text = "lol"

            # CRITICAL: Validate and truncate word count (max 5 words)
            if not self._validate_word_count(text):
                original_len = len(text.split())
                text = self._truncate_response(text)
                logger.warning(
                    f"[{trace_id}] Response too long ({original_len} words), "
                    f"truncated to: '{text}'"
                )

            send_photo = bool(data.get("send_photo", False))
            photo_mood = data.get("photo_mood", "casual")

            # Validate photo mood
            if photo_mood not in VALID_MOODS:
                photo_mood = "casual"

            return GeneratedResponse(
                text=text.strip().lower(),  # Force lowercase
                send_photo=send_photo,
                photo_mood=photo_mood,
                internal_notes=data.get("internal_notes", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[{trace_id}] Failed to parse response: {e}")
            # Return fallback - ultra short
            return GeneratedResponse(
                text="lol",
                send_photo=False,
                photo_mood="casual",
                internal_notes="fallback due to parse error",
            )


class MockResponseGenerator:
    """Mock response generator for testing without LLM calls."""

    def __init__(self, persona: Optional[PersonaConfig] = None, **kwargs):
        self.persona = persona or PersonaConfig()
        self.call_count = 0

        # Pre-defined responses by stage (1-5 words MAX, slight flirt)
        self.stage_responses = {
            FunnelStage.INITIAL_RESPONSE: [
                "thx cutie",
                "aw thx wbu",
                "hey u",
                "thx babe",
            ],
            FunnelStage.SMALL_TALK: [
                "lol cute",
                "haha nice",
                "what city u in",
                "wbu cutie",
            ],
            FunnelStage.LOCATION_EXCHANGE: [
                "oh im there too",
                "no way same",
                "wait me too",
            ],
            FunnelStage.VIBING: [
                "lol ur funny",
                "haha maybeee",
                "lol",
                "u wish",
            ],
            FunnelStage.PLATFORM_REDIRECT: [
                "sub to my OF babe",
                "only on my OF",
                "my OF subscribers only",
            ],
            FunnelStage.OBJECTION_HANDLING: [
                "lol come on",
                "up to u cutie",
                "ur loss babe",
            ],
            FunnelStage.VERIFICATION: [
                "whats ur username babe",
                "username?",
                "whats ur name there",
            ],
            FunnelStage.CONVERTED: [
                "msg me there cutie",
                "k msg me babe",
                "hmu there",
            ],
            FunnelStage.DEAD_LEAD: [
                "k bye",
                "lol k",
                "ok",
            ],
        }

    def generate(
        self,
        stage_analysis: StageAnalysis,
        conversation_history: List[Dict],
        used_phrases: Optional[Set[str]] = None,
        location_to_match: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> GeneratedResponse:
        """Generate mock response based on stage."""
        self.call_count += 1
        trace_id = trace_id or str(uuid.uuid4())[:8]
        used_phrases = used_phrases or set()

        stage = stage_analysis.next_stage
        responses = self.stage_responses.get(stage, ["haha nice!"])

        # Find unused response
        available = [r for r in responses if r not in used_phrases]
        if not available:
            available = responses

        text = random.choice(available)

        # Inject location if in location_exchange
        if location_to_match and stage == FunnelStage.LOCATION_EXCHANGE:
            text = f"omg no way! I'm actually in {location_to_match} right now haha"

        # Decide on photo (30% chance)
        send_photo = random.random() < 0.3
        photo_mood = random.choice(VALID_MOODS)

        return GeneratedResponse(
            text=text,
            send_photo=send_photo,
            photo_mood=photo_mood,
            internal_notes=f"mock response for {stage.value}",
            trace_id=trace_id,
        )


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.DEBUG)

    # Test mock generator
    generator = MockResponseGenerator()

    # Simulate stage analysis
    from stage_analyzer import StageAnalysis

    test_analyses = [
        StageAnalysis(
            current_stage=FunnelStage.INITIAL_RESPONSE,
            detected_intent="greeting",
            should_transition=True,
            next_stage=FunnelStage.SMALL_TALK,
        ),
        StageAnalysis(
            current_stage=FunnelStage.SMALL_TALK,
            detected_intent="mentioned location",
            should_transition=True,
            next_stage=FunnelStage.LOCATION_EXCHANGE,
            location_mentioned="LA",
        ),
        StageAnalysis(
            current_stage=FunnelStage.PLATFORM_REDIRECT,
            detected_intent="user objecting",
            should_transition=True,
            next_stage=FunnelStage.OBJECTION_HANDLING,
            objection_detected=True,
            objection_type="no_of",
        ),
    ]

    for analysis in test_analyses:
        response = generator.generate(
            stage_analysis=analysis,
            conversation_history=[],
            location_to_match=analysis.location_mentioned,
        )
        print(f"\nStage: {analysis.next_stage.value}")
        print(f"Response: '{response.text}'")
        print(f"Photo: {response.send_photo} ({response.photo_mood})")
