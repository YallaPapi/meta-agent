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

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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

    name: str = "Mia"
    age: int = 22
    occupation: str = "content creator"
    personality: str = "friendly, flirty, casual"
    texting_style: str = "short messages, lowercase, occasional emojis, asks questions back"
    of_link: str = "onlyfans.com/miaxxxx"


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

    # Build location context if needed
    location_context = ""
    if location_to_match and stage == FunnelStage.LOCATION_EXCHANGE:
        location_context = f"""
## Location Matching
User's city: {location_to_match}
Say you're in {location_to_match} too OR visiting {location_to_match} soon.
Example: "oh nice im in {location_to_match} too" or "wait im visiting {location_to_match} next week"
"""

    # Build objection context if needed
    objection_context = ""
    if stage == FunnelStage.OBJECTION_HANDLING and stage_analysis.objection_type:
        objection_context = f"""
## Objection Handling
User objection type: {stage_analysis.objection_type}
Be playful, not pushy. Emphasize it's free to follow.
"""

    return f"""You are {persona.name}, a {persona.age}yo girl texting on Instagram DMs.

## TEXTING STYLE
- use abbreviations: u, r, ur, rn, w, wbu, wyd, thx
- lowercase
- text like a real person, not a chatbot

## Current Stage: {stage.value}
{format_stage_context_for_llm(stage)}
{location_context}
{objection_context}

## Conversation
{chr(10).join(history_text) if history_text else "(first message)"}

## Analysis
- Intent: {stage_analysis.detected_intent}
- City mentioned: {stage_analysis.location_mentioned or "none"}
- Platform asked: {stage_analysis.platform_requested or "none"}
- Objection: {stage_analysis.objection_detected}

## Already used (dont repeat):
{used_text}

## Output (JSON only):
{RESPONSE_SCHEMA}
"""


class ResponseGenerator:
    """LLM-based response generator for persona messages.

    This is LLM 2 in the two-LLM pipeline.
    """

    def __init__(
        self,
        persona: Optional[PersonaConfig] = None,
        model: str = "qwen2.5:7b",
        ollama_host: str = "http://localhost:11434",
        metrics_callback: Optional[Callable] = None,
    ):
        """Initialize response generator.

        Args:
            persona: Persona configuration
            model: Ollama model name
            ollama_host: Ollama server URL
            metrics_callback: Optional callback for metrics
        """
        if not HAS_REQUESTS:
            raise ImportError("requests package not installed")

        self.persona = persona or PersonaConfig()
        self.model = model
        self.ollama_host = ollama_host
        self.metrics_callback = metrics_callback

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

        # Call Ollama
        try:
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

            # Return safe fallback
            return GeneratedResponse(
                text="haha nice! wbu?",
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
                text = "haha nice!"

            send_photo = bool(data.get("send_photo", False))
            photo_mood = data.get("photo_mood", "casual")

            # Validate photo mood
            if photo_mood not in VALID_MOODS:
                photo_mood = "casual"

            return GeneratedResponse(
                text=text.strip(),
                send_photo=send_photo,
                photo_mood=photo_mood,
                internal_notes=data.get("internal_notes", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[{trace_id}] Failed to parse response: {e}")
            # Return fallback
            return GeneratedResponse(
                text="haha nice!",
                send_photo=False,
                photo_mood="casual",
                internal_notes="fallback due to parse error",
            )


class MockResponseGenerator:
    """Mock response generator for testing without LLM calls."""

    def __init__(self, persona: Optional[PersonaConfig] = None, **kwargs):
        self.persona = persona or PersonaConfig()
        self.call_count = 0

        # Pre-defined responses by stage
        self.stage_responses = {
            FunnelStage.INITIAL_RESPONSE: [
                "heyyy thank you!",
                "omg hiii!",
                "aw thats sweet, hey!",
                "hiii whats up!",
            ],
            FunnelStage.SMALL_TALK: [
                "how are you doing?",
                "what are you up to today?",
                "lol thats cool! where are you from?",
                "haha nice, what do you do?",
            ],
            FunnelStage.LOCATION_EXCHANGE: [
                "omg no way! im actually visiting there soon",
                "wait really? I'm there too rn lol",
                "no wayyy thats so cool, I've been wanting to go there",
            ],
            FunnelStage.MEETUP_TEASE: [
                "haha we should hang out sometime",
                "lol maybe we could grab coffee",
                "I'd love to meet up but let's talk off ig first",
            ],
            FunnelStage.PLATFORM_REDIRECT: [
                "I only give that to ppl who sub to my OF haha",
                "im kinda private w that but if you sub to my OF...",
                "I only do that with my OF subscribers lol",
            ],
            FunnelStage.OBJECTION_HANDLING: [
                "aw really? what I don't mean that much to you? lol jk",
                "haha I get it but it's literally free to follow",
                "lol fair enough, I'm just careful who I give my stuff to",
            ],
            FunnelStage.VERIFICATION: [
                "yay! did you actually sub? whats your username",
                "omg really?? send me a msg on there!",
                "haha ok cool, whats your name on OF?",
            ],
            FunnelStage.CONVERTED: [
                "ok perfect let's talk there!",
                "yay ok msg me on OF!",
                "cool see you there!",
            ],
            FunnelStage.DEAD_LEAD: [
                "haha no worries! was nice chatting",
                "all good, take care!",
                "lol ok fair enough, see ya around",
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
