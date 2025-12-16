"""Persona LLM - Structured JSON output from LLM with persona voice.

Handles:
- System prompt with persona definition
- Conversation history formatting
- Structured JSON response parsing with validation
- Goal status tracking
- Retry logic with exponential backoff
- Error tracking and metrics
"""

import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

# Import based on provider
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

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config import PersonaConfig
from prompt_builder import (
    PersonaDefinition,
    PersonaPromptBuilder,
    VALID_MOODS,
    VALID_GOALS,
)
from errors import (
    LLMCallError,
    LLMEmptyResponseError,
    LLMInvalidJSONError,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    text: str
    send_photo: bool
    photo_mood: str
    goal_status: str
    raw_response: Optional[str] = None
    trace_id: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_backoff_sec: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_sec: float = 30.0
    jitter: float = 0.1  # Random jitter factor (0.1 = +/- 10%)

    def get_backoff(self, attempt: int) -> float:
        """Calculate backoff with exponential growth and jitter."""
        backoff = min(
            self.initial_backoff_sec * (self.backoff_multiplier ** attempt),
            self.max_backoff_sec
        )
        # Add jitter
        jitter_range = backoff * self.jitter
        return backoff + random.uniform(-jitter_range, jitter_range)


class PersonaLLM:
    """LLM client with persona system prompt, structured output, and retry logic."""

    def __init__(
        self,
        persona: PersonaConfig,
        api_key: str = "",
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        ollama_host: str = "http://localhost:11434",
        retry_config: Optional[RetryConfig] = None,
        metrics_callback: Optional[Callable] = None,
    ):
        """Initialize LLM client.

        Args:
            persona: Persona configuration
            api_key: API key for LLM provider (not needed for Ollama)
            provider: "anthropic", "openai", or "ollama"
            model: Model name
            ollama_host: Ollama server URL (default: http://localhost:11434)
            retry_config: Configuration for retry logic
            metrics_callback: Optional callback for recording metrics
        """
        self.persona = persona
        self.provider = provider
        self.model = model
        self.ollama_host = ollama_host
        self.retry_config = retry_config or RetryConfig()
        self.metrics_callback = metrics_callback

        # Use new prompt builder for cleaner separation
        persona_def = PersonaDefinition(
            name=persona.name,
            age=persona.age,
            occupation=persona.occupation,
            university=persona.university,
            personality=persona.personality,
            texting_style=persona.texting_style,
            goal=persona.goal,
        )
        builder = PersonaPromptBuilder(persona_def)
        self.system_prompt = builder.build_system_prompt()

        if provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package not installed")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed")
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "ollama":
            if not HAS_REQUESTS:
                raise ImportError("requests package not installed")
            self.client = None  # We'll use requests directly
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(
        self,
        conversation_history: List[Dict],
        goal_status: str = "chatting",
        max_tokens: int = 300,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response for the conversation with retry logic.

        Args:
            conversation_history: List of {role, content} messages
            goal_status: Current goal status
            max_tokens: Max tokens in response
            user_id: User ID for error context
            trace_id: Trace ID for debugging (generated if not provided)

        Returns:
            LLMResponse with structured data
        """
        trace_id = trace_id or str(uuid.uuid4())[:8]
        messages = self._format_messages(conversation_history, goal_status)

        logger.info(f"[{trace_id}] LLM generate: model={self.model}, goal={goal_status}")

        last_error = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                start_time = time.time()

                # Call LLM based on provider
                if self.provider == "anthropic":
                    raw = self._call_anthropic(messages, max_tokens)
                elif self.provider == "ollama":
                    raw = self._call_ollama(messages, max_tokens, trace_id)
                else:
                    raw = self._call_openai(messages, max_tokens)

                latency_ms = (time.time() - start_time) * 1000

                # Check for empty response
                if not raw or not raw.strip():
                    raise LLMEmptyResponseError(
                        message="LLM returned empty response",
                        model=self.model,
                        attempt=attempt,
                        user_id=user_id,
                        trace_id=trace_id,
                    )

                # Parse and validate response
                response = self._parse_response(raw, trace_id, user_id)
                response.latency_ms = latency_ms
                response.trace_id = trace_id

                # Record success metrics
                if self.metrics_callback:
                    self.metrics_callback("llm_call", success=True, latency_ms=latency_ms)

                logger.info(f"[{trace_id}] LLM success: latency={latency_ms:.0f}ms, goal={response.goal_status}")
                return response

            except LLMEmptyResponseError as e:
                last_error = e
                e.log()
                if self.metrics_callback:
                    self.metrics_callback("llm_empty_response")
                    self.metrics_callback("llm_call", success=False)

                if attempt < self.retry_config.max_retries:
                    backoff = self.retry_config.get_backoff(attempt)
                    logger.warning(f"[{trace_id}] Empty response, retrying in {backoff:.1f}s (attempt {attempt + 1}/{self.retry_config.max_retries})")
                    if self.metrics_callback:
                        self.metrics_callback("llm_retry", reason="empty_response")
                    time.sleep(backoff)
                    continue

            except LLMInvalidJSONError as e:
                last_error = e
                e.log()
                if self.metrics_callback:
                    self.metrics_callback("llm_json_error")
                    self.metrics_callback("llm_call", success=False)

                if attempt < self.retry_config.max_retries:
                    backoff = self.retry_config.get_backoff(attempt)
                    logger.warning(f"[{trace_id}] JSON parse error, retrying in {backoff:.1f}s (attempt {attempt + 1}/{self.retry_config.max_retries})")
                    if self.metrics_callback:
                        self.metrics_callback("llm_retry", reason="json_error")
                    time.sleep(backoff)
                    continue

            except (requests.exceptions.RequestException, Exception) as e:
                # Network or API errors
                status_code = getattr(getattr(e, 'response', None), 'status_code', None)
                last_error = LLMCallError(
                    message=str(e),
                    model=self.model,
                    attempt=attempt,
                    max_attempts=self.retry_config.max_retries,
                    status_code=status_code,
                    user_id=user_id,
                    trace_id=trace_id,
                )
                last_error.log()

                if self.metrics_callback:
                    self.metrics_callback("llm_call", success=False)

                # Check if retryable
                if status_code and status_code in (400, 401, 403):
                    # Non-retryable errors
                    logger.error(f"[{trace_id}] Non-retryable error: {status_code}")
                    break

                if attempt < self.retry_config.max_retries:
                    backoff = self.retry_config.get_backoff(attempt)
                    logger.warning(f"[{trace_id}] LLM call failed, retrying in {backoff:.1f}s (attempt {attempt + 1}/{self.retry_config.max_retries})")
                    if self.metrics_callback:
                        self.metrics_callback("llm_retry", reason="api_error")
                    time.sleep(backoff)
                    continue

        # All retries exhausted - return fallback response
        logger.error(f"[{trace_id}] All retries exhausted, returning fallback response")
        return LLMResponse(
            text="haha nice! wbu?",
            send_photo=False,
            photo_mood="casual",
            goal_status=goal_status,  # Keep current goal
            raw_response=None,
            trace_id=trace_id,
            latency_ms=0.0,
        )

    def _format_messages(
        self,
        history: List[Dict],
        goal_status: str,
    ) -> List[Dict]:
        """Format conversation history for LLM.

        Args:
            history: Raw conversation history
            goal_status: Current goal status

        Returns:
            Formatted messages for LLM API
        """
        messages = []

        # Add history (last 20 messages max)
        for msg in history[-20:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # Add goal context to last message
        if messages:
            last = messages[-1]
            last["content"] = f"{last['content']}\n\n[Current goal_status: {goal_status}]"

        return messages

    def _call_anthropic(self, messages: List[Dict], max_tokens: int) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=messages,
        )
        return response.content[0].text

    def _call_openai(self, messages: List[Dict], max_tokens: int) -> str:
        """Call OpenAI API."""
        # Add system message at start
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_ollama(self, messages: List[Dict], max_tokens: int, trace_id: str = "") -> str:
        """Call Ollama API with timeout and error handling.

        Uses the /api/chat endpoint for chat completions.
        """
        # Add system message at start
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "format": "json",  # Request JSON output
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{trace_id}] Could not connect to Ollama at {self.ollama_host}")
            raise LLMCallError(
                message=f"Connection error: {e}",
                model=self.model,
                trace_id=trace_id,
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"[{trace_id}] Ollama request timed out (60s)")
            raise LLMCallError(
                message="Request timeout (60s)",
                model=self.model,
                trace_id=trace_id,
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            logger.error(f"[{trace_id}] Ollama HTTP error: {status_code}")
            raise LLMCallError(
                message=str(e),
                model=self.model,
                status_code=status_code,
                trace_id=trace_id,
            )

    def _parse_response(self, raw: str, trace_id: str = "", user_id: Optional[str] = None) -> LLMResponse:
        """Parse structured JSON response from LLM.

        Args:
            raw: Raw LLM output
            trace_id: Trace ID for error context
            user_id: User ID for error context

        Returns:
            LLMResponse with validated fields

        Raises:
            LLMInvalidJSONError: If JSON parsing fails
        """
        # Try to extract JSON from response
        try:
            # Look for JSON in the response (handle markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMInvalidJSONError(
                message=f"Failed to parse JSON: {e}",
                raw_response=raw,
                model=self.model,
                user_id=user_id,
                trace_id=trace_id,
            )

        # Validate required fields
        missing_fields = []
        if "text" not in data:
            missing_fields.append("text")

        if missing_fields:
            raise LLMInvalidJSONError(
                message=f"Missing required fields: {missing_fields}",
                raw_response=raw,
                model=self.model,
                missing_fields=missing_fields,
                user_id=user_id,
                trace_id=trace_id,
            )

        # Extract and validate fields
        text = data.get("text", "")
        send_photo = bool(data.get("send_photo", False))
        photo_mood = data.get("photo_mood", "casual")
        goal_status = data.get("goal_status", "chatting")

        # Validate mood
        if photo_mood not in VALID_MOODS:
            logger.warning(f"[{trace_id}] Invalid mood: {photo_mood}, defaulting to casual")
            photo_mood = "casual"

        # Validate goal
        if goal_status not in VALID_GOALS:
            logger.warning(f"[{trace_id}] Invalid goal: {goal_status}, defaulting to chatting")
            goal_status = "chatting"

        # Validate text is not empty
        if not text or not text.strip():
            raise LLMEmptyResponseError(
                message="Empty text field in JSON response",
                model=self.model,
                user_id=user_id,
                trace_id=trace_id,
            )

        return LLMResponse(
            text=text,
            send_photo=send_photo,
            photo_mood=photo_mood,
            goal_status=goal_status,
            raw_response=raw,
        )


class MockPersonaLLM(PersonaLLM):
    """Mock LLM for testing without API calls."""

    def __init__(self, persona: PersonaConfig, **kwargs):
        self.persona = persona
        self.system_prompt = persona.to_system_prompt()
        self.call_count = 0
        self.model = "mock"

    def generate(
        self,
        conversation_history: List[Dict],
        goal_status: str = "chatting",
        max_tokens: int = 300,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> LLMResponse:
        """Return mock responses for testing."""
        self.call_count += 1
        trace_id = trace_id or str(uuid.uuid4())[:8]

        # Simple mock logic based on goal
        if goal_status == "chatting":
            return LLMResponse(
                text="omg that's so cool! where are you from btw?",
                send_photo=True,
                photo_mood="excited",
                goal_status="asking_location",
                trace_id=trace_id,
            )
        elif goal_status == "asking_location":
            return LLMResponse(
                text="no way! I've always wanted to go there haha",
                send_photo=False,
                photo_mood="happy",
                goal_status="got_location",
                trace_id=trace_id,
            )
        else:
            return LLMResponse(
                text="haha nice!",
                send_photo=False,
                photo_mood="casual",
                goal_status=goal_status,
                trace_id=trace_id,
            )


if __name__ == "__main__":
    # Test with mock
    logging.basicConfig(level=logging.DEBUG)

    from config import PersonaConfig

    persona = PersonaConfig()
    llm = MockPersonaLLM(persona)

    # Simulate conversation
    history = [
        {"role": "user", "content": "hey! how are you?"},
    ]

    response = llm.generate(history, goal_status="chatting")
    print(f"Text: {response.text}")
    print(f"Photo: {response.send_photo} ({response.photo_mood})")
    print(f"Goal: {response.goal_status}")
    print(f"Trace ID: {response.trace_id}")
