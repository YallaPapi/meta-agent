"""Two-LLM Pipeline - Orchestrates the stage analyzer and response generator.

This module implements the core two-LLM architecture:
1. LLM 1 (Stage Analyzer): Analyzes user message and determines funnel stage
2. LLM 2 (Response Generator): Generates natural, varied response for the stage

Flow:
    user_message -> Stage Analyzer -> StageAnalysis -> Response Generator -> Response

The pipeline:
- Does NOT use keyword matching (all analysis is LLM-based)
- Tracks response variety to prevent repetition
- Handles location matching for the location_exchange stage
- Manages funnel stage transitions with validation
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from funnel_stages import (
    FunnelStage,
    is_valid_transition,
    is_terminal_stage,
)
from stage_analyzer import (
    StageAnalyzer,
    StageAnalysis,
    MockStageAnalyzer,
)
from response_generator import (
    ResponseGenerator,
    GeneratedResponse,
    PersonaConfig,
    MockResponseGenerator,
)
from conversation_state import ConversationState, ConversationStateManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from the two-LLM pipeline."""

    text: str
    send_photo: bool
    photo_mood: str
    stage_before: FunnelStage
    stage_after: FunnelStage
    stage_transitioned: bool

    # Analysis details
    detected_intent: str = ""
    location_mentioned: Optional[str] = None
    objection_detected: bool = False
    subscription_claimed: bool = False

    # Timing
    stage_analysis_ms: float = 0.0
    response_generation_ms: float = 0.0
    total_latency_ms: float = 0.0

    trace_id: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for the two-LLM pipeline."""

    # Response generator settings (LLM 2)
    response_provider: str = "anthropic"  # "anthropic" (Haiku) or "ollama"
    response_model: str = "claude-3-haiku-20240307"  # Haiku model

    # Stage analyzer settings (LLM 1) - now also uses Haiku
    analyzer_provider: str = "anthropic"  # "anthropic" (Haiku) or "ollama"
    analyzer_model: str = "claude-3-haiku-20240307"
    ollama_host: str = "http://localhost:11434"

    state_dir: str = "./state"
    persona: Optional[PersonaConfig] = None
    mock_mode: bool = False


class TwoLLMPipeline:
    """Orchestrates the two-LLM pipeline for DM conversations.

    This is the main entry point for processing user messages through
    the stage analyzer and response generator.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        metrics_callback: Optional[Callable] = None,
    ):
        """Initialize the two-LLM pipeline.

        Args:
            config: Pipeline configuration
            metrics_callback: Optional callback for recording metrics
        """
        self.config = config or PipelineConfig()
        self.metrics_callback = metrics_callback

        # Initialize state manager
        self.state_manager = ConversationStateManager(self.config.state_dir)

        # Initialize LLM components
        if self.config.mock_mode:
            self.stage_analyzer = MockStageAnalyzer()
            self.response_generator = MockResponseGenerator(
                persona=self.config.persona
            )
        else:
            # Stage Analyzer: Uses Haiku (fast/accurate)
            self.stage_analyzer = StageAnalyzer(
                provider=self.config.analyzer_provider,
                model=self.config.analyzer_model,
                ollama_host=self.config.ollama_host,
                metrics_callback=metrics_callback,
            )
            # Response Generator: Uses Anthropic Haiku (fast/cheap)
            self.response_generator = ResponseGenerator(
                persona=self.config.persona,
                provider=self.config.response_provider,
                model=self.config.response_model,
                ollama_host=self.config.ollama_host,
                metrics_callback=metrics_callback,
            )

    def process_message(
        self,
        user_id: str,
        user_message: str,
        trace_id: Optional[str] = None,
    ) -> PipelineResult:
        """Process a user message through the two-LLM pipeline.

        This is the main entry point. The pipeline:
        1. Loads conversation state
        2. Runs stage analysis (LLM 1)
        3. Updates state based on analysis
        4. Runs response generation (LLM 2)
        5. Saves updated state
        6. Returns the generated response

        Args:
            user_id: Instagram user ID
            user_message: The user's latest message
            trace_id: Optional trace ID for debugging

        Returns:
            PipelineResult with response and stage information
        """
        trace_id = trace_id or str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(f"[{trace_id}] Processing message for {user_id}: '{user_message[:50]}...'")

        # Load conversation state
        state = self.state_manager.load_state(user_id)
        stage_before = state.get_funnel_stage()

        # Add user message to history
        state.add_message(role="user", content=user_message)

        # === LLM 1: Stage Analysis ===
        analysis = self.stage_analyzer.analyze(
            current_stage=stage_before,
            conversation_history=state.history,
            user_message=user_message,
            trace_id=trace_id,
        )

        # Update state based on analysis
        self._update_state_from_analysis(state, analysis)
        stage_after = state.get_funnel_stage()

        # === LLM 2: Response Generation ===
        response = self.response_generator.generate(
            stage_analysis=analysis,
            conversation_history=state.history,
            used_phrases=state.get_used_phrases(),
            location_to_match=analysis.location_mentioned or state.location_mentioned,
            trace_id=trace_id,
        )

        # Add response to history and cache
        state.add_message(role="assistant", content=response.text)
        state.add_used_phrase(response.text)

        # Save state
        self.state_manager.save_state(user_id, state)

        total_time = (time.time() - start_time) * 1000

        logger.info(
            f"[{trace_id}] Pipeline complete: "
            f"stage={stage_before.value}->{stage_after.value}, "
            f"latency={total_time:.0f}ms"
        )

        if self.metrics_callback:
            self.metrics_callback(
                "pipeline_complete",
                stage_before=stage_before.value,
                stage_after=stage_after.value,
                transitioned=(stage_before != stage_after),
                total_latency_ms=total_time,
            )

        return PipelineResult(
            text=response.text,
            send_photo=response.send_photo,
            photo_mood=response.photo_mood,
            stage_before=stage_before,
            stage_after=stage_after,
            stage_transitioned=(stage_before != stage_after),
            detected_intent=analysis.detected_intent,
            location_mentioned=analysis.location_mentioned,
            objection_detected=analysis.objection_detected,
            subscription_claimed=analysis.subscription_claimed,
            stage_analysis_ms=analysis.latency_ms,
            response_generation_ms=response.latency_ms,
            total_latency_ms=total_time,
            trace_id=trace_id,
        )

    def _update_state_from_analysis(
        self,
        state: ConversationState,
        analysis: StageAnalysis,
    ):
        """Update conversation state based on stage analysis.

        Args:
            state: Conversation state to update
            analysis: Stage analysis from LLM 1
        """
        current_stage = state.get_funnel_stage()

        # Handle stage transition
        if analysis.should_transition:
            next_stage = analysis.next_stage

            # Validate transition
            if is_valid_transition(current_stage, next_stage):
                state.set_funnel_stage(next_stage)
                logger.debug(
                    f"Stage transition: {current_stage.value} -> {next_stage.value}"
                )
            else:
                logger.warning(
                    f"Invalid transition blocked: {current_stage.value} -> {next_stage.value}"
                )

        # Update location if mentioned
        if analysis.location_mentioned:
            state.location_mentioned = analysis.location_mentioned
            state.city = analysis.location_mentioned  # Legacy field
            logger.debug(f"Location detected: {analysis.location_mentioned}")

        # Track objections
        if analysis.objection_detected:
            state.objections_count += 1
            logger.debug(f"Objection #{state.objections_count} detected")

        # Track subscription claims - FORCE transition to verification
        if analysis.subscription_claimed:
            state.subscription_claimed = True
            logger.debug("Subscription claimed by user")
            # Force transition to verification if not already there
            current = state.get_funnel_stage()
            if current not in (FunnelStage.VERIFICATION, FunnelStage.CONVERTED):
                if is_valid_transition(current, FunnelStage.VERIFICATION):
                    state.set_funnel_stage(FunnelStage.VERIFICATION)
                    # Also update the analysis so response generator uses correct stage
                    analysis.next_stage = FunnelStage.VERIFICATION
                    analysis.should_transition = True
                    logger.info(f"FORCED transition to verification (subscription claimed)")

        # Check for terminal states
        final_stage = state.get_funnel_stage()
        if final_stage == FunnelStage.CONVERTED:
            state.converted = True
            logger.info("Conversation converted successfully!")
        elif final_stage == FunnelStage.DEAD_LEAD:
            logger.info("Conversation marked as dead lead")

    def get_conversation_summary(self, user_id: str) -> dict:
        """Get a summary of a conversation.

        Args:
            user_id: Instagram user ID

        Returns:
            Dictionary with conversation summary
        """
        state = self.state_manager.load_state(user_id)
        return {
            "user_id": user_id,
            "funnel_stage": state.funnel_stage,
            "messages_count": len(state.history),
            "location_mentioned": state.location_mentioned,
            "location_matched": state.location_matched,
            "objections_count": state.objections_count,
            "link_sent": state.link_sent,
            "subscription_claimed": state.subscription_claimed,
            "converted": state.converted,
            "last_interaction": state.last_interaction,
        }

    def reset_conversation(self, user_id: str):
        """Reset a conversation to initial state.

        Args:
            user_id: Instagram user ID
        """
        self.state_manager.delete_state(user_id)
        logger.info(f"Reset conversation for {user_id}")


def create_pipeline(
    mock_mode: bool = False,
    response_provider: str = "anthropic",
    response_model: str = "claude-3-haiku-20240307",
    analyzer_provider: str = "anthropic",
    analyzer_model: str = "claude-3-haiku-20240307",
    ollama_host: str = "http://localhost:11434",
    state_dir: str = "./state",
    persona: Optional[PersonaConfig] = None,
) -> TwoLLMPipeline:
    """Factory function to create a configured pipeline.

    Args:
        mock_mode: If True, use mock LLMs for testing
        response_provider: "anthropic" (Haiku) or "ollama" for response gen
        response_model: Model for response generation
        analyzer_provider: "anthropic" (Haiku) or "ollama" for stage analysis
        analyzer_model: Model for stage analysis
        ollama_host: Ollama server URL
        state_dir: Directory for state persistence
        persona: Persona configuration

    Returns:
        Configured TwoLLMPipeline instance
    """
    config = PipelineConfig(
        response_provider=response_provider,
        response_model=response_model,
        analyzer_provider=analyzer_provider,
        analyzer_model=analyzer_model,
        ollama_host=ollama_host,
        state_dir=state_dir,
        persona=persona,
        mock_mode=mock_mode,
    )
    return TwoLLMPipeline(config)


if __name__ == "__main__":
    # Demo
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Use mock mode by default for testing
    mock_mode = "--real" not in sys.argv

    print(f"Running in {'REAL' if not mock_mode else 'MOCK'} mode")
    print("=" * 60)

    pipeline = create_pipeline(
        mock_mode=mock_mode,
        state_dir="./test_state",
    )

    # Simulate a conversation
    test_messages = [
        ("user1", "hey! love your pics"),
        ("user1", "im doing good, wbu? where are you from?"),
        ("user1", "I'm in LA"),
        ("user1", "yeah that sounds cool! whats your snap?"),
        ("user1", "nah I don't really use OF"),
        ("user1", "ok fine I'll check it out"),
        ("user1", "done, I subscribed"),
    ]

    for user_id, message in test_messages:
        print(f"\n[USER]: {message}")
        result = pipeline.process_message(user_id, message)
        print(f"[BOT]: {result.text}")
        print(f"  Stage: {result.stage_before.value} -> {result.stage_after.value}")
        print(f"  Photo: {result.send_photo} ({result.photo_mood})")
        print(f"  Latency: {result.total_latency_ms:.0f}ms")

    # Print summary
    print("\n" + "=" * 60)
    print("Conversation Summary:")
    print(pipeline.get_conversation_summary("user1"))
