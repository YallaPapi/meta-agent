"""Funnel Stages - Defines the OF conversion funnel stages and transitions.

The conversation funnel guides users through these stages:
1. initial_response - First response to user's DM
2. small_talk - Building rapport, ask what city they're in
3. location_exchange - Match their location ("im in X too")
4. vibing - Keep chatting naturally until THEY ask for contact
5. platform_redirect - When THEY ask for snap/number, redirect to OF
6. objection_handling - Handling hesitation or "no"
7. verification - Verifying they subscribed
8. converted - Successfully converted
9. dead_lead - Conversation ended without conversion
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class FunnelStage(str, Enum):
    """The 9 stages of the OF conversion funnel."""

    INITIAL_RESPONSE = "initial_response"
    SMALL_TALK = "small_talk"
    LOCATION_EXCHANGE = "location_exchange"
    VIBING = "vibing"
    PLATFORM_REDIRECT = "platform_redirect"
    OBJECTION_HANDLING = "objection_handling"
    VERIFICATION = "verification"
    CONVERTED = "converted"
    DEAD_LEAD = "dead_lead"


# Valid stage transitions - defines which stages can transition to which
STAGE_TRANSITIONS: Dict[FunnelStage, List[FunnelStage]] = {
    FunnelStage.INITIAL_RESPONSE: [
        FunnelStage.INITIAL_RESPONSE,  # Stay if still initial greeting
        FunnelStage.SMALL_TALK,  # Normal progression
        FunnelStage.DEAD_LEAD,  # They leave immediately
    ],
    FunnelStage.SMALL_TALK: [
        FunnelStage.SMALL_TALK,  # Continue chatting
        FunnelStage.LOCATION_EXCHANGE,  # They mention a city
        FunnelStage.VIBING,  # Skip location if vibe is good
        FunnelStage.PLATFORM_REDIRECT,  # They ask for contact early
        FunnelStage.DEAD_LEAD,  # Conversation dies
    ],
    FunnelStage.LOCATION_EXCHANGE: [
        FunnelStage.LOCATION_EXCHANGE,  # Still discussing location
        FunnelStage.VIBING,  # After you match their location, keep vibing
        FunnelStage.PLATFORM_REDIRECT,  # They ask for contact after location
        FunnelStage.DEAD_LEAD,  # Conversation dies
    ],
    FunnelStage.VIBING: [
        FunnelStage.VIBING,  # Keep chatting naturally
        FunnelStage.PLATFORM_REDIRECT,  # THEY ask for snap/number/whatsapp
        FunnelStage.DEAD_LEAD,  # Conversation dies
    ],
    FunnelStage.PLATFORM_REDIRECT: [
        FunnelStage.PLATFORM_REDIRECT,  # Still redirecting
        FunnelStage.OBJECTION_HANDLING,  # They object/hesitate
        FunnelStage.VERIFICATION,  # They claim to have subscribed
        FunnelStage.DEAD_LEAD,  # They refuse and leave
    ],
    FunnelStage.OBJECTION_HANDLING: [
        FunnelStage.OBJECTION_HANDLING,  # Still handling objections
        FunnelStage.PLATFORM_REDIRECT,  # Try redirecting again
        FunnelStage.VERIFICATION,  # They subscribed after objection
        FunnelStage.DEAD_LEAD,  # They refuse
    ],
    FunnelStage.VERIFICATION: [
        FunnelStage.VERIFICATION,  # Still verifying
        FunnelStage.CONVERTED,  # Verified subscription
        FunnelStage.OBJECTION_HANDLING,  # They lied, didn't subscribe
        FunnelStage.DEAD_LEAD,  # Conversation dies
    ],
    FunnelStage.CONVERTED: [
        FunnelStage.CONVERTED,  # Stay converted
    ],
    FunnelStage.DEAD_LEAD: [
        FunnelStage.DEAD_LEAD,  # Stay dead
        FunnelStage.INITIAL_RESPONSE,  # Conversation restart after cold period
        FunnelStage.VERIFICATION,  # They come back and subscribe
        FunnelStage.PLATFORM_REDIRECT,  # They reconsider
    ],
}


@dataclass
class StageGuidelines:
    """Guidelines for a specific funnel stage."""

    stage: FunnelStage
    trigger: str
    goal: str
    examples: List[str]
    notes: Optional[str] = None


# Stage guidelines for LLM prompts
STAGE_GUIDELINES: Dict[FunnelStage, StageGuidelines] = {
    FunnelStage.INITIAL_RESPONSE: StageGuidelines(
        stage=FunnelStage.INITIAL_RESPONSE,
        trigger="User sends first DM (usually compliment, emoji, or pickup line)",
        goal="Respond - you get 1000s of these DMs daily so you're not excited, just chill",
        examples=[
            "thx hru",
            "hey wyd",
            "thx ❤️ wbu",
        ],
        notes="Stay chill and unbothered - random compliments from strangers arent exciting to you",
    ),
    FunnelStage.SMALL_TALK: StageGuidelines(
        stage=FunnelStage.SMALL_TALK,
        trigger="Conversation has started, building rapport",
        goal="Chat naturally, eventually ask what city theyre in",
        examples=[
            "nice what do u do",
            "lol thats cool",
            "what city u in",
            "User: 'can we fuck' -> 'lol chill we just met, what city u in'",
            "User: 'send nudes' -> 'haha slow down, wbu whats ur deal'",
        ],
        notes="Forward/sexual messages = INTEREST. Deflect and keep chatting. Ask for CITY not 'where from'",
    ),
    FunnelStage.LOCATION_EXCHANGE: StageGuidelines(
        stage=FunnelStage.LOCATION_EXCHANGE,
        trigger="They mentioned a city OR you need to ask what city theyre in",
        goal="Get their CITY and MATCH IT - say you're there or visiting soon",
        examples=[
            "User: 'LA' -> 'oh nice im in LA too rn'",
            "User: 'Miami' -> 'wait fr im visiting miami next week'",
            "User: 'NYC' -> 'no way im gonna be in nyc soon'",
            "(if no city yet) -> 'what city u in'",
        ],
        notes="ALWAYS match their city - either you live there or visiting soon",
    ),
    FunnelStage.VIBING: StageGuidelines(
        stage=FunnelStage.VIBING,
        trigger="Location matched, now just chatting naturally",
        goal="Keep the vibe going. DO NOT suggest meetups. Wait for THEM to ask for contact.",
        examples=[
            "lol nice",
            "what do u do for fun",
            "im so bored rn",
            "haha same",
        ],
        notes="DO NOT suggest meeting up or ask for their contact. Let THEM bring it up. Just vibe.",
    ),
    FunnelStage.PLATFORM_REDIRECT: StageGuidelines(
        stage=FunnelStage.PLATFORM_REDIRECT,
        trigger="THEY ask for WhatsApp, Telegram, Snapchat, phone number, or to meet up",
        goal="Redirect to OnlyFans subscription - that's how you know they're real",
        examples=[
            "User: 'what's your snap?' -> 'i only give that to my OF subscribers haha'",
            "User: 'can I get your number?' -> 'i share that stuff on my OF, thats how i know ur real'",
            "User: 'lets meet up' -> 'maybe haha sub to my OF first so i know ur not a creep lol'",
        ],
        notes="Do NOT say it's free - OF costs money. Say it's how you verify people are real.",
    ),
    FunnelStage.OBJECTION_HANDLING: StageGuidelines(
        stage=FunnelStage.OBJECTION_HANDLING,
        trigger="User expresses hesitation, says no, or makes excuses about OF",
        goal="Overcome objection without being pushy, stay playful, emphasize it's about trust/verification",
        examples=[
            "User: 'nah I don't have OF' -> 'aw cmon its not that serious lol i just wanna know ur real'",
            "User: 'I don't pay for that' -> 'haha i get it but thats just how i know who im talking to'",
            "User: 'that's weird' -> 'lol i mean i get random creeps in my dms all day, gotta be careful'",
        ],
        notes="Do NOT say it's free. Frame it as trust/verification - you need to know they're real.",
    ),
    FunnelStage.VERIFICATION: StageGuidelines(
        stage=FunnelStage.VERIFICATION,
        trigger="User claims they subscribed",
        goal="Verify and transition to OF platform",
        examples=[
            "User: 'ok done' -> 'yay! did you actually sub? what's your username on there'",
            "User: 'I subscribed' -> 'omg really?? send me a msg on there so I know it's you'",
            "User: 'just did it' -> 'haha ok cool, whats your name on OF so I can find you'",
        ],
    ),
    FunnelStage.CONVERTED: StageGuidelines(
        stage=FunnelStage.CONVERTED,
        trigger="Verified subscription",
        goal="Close conversation, move to OF",
        examples=[
            "ok perfect let's talk there, send me a dm",
            "yay ok msg me on OF, way easier to chat there",
            "cool see you there!",
        ],
    ),
    FunnelStage.DEAD_LEAD: StageGuidelines(
        stage=FunnelStage.DEAD_LEAD,
        trigger="User EXPLICITLY says stop/leave me alone, or refuses subscription with finality after multiple attempts",
        goal="End gracefully, don't burn bridges",
        examples=[
            "haha no worries! was nice chatting anyway",
            "all good, take care!",
            "lol ok fair enough, see ya around",
        ],
        notes="ONLY use for explicit rejection. Forward/sexual messages are NOT dead leads - they show interest!",
    ),
}


def is_valid_transition(current: FunnelStage, next_stage: FunnelStage) -> bool:
    """Check if a stage transition is valid.

    Args:
        current: Current funnel stage
        next_stage: Proposed next stage

    Returns:
        True if the transition is valid, False otherwise
    """
    valid_next = STAGE_TRANSITIONS.get(current, [])
    return next_stage in valid_next


def get_valid_transitions(current: FunnelStage) -> List[FunnelStage]:
    """Get list of valid next stages from current stage.

    Args:
        current: Current funnel stage

    Returns:
        List of valid next stages
    """
    return STAGE_TRANSITIONS.get(current, [])


def get_stage_guidelines(stage: FunnelStage) -> StageGuidelines:
    """Get guidelines for a specific stage.

    Args:
        stage: Funnel stage to get guidelines for

    Returns:
        StageGuidelines for the stage
    """
    return STAGE_GUIDELINES[stage]


def format_stage_context_for_llm(stage: FunnelStage) -> str:
    """Format stage guidelines as context for LLM prompt.

    Args:
        stage: Current funnel stage

    Returns:
        Formatted string for LLM context
    """
    guidelines = STAGE_GUIDELINES[stage]
    valid_next = get_valid_transitions(stage)

    lines = [
        f"## Current Stage: {stage.value}",
        f"**Trigger**: {guidelines.trigger}",
        f"**Goal**: {guidelines.goal}",
        "",
        "**Example Responses**:",
    ]

    for example in guidelines.examples:
        lines.append(f"- {example}")

    if guidelines.notes:
        lines.append("")
        lines.append(f"**Note**: {guidelines.notes}")

    lines.append("")
    lines.append(f"**Valid Next Stages**: {', '.join(s.value for s in valid_next)}")

    return "\n".join(lines)


def is_terminal_stage(stage: FunnelStage) -> bool:
    """Check if a stage is a terminal stage (conversation complete).

    Args:
        stage: Stage to check

    Returns:
        True if terminal (converted or dead_lead)
    """
    return stage in (FunnelStage.CONVERTED, FunnelStage.DEAD_LEAD)


def is_conversion_complete(stage: FunnelStage) -> bool:
    """Check if conversion was successful.

    Args:
        stage: Current stage

    Returns:
        True if stage is CONVERTED
    """
    return stage == FunnelStage.CONVERTED


if __name__ == "__main__":
    # Demo: print all stages and their guidelines
    for stage in FunnelStage:
        print(f"\n{'='*60}")
        print(format_stage_context_for_llm(stage))
