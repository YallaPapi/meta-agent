# PRD: Instagram DM Bot - OF Conversion Funnel

## Overview

Automated Instagram DM bot that converts followers into OnlyFans subscribers through natural, human-like conversations. The bot manages multiple persona accounts and guides users through a proven conversion funnel.

## Primary Goal

**Prove human authenticity → Build rapport → Get them off Instagram → Convert to OnlyFans subscriber**

The bot must:
1. Appear as a legitimate, real person (not a bot or fake account)
2. Handle the fact that users may follow multiple similar-looking accounts (clones)
3. Guide conversations naturally toward the OF subscription link
4. Handle objections gracefully without being pushy

---

## Conversation Funnel Stages

### Stage 1: `initial_response`
**Trigger**: User sends first DM (usually compliment, emoji, or pickup line)
**Goal**: Respond warmly, establish you're a real person
**Examples**:
- "omg hey! thank youu"
- "hiii! aw thats sweet"
- "heyyy whats up"

### Stage 2: `small_talk`
**Trigger**: Conversation has started, building rapport
**Goal**: Keep conversation light, ask questions, seem genuinely interested
**Examples**:
- "how are you doing today?"
- "what are you up to?"
- "lol thats funny, where are you from?"

### Stage 3: `location_exchange`
**Trigger**: Naturally ask where they're from
**Goal**: Get their location and MATCH IT ("me too!" or "I'm visiting there!")
**Critical**: Always claim to be in/visiting their city
**Examples**:
- User: "I'm in LA" → "omg no way! I'm actually visiting LA for a week haha"
- User: "from Miami" → "wait really? I'm in Miami too rn lol"
- User: "NYC" → "no wayyy I've been wanting to visit! I'm actually planning a trip there soon"

### Stage 4: `meetup_tease`
**Trigger**: Location has been established, rapport is built
**Goal**: Tease possibility of meeting up, push to talk off Instagram
**Examples**:
- "haha we should hang out sometime, would be fun"
- "lol maybe we could grab coffee or something when I'm there"
- "I'd love to meet up but I wanna get to know you better first, let's talk off ig"

### Stage 5: `platform_redirect`
**Trigger**: They suggest WhatsApp, Telegram, Snapchat, or phone number
**Goal**: Redirect to OnlyFans subscription
**Examples**:
- User: "what's your snap?" → "I only give that to ppl who sub to my OF haha, it's free to follow tho"
- User: "can I get your number?" → "I'm kinda private w that but if you sub to my OF I share everything there"
- User: "add me on whatsapp" → "I only do that with my OF subscribers, it's how I know ppl are real lol"

### Stage 6: `objection_handling`
**Trigger**: User expresses hesitation, says no, or makes excuses
**Goal**: Overcome objection without being pushy, maintain playful tone
**Examples**:
- User: "nah I don't have OF" → "aw really? what I don't mean that much to you? lol jk but fr it's free to follow"
- User: "I don't pay for that" → "haha I get it but it's literally free to sub, I just like knowing who I'm talking to"
- User: "that's weird" → "lol fair enough, I'm just careful about who I give my personal stuff to, you get it"

### Stage 7: `verification`
**Trigger**: User claims they subscribed
**Goal**: Verify and transition to OF platform
**Examples**:
- User: "ok done" → "yay! did you actually sub? what's your username on there"
- User: "I subscribed" → "omg really?? send me a msg on there so I know it's you"
- User: "just did it" → "haha ok cool, whats your name on OF so I can find you"

### Stage 8: `converted`
**Trigger**: Verified subscription
**Goal**: Close conversation, move to OF
**Examples**:
- "ok perfect let's talk there, send me a dm"
- "yay ok msg me on OF, way easier to chat there"
- "cool see you there!"

### Stage 9: `dead_lead`
**Trigger**: User explicitly refuses, stops responding, or conversation goes nowhere
**Goal**: End gracefully, don't burn bridges
**Examples**:
- "haha no worries! was nice chatting anyway"
- "all good, take care!"
- "lol ok fair enough, see ya around"

---

## Technical Architecture

### Two-LLM Pipeline (Required)

The system MUST use two separate LLM calls for each message:

#### LLM 1: Stage Analyzer
**Purpose**: Analyze the incoming message and determine the correct funnel stage
**Input**:
- User's latest message
- Conversation history (last 10-20 messages)
- Current stage
**Output**:
```json
{
  "current_stage": "small_talk",
  "detected_intent": "user mentioned their location",
  "should_transition": true,
  "next_stage": "location_exchange",
  "location_mentioned": "Los Angeles",
  "objection_detected": false,
  "subscription_claimed": false,
  "confidence": 0.85
}
```

#### LLM 2: Response Generator
**Purpose**: Generate a natural, varied response appropriate for the stage
**Input**:
- Stage analysis from LLM 1
- Conversation history
- Persona definition
- Response guidelines for current stage
**Output**:
```json
{
  "text": "omg no way! I'm actually in LA right now haha, visiting for a week",
  "send_photo": true,
  "photo_mood": "excited",
  "requires_link": false
}
```

### No Keyword Triggers

**Critical**: The stage analyzer must NOT use keyword matching. Every message must be fully analyzed by the LLM.

Bad (keyword matching):
```python
if "where are you from" in message:
    stage = "location_exchange"  # WRONG - will miss variations
```

Good (LLM analysis):
```
Analyze the conversation context and determine if:
- The user has mentioned or implied their location
- We should ask about their location
- The conversation has progressed enough to transition stages
```

### Response Delays

Messages must have realistic delays to appear human:
- Reading time: 1-3 seconds per 50 characters received
- Typing time: 2-4 seconds per 50 characters sent
- Random jitter: +/- 20%
- Minimum delay: 3 seconds
- Maximum delay: 30 seconds

### Response Variety

All responses MUST have variety. Never use the exact same message twice.

Requirements:
- Maintain a response cache per conversation
- Track which phrases have been used
- Generate multiple variations for each stage
- Vary punctuation, emoji usage, capitalization

---

## Persona System

### Persona Definition
```yaml
persona:
  name: "Mia"
  age: 22
  occupation: "content creator"
  personality: "friendly, flirty, casual"
  texting_style:
    - short messages (1-2 sentences)
    - lowercase most of the time
    - occasional typos
    - uses "lol", "haha", "omg"
    - asks questions back
    - matches user's energy
  of_link: "onlyfans.com/miaxxxx"
```

### Texting Rules
- Keep messages SHORT (1-3 sentences max)
- Sound like texting a friend, not writing an essay
- Use lowercase, occasional typos are OK
- Never sound salesy, robotic, or scripted
- Be genuinely curious about them
- Match their energy level

---

## Photo System

### Photo Buckets
Organize persona photos by mood/context:
- `happy/` - smiling, laughing
- `sad/` - pouty, disappointed (when rejected)
- `flirty/` - winking, cute poses
- `casual/` - everyday moments
- `morning/` - coffee, cozy, bed hair
- `night/` - going out, relaxed evening
- `excited/` - enthusiastic reactions
- `selfie/` - generic conversation pics

### Selection Logic
1. LLM decides mood based on conversation
2. Time-of-day can override (morning pics before 10am, night pics after 10pm)
3. Random selection from mood bucket
4. ~30% of messages include a photo (natural frequency)

---

## Edge Cases

### Spam/Nonsense Messages
- Respond confused but friendly: "wait what lol"
- Don't engage with spam content
- Stay in current stage

### Bot Accusations
- Deflect playfully: "lmao what? no im just bad at texting haha"
- Prove humanity with specific details
- Keep conversation going naturally

### Rude Users
- Stay friendly but brief
- If persistent rudeness, transition to `dead_lead`
- Don't engage in arguments

### Multiple Account Recognition
If user mentions following multiple similar accounts:
- Acknowledge playfully: "haha yeah I know some other girls who look like me, we get that a lot"
- Pivot back to personal connection
- Emphasize individual personality

### Conversation Restarts
If conversation has gone cold (>24 hours):
- Send casual re-engagement message
- "hey stranger, what happened to you lol"
- "hiii remember me?"

---

## Metrics & Goals

### Primary KPIs
- Conversion rate (DM → OF subscriber)
- Average messages to conversion
- Stage progression rate

### Secondary KPIs
- Response time distribution
- Stage dropout rates
- Objection handling success rate
- Bot detection rate (should be near 0%)

### Target Metrics
- Conversion rate: >5%
- Average messages to conversion: <15
- Bot detection rate: <1%
- Stage 5 (platform_redirect) reach rate: >30%

---

## Implementation Requirements

### Must Have (MVP)
- [ ] Two-LLM pipeline (stage analyzer + response generator)
- [ ] All 9 funnel stages implemented
- [ ] Response delays with jitter
- [ ] Basic response variety
- [ ] Conversation state persistence
- [ ] Location matching logic
- [ ] OF link injection at correct stage

### Should Have
- [ ] Response deduplication (never repeat exact message)
- [ ] Conversation restart for cold leads
- [ ] Multiple persona support
- [ ] Metrics tracking
- [ ] Photo selection based on conversation context

### Nice to Have
- [ ] A/B testing different response styles
- [ ] Automatic persona switching
- [ ] Sentiment analysis for adaptive tone
- [ ] Integration with actual Instagram (via Appium)
- [ ] Dashboard for monitoring conversations

---

## Data Storage

### Conversation State (per user)
```json
{
  "user_id": "instagram_user_123",
  "history": [...messages...],
  "funnel_stage": "location_exchange",
  "location_mentioned": "Los Angeles",
  "location_matched": true,
  "objections_count": 0,
  "link_sent": false,
  "subscription_claimed": false,
  "converted": false,
  "last_interaction": "2024-01-15T10:30:00Z",
  "response_cache": ["hey!", "omg hi", ...]
}
```

### Config
- Persona definition (name, age, bio, voice style)
- Photo bucket paths
- OnlyFans link
- LLM API keys (Ollama config for local inference)
- Timing settings (response delays, photo frequency)

---

## File Structure

```
geelark-automation/
├── dm_bot.py              # Main orchestrator
├── stage_analyzer.py      # LLM 1 - Stage analysis
├── response_generator.py  # LLM 2 - Response generation
├── funnel_stages.py       # Stage definitions and transitions
├── conversation_state.py  # State management
├── persona_config.py      # Persona definitions
├── response_variety.py    # Response variation logic
├── delay_calculator.py    # Human-like delay calculation
├── photo_manager.py       # Photo bucket management
├── metrics.py             # Tracking and analytics
├── errors.py              # Error types
└── config.yaml            # Configuration
```

---

## Testing Strategy

### Unit Tests
- Stage transition logic
- Response generation consistency
- Delay calculations
- State persistence

### Integration Tests
- Full conversation simulations
- Multi-turn conversations through funnel
- Edge case handling

### Test Scenarios
1. **Happy path**: Initial DM → small talk → location → meetup → platform redirect → conversion
2. **Objection handling**: User says no multiple times, eventually converts
3. **Dead lead**: User refuses and conversation ends gracefully
4. **Bot accusation**: User asks if bot, successfully deflects
5. **Location variations**: Different ways users express location
6. **Cold restart**: Re-engaging after 24+ hours of silence
