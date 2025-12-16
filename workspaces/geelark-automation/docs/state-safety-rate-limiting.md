# State Safety & Rate Limiting

## Phase 4-5 Review: Robustness Improvements

This document covers the safety and rate limiting features added to conversation state management and UI automation.

---

## State Safety Features

### 1. Atomic Writes

**Problem**: If the bot crashes while writing a state file, the JSON can be left in a corrupted/partial state.

**Solution**: Temp file + rename pattern

```python
# Write to temp file first
fd, temp_path = tempfile.mkstemp(suffix=".json", dir=state_dir)
with os.fdopen(fd, "w") as f:
    json.dump(state.to_dict(), f)

# Atomic rename
os.replace(temp_path, state_path)
```

**Behavior**:
- On success: State file is atomically updated
- On crash during write: Temp file is orphaned, original state preserved
- On crash during rename: Either old or new state (not partial)

### 2. Max History Length

**Problem**: Unbounded conversation history grows forever, consuming memory and disk.

**Solution**: `MAX_HISTORY_LENGTH = 100` messages

```python
if len(self.history) > MAX_HISTORY_LENGTH:
    self.history = self.history[-MAX_HISTORY_LENGTH:]
```

**Behavior**:
- Old messages are automatically trimmed
- Most recent 100 messages kept
- Warning logged when trimming occurs

### 3. Message Truncation

**Problem**: Extremely long messages (spam, copy-paste) bloat state files.

**Solution**: `MAX_MESSAGE_LENGTH = 2000` characters

```python
if len(content) > MAX_MESSAGE_LENGTH:
    content = content[:MAX_MESSAGE_LENGTH] + "..."
```

### 4. Corrupted File Recovery

**Problem**: State file could become corrupted (disk error, encoding issue).

**Solution**: Backup corrupted files and start fresh

```python
except json.JSONDecodeError as e:
    self._backup_corrupted(state_path)  # Save to .corrupted.YYYYMMDD_HHMMSS.json
    state = ConversationState(user_id=user_id)  # Start fresh
```

**Behavior**:
- Corrupted file renamed to `.corrupted.<timestamp>.json`
- New empty state created
- Error logged with details

---

## Rate Limiting

### Per-Conversation Limits

Added to `ConversationState`:

```python
messages_sent_today: int = 0
messages_sent_hour: int = 0
last_message_date: str = ""  # YYYY-MM-DD
last_message_hour: str = ""  # YYYY-MM-DD-HH

def can_send_message(self, max_per_hour=10, max_per_day=50) -> bool:
    """Check if rate limits allow sending."""
```

### Default Limits

| Limit | Default | Description |
|-------|---------|-------------|
| Per hour | 10 | Max messages per conversation per hour |
| Per day | 50 | Max messages per conversation per day |

### Integration with DMBot

```python
# In dm_bot.py:process_conversation()
state = self.state_manager.load_state(user_id)

# Check rate limit before responding
if not state.can_send_message():
    logger.warning(f"Rate limited: {user_id}")
    return None  # Skip this conversation

# ... generate and send response
```

---

## UI Automation Robustness

### Screen Geometry Injection

Replaced hardcoded coordinates with injectable `ScreenGeometry`:

```python
@dataclass
class ScreenGeometry:
    width: int = 720
    height: int = 1280

    @property
    def center_x(self) -> int:
        return self.width // 2

    @property
    def dm_button_x(self) -> int:
        return int(self.width * 0.94)  # 94% from left
```

### Coordinate Strategy

All fallback coordinates now use percentage-based calculations:

| Element | X Position | Y Position |
|---------|------------|------------|
| DM button | `width * 0.94` | `height * 0.09` |
| Message input | `width * 0.5` | `height * 0.9` |
| Send button | `width * 0.94` | `height * 0.9` |
| Gallery button | `width * 0.08` | `height * 0.9` |
| First photo | `width * 0.14` | `height * 0.31` |

### Element Discovery Priority

1. **Content-desc** (accessibility label) - Most stable
2. **Resource ID** - Stable but app-version dependent
3. **Coordinate fallback** - Last resort, screen-size aware

---

## Safe DM Automation Checklist

### Before Running

- [ ] Set appropriate rate limits in config
- [ ] Verify screen resolution matches ScreenGeometry
- [ ] Ensure state directory has write permissions
- [ ] Test with `--simulate` flag first

### Runtime Safety

- [ ] Rate limiting enabled (`can_send_message()` check)
- [ ] Human-like delays between messages (2-8s)
- [ ] Photo randomization (~30% probability)
- [ ] Typing delay based on message length

### Error Recovery

- [ ] Corrupted state files backed up automatically
- [ ] New state created on load failure
- [ ] Rate limit counters reset on new day/hour
- [ ] Exception logging with user context

---

## Configuration Recommendations

### Conservative (Low Risk)
```yaml
max_messages_per_hour: 5
max_messages_per_day: 20
photo_probability: 0.20
min_response_delay: 5.0
max_response_delay: 15.0
poll_interval: 60
```

### Standard (Balanced)
```yaml
max_messages_per_hour: 10
max_messages_per_day: 50
photo_probability: 0.30
min_response_delay: 2.0
max_response_delay: 8.0
poll_interval: 30
```

### Aggressive (Higher Risk)
```yaml
max_messages_per_hour: 20
max_messages_per_day: 100
photo_probability: 0.40
min_response_delay: 1.0
max_response_delay: 5.0
poll_interval: 15
```

---

*Generated as part of State Safety & Rate Limiting Review (Phases 4-5)*
*Review prompts: `04-conversation-state-file-safety.md`, `05-dm-ui-automation-robustness-rate-limiting.md`*
