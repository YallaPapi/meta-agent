# Coupling & Cohesion Analysis Report

## Phase 2 Review: Module Separation Quality

This report analyzes the current DM Bot module structure for coupling issues and cohesion quality, per the review prompt `reviews/02-coupling-cohesion-dm-modules.md`.

---

## 1. DMBot Orchestrator Analysis (`dm_bot.py`)

### Tasks Performed Directly vs Delegated

| Task | Direct or Delegated | Implementation |
|------|---------------------|----------------|
| UI Navigation | **Delegated** | `self.ui_controller.open_dm_inbox()`, etc. |
| Persona Prompt Construction | **Delegated** | `self.llm.generate()` handles internally |
| LLM Response Parsing | **Delegated** | `PersonaLLM._parse_response()` |
| Reading State | **Delegated** | `self.state_manager.load_state()` |
| Writing State | **Delegated** | `self.state_manager.save_state()` |
| Photo Selection | **Delegated** | `self.photo_manager.select_photo()` |
| Timing/Delays | **Direct** | `_calculate_response_delay()`, inline `time.sleep()` |
| City Extraction | **Direct** | `_extract_city()` - stub implementation |
| Goal Status Interpretation | **Direct** | Checks `got_location` to trigger city extraction |

### Assessment: **Good Separation**

DMBot properly acts as an orchestrator, delegating most specialized tasks. The only direct business logic is:
- Response delay calculation (appropriate - timing is orchestration concern)
- City extraction (TODO item - should eventually move to LLM or NLP service)

---

## 2. Supporting Module Analysis

### persona_llm.py

| Aspect | Details |
|--------|---------|
| **Responsibilities** | LLM API calls, prompt construction, JSON response parsing, response validation |
| **Imports From** | `config.PersonaConfig`, `anthropic`, `openai` |
| **Imported By** | `dm_bot.py` |
| **Cohesion** | HIGH - focused on LLM interaction only |
| **Coupling** | LOW - only depends on PersonaConfig for prompt template |

**Issues Found:**
- None significant. Module is well-isolated.

### conversation_state.py

| Aspect | Details |
|--------|---------|
| **Responsibilities** | State persistence, JSON file I/O, history management, stats aggregation |
| **Imports From** | Standard library only (`json`, `pathlib`, `dataclasses`) |
| **Imported By** | `dm_bot.py`, `main.py` |
| **Cohesion** | HIGH - focused on state management |
| **Coupling** | VERY LOW - no external dependencies |

**Issues Found:**
- None. Well-isolated module with clean interface.

### photo_manager.py

| Aspect | Details |
|--------|---------|
| **Responsibilities** | Photo bucket organization, mood-based selection, time-of-day override, stats |
| **Imports From** | Standard library only (`pathlib`, `random`, `datetime`) |
| **Imported By** | `dm_bot.py`, `main.py` |
| **Cohesion** | HIGH - focused on photo selection |
| **Coupling** | VERY LOW - no external dependencies |

**Issues Found:**
- None. Well-isolated module.

### appium_ui_controller.py

| Aspect | Details |
|--------|---------|
| **Responsibilities** | Instagram UI navigation, element discovery, DM interactions, screenshot capture |
| **Imports From** | `appium`, `config.Config` (for screen coordinates) |
| **Imported By** | `dm_bot.py`, `main.py` |
| **Cohesion** | MEDIUM-HIGH - handles all UI, but DM methods are distinct from general UI |
| **Coupling** | MEDIUM - depends on `config.Config` for screen coordinates |

**Issues Found:**
1. **Config coupling**: Uses `Config.SCREEN_CENTER_X`, `Config.FEED_BOTTOM_Y`, etc.
2. **Hardcoded element IDs**: Instagram-specific IDs like `com.instagram.android:id/row_thread_composer_edittext`
3. **Coordinate fallbacks**: Hardcoded fallback coordinates for when element IDs fail

### config.py

| Aspect | Details |
|--------|---------|
| **Responsibilities** | Persona definition, system prompt generation, settings, YAML loading |
| **Imports From** | Standard library (`yaml`, `os`, `dataclasses`) |
| **Imported By** | All other modules |
| **Cohesion** | MEDIUM - mixes persona logic with general config |
| **Coupling** | Low external, but creates dependency for all modules |

**Issues Found:**
1. **Dual responsibility**: `PersonaConfig` handles both configuration AND prompt generation
2. **Embedded JSON schema**: The persona prompt contains the JSON output schema inline

---

## 3. Coupling Issues Identified

### Issue 1: PersonaConfig Dual Responsibility
**Location**: `config.py:PersonaConfig`
**Problem**: `PersonaConfig` is both a data container AND generates the system prompt with embedded JSON schema.
**Impact**: Changing the prompt format requires modifying the config class.
**Severity**: MEDIUM

### Issue 2: Hardcoded UI Coordinates
**Location**: `appium_ui_controller.py` DM methods
**Problem**: Fallback coordinates like `self.tap(680, 120)` are hardcoded for specific screen resolution.
**Impact**: Will break on devices with different screen sizes.
**Severity**: HIGH for production use

### Issue 3: Screen Constants Coupling
**Location**: `appium_ui_controller.py` uses `Config.SCREEN_CENTER_X`, etc.
**Problem**: UI controller depends on a global Config class for screen geometry.
**Impact**: Makes testing harder, creates hidden dependency.
**Severity**: LOW-MEDIUM

### Issue 4: Goal Logic Split
**Location**: `dm_bot.py:_extract_city()` vs `persona_llm.py` goal_status
**Problem**: The LLM determines goal_status, but city extraction is handled separately in DMBot.
**Impact**: Business logic for goals is split across modules.
**Severity**: LOW (acceptable for now, but should unify)

---

## 4. Cohesion Assessment

| Module | Cohesion Level | Notes |
|--------|----------------|-------|
| `dm_bot.py` | **HIGH** | Clear orchestration role, delegates well |
| `persona_llm.py` | **HIGH** | Single responsibility: LLM interaction |
| `conversation_state.py` | **HIGH** | Single responsibility: state persistence |
| `photo_manager.py` | **HIGH** | Single responsibility: photo selection |
| `appium_ui_controller.py` | **MEDIUM-HIGH** | UI layer is cohesive, DM methods could be subclass |
| `config.py` | **MEDIUM** | Mixes config loading with prompt generation |

---

## 5. Proposed Refactors

### Refactor 1: Extract Prompt Builder from PersonaConfig

**Current**: `PersonaConfig.to_system_prompt()` generates the full prompt inline.

**Proposed**: Create a `PromptBuilder` class that takes a `PersonaConfig` and produces prompts.

```python
# New file: prompt_builder.py
class PersonaPromptBuilder:
    def __init__(self, persona: PersonaConfig):
        self.persona = persona

    def build_system_prompt(self) -> str:
        """Build system prompt with embedded JSON contract."""
        return f"""You are {self.persona.name}...

{self._build_json_schema()}

{self._build_goal_instructions()}
"""

    def _build_json_schema(self) -> str:
        """Return the JSON output schema."""
        ...

    def _build_goal_instructions(self) -> str:
        """Return goal progression instructions."""
        ...
```

**Benefit**: Separates config data from prompt logic. Allows A/B testing different prompts.

### Refactor 2: Extract Screen Geometry to Injectable Config

**Current**: `AppiumUIController` uses `Config.SCREEN_CENTER_X` directly.

**Proposed**: Pass screen geometry at construction time or use a ScreenConfig object.

```python
@dataclass
class ScreenGeometry:
    width: int = 720
    height: int = 1280

    @property
    def center_x(self) -> int:
        return self.width // 2

    @property
    def center_y(self) -> int:
        return self.height // 2

class AppiumUIController:
    def __init__(self, driver, screen: ScreenGeometry = None):
        self._driver = driver
        self._screen = screen or ScreenGeometry()
```

**Benefit**: Makes screen size configurable, improves testability.

### Refactor 3: Create DMUIHelper Subclass

**Current**: `AppiumUIController` has both general UI methods and DM-specific methods mixed together.

**Proposed**: Create `InstagramDMController` that extends `AppiumUIController`.

```python
class AppiumUIController:
    """Base UI controller with generic Android methods."""
    def tap(self, x, y): ...
    def swipe(self, ...): ...
    def dump_ui(self): ...

class InstagramDMController(AppiumUIController):
    """Instagram DM-specific automation."""
    DM_INBOX_ID = "com.instagram.android:id/action_bar_inbox_button"
    MESSAGE_INPUT_ID = "com.instagram.android:id/row_thread_composer_edittext"

    def open_dm_inbox(self): ...
    def send_dm_message(self, text): ...
```

**Benefit**: Cleaner separation, easier to add other Instagram features (Stories, Posts) without bloating the base class.

---

## 6. Anti-Coupling Rules

To maintain good separation going forward:

| Rule | Description |
|------|-------------|
| **UI → No LLM** | `appium_ui_controller.py` MUST NOT import persona_llm |
| **State → No UI** | `conversation_state.py` MUST NOT import appium_ui_controller |
| **LLM → No UI** | `persona_llm.py` MUST NOT import appium_ui_controller |
| **Photo → No LLM** | `photo_manager.py` MUST NOT import persona_llm |
| **Config → Standalone** | `config.py` should only import standard library |
| **Bot → All** | Only `dm_bot.py` should import all other modules |

### Import Graph (Current - Correct)

```
main.py
   └── dm_bot.py
          ├── config.py
          ├── conversation_state.py
          ├── persona_llm.py ─── config.py
          ├── photo_manager.py
          └── appium_ui_controller.py ─── config.py
```

---

## 7. Conclusion

The current module structure is **fundamentally sound** with good separation of concerns. The DMBot properly orchestrates, and supporting modules are largely decoupled.

**Priority fixes:**
1. **HIGH**: Extract hardcoded coordinates to injectable screen config
2. **MEDIUM**: Separate PromptBuilder from PersonaConfig
3. **LOW**: Consider InstagramDMController subclass for future extensibility

---

*Generated as part of DM Bot Coupling & Cohesion Review (Phase 2)*
*Review prompt: `reviews/02-coupling-cohesion-dm-modules.md`*
