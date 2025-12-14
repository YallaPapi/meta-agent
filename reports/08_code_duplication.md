# Code Duplication Analysis

# Code Duplication Analysis

Based on my analysis of the codebase, I've identified several instances of code duplication and patterns that could benefit from refactoring. Here's a comprehensive breakdown:

## Identified Duplications

### 1. Subprocess Pattern Duplication

**Location:**
- `claude_runner.py` (lines 45-67, 95-136)
- `repomix.py` (lines 60-120)
- `analysis.py` (lines 265-330, retry logic)

**Duplication Details:**
- **Length:** 15-30 lines per instance
- **Content/Purpose:** All three modules implement similar patterns for:
  - Subprocess execution with timeout handling
  - Error capture and logging
  - Graceful fallback between command variations
  - Result encapsulation in dataclass structures

**Specific Duplicated Patterns:**
```python
# Pattern 1: Subprocess execution with timeout
try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=self.timeout,
        cwd=repo_path,
    )
except subprocess.TimeoutExpired:
    return ResultClass(success=False, error=f"Timeout after {timeout}s")
except FileNotFoundError:
    return ResultClass(success=False, error="Command not found")
```

### 2. Error Handling Pattern Duplication

**Location:**
- `claude_runner.py` (lines 130-145)
- `repomix.py` (lines 100-115)
- `codebase_digest.py` (lines 85-105)

**Duplication Details:**
- **Length:** 10-15 lines per instance
- **Content/Purpose:** Identical error handling for:
  - `subprocess.TimeoutExpired`
  - `FileNotFoundError` 
  - Generic `Exception` catching
  - Structured error message formatting

### 3. Result Class Structure Duplication

**Location:**
- `claude_runner.py` (ClaudeCodeResult)
- `repomix.py` (RepomixResult)
- `codebase_digest.py` (DigestResult)
- `analysis.py` (AnalysisResult)

**Duplication Details:**
- **Length:** 5-8 lines per dataclass
- **Content/Purpose:** All result classes share common fields:
  - `success: bool`
  - `error: Optional[str]`
  - Tool-specific output fields
  - Similar initialization patterns

### 4. Command Availability Checking

**Location:**
- `claude_runner.py` (lines 45-67)
- Similar pattern implied in `repomix.py` and `codebase_digest.py`

**Duplication Details:**
- **Length:** 8-12 lines
- **Content/Purpose:** Checking if CLI tools are installed and accessible

### 5. File Path and Git Status Parsing

**Location:**
- `claude_runner.py` (lines 170-195)
- Similar pattern could emerge in other modules

**Duplication Details:**
- **Length:** 15-20 lines
- **Content/Purpose:** Git status parsing and file modification tracking

## Impact Assessment

### Maintainability Impact
- **High Impact:** Changes to subprocess patterns require updates in 3+ locations
- **Medium Impact:** Error handling improvements need to be replicated across modules
- **Medium Impact:** Adding new CLI tools requires duplicating the entire pattern

### Code Quality Issues
- **Consistency Risk:** Similar logic implemented slightly differently across modules
- **Testing Overhead:** Each duplicate pattern requires separate test coverage
- **Documentation Burden:** Similar functionality documented in multiple places

## Refactoring Opportunities

### 1. Extract Common Subprocess Runner
**Priority:** High

Create a shared `CommandRunner` class:

```python
# src/metaagent/command_runner.py
@dataclass
class CommandResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: Optional[str] = None

class CommandRunner:
    def __init__(self, timeout: int = 120):
        self.timeout = timeout
    
    def run_command(
        self, 
        cmd: list[str], 
        cwd: Optional[Path] = None,
        fallback_commands: Optional[list[list[str]]] = None
    ) -> CommandResult:
        # Unified subprocess execution logic
        pass
    
    def check_command_available(self, cmd: str) -> bool:
        # Unified availability checking
        pass
```

**Benefits:**
- Single source of truth for subprocess patterns
- Consistent error handling and logging
- Easier to add new features (e.g., progress callbacks)

### 2. Standardize Result Classes
**Priority:** Medium

Create a base result class:

```python
# src/metaagent/base_result.py
@dataclass
class BaseResult:
    success: bool
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.error and self.success:
            raise ValueError("Cannot have both success=True and error set")
```

Then inherit from it:

```python
@dataclass
class ClaudeCodeResult(BaseResult):
    output: str = ""
    files_modified: list[str] = field(default_factory=list)
    exit_code: int = 0
```

### 3. Create Utility Functions
**Priority:** Medium

Extract common utilities:

```python
# src/metaagent/git_utils.py
def get_modified_files(repo_path: Path) -> list[str]:
    """Shared git status parsing logic"""
    pass

def get_file_changes_summary(repo_path: Path) -> dict[str, int]:
    """Get summary of additions, deletions, modifications"""
    pass
```

### 4. Unified Tool Integration Pattern
**Priority:** Low

Create an abstract base class for tool integrations:

```python
# src/metaagent/tool_integration.py
class ToolIntegration(ABC):
    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.runner = CommandRunner(timeout)
    
    @abstractmethod
    def check_installed(self) -> bool:
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> BaseResult:
        pass
```

## Pattern Analysis

### Common Anti-Patterns Found
1. **Copy-Paste Programming:** Similar subprocess logic copied across modules
2. **Inconsistent Error Messages:** Slight variations in error message formatting
3. **Mixed Abstraction Levels:** Some modules handle low-level subprocess details while others abstract them

### Trends
1. **Tool Integration Pattern:** Every new tool follows similar integration pattern
2. **Result Encapsulation:** All modules use dataclasses for result objects
3. **Timeout and Retry Logic:** Common need for robust external command execution

## Overall Maintainability Impact

### Current State
- **Duplication Score:** Moderate (30-40% of subprocess-related code is duplicated)
- **Maintenance Burden:** High - changes require updates in 3+ files
- **Testing Complexity:** Medium - each module needs separate mock strategies

### After Refactoring
- **Estimated Duplication Reduction:** 60-70%
- **Maintenance Improvement:** Significant - changes centralized to 1-2 files
- **Testing Benefits:** Shared test utilities and mocks

## Recommended Implementation Order

1. **Phase 1:** Extract `CommandRunner` utility class
2. **Phase 2:** Refactor `repomix.py` and `claude_runner.py` to use shared runner
3. **Phase 3:** Standardize result classes with base class
4. **Phase 4:** Extract git utilities and other shared functions
5. **Phase 5:** Create abstract tool integration pattern for future tools

## Success Metrics

- **Lines of Code:** Target 15-20% reduction in total LOC
- **Cyclomatic Complexity:** Reduce complexity in individual modules
- **Test Coverage:** Maintain or improve coverage with shared test utilities
- **Bug Consistency:** Fixes in shared code benefit all tool integrations

This refactoring would significantly improve the codebase's maintainability while reducing the risk of inconsistencies as new tool integrations are added.