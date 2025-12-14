# Coupling and Cohesion Analysis

# Coupling and Cohesion Analysis

## Executive Summary

The meta-agent codebase exhibits **moderate to high coupling** with several concerning tight dependencies, particularly around subprocess spawning and external tool orchestration. The **core problem identified** - spawning separate Claude Code processes instead of returning analysis to the current session - is a direct consequence of architectural decisions that prioritize external tool integration over internal workflow coordination.

## 1. Module Dependencies Analysis

### 1.1 Dependency Graph

```
cli.py
├── orchestrator.py (HIGH COUPLING)
│   ├── analysis.py (MEDIUM)
│   ├── repomix.py (HIGH - subprocess)
│   ├── claude_runner.py (HIGH - subprocess)
│   ├── plan_writer.py (MEDIUM)
│   └── config.py (LOW)
├── config.py (LOW)
└── prompts.py (imported via orchestrator)

claude_runner.py (ISOLATED - subprocess only)
```

### 1.2 Coupling Severity Assessment

**HIGH COUPLING:**
- `orchestrator.py` → `claude_runner.py` (subprocess spawning)
- `orchestrator.py` → `repomix.py` (subprocess spawning)
- `cli.py` → `orchestrator.py` (direct instantiation and execution)

**MEDIUM COUPLING:**
- `orchestrator.py` → `analysis.py` (interface-based)
- `orchestrator.py` → `plan_writer.py` (data passing)

**LOW COUPLING:**
- All modules → `config.py` (configuration injection)

## 2. Coupling Analysis

### 2.1 Critical Coupling Issues

#### **Issue #1: Subprocess Spawning Anti-Pattern**
```python
# orchestrator.py lines (inferred from design)
def run_implementation(self, plan_path: Path):
    runner = ClaudeCodeRunner(
        timeout=self.config.claude_code_timeout,
        model=self.config.claude_code_model
    )
    
    # PROBLEM: This spawns a separate subprocess
    result = runner.implement(
        repo_path=self.config.repo_path,
        prompt=implementation_prompt,
        plan_file=plan_path
    )
```

**Coupling Type:** Tight temporal coupling + process boundary coupling
**Impact:** Analysis results are lost to the current Claude session

#### **Issue #2: Hard-coded External Dependencies**
```python
# claude_runner.py lines 99-112
result = subprocess.run(
    [
        "claude",
        "--print",
        "--model", self.model,
        "--max-turns", str(self.max_turns),
        "--dangerously-skip-permissions",
        "-p", full_prompt,
    ],
    cwd=repo_path,
    capture_output=True,
    text=True,
    timeout=self.timeout,
)
```

**Coupling Type:** Infrastructure coupling - tightly bound to Claude CLI installation
**Impact:** Cannot adapt to different execution contexts

#### **Issue #3: Orchestrator God Object**
The `orchestrator.py` module (not shown but inferred) likely contains:
- Configuration management logic
- External tool coordination
- Analysis workflow orchestration  
- Plan generation coordination

**Coupling Type:** Functional coupling across multiple concerns

### 2.2 Data Coupling Examples

**Positive Example - Low Coupling:**
```python
# plan_writer.py lines 67-76
def write_plan(
    self,
    prd_content: str,
    profile_name: str, 
    stage_results: list[StageResult],
    output_filename: str = "mvp_improvement_plan.md",
) -> Path:
```

**Analysis:** Clean data coupling - function receives only required data and returns a clear result.

## 3. Cohesion Analysis

### 3.1 High Cohesion Modules ✅

#### **analysis.py**
- **Cohesion Type:** Functional cohesion
- **Responsibilities:** 
  - LLM API integration
  - Response parsing and validation
  - Error handling and retry logic
- **Evidence:** All functions serve the single purpose of analysis execution

#### **plan_writer.py** 
- **Cohesion Type:** Functional cohesion
- **Responsibilities:**
  - Plan document generation
  - Task aggregation and formatting
  - Markdown output generation
- **Evidence:** Lines 126-162 show focused task normalization logic

### 3.2 Moderate Cohesion Issues

#### **config.py**
- **Cohesion Type:** Logical cohesion (acceptable)
- **Mixed Responsibilities:**
  - API key management
  - Path configuration  
  - Timeout settings
  - Claude Code specific settings
  - Git/commit settings
  - Retry configuration

```python
# config.py lines 15-35 (partial)
class Config:
    # API Keys
    perplexity_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Runtime Settings  
    log_level: str = "INFO"
    mock_mode: bool = False
    
    # Claude Code Settings
    claude_code_timeout: int = 600
    claude_code_model: str = "claude-sonnet-4-20250514"
    
    # Git/Commit Settings
    auto_commit: bool = True
    auto_push: bool = False
```

**Analysis:** While logically related as "configuration," these span multiple system concerns.

### 3.3 Low Cohesion Modules ⚠️

#### **claude_runner.py**
- **Cohesion Type:** Coincidental cohesion
- **Mixed Responsibilities:**
  - Claude CLI subprocess management
  - Command argument construction  
  - Git status parsing (lines 164-185)
  - Mock implementation for testing
  - File modification tracking

**Evidence of Low Cohesion:**
```python
# lines 164-185 - Git operations mixed with CLI execution
def _get_modified_files(self, repo_path: Path) -> list[str]:
    """Get list of modified files from git status."""
    # ...git subprocess logic...
    
# lines 99-130 - CLI subprocess management  
def implement(self, repo_path: Path, prompt: str, plan_file: Optional[Path] = None):
    # ...claude CLI execution...
```

## 4. Root Cause of Core Problem

### 4.1 Architecture Decision Analysis

The **subprocess spawning issue** stems from an architectural decision to treat Claude Code as a **black-box external tool** rather than integrating with the current execution context.

**Design Pattern in Use:**
```
Meta-Agent → [Process Boundary] → Claude Code CLI → [Results Lost]
```

**Desired Pattern:**
```  
Meta-Agent → Analysis Results → Current Claude Session
```

### 4.2 Contributing Coupling Factors

1. **Infrastructure Coupling:** Hard dependency on `claude` CLI binary
2. **Process Boundary Coupling:** Results cannot cross subprocess boundaries  
3. **Temporal Coupling:** Implementation must complete before results are available
4. **Configuration Coupling:** CLI args embedded in runner logic

## 5. Concrete Improvement Recommendations

### 5.1 Immediate Fix: Result Return Interface

**Problem:** `ClaudeCodeRunner.implement()` returns results that are ignored by orchestrator.

**Solution:** Modify orchestrator to capture and return results instead of spawning:

```python
# Proposed change to orchestrator pattern
class MetaAgentOrchestrator:
    def refine(self, profile: str) -> RefinementResult:
        """Return results instead of spawning processes."""
        analysis_results = self._run_analysis_stages(profile)
        improvement_plan = self._generate_plan(analysis_results)
        
        # RETURN results instead of spawning Claude Code
        return RefinementResult(
            analysis_results=analysis_results,
            improvement_plan=improvement_plan,
            plan_file_path=plan_path
        )
```

### 5.2 Decouple External Tool Dependencies

**Current High Coupling:**
```python
# claude_runner.py - hardcoded CLI integration  
subprocess.run(["claude", "--print", ...])
```

**Proposed Abstraction:**
```python
class CodeImplementationStrategy(ABC):
    @abstractmethod
    def implement(self, plan: ImprovementPlan) -> ImplementationResult:
        pass

class ClaudeCodeStrategy(CodeImplementationStrategy):
    # Existing subprocess logic
    
class CurrentSessionStrategy(CodeImplementationStrategy):
    def implement(self, plan: ImprovementPlan) -> ImplementationResult:
        # Return plan for current session to execute
        return ImplementationResult(
            action="return_to_session",
            plan=plan,
            instructions=self._generate_instructions(plan)
        )
```

### 5.3 Split Orchestrator Responsibilities

**Current God Object (inferred):**
```python
class MetaAgentOrchestrator:
    # Too many responsibilities:
    # - Configuration management  
    # - External tool coordination
    # - Analysis workflow
    # - Plan generation
    # - Implementation execution
```

**Proposed Separation:**
```python
class AnalysisWorkflow:
    """High cohesion - only analysis orchestration"""
    
class PlanGenerator:  
    """High cohesion - only plan generation"""
    
class ToolCoordinator:
    """High cohesion - only external tool management"""

class MetaAgentFacade:
    """Coordinates the above with minimal coupling"""
```

### 5.4 Configuration Cohesion Improvement

**Split large Config class:**
```python
@dataclass
class ApiConfig:
    perplexity_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    timeout: int = 120

@dataclass  
class ExecutionConfig:
    mock_mode: bool = False
    dry_run: bool = False
    auto_implement: bool = False

@dataclass
class PathConfig:
    repo_path: Path
    config_dir: Path  
    prd_path: Optional[Path] = None
```

## 6. Implementation Priority

### Phase 1: Critical (Addresses Core Problem)
1. **Modify orchestrator to return results instead of spawning**
2. **Add `CurrentSessionStrategy` for Claude Code integration**  
3. **Update CLI to output analysis results to current session**

### Phase 2: Architecture Cleanup  
1. **Split orchestrator responsibilities**
2. **Implement `CodeImplementationStrategy` abstraction**
3. **Refactor configuration cohesion**

### Phase 3: Polish
1. **Add dependency injection for loose coupling**
2. **Extract interfaces for all external dependencies**
3. **Implement comprehensive mocking strategies**

## Conclusion

The codebase demonstrates **good functional cohesion** in focused modules like `analysis.py` and `plan_writer.py`, but suffers from **high coupling** around external tool integration and **low cohesion** in utility modules like `claude_runner.py`.

The **core subprocess spawning problem** is a direct result of treating Claude Code as a black-box external dependency rather than designing for integration with the current execution context. The recommended solutions focus on **returning analysis results** to the calling session while maintaining the existing external tool capabilities through strategy patterns.