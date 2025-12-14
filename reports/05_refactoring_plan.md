# Refactoring Plan

Looking at the codebase, I can see the core problem clearly: **`claude_runner.py` spawns a separate Claude Code subprocess via `subprocess.run()` instead of returning analysis reports to the current session.**

Let me provide a comprehensive refactoring plan to address this issue:

## Analysis of Current Implementation

The current `claude_runner.py` has these problematic patterns:

1. **Subprocess spawning**: Uses `subprocess.run(["claude", ...])` to launch a separate Claude Code session
2. **Black box execution**: No structured output or progress tracking from the subprocess
3. **Lost context**: The spawned subprocess loses all context from the current analysis session
4. **Poor integration**: Results are only tracked via git file changes, not structured reports

## Refactoring Strategy

### 1. Transform ClaudeCodeRunner.implement() to Return Structured Reports

**Current problematic method:**
```python
def implement(self, repo_path: Path, prompt: str, plan_file: Optional[Path] = None) -> ClaudeCodeResult:
    # Currently spawns subprocess - THIS IS THE PROBLEM
    result = subprocess.run(["claude", "--print", ...])
```

**Refactored approach:**
```python
def implement(self, repo_path: Path, prompt: str, plan_file: Optional[Path] = None) -> ImplementationReport:
    """Analyze implementation requirements and return structured report.
    
    NO LONGER spawns subprocess - returns analysis for current session.
    """
    # Parse the prompt and plan to extract tasks
    tasks = self._extract_tasks_from_plan(plan_file)
    
    # Analyze each task and create implementation recommendations
    recommendations = []
    for task in tasks:
        rec = self._analyze_task_requirements(task, repo_path)
        recommendations.append(rec)
    
    return ImplementationReport(
        tasks=tasks,
        recommendations=recommendations,
        implementation_plan=self._create_implementation_plan(recommendations),
        estimated_changes=self._estimate_file_changes(recommendations),
        success=True
    )
```

### 2. New Data Structures for Structured Results

```python
@dataclass
class TaskAnalysis:
    """Analysis of a single implementation task."""
    task_id: str
    title: str
    description: str
    priority: str
    estimated_complexity: str  # low, medium, high
    affected_files: list[str]
    implementation_steps: list[str]
    dependencies: list[str]
    risks: list[str]

@dataclass
class ImplementationRecommendation:
    """Recommendation for implementing a specific change."""
    target_file: Path
    change_type: str  # create, modify, delete
    description: str
    code_snippet: Optional[str] = None
    rationale: str = ""
    prerequisites: list[str] = field(default_factory=list)

@dataclass
class ImplementationReport:
    """Structured report from implementation analysis."""
    tasks: list[TaskAnalysis]
    recommendations: list[ImplementationRecommendation]
    implementation_plan: str  # Markdown formatted plan
    estimated_changes: dict[str, int]  # file -> estimated lines changed
    success: bool
    error: Optional[str] = None
    total_estimated_effort: str = ""  # e.g., "2-4 hours"
```

### 3. Updated ClaudeCodeRunner Implementation

```python
class ClaudeCodeRunner:
    """Analyzes implementation requirements and generates structured reports.
    
    NO LONGER spawns subprocesses - works within current Claude session.
    """
    
    def __init__(self, analysis_engine: Optional[AnalysisEngine] = None):
        """Initialize with optional analysis engine for task breakdown."""
        self.analysis_engine = analysis_engine
        
    def implement(
        self,
        repo_path: Path,
        prompt: str,
        plan_file: Optional[Path] = None,
    ) -> ImplementationReport:
        """Analyze implementation requirements and return structured report."""
        try:
            # Extract tasks from the improvement plan
            tasks = self._extract_tasks_from_plan(plan_file) if plan_file else []
            
            # If no plan file, extract tasks from prompt
            if not tasks:
                tasks = self._extract_tasks_from_prompt(prompt)
            
            # Analyze each task for implementation requirements
            recommendations = []
            for task in tasks:
                recs = self._analyze_task_implementation(task, repo_path)
                recommendations.extend(recs)
            
            # Create implementation plan
            plan = self._create_implementation_plan(tasks, recommendations)
            
            # Estimate effort and changes
            changes = self._estimate_file_changes(recommendations)
            effort = self._estimate_total_effort(tasks)
            
            return ImplementationReport(
                tasks=tasks,
                recommendations=recommendations,
                implementation_plan=plan,
                estimated_changes=changes,
                total_estimated_effort=effort,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Implementation analysis failed: {e}")
            return ImplementationReport(
                tasks=[],
                recommendations=[],
                implementation_plan="",
                estimated_changes={},
                success=False,
                error=str(e)
            )
    
    def _extract_tasks_from_plan(self, plan_file: Path) -> list[TaskAnalysis]:
        """Extract and analyze tasks from improvement plan."""
        if not plan_file.exists():
            return []
            
        content = plan_file.read_text()
        tasks = []
        
        # Parse markdown checkboxes and task descriptions
        import re
        task_pattern = re.compile(r'- \[ \] \*\*(.+?)\*\*(?:\s*\(`(.+?)`\))?\s*\n(?:\s*- (.+))?')
        
        for match in task_pattern.finditer(content):
            title = match.group(1).strip()
            file_ref = match.group(2) or ""
            description = match.group(3) or ""
            
            # Analyze task complexity and requirements
            task = TaskAnalysis(
                task_id=f"task-{len(tasks)+1}",
                title=title,
                description=description,
                priority=self._extract_priority_from_context(match.string, match.start()),
                estimated_complexity=self._estimate_task_complexity(title, description),
                affected_files=[file_ref] if file_ref else [],
                implementation_steps=self._generate_implementation_steps(title, description),
                dependencies=[],
                risks=self._identify_task_risks(title, description)
            )
            tasks.append(task)
            
        return tasks
    
    def _analyze_task_implementation(self, task: TaskAnalysis, repo_path: Path) -> list[ImplementationRecommendation]:
        """Analyze how to implement a specific task."""
        recommendations = []
        
        # Analyze existing code if files are specified
        for file_path in task.affected_files:
            full_path = repo_path / file_path
            if full_path.exists():
                # Analyze existing file for modification
                rec = self._analyze_file_modification(full_path, task)
                if rec:
                    recommendations.append(rec)
            else:
                # New file creation
                rec = self._analyze_file_creation(full_path, task)
                if rec:
                    recommendations.append(rec)
        
        # If no specific files, infer from task description
        if not task.affected_files:
            inferred_recs = self._infer_file_changes(task, repo_path)
            recommendations.extend(inferred_recs)
            
        return recommendations
```

### 4. Updated Orchestrator Integration

```python
class Orchestrator:
    """Updated orchestrator that consumes structured reports."""
    
    def run_implementation_analysis(
        self,
        improvement_plan_path: Path,
        profile: str = "automation_agent"
    ) -> ImplementationReport:
        """Analyze implementation requirements without spawning subprocess."""
        
        # Initialize Claude Code runner (NO subprocess spawning)
        claude_runner = ClaudeCodeRunner(analysis_engine=self.analysis_engine)
        
        # Generate implementation analysis report
        prompt = self._build_implementation_prompt(improvement_plan_path)
        report = claude_runner.implement(
            repo_path=self.config.repo_path,
            prompt=prompt,
            plan_file=improvement_plan_path
        )
        
        return report
    
    def run_full_refinement(self, profile_name: str) -> RefinementResult:
        """Run complete refinement cycle with structured reports."""
        
        # Step 1: Run analysis stages (existing code)
        stage_results = self._run_analysis_stages(profile_name)
        
        # Step 2: Generate improvement plan (existing code)
        plan_path = self._write_improvement_plan(stage_results)
        
        # Step 3: NEW - Generate implementation report
        implementation_report = self.run_implementation_analysis(plan_path, profile_name)
        
        # Step 4: Return structured results for consumption
        return RefinementResult(
            profile_name=profile_name,
            stage_results=stage_results,
            improvement_plan_path=plan_path,
            implementation_report=implementation_report,
            success=True
        )

@dataclass 
class RefinementResult:
    """Complete result from refinement process."""
    profile_name: str
    stage_results: list[StageResult]
    improvement_plan_path: Path
    implementation_report: ImplementationReport
    success: bool
    error: Optional[str] = None
```

### 5. New Usage Patterns

**As a Library (returns structured data):**
```python
from metaagent.orchestrator import Orchestrator

orchestrator = Orchestrator(config)
result = orchestrator.run_full_refinement("automation_agent")

if result.success:
    print(f"Analysis complete: {len(result.implementation_report.tasks)} tasks identified")
    print(f"Estimated effort: {result.implementation_report.total_estimated_effort}")
    
    # Consumer can now process structured results
    for task in result.implementation_report.tasks:
        print(f"Task: {task.title} (Priority: {task.priority})")
        for rec in result.implementation_report.recommendations:
            if rec.target_file in task.affected_files:
                print(f"  Change {rec.target_file}: {rec.description}")
```

**File Output Mode (existing workflow):**
```python
# Write structured reports to files for manual consumption
orchestrator.write_implementation_guide(result, "docs/implementation_guide.md")
orchestrator.write_task_breakdown(result, "docs/task_breakdown.md")
```

**Taskmaster Integration:**
```python
# Integration with Taskmaster for task creation
def create_taskmaster_tasks(implementation_report: ImplementationReport):
    """Create Taskmaster tasks from implementation report."""
    for task in implementation_report.tasks:
        subprocess.run([
            "task-master", "add-task", 
            f"--prompt={task.title}: {task.description}",
            f"--priority={task.priority}"
        ])
```

### 6. Benefits of This Refactoring

1. **No subprocess spawning** - Everything runs in current Claude session
2. **Structured reports** - Rich data structures instead of black box execution  
3. **Flexible consumption** - Can be used as library, file output, or Taskmaster integration
4. **Better error handling** - Detailed error reporting and partial success handling
5. **Incremental processing** - Can analyze tasks individually and provide progress
6. **Context preservation** - All analysis context stays within current session

### 7. Migration Path

1. **Phase 1**: Implement new data structures and analysis methods
2. **Phase 2**: Update `ClaudeCodeRunner.implement()` to return reports instead of spawning subprocess
3. **Phase 3**: Update orchestrator to consume structured reports
4. **Phase 4**: Add new output modes (library, files, Taskmaster)
5. **Phase 5**: Remove all subprocess spawning code

This refactoring transforms the meta-agent from a "subprocess orchestrator" into a true "analysis and planning system" that works seamlessly within the current Claude session while providing much richer, more actionable results.