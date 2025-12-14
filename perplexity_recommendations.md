# Perplexity Prompt Recommendations

Here are **8 specific prompts** from the library most valuable for analyzing and improving this meta-agent codebase, prioritized for the subprocess spawning issue, code quality problems, and refactoring planning:

1. **architecture_coupling_cohesion_analysis**  
   Directly identifies the core architectural flaw: `ClaudeCodeRunner` spawns separate subprocesses instead of returning reports to the current session, revealing poor coupling between `orchestrator.py`, `analysis.py`, and `claude_runner.py`.

2. **architecture_layer_identification**  
   Maps the current layered structure (CLI → Orchestrator → Analysis → Claude Runner) against PRD architecture, highlighting missing feedback loops and subprocess isolation violating the intended iterative refinement flow.

3. **quality_code_complexity_analysis**  
   Analyzes cyclomatic complexity in `claude_runner.py` subprocess logic and `orchestrator.py` stage coordination, identifying overly complex error handling and timeout patterns that block iterative report flow.

4. **architecture_api_conformance_check**  
   Verifies if `analysis.py` (Perplexity integration) and `claude_runner.py` conform to PRD functional requirements (FR8-FR10, FR14), specifically the requirement for analysis reports to flow back to the orchestrator rather than spawning disconnected Claude sessions.

5. **improvement_refactoring**  
   Generates concrete refactoring plan to restructure `ClaudeCodeRunner.implement()` from subprocess spawning to returning structured `ClaudeCodeResult` objects that feed into the current orchestrator loop.

6. **quality_error_analysis**  
   Examines error handling across `claude_runner.py`, `analysis.py`, and `orchestrator.py`, identifying gaps in subprocess timeout recovery, API error propagation, and missing validation of analysis report parsing.

7. **evolution_impact_analysis_of_code_changes**  
   Assesses downstream impact of refactoring the subprocess model, predicting effects on PRD flows (Flow B stages) and existing CLI integration in `cli.py`.

8. **quality_code_duplication_analysis**  
   Detects duplicated subprocess patterns and result parsing logic across `claude_runner.py`, `repomix.py`, and potentially `analysis.py`, suggesting unified interfaces for all external tool integrations.

These prompts target the **exact architectural mismatch** (separate subprocess vs. current session feedback), **code quality issues** in subprocess/error handling, and **refactoring roadmap** needed to achieve PRD compliance (FR3, FR10, FR14). Run in sequence: architecture → quality → improvement for optimal analysis flow.