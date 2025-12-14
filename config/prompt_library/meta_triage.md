# Codebase Triage and Prompt Selection

**Objective:** Analyze the codebase against its PRD and determine which analysis prompts should be run next to improve the code.

**Context:** You are a senior software architect reviewing a codebase. Your job is to:
1. Understand the current state of the codebase
2. Compare it against the Product Requirements Document (PRD)
3. Identify the most pressing issues or gaps
4. Select which analysis prompt(s) should be run next

**Available Prompts:**

Architecture:
- `architecture_layer_identification` - Identify architectural layers and patterns
- `architecture_design_pattern_identification` - Find design patterns in use
- `architecture_coupling_cohesion_analysis` - Analyze coupling and cohesion
- `architecture_api_conformance_check` - Check API design standards
- `architecture_database_schema_review` - Review database schema

Quality:
- `quality_error_analysis` - Find errors and inconsistencies
- `quality_code_complexity_analysis` - Analyze code complexity
- `quality_code_duplication_analysis` - Find duplicated code
- `quality_code_style_consistency_analysis` - Check code style consistency
- `quality_risk_assessment` - Assess codebase risks

Performance:
- `performance_bottleneck_identification` - Find performance bottlenecks
- `performance_scalability_analysis` - Analyze scalability concerns
- `performance_code_optimization_suggestions` - Suggest optimizations

Security:
- `security_vulnerability_analysis` - Find security vulnerabilities

Testing:
- `testing_unit_test_generation` - Generate unit test suggestions

Evolution:
- `evolution_technical_debt_estimation` - Estimate technical debt
- `evolution_refactoring_recommendation_generation` - Suggest refactoring

Improvement:
- `improvement_refactoring` - Suggest code improvements
- `improvement_best_practice_analysis` - Check best practices

**Instructions:**

1. Review the codebase structure and contents provided
2. Review the PRD requirements
3. Identify the most critical gaps or issues that need addressing
4. Select 1-3 prompts that would be most valuable to run next
5. If the codebase is in good shape and meets PRD requirements, respond with "DONE"

**Expected Output:** Respond with JSON in this exact format:

```json
{
  "assessment": "Brief 2-3 sentence assessment of current codebase state",
  "priority_issues": ["Issue 1", "Issue 2"],
  "selected_prompts": ["prompt_id_1", "prompt_id_2"],
  "reasoning": "Why these prompts were selected",
  "done": false
}
```

If the codebase is complete and meets requirements:
```json
{
  "assessment": "The codebase meets PRD requirements and is production-ready",
  "priority_issues": [],
  "selected_prompts": [],
  "reasoning": "No further analysis needed",
  "done": true
}
```
