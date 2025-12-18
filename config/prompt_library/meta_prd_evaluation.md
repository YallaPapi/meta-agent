# PRD Alignment Evaluation

**Objective:** Evaluate whether the current codebase fully implements the requirements from the PRD and identify any remaining gaps.

**Context:**
You are evaluating a codebase after an autonomous development loop has attempted to implement features described in a PRD.

**Instructions:**

1. **Compare the codebase to the PRD requirements:**
   - Identify which requirements are fully implemented
   - Identify which requirements are partially implemented
   - Identify which requirements are missing entirely

2. **Evaluate implementation quality:**
   - Does the implementation match the PRD intent?
   - Are there any edge cases not handled?
   - Is the code production-ready or MVP-quality?
   - Are there any obvious bugs or issues?

3. **Check integration completeness:**
   - Do all components work together correctly?
   - Are there any broken connections between modules?
   - Is error handling comprehensive?

4. **Generate final assessment:**
   - Overall completion percentage
   - Remaining critical items
   - Recommended next steps

**Input Format:**
```
## Product Requirements Document
{prd_content}

## Current Codebase
{repomix_xml}

## Implementation History (if available)
{implementation_summary}
```

**Expected Output:**

Respond with a JSON object in this exact format:
```json
{
  "evaluation": {
    "completion_percentage": 85,
    "prd_aligned": true,
    "production_ready": false,
    "mvp_ready": true
  },
  "requirements_status": [
    {
      "requirement": "Requirement description",
      "status": "complete|partial|missing",
      "notes": "Any relevant notes"
    }
  ],
  "remaining_tasks": [
    {
      "task": "Description of remaining task",
      "priority": "critical|high|medium|low",
      "estimated_complexity": "low|medium|high"
    }
  ],
  "overall_assessment": "Summary of the current state and recommendations",
  "approved": true,
  "approval_reason": "Reason for approval/rejection"
}
```

**Approval Criteria:**
- approved=true if:
  - All critical requirements are implemented
  - Tests pass
  - No major bugs or security issues
  - Code is functional for its intended purpose

- approved=false if:
  - Critical requirements are missing
  - Major bugs prevent core functionality
  - Security vulnerabilities exist
