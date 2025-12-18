"""Ollama integration for local LLM-based triage."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default Ollama endpoint
OLLAMA_BASE_URL = "http://localhost:11434"

# Recommended models for triage (balance of speed and quality)
RECOMMENDED_MODELS = [
    "qwen2.5:7b",    # Best balance of speed and instruction-following (4.7GB)
    "qwen2.5:14b",   # Better quality if you have VRAM (9GB)
    "llama3.1:8b",   # Good alternative (4.7GB)
]


@dataclass
class OllamaResult:
    """Result from an Ollama analysis."""

    success: bool
    content: str = ""
    error: Optional[str] = None
    model: Optional[str] = None
    tokens_used: int = 0


@dataclass
class TriageOutput:
    """Structured output from Ollama triage."""

    success: bool
    assessment: str = ""
    selected_prompts: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    def get_all_relevant_files(self) -> list[str]:
        """Get all unique relevant files across all selected prompts."""
        files = set()
        for prompt in self.selected_prompts:
            for f in prompt.get("relevant_files", []):
                files.add(f)
        return sorted(files)


class OllamaEngine:
    """Engine for running local LLM analysis via Ollama."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = 300,
    ):
        """Initialize the Ollama engine.

        Args:
            model: The Ollama model to use.
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        if self._available is not None:
            return self._available

        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                # Check if our model (or base name) is available
                model_base = self.model.split(":")[0]
                self._available = any(
                    m.startswith(model_base) for m in models
                )
                if not self._available:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Available: {models}. Run: ollama pull {self.model}"
                    )
            else:
                self._available = False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False

        return self._available

    def generate(self, prompt: str) -> OllamaResult:
        """Generate a response from Ollama.

        Args:
            prompt: The prompt to send.

        Returns:
            OllamaResult with the response or error.
        """
        if not self.is_available():
            return OllamaResult(
                success=False,
                error="Ollama not available. Install from https://ollama.com and run: ollama pull " + self.model,
            )

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for consistent triage
                        "num_predict": 4096,  # Enough for JSON output
                    },
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return OllamaResult(
                    success=False,
                    error=f"Ollama returned status {response.status_code}: {response.text}",
                )

            data = response.json()
            return OllamaResult(
                success=True,
                content=data.get("response", ""),
                model=self.model,
                tokens_used=data.get("eval_count", 0),
            )

        except httpx.TimeoutException:
            return OllamaResult(
                success=False,
                error=f"Ollama request timed out after {self.timeout}s",
            )
        except Exception as e:
            return OllamaResult(
                success=False,
                error=f"Ollama request failed: {e}",
            )

    def select_files(
        self,
        feature_request: str,
        code_context: str,
        prd_content: str = "",
    ) -> list[str]:
        """Select relevant files for a feature request.

        This is a lightweight operation - just file selection, no prompt selection.
        Used in feature-focused mode where Perplexity does the heavy lifting.

        Args:
            feature_request: The feature to implement or bug to fix.
            code_context: The full packed codebase from Repomix.
            prd_content: Optional PRD content for context.

        Returns:
            List of relevant file paths.
        """
        prompt = f"""# File Selection for Feature Implementation

You are a software architect. Given a feature request and a codebase,
identify which files are most relevant to implementing this feature.

## Feature Request

**{feature_request}**

## Codebase

{code_context}

## Your Task

List the files that would need to be modified or referenced to implement this feature.
Focus on the most relevant 3-8 files.

## Output Format

Respond with a JSON array of file paths ONLY. No other text.

Example: ["file1.py", "file2.py", "config.json"]

JSON array:"""

        result = self.generate(prompt)

        if not result.success:
            logger.warning(f"Ollama file selection failed: {result.error}")
            return []

        # Parse the JSON array
        try:
            # Look for JSON array in response
            content = result.content.strip()
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                files = json.loads(json_match.group(0))
                if isinstance(files, list):
                    return [f for f in files if isinstance(f, str)]
            return []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse file list: {result.content[:200]}")
            return []

    def triage(
        self,
        prd_content: str,
        code_context: str,
        prompt_index: str,
        focus: Optional[str] = None,
    ) -> TriageOutput:
        """Run triage to select prompts and relevant files.

        Args:
            prd_content: The PRD content.
            code_context: The full packed codebase from Repomix.
            prompt_index: Formatted list of available prompts.
            focus: Optional custom focus to steer prompt selection
                (e.g., "electron frontend with gamified UX").

        Returns:
            TriageOutput with selected prompts and files.
        """
        triage_prompt = self._build_triage_prompt(
            prd_content, code_context, prompt_index, focus
        )

        result = self.generate(triage_prompt)

        if not result.success:
            return TriageOutput(success=False, error=result.error)

        return self._parse_triage_output(result.content)

    def _build_triage_prompt(
        self,
        prd_content: str,
        code_context: str,
        prompt_index: str,
        focus: Optional[str] = None,
    ) -> str:
        """Build the triage prompt for Ollama."""
        focus_section = ""
        if focus:
            focus_section = f"""
## Focus Area

When selecting prompts and analyzing the codebase, prioritize improvements related to:

**{focus}**

Select prompts that will help achieve this vision. Focus on files and code areas
that are most relevant to this goal.

"""

        return f"""# Codebase Analysis and Triage
{focus_section}

You are a senior software architect. Analyze this codebase against its PRD and determine:
1. Which analysis prompts are most relevant for improving this codebase
2. Which files are relevant to each selected prompt

## Product Requirements Document (PRD)

{prd_content}

## Full Codebase

{code_context}

## Available Analysis Prompts

Select the prompts that would be most valuable based on issues you find:

{prompt_index}

## Your Task

1. Compare the codebase against the PRD
2. Identify the most pressing issues or gaps
3. Select 2-5 analysis prompts from the list above that would help address these issues
4. For each prompt, list the ACTUAL files from the codebase that are relevant

IMPORTANT:
- Use REAL prompt IDs from the "Available Analysis Prompts" list above (e.g., "quality_error_analysis", "security_vulnerability_analysis")
- Use REAL file paths from the codebase (e.g., "spoof_videos.py", "create_va_chunks.py")
- Do NOT use placeholder values like "path/to/file1.py"

## Output Format

Respond with valid JSON. Fill in ALL fields with your analysis:

```json
{{
  "assessment": "[Write 2-3 sentences about the codebase state here]",
  "selected_prompts": [
    {{
      "prompt_id": "[Pick a real prompt_id from the list above]",
      "reasoning": "[Explain why you picked this prompt]",
      "relevant_files": ["[real_file.py from codebase]"]
    }}
  ]
}}
```

Remember: Replace everything in [brackets] with real values. Do not copy the brackets.

JSON response:"""

    def expand_feature(
        self,
        feature_request: str,
        code_context: str,
        prd_content: str,
    ) -> FeatureExpansion:
        """Expand a rough feature request into a detailed specification.

        Takes a user's rough feature idea and analyzes the codebase to produce:
        - Detailed feature requirements
        - Suggested additional functionality
        - Relevant files that need modification
        - Integration points in the code

        Args:
            feature_request: User's rough feature description
                (e.g., "add analytics tracking").
            code_context: The full packed codebase from Repomix.
            prd_content: The PRD content for context.

        Returns:
            FeatureExpansion with detailed spec and implementation guidance.
        """
        prompt = self._build_feature_expansion_prompt(
            feature_request, code_context, prd_content
        )

        result = self.generate(prompt)

        if not result.success:
            return FeatureExpansion(success=False, error=result.error)

        return self._parse_feature_expansion(result.content, feature_request)

    def _build_feature_expansion_prompt(
        self,
        feature_request: str,
        code_context: str,
        prd_content: str,
    ) -> str:
        """Build the feature expansion prompt for Ollama."""
        return f"""# Feature Design and Expansion

You are a senior software architect. A developer has requested a new feature.
Your job is to analyze the codebase and expand their rough idea into a
detailed feature specification.

## The Feature Request

**"{feature_request}"**

## Current Product Requirements (PRD)

{prd_content}

## Current Codebase

{code_context}

## Your Task

1. Understand what the developer wants to achieve
2. Analyze the codebase to understand the current architecture
3. Expand the feature request into detailed requirements
4. Identify what ADDITIONAL functionality would be valuable (things they didn't ask for but should have)
5. Find the specific files and code locations that need modification
6. Note any integration points or dependencies

## Output Format

Respond with valid JSON:

```json
{{
  "feature_name": "[Clear name for this feature]",
  "description": "[2-3 sentence description of the full feature]",
  "metrics_to_track": [
    "[Specific metric 1 the user asked for]",
    "[Specific metric 2 the user asked for]",
    "[etc...]"
  ],
  "suggested_additions": [
    "[Additional metric/feature 1 they didn't ask for but should track]",
    "[Additional metric/feature 2]",
    "[Additional metric/feature 3]",
    "[etc... aim for 5-10 suggestions based on the codebase]"
  ],
  "relevant_files": [
    "[file1.py that needs modification]",
    "[file2.py that needs modification]"
  ],
  "integration_points": [
    {{
      "file": "[filename.py]",
      "location": "[function or class name]",
      "description": "[What needs to happen here]"
    }}
  ],
  "implementation_notes": "[Any important considerations, gotchas, or architectural decisions]"
}}
```

Remember:
- Use REAL file names from the codebase above
- Be specific about what to track and where
- Suggest things the developer didn't think of

JSON response:"""

    def _parse_feature_expansion(
        self, content: str, original_request: str
    ) -> FeatureExpansion:
        """Parse the feature expansion response from Ollama."""
        try:
            # Look for JSON in code blocks
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return FeatureExpansion(
                        success=False,
                        error=f"No JSON found in response: {content[:500]}",
                    )

            data = json.loads(json_str)

            return FeatureExpansion(
                success=True,
                feature_name=data.get("feature_name", original_request),
                description=data.get("description", ""),
                metrics_to_track=data.get("metrics_to_track", []),
                suggested_additions=data.get("suggested_additions", []),
                relevant_files=data.get("relevant_files", []),
                integration_points=data.get("integration_points", []),
                implementation_notes=data.get("implementation_notes", ""),
            )

        except json.JSONDecodeError as e:
            return FeatureExpansion(
                success=False,
                error=f"Failed to parse JSON: {e}. Response: {content[:500]}",
            )

    def _parse_triage_output(self, content: str) -> TriageOutput:
        """Parse the triage response from Ollama."""
        # Try to extract JSON from the response
        try:
            # Look for JSON in code blocks
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return TriageOutput(
                        success=False,
                        error=f"No JSON found in response: {content[:500]}",
                    )

            data = json.loads(json_str)

            return TriageOutput(
                success=True,
                assessment=data.get("assessment", ""),
                selected_prompts=data.get("selected_prompts", []),
            )

        except json.JSONDecodeError as e:
            return TriageOutput(
                success=False,
                error=f"Failed to parse JSON: {e}. Response: {content[:500]}",
            )


@dataclass
class FeatureExpansion:
    """Expanded feature specification from Ollama."""

    success: bool
    feature_name: str = ""
    description: str = ""
    metrics_to_track: list[str] = field(default_factory=list)
    suggested_additions: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    integration_points: list[dict] = field(default_factory=list)
    implementation_notes: str = ""
    error: Optional[str] = None


def check_ollama_status() -> dict:
    """Check Ollama installation status and available models.

    Returns:
        Dict with status information.
    """
    result = {
        "installed": False,
        "running": False,
        "models": [],
        "recommended_model": None,
        "install_instructions": "Install from https://ollama.com",
    }

    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            result["installed"] = True
            result["running"] = True
            data = response.json()
            result["models"] = [m["name"] for m in data.get("models", [])]

            # Check for recommended models
            for model in RECOMMENDED_MODELS:
                model_base = model.split(":")[0]
                if any(m.startswith(model_base) for m in result["models"]):
                    result["recommended_model"] = model
                    break

            if not result["recommended_model"] and result["models"]:
                result["recommended_model"] = result["models"][0]

    except httpx.ConnectError:
        result["install_instructions"] = (
            "Ollama installed but not running. Start with: ollama serve"
        )
        result["installed"] = True  # Assume installed if we get connection refused
    except Exception:
        pass

    return result
