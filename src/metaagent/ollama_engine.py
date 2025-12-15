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
    "llama3.2:3b",   # Fast, good for triage (2GB)
    "llama3.1:8b",   # Better quality (4.7GB)
    "mistral:7b",    # Good alternative (4.1GB)
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
        model: str = "llama3.2:3b",
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

    def triage(
        self,
        prd_content: str,
        code_context: str,
        prompt_index: str,
    ) -> TriageOutput:
        """Run triage to select prompts and relevant files.

        Args:
            prd_content: The PRD content.
            code_context: The full packed codebase from Repomix.
            prompt_index: Formatted list of available prompts.

        Returns:
            TriageOutput with selected prompts and files.
        """
        triage_prompt = self._build_triage_prompt(
            prd_content, code_context, prompt_index
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
    ) -> str:
        """Build the triage prompt for Ollama."""
        return f"""# Codebase Analysis and Triage

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

Respond with valid JSON only:

```json
{{
  "assessment": "Your actual assessment of this specific codebase",
  "selected_prompts": [
    {{
      "prompt_id": "actual_prompt_id_from_list_above",
      "reasoning": "Your actual reasoning for this codebase",
      "relevant_files": ["actual_file.py", "another_real_file.py"]
    }}
  ]
}}
```

JSON response:"""

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
