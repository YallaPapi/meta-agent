"""Utilities for parsing and extracting content from Repomix output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedFile:
    """A file extracted from Repomix output."""

    path: str
    content: str
    language: Optional[str] = None


def parse_repomix_files(repomix_content: str) -> dict[str, ExtractedFile]:
    """Parse Repomix markdown output into individual files.

    Repomix outputs markdown in this format:

    ## File: path/to/file.py

    ```python
    file contents here
    ```

    Args:
        repomix_content: The full Repomix markdown output.

    Returns:
        Dict mapping file paths to ExtractedFile objects.
    """
    files = {}

    # Pattern to match file sections
    # Handles both "## File: path" and "### File: path" formats
    file_pattern = re.compile(
        r"#{2,3}\s+(?:File:\s*)?([^\n]+?)\s*\n+"  # File header
        r"```(\w*)\n"  # Code block start with optional language
        r"(.*?)"  # File content (non-greedy)
        r"\n```",  # Code block end
        re.DOTALL,
    )

    for match in file_pattern.finditer(repomix_content):
        file_path = match.group(1).strip()
        language = match.group(2) or None
        content = match.group(3)

        # Clean up the file path (remove any markdown formatting)
        file_path = file_path.strip("`").strip()

        files[file_path] = ExtractedFile(
            path=file_path,
            content=content,
            language=language,
        )

    return files


def extract_files(
    repomix_content: str,
    file_paths: list[str],
) -> str:
    """Extract specific files from Repomix output and format for analysis.

    Args:
        repomix_content: The full Repomix markdown output.
        file_paths: List of file paths to extract.

    Returns:
        Formatted markdown string with only the requested files.
    """
    all_files = parse_repomix_files(repomix_content)

    extracted_parts = []
    found_files = []
    missing_files = []

    for file_path in file_paths:
        # Try exact match first
        if file_path in all_files:
            found_files.append(file_path)
            f = all_files[file_path]
            lang = f.language or _guess_language(file_path)
            extracted_parts.append(
                f"## File: {f.path}\n\n```{lang}\n{f.content}\n```"
            )
        else:
            # Try partial match (e.g., "auth/login.py" matches "src/auth/login.py")
            matched = False
            for stored_path, f in all_files.items():
                if stored_path.endswith(file_path) or file_path in stored_path:
                    found_files.append(stored_path)
                    lang = f.language or _guess_language(stored_path)
                    extracted_parts.append(
                        f"## File: {f.path}\n\n```{lang}\n{f.content}\n```"
                    )
                    matched = True
                    break

            if not matched:
                missing_files.append(file_path)

    # Build output
    output_parts = []

    if missing_files:
        output_parts.append(
            f"<!-- Note: Could not find files: {', '.join(missing_files)} -->\n"
        )

    output_parts.append(f"# Relevant Code ({len(found_files)} files)\n")
    output_parts.extend(extracted_parts)

    return "\n\n".join(output_parts)


def get_file_list(repomix_content: str) -> list[str]:
    """Get a list of all file paths in the Repomix output.

    Args:
        repomix_content: The full Repomix markdown output.

    Returns:
        List of file paths.
    """
    files = parse_repomix_files(repomix_content)
    return sorted(files.keys())


def get_file_tree(repomix_content: str) -> str:
    """Generate a simple file tree from Repomix output.

    Args:
        repomix_content: The full Repomix markdown output.

    Returns:
        Formatted file tree string.
    """
    files = sorted(parse_repomix_files(repomix_content).keys())

    if not files:
        return "(no files found)"

    lines = ["```"]
    for f in files:
        # Simple indentation based on path depth
        depth = f.count("/")
        indent = "  " * depth
        name = f.split("/")[-1]
        lines.append(f"{indent}{name}")
    lines.append("```")

    return "\n".join(lines)


def _guess_language(file_path: str) -> str:
    """Guess the language from file extension."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sql": "sql",
        ".md": "markdown",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
    }

    for ext, lang in ext_map.items():
        if file_path.endswith(ext):
            return lang

    return ""
