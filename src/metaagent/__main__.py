"""Main entry point for running meta-agent as a module.

Usage:
    python -m metaagent --help
    python -m metaagent loop --prd docs/prd.md
    python -m metaagent --gui
"""

from __future__ import annotations

from .cli import app

if __name__ == "__main__":
    app()
