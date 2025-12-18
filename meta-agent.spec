# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for meta-agent CLI.

Build the standalone executable with:
    pyinstaller meta-agent.spec

This creates a single-file executable at dist/meta-agent(.exe)
"""

import sys
from pathlib import Path

# Determine platform
IS_WINDOWS = sys.platform.startswith('win')
IS_MAC = sys.platform == 'darwin'

# Get project root
PROJECT_ROOT = Path(SPECPATH)

# Data files to include
datas = [
    # Config directory (prompts, profiles, prompt_library)
    (str(PROJECT_ROOT / 'config'), 'config'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'metaagent',
    'metaagent.cli',
    'metaagent.orchestrator',
    'metaagent.analysis',
    'metaagent.config',
    'metaagent.license',
    'metaagent.task_manager',
    'metaagent.local_manager',
    'metaagent.grok_client',
    'metaagent.ollama_engine',
    'metaagent.claude_impl',
    'metaagent.claude_runner',
    'metaagent.plan_writer',
    'metaagent.prompts',
    'metaagent.repomix',
    'metaagent.repomix_parser',
    'metaagent.workspace_manager',
    'metaagent.tokens',
    # Dependencies
    'typer',
    'rich',
    'rich.console',
    'rich.logging',
    'rich.table',
    'rich.panel',
    'rich.markdown',
    'yaml',
    'httpx',
    'anthropic',
    'git',
    'dotenv',
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
    'regex',
]

# Excluded modules to reduce size
excludes = [
    'tkinter',
    'matplotlib',
    'numpy',
    'pandas',
    'scipy',
    'PIL',
    'cv2',
    'tensorflow',
    'torch',
    'pytest',
    'setuptools',
    'wheel',
]

a = Analysis(
    [str(PROJECT_ROOT / 'src' / 'metaagent' / 'cli.py')],
    pathex=[str(PROJECT_ROOT / 'src')],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='meta-agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # CLI tool needs console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if you have one: 'assets/icon.ico'
)
