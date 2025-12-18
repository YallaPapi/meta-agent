#!/usr/bin/env python3
"""Build script for meta-agent standalone executable.

Usage:
    python scripts/build.py          # Build standalone exe
    python scripts/build.py --clean  # Clean and rebuild
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build meta-agent executable")
    parser.add_argument("--clean", action="store_true", help="Clean build dirs first")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    spec_file = project_root / "meta-agent.spec"
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"

    # Check spec file exists
    if not spec_file.exists():
        print(f"Error: Spec file not found: {spec_file}")
        sys.exit(1)

    # Clean if requested
    if args.clean:
        print("Cleaning build directories...")
        for dir_path in [dist_dir, build_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  Removed: {dir_path}")

    # Check PyInstaller is installed
    try:
        subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        print("Error: PyInstaller not installed. Install with:")
        print("  pip install pyinstaller")
        sys.exit(1)

    # Build
    print(f"Building meta-agent from: {spec_file}")
    print("This may take a few minutes...")

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec_file)],
        cwd=project_root,
    )

    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)

    # Check output
    exe_name = "meta-agent.exe" if sys.platform.startswith("win") else "meta-agent"
    exe_path = dist_dir / exe_name

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\nBuild successful!")
        print(f"Executable: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"\nTest with:")
        print(f"  {exe_path} --help")
    else:
        print(f"Warning: Expected executable not found at {exe_path}")


if __name__ == "__main__":
    main()
