@echo off
REM Meta-Agent Windows Installer
REM Installs meta-agent with all dependencies
REM Usage: Double-click or run from command prompt

echo ============================================================
echo              Meta-Agent Installer for Windows
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check Python version is 3.10+
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10 or higher is required.
    echo Please upgrade Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Installing meta-agent...
echo.

REM Install in editable mode if in repo, otherwise install from current dir
if exist "pyproject.toml" (
    pip install -e ".[dev]"
) else (
    pip install metaagent
)

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo Try running as Administrator or check your internet connection.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo                Installation Complete!
echo ============================================================
echo.
echo You can now use meta-agent:
echo.
echo   metaagent --help              Show all commands
echo   metaagent --gui               Launch graphical interface
echo   metaagent loop --prd FILE     Run autonomous task loop
echo.
echo For more info, see the README.md or visit:
echo   https://github.com/yourrepo/meta-agent
echo.
pause
