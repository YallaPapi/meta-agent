# Meta-Agent Windows Installer (PowerShell)
# Installs meta-agent with all dependencies
# Usage: Right-click > Run with PowerShell, or: powershell -ExecutionPolicy Bypass -File install.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "             Meta-Agent Installer for Windows" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.10+ from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Python version
$versionCheck = python -c "import sys; print('ok' if sys.version_info >= (3, 10) else 'old')" 2>&1
if ($versionCheck -ne "ok") {
    Write-Host "ERROR: Python 3.10 or higher is required." -ForegroundColor Red
    Write-Host "Please upgrade Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Installing meta-agent..." -ForegroundColor Yellow
Write-Host ""

# Install
if (Test-Path "pyproject.toml") {
    # In development mode
    pip install -e ".[dev]"
} else {
    # Standard install
    pip install metaagent
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Installation failed!" -ForegroundColor Red
    Write-Host "Try running as Administrator or check your internet connection." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "               Installation Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now use meta-agent:" -ForegroundColor White
Write-Host ""
Write-Host "  metaagent --help              " -NoNewline -ForegroundColor Cyan
Write-Host "Show all commands"
Write-Host "  metaagent --gui               " -NoNewline -ForegroundColor Cyan
Write-Host "Launch graphical interface"
Write-Host "  metaagent loop --prd FILE     " -NoNewline -ForegroundColor Cyan
Write-Host "Run autonomous task loop"
Write-Host ""
Write-Host "For more info, see the README.md or visit:" -ForegroundColor White
Write-Host "  https://github.com/yourrepo/meta-agent" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"
