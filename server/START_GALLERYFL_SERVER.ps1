$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  GalleryFL Central Aggregator" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
        & py -3.11 --version *> $null
        if ($LASTEXITCODE -eq 0) { $python = @("py", "-3.11") }
    } catch {}
}
if (-not $python -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $python = @("python")
}
if (-not $python) {
    throw "Python 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/ and enable Add Python to PATH."
}

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "[1/3] Creating the private Python environment..."
    if ($python.Count -eq 2) { & $python[0] $python[1] -m venv venv }
    else { & $python[0] -m venv venv }
} else {
    Write-Host "[1/3] Existing Python environment found."
}

Write-Host "[2/3] Installing or checking server dependencies..."
& .\venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt

Write-Host "[3/3] Starting GalleryFL..."
$env:FGT_PERSIST_CONFIG = "1"
Write-Host "Dashboard: http://localhost:8000/dashboard/" -ForegroundColor Green
Write-Host "Keep this window open. Press Ctrl+C to stop the server."
Start-Process "http://localhost:8000/dashboard/"
& .\venv\Scripts\python.exe .\run_server.py
