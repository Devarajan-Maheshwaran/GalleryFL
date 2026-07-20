@echo off
setlocal
cd /d "%~dp0"
title GalleryFL Central Aggregator

echo =========================================
echo   GalleryFL Central Aggregator
echo =========================================
echo.

set "PYTHON_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 --version >nul 2>nul
    if %errorlevel%==0 set "PYTHON_CMD=py -3.11"
)
if not defined PYTHON_CMD (
    where python >nul 2>nul
    if %errorlevel%==0 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo ERROR: Python 3.11 or 3.12 is required.
    echo Download Python from https://www.python.org/downloads/
    echo During installation, select "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [1/3] Creating the private Python environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 goto :failed
) else (
    echo [1/3] Existing Python environment found.
)

call "venv\Scripts\activate.bat"

echo [2/3] Installing or checking server dependencies...
python -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :failed

echo [3/3] Starting GalleryFL...
echo.
echo Dashboard: http://localhost:8000/dashboard/
echo Keep this window open while GalleryFL is running.
echo Press Ctrl+C to stop the server.
echo.
start "" "http://localhost:8000/dashboard/"
python run_server.py
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo GalleryFL could not start. Review the error shown above.
pause
exit /b 1
