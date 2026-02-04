@echo off
REM Quick start script for Claude API Router (Windows)

setlocal enabledelayedexpansion

echo.
echo ================================================
echo Claude Code API Router - Quick Start (Windows)
echo ================================================
echo.

REM Check if .env exists
if not exist ".env" (
    echo WARNING: .env file not found
    echo Creating .env from .env.example...
    copy .env.example .env
    echo.
    echo Please edit .env with your Claude API keys
    echo Opening .env...
    start notepad .env
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed
    exit /b 1
)

echo:
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Install dependencies
echo Installing dependencies...
pip install -q -r requirements.txt
echo Dependencies installed
echo.

echo ================================================
echo Configuration:
echo ================================================

REM Load .env variables (simple approach)
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="SERVER_HOST" set SERVER_HOST=%%b
    if "%%a"=="SERVER_PORT" set SERVER_PORT=%%b
)

if not defined SERVER_HOST set SERVER_HOST=0.0.0.0
if not defined SERVER_PORT set SERVER_PORT=8000

echo Host:      %SERVER_HOST%
echo Port:      %SERVER_PORT%
echo.

echo ================================================
echo Starting server...
echo ================================================
echo.
echo Server will be available at: http://localhost:%SERVER_PORT%
echo.
echo Health Check:    curl http://localhost:%SERVER_PORT%/health
echo Metrics:         curl http://localhost:%SERVER_PORT%/metrics
echo Accounts:        curl http://localhost:%SERVER_PORT%/accounts
echo.
echo Press Ctrl+C to stop
echo ================================================
echo.

python run.py

endlocal
