@echo off
rem Launch Claude Code routed through Ollama Router.
rem Usage:   scripts\claude.cmd [claude args...]
rem Env:     ROUTER_URL   (default: http://localhost:8000)
rem          ROUTER_TOKEN (default: ollama)

setlocal

if not defined ROUTER_URL set "ROUTER_URL=http://localhost:8000"
if not defined ROUTER_TOKEN set "ROUTER_TOKEN=ollama"

where claude >nul 2>&1
if errorlevel 1 (
    echo [error] claude not found in PATH. Install: https://docs.claude.com/en/docs/claude-code 1>&2
    exit /b 127
)

set "ANTHROPIC_BASE_URL=%ROUTER_URL%"
set "ANTHROPIC_AUTH_TOKEN=%ROUTER_TOKEN%"

claude %*
exit /b %ERRORLEVEL%
