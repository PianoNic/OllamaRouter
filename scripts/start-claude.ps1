# Launch Claude Code routed through Ollama Router.
# Usage:   .\scripts\start-claude.ps1 [claude args...]
# Env:     $env:ROUTER_URL   (default: http://localhost:8000)
#          $env:ROUTER_TOKEN (default: ollama)

[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Args)

$ErrorActionPreference = 'Stop'

$routerUrl = if ($env:ROUTER_URL) { $env:ROUTER_URL } else { 'http://localhost:8000' }
$routerToken = if ($env:ROUTER_TOKEN) { $env:ROUTER_TOKEN } else { 'ollama' }

try {
    $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "$routerUrl/health"
}
catch {
    Write-Warning "Router not reachable at $routerUrl - start it with: docker compose up -d"
}

$claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
if (-not $claudeCmd) {
    Write-Error "claude not found in PATH. Install: https://docs.claude.com/en/docs/claude-code"
    exit 127
}

$env:ANTHROPIC_BASE_URL = $routerUrl
$env:ANTHROPIC_AUTH_TOKEN = $routerToken

& $claudeCmd.Source @Args
exit $LASTEXITCODE
