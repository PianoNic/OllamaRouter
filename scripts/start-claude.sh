#!/usr/bin/env sh
# Launch Claude Code routed through Ollama Router.
# Usage:   ./scripts/claude.sh [claude args...]
# Env:     ROUTER_URL  (default: http://localhost:8000)
#          ROUTER_TOKEN (default: ollama)

set -eu

ROUTER_URL="${ROUTER_URL:-http://localhost:8000}"
ROUTER_TOKEN="${ROUTER_TOKEN:-ollama}"

# Quick reachability check — warn but don't block.
if command -v curl >/dev/null 2>&1; then
    if ! curl -fsS -o /dev/null --max-time 2 "$ROUTER_URL/health"; then
        printf '[warn] Router not reachable at %s — start it with: docker compose up -d\n' "$ROUTER_URL" >&2
    fi
fi

if ! command -v claude >/dev/null 2>&1; then
    printf '[error] `claude` not found in PATH. Install: https://docs.claude.com/en/docs/claude-code\n' >&2
    exit 127
fi

ANTHROPIC_BASE_URL="$ROUTER_URL" \
ANTHROPIC_AUTH_TOKEN="$ROUTER_TOKEN" \
exec claude "$@"
