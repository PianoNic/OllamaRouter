#!/usr/bin/env python3
"""
Script to start the Claude API Router
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import app
import uvicorn


def main():
    """Start the server"""
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    print("=" * 60)
    print("Claude Code API Router")
    print("=" * 60)
    print(f"Starting server on {host}:{port}")
    print()
    print("Endpoints:")
    print(f"  Health Check:    GET  http://{host}:{port}/health")
    print(f"  Messages:        POST http://{host}:{port}/v1/messages")
    print(f"  Streaming:       POST http://{host}:{port}/v1/messages/stream")
    print(f"  Metrics:         GET  http://{host}:{port}/metrics")
    print(f"  Accounts:        GET  http://{host}:{port}/accounts")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
