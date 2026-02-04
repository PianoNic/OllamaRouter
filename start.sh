#!/bin/bash
# Quick start script for Claude API Router

set -e

echo "================================================"
echo "Claude Code API Router - Quick Start"
echo "================================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "❌ Please edit .env with your Claude API keys:"
    echo "   vi .env"
    echo ""
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Load .env
set -a
source .env
set +a

echo "================================================"
echo "Configuration:"
echo "================================================"
echo "Host:      ${SERVER_HOST:-0.0.0.0}"
echo "Port:      ${SERVER_PORT:-8000}"
echo ""

if [ -z "$CLAUDE_ACCOUNTS" ]; then
    echo "⚠️  CLAUDE_ACCOUNTS not configured"
    echo "Please set CLAUDE_ACCOUNTS in your .env file"
    exit 1
fi

echo "✓ Claude accounts configured"
echo ""

echo "================================================"
echo "Starting server..."
echo "================================================"
echo ""
echo "Server will be available at: http://localhost:${SERVER_PORT:-8000}"
echo ""
echo "Health Check:    curl http://localhost:${SERVER_PORT:-8000}/health"
echo "Metrics:         curl http://localhost:${SERVER_PORT:-8000}/metrics"
echo "Accounts:        curl http://localhost:${SERVER_PORT:-8000}/accounts"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

python3 run.py
