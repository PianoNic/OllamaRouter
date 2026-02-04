# Ollama Router for Claude Code

FastAPI router that proxies Claude Code requests to multiple Ollama Cloud instances with automatic failover on rate limits.

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Add API Keys
Edit `apikeys.txt` - add one API key per line:
```
olm_key1...
olm_key2...
olm_key3...
```

Get keys from: https://ollama.com/settings/keys

### 3. Run
```bash
python main.py
```

### 4. Use
```bash
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_BASE_URL=http://localhost:8000
claude --model gpt-oss:120b-cloud
```

## How It Works

```
Claude Code Request
        ↓
Router (port 8000)
        ↓
Select API Key
        ↓
Send to Ollama Cloud
        ↓
Rate Limited (429)?
├─ YES → Try next key
└─ NO  → Return response
```

## Monitoring

```bash
# Health
curl http://localhost:8000/health

# Instances
curl http://localhost:8000/instances

# Metrics
curl http://localhost:8000/metrics
```

## Configuration

### .env File
```env
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
ANTHROPIC_AUTH_TOKEN=ollama
ANTHROPIC_BASE_URL=http://localhost:8000
```

### apikeys.txt
One API key per line, comments with `#`:
```
# Production keys
olm_abc123...
olm_def456...

# Backup keys
olm_ghi789...
```

## Models

- `gpt-oss:120b-cloud` - Best reasoning
- `qwen3-coder` - Code optimized
- `minimax-m2.1:cloud` - Largest context (256K)
- `glm-4.7:cloud` - Very fast

## Docker

```bash
docker-compose up
```

## Features

- ✅ Multiple Ollama Cloud accounts
- ✅ Automatic rate limit failover
- ✅ Claude Code compatible
- ✅ Health checks & metrics
- ✅ Streaming support
- ✅ Bearer token authentication

## API Endpoints

- `POST /api/chat` - Chat endpoint
- `POST /api/generate` - Generate endpoint
- `GET /api/tags` - List models
- `GET /health` - Health check
- `GET /metrics` - Instance metrics
- `GET /instances` - List instances

## Troubleshooting

**Connection refused**: Make sure `python main.py` is running

**401 Unauthorized**: Check API keys in `apikeys.txt`

**429 Too Many Requests**: Router auto-switches accounts. Wait 30s if all limited.

**Model not found**: Verify model name is correct

## Unlimited Usage

With 3 accounts (30 req/min each = 90 req/min total):
- Effectively unlimited for most use cases
- Automatic failover when one account hits limit
- Scales with more accounts

## Support

See `QUICKSTART.md` for quick reference
