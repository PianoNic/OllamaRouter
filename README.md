# Ollama Router

A FastAPI-based router that proxies API requests to multiple Ollama Cloud instances with automatic rate-limit handling and load balancing. Compatible with Claude Code and Anthropic SDK.

## Features

- 🔄 **Load Balancing** - Distributes requests across multiple Ollama instances
- ⚡ **Rate Limit Handling** - Automatically switches instances on 429 errors
- 🔧 **Tool Calling Support** - Full Anthropic tool calling with multi-turn agent loops
- 📊 **Live Dashboard** - Real-time metrics and usage statistics with smooth animations
- 💾 **Persistent Database** - SQLite database for token tracking across restarts
- 🎯 **Model Mapping** - All Claude models route to the same Ollama model
- 📈 **Token Tracking** - Monitor uploaded/downloaded tokens per account
- 🛡️ **Request Deduplication** - Prevents duplicate responses within 2-second window

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Multiple Ollama Cloud API keys (get them at https://ollama.com/settings/keys)
- Claude Code or any Anthropic SDK client

### 1. Setup API Keys

Create an `apikeys.txt` file with one API key per line:

```
olm_key1_abc123xyz...
olm_key2_def456xyz...
olm_key3_ghi789xyz...
olm_key4_jkl012xyz...
```

Each key is automatically assigned to an account (account_1, account_2, etc.). Each account has its own rate limit (typically 30 requests/minute).

### 2. Start the Router

```bash
docker compose up -d --build
```

The router will start on `http://localhost:8000` and the dashboard on `http://localhost:8000/dashboard`

Logs available via:
```bash
docker compose logs -f ollama-router
```

### 3. Configure Claude Code

Create a `.claude-code` config file in your project root:

```json
{
  "permissions": {
    "allow": [],
    "deny": [],
    "ask": []
  },
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_AUTH_TOKEN": "ollama"
  }
}
```

### 4. Test with Python

```python
import anthropic

client = anthropic.Anthropic(
    api_key="ollama",
    base_url="http://localhost:8000"
)

# Any Claude model name works - they all route to Ollama
response = client.messages.create(
    model="claude-3-5-sonnet",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.content[0].text)
```

## API Endpoints

### Health Check
```
GET http://localhost:8000/health
Response: {"status": "healthy", "accounts": [...]}
```

### Messages (Anthropic API v1 compatible)
```
POST http://localhost:8000/v1/messages
Content-Type: application/json

{
  "model": "claude-3-5-sonnet",
  "messages": [...],
  "max_tokens": 1024
}
```

### Chat Endpoint
```
POST http://localhost:8000/api/chat
```

### Generate Endpoint
```
POST http://localhost:8000/api/generate
```

### Models List
```
GET http://localhost:8000/api/tags
```

### Live Dashboard
```
GET http://localhost:8000/dashboard
```

## Dashboard Features

**Real-time Metrics** (updates every 3 seconds, configurable):
- **Healthy Accounts** - Number of available instances
- **Total Requests** - All requests since startup
- **Tokens Uploaded** - Total input tokens sent to Ollama
- **Tokens Downloaded** - Total output tokens received
- **Tool Calls** - Number of tool invocations made
- **Rate Limited** - Currently unavailable accounts

**Per-Account Status:**
- Individual token usage
- Request count per account
- Current rate limit status
- Color-coded health indicators

**Interactive Controls:**
- Adjust refresh interval from 1-60 seconds
- Interval preference saved in browser
- Smooth animated number counters with easing

## Configuration

### Environment Variables

In `docker-compose.yml` or `.env`:

```env
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
ANTHROPIC_AUTH_TOKEN=ollama
```

### Model Mapping

All Claude models are routed to a single Ollama model. To change the target, edit `main.py`:

```python
DEFAULT_MODEL = "minimax-m2.1:cloud"  # Change this to any Ollama model
```

Available Ollama Cloud models:
- `minimax-m2.1:cloud` - Largest context (256K tokens)
- `qwen3-coder` - Code optimized
- `gpt-oss:120b-cloud` - Best reasoning
- `glm-4.7:cloud` - Very fast

## Data Persistence

All data survives container restarts:

- **Database**: `./data/ollama_metrics.db` (SQLite)
- **Logs**: `./logs/` directory
- **API Keys**: `./apikeys.txt`

These are mounted as Docker volumes in `docker-compose.yml`.

## Rate Limiting

**How it works:**
1. Each account has a rate limit (typically 30 requests/minute)
2. When an account hits 429, it's marked as rate-limited for 30 seconds
3. Router automatically switches to next available account
4. With N accounts, effective rate limit = N × 30 req/min

**Example with 4 accounts:**
- Individual limit: 30 req/min each
- Combined limit: 120 req/min (effectively unlimited for most use cases)
- Automatic failover when any account is limited

## Request Deduplication

Prevents identical requests from being processed twice:
- Uses MD5 hash of request body
- Prevents duplicates within 2-second window
- Useful for avoiding duplicate LLM responses
- Transparent to client (no performance impact)

## Tool Calling Support

Full Anthropic tool calling with multi-turn agent loops:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet",
    max_tokens=1024,
    tools=tools,
    messages=[...]
)

# Tool calling works end-to-end
# Router converts between Anthropic and Ollama formats automatically
```

## Troubleshooting

### Container won't start
```bash
docker compose logs ollama-router
```

### Database reset (WARNING: deletes all metrics)
```bash
rm -rf data/ollama_metrics.db
docker compose restart
```

### API key authorization failed
- Verify keys in `apikeys.txt` are valid
- Check they have proper formatting (one per line)
- Get new keys at https://ollama.com/settings/keys

### All accounts rate-limited
- Wait 30 seconds (default rate limit window)
- Add more API keys to `apikeys.txt`
- Reduce request frequency

### Duplicate responses
- Router includes deduplication (2-second window)
- Check logs for actual duplicate requests

### High latency
- Check dashboard for rate-limited accounts
- Verify internet connection to Ollama Cloud
- Try adjusting refresh interval on dashboard

## Token Counting

The router tracks tokens for all requests:

**Input Tokens (Uploaded):** Sent to Ollama
**Output Tokens (Downloaded):** Received from Ollama

Visible per-account and per-request in dashboard.

## Performance Tips

1. **Use multiple API keys** - More accounts = better distribution
2. **Monitor dashboard** - Check rate limit status regularly
3. **Adjust refresh interval** - Balance between UI responsiveness and load
4. **Cache responses** - Reduce redundant requests on client side
5. **Use streaming** - For long responses, stream to reduce memory usage

## Stop the Router

```bash
docker compose down
```

Data persists - restart anytime with `docker compose up -d`

## Reset Everything

```bash
docker compose down -v
rm -rf data logs
```

This removes containers, volumes, database, and logs. (Keeps `apikeys.txt`)

## Development

### Local setup without Docker
```bash
pip install -r requirements.txt
python main.py
```

### Tests
```bash
python test_router.py
```

## Architecture Details

- **Framework**: FastAPI with async/await
- **Streaming**: Server-Sent Events (SSE) for real-time responses
- **Database**: SQLite + Peewee ORM
- **Load Balancing**: Round-robin with rate limit awareness
- **Tool Conversion**: 3-layer system for Anthropic ↔ Ollama compatibility
- **Frontend**: Vanilla HTML/CSS/JavaScript (no build step required)

## API Compatibility

✅ Anthropic API v1 Messages format
✅ Tool calling with multi-turn loops
✅ Streaming responses (text/event-stream)
✅ Token counting
✅ Rate limit handling (429 responses)
✅ Stop sequence handling
✅ Bearer token authentication

## Support

For issues:
1. Check logs: `docker compose logs -f ollama-router`
2. Verify API keys in `apikeys.txt`
3. Check dashboard health at `http://localhost:8000/dashboard`
4. Ensure Docker is running: `docker ps`

## License

MIT

## Next Steps

- Add more API keys to `apikeys.txt` for higher throughput
- Monitor the dashboard at `http://localhost:8000/dashboard`
- Try different Ollama models by editing `main.py`
- Integrate with your Claude Code workflow
