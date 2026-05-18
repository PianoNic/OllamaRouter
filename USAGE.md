# Ollama Router — Endpoint Reference

The router exposes three API surfaces against the same pool of Ollama Cloud
accounts. All three rotate accounts on rate-limits automatically.

Default base URL: `http://localhost:8000`

| Surface         | Path prefix                                           | Use with                                  |
| --------------- | ----------------------------------------------------- | ----------------------------------------- |
| OpenAI-compat   | `/v1/chat/completions`, `/v1/completions`, …          | `openai` Python/JS SDK, LangChain, LiteLLM |
| Ollama-native   | `/api/chat`, `/api/generate`, `/api/tags`             | `ollama` CLI / SDK, any Ollama client     |
| Anthropic-compat | `/v1/messages`, `/v1/messages/count_tokens`          | Claude Code CLI, Anthropic SDK            |

The model you ask for is forwarded to Ollama. The `DEFAULT_MODEL` env var
(currently `glm-4.7:cloud`) is used only by the Anthropic-compat layer where
Claude model names don't map to anything real upstream.

---

## OpenAI-compatible

These mirror OpenAI's REST shape and are proxied to Ollama's own OpenAI surface.

### Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="anything",        # router doesn't authenticate locally
)

resp = client.chat.completions.create(
    model="glm-4.7:cloud",
    messages=[{"role": "user", "content": "ping"}],
)
print(resp.choices[0].message.content)
```

### Streaming
```python
stream = client.chat.completions.create(
    model="glm-4.7:cloud",
    messages=[{"role": "user", "content": "count to 3"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### curl
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7:cloud",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

### Endpoints
| Method | Path                       | Notes                                      |
| ------ | -------------------------- | ------------------------------------------ |
| POST   | `/v1/chat/completions`     | Chat completion, supports `stream: true`   |
| POST   | `/v1/completions`          | Legacy text completion                     |
| POST   | `/v1/embeddings`           | Embedding vectors                          |
| GET    | `/v1/models`               | Merged model list across all accounts      |

---

## Ollama-native

Same shape Ollama itself exposes, so any tool that talks to Ollama works
unmodified — just point it at the router.

### `ollama` CLI
```bash
export OLLAMA_HOST=http://localhost:8000
ollama run glm-4.7:cloud "hello"
```

### Python SDK
```python
import ollama

client = ollama.Client(host="http://localhost:8000")
response = client.chat(
    model="glm-4.7:cloud",
    messages=[{"role": "user", "content": "hello"}],
)
print(response["message"]["content"])
```

### curl
```bash
curl http://localhost:8000/api/chat \
  -d '{
    "model": "glm-4.7:cloud",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false
  }'
```

### Endpoints
| Method | Path             | Notes                                              |
| ------ | ---------------- | -------------------------------------------------- |
| POST   | `/api/chat`      | Chat with messages, supports tools and streaming   |
| POST   | `/api/generate`  | Single-prompt completion, supports streaming       |
| GET    | `/api/tags`      | Merged model list across all accounts              |

---

## Anthropic-compatible (Claude Code CLI)

The router translates the Anthropic Messages API to Ollama's chat format and
back, including tool calls.

### Claude Code CLI
```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_AUTH_TOKEN=anything
claude
```

### Anthropic Python SDK
```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="anything",
)

msg = client.messages.create(
    model="claude-3-5-sonnet-20241022",   # mapped to DEFAULT_MODEL upstream
    max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content[0].text)
```

### curl
```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 1024
  }'
```

### Endpoints
| Method | Path                          | Notes                                    |
| ------ | ----------------------------- | ---------------------------------------- |
| POST   | `/v1/messages`                | Messages API, full tool calling + SSE    |
| POST   | `/v1/messages/count_tokens`   | Returns estimated input-token count      |

---

## Admin / observability

| Method | Path                                  | Notes                                       |
| ------ | ------------------------------------- | ------------------------------------------- |
| GET    | `/`                                   | Serves the React dashboard (`frontend/dist`) |
| GET    | `/health`                             | `{status, instances, timestamp}`            |
| GET    | `/metrics`                            | Per-instance in-memory counters             |
| GET    | `/instances`                          | Configured accounts                         |
| GET    | `/instances/{name}/metrics`           | Single account                              |
| GET    | `/dashboard`                          | Full dashboard JSON snapshot                |
| WS     | `/ws/dashboard`                       | Event-driven live updates                   |

---

## Configuration

| Env var                       | Default                           | Purpose                                    |
| ----------------------------- | --------------------------------- | ------------------------------------------ |
| `SERVER_HOST`                 | `0.0.0.0`                         | uvicorn bind host                          |
| `SERVER_PORT`                 | `8000`                            | uvicorn bind port                          |
| `DATA_DIR`                    | `./data`                          | sqlite DB location                         |
| `APIKEYS_FILE`                | `./apikeys.txt`                   | one Ollama Cloud key per line              |
| `DEFAULT_MODEL`               | `glm-4.7:cloud`                   | Used by `/v1/messages` when client sends a Claude model name |
| `RATE_LIMIT_COOLDOWN_SECONDS` | `30`                              | how long before a flagged account is retried |
| `HTTP_TIMEOUT_SECONDS`        | `300`                             | upstream request timeout                   |
| `FRONTEND_DIST`               | `./frontend/dist`                 | where the built React app lives            |
