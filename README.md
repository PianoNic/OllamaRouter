# Ollama Router

Multi-account router for Ollama Cloud. Aggregates several API keys, rotates on rate-limits, and exposes three API surfaces (Ollama-native, OpenAI-compatible, Anthropic-compatible) plus a live dashboard.

![Dashboard](docs/dashboard.png)

---

## Setup

Create `apikeys.txt` (one key per line):

```
olm_key1...
olm_key2...
```

Start:

```bash
docker compose up -d --build
```

Open [http://localhost:8000](http://localhost:8000).

---

## Endpoints

The router exposes the same surface as Ollama itself, the OpenAI REST shape, and Anthropic's Messages API. The Anthropic path is forwarded straight to Ollama's own `/v1/messages` (added in Ollama v0.14) — no translation in between.

| Surface          | Paths                                                       |
| ---------------- | ----------------------------------------------------------- |
| Ollama-native    | `/api/chat`, `/api/generate`, `/api/tags`                   |
| OpenAI-compatible | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` |
| Anthropic-compatible | `/v1/messages`, `/v1/messages/count_tokens`            |
| Admin            | `/`, `/dashboard`, `/health`, `/metrics`, `/instances`, `/ws/dashboard` |
| OpenAPI          | `/docs`, `/redoc`, `/openapi.json`                          |

Full reference and code examples: [`USAGE.md`](USAGE.md), or the **Docs** tab in the dashboard.

---

## Clients

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_AUTH_TOKEN=anything
claude
```

### `ollama` CLI

```bash
export OLLAMA_HOST=http://localhost:8000
ollama run glm-4.7:cloud "hello"
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
resp = client.chat.completions.create(
    model="glm-4.7:cloud",
    messages=[{"role": "user", "content": "hello"}],
)
```

---

## Configuration

Set via env vars (see `compose.yml`):

| Var                           | Default          |
| ----------------------------- | ---------------- |
| `DEFAULT_MODEL`               | `glm-4.7:cloud`  |
| `RATE_LIMIT_COOLDOWN_SECONDS` | `30`             |
| `HTTP_TIMEOUT_SECONDS`        | `300`            |
| `APIKEYS_FILE`                | `/app/apikeys.txt` |
| `DATA_DIR`                    | `/app/data`      |

---

## Stop

```bash
docker compose down
```
