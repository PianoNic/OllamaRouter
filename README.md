# Ollama Router

Load-balanced Ollama Cloud router with Anthropic API compatibility.

---

<img width="1866" height="912" alt="image" src="https://github.com/user-attachments/assets/5c57cf31-f8fd-417e-a2ec-ffd5bb11b79c" />


## Setup

Create `apikeys.txt`:

```
olm_key1...
olm_key2...
```

Start:

```bash
docker compose up -d --build
```

Router → [http://localhost:8000](http://localhost:8000)
Dashboard → [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

---

## Claude Code Config

`.claude-code`

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_AUTH_TOKEN": "ollama"
  }
}
```

---

## Example (Python)

```python
import anthropic

client = anthropic.Anthropic(
    api_key="ollama",
    base_url="http://localhost:8000"
)

r = client.messages.create(
    model="claude-3-5-sonnet",
    max_tokens=1024,
    messages=[{"role":"user","content":"hello"}]
)

print(r.content[0].text)
```

---

## API

```
POST /v1/messages
GET  /health
GET  /dashboard
```

---

## Change Model

Edit in `main.py`:

```python
DEFAULT_MODEL = "minimax-m2.1:cloud"
```

---

## Stop

```bash
docker compose down
```
