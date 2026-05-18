import {
  Boxes,
  ExternalLink,
  FileJson,
  Server,
  SquareTerminal,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

interface Endpoint {
  method: "GET" | "POST" | "WS";
  path: string;
  description: string;
}

interface ApiSurface {
  title: string;
  blurb: string;
  icon: typeof Server;
  endpoints: Endpoint[];
  example: { label: string; lang: string; code: string };
}

const surfaces: ApiSurface[] = [
  {
    title: "OpenAI-compatible",
    blurb: "Works with the openai SDK, LangChain, LiteLLM, and any other client that speaks the OpenAI REST shape.",
    icon: Boxes,
    endpoints: [
      { method: "POST", path: "/v1/chat/completions", description: "Chat completion. Supports stream: true." },
      { method: "POST", path: "/v1/completions", description: "Legacy text completion." },
      { method: "POST", path: "/v1/embeddings", description: "Embedding vectors." },
      { method: "GET", path: "/v1/models", description: "Merged model list across all accounts." },
    ],
    example: {
      label: "Python (openai SDK)",
      lang: "python",
      code: `from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="anything",
)

resp = client.chat.completions.create(
    model="glm-4.7:cloud",
    messages=[{"role": "user", "content": "ping"}],
)
print(resp.choices[0].message.content)`,
    },
  },
  {
    title: "Ollama-native",
    blurb: "The same shape Ollama itself exposes. Point any Ollama client at the router and it just works.",
    icon: Server,
    endpoints: [
      { method: "POST", path: "/api/chat", description: "Chat with messages. Tool calls and streaming supported." },
      { method: "POST", path: "/api/generate", description: "Single-prompt completion. Streaming supported." },
      { method: "GET", path: "/api/tags", description: "Merged model list across all accounts." },
    ],
    example: {
      label: "ollama CLI",
      lang: "bash",
      code: `export OLLAMA_HOST=http://localhost:8000
ollama run glm-4.7:cloud "hello"`,
    },
  },
  {
    title: "Anthropic-compatible (Claude Code)",
    blurb: "Forwards directly to Ollama's native /v1/messages (Ollama v0.14+). No translation, no model renaming.",
    icon: SquareTerminal,
    endpoints: [
      { method: "POST", path: "/v1/messages", description: "Messages API with full tool calling and SSE streaming." },
      { method: "POST", path: "/v1/messages/count_tokens", description: "Estimated input-token count." },
    ],
    example: {
      label: "Claude Code CLI",
      lang: "bash",
      code: `export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_AUTH_TOKEN=anything
claude`,
    },
  },
];

const adminEndpoints: Endpoint[] = [
  { method: "GET", path: "/", description: "Dashboard UI." },
  { method: "GET", path: "/health", description: "Liveness probe." },
  { method: "GET", path: "/metrics", description: "Per-instance in-memory counters." },
  { method: "GET", path: "/instances", description: "Configured accounts." },
  { method: "GET", path: "/dashboard", description: "Full dashboard snapshot (JSON)." },
  { method: "WS", path: "/ws/dashboard", description: "Event-driven live updates." },
];

export function Docs() {
  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center gap-2">
        <a
          href="/docs"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm font-medium hover:bg-accent"
        >
          <FileJson className="h-4 w-4 text-primary" />
          Swagger UI
          <ExternalLink className="h-3 w-3 text-muted-foreground" />
        </a>
        <a
          href="/redoc"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm font-medium hover:bg-accent"
        >
          <FileJson className="h-4 w-4 text-primary" />
          ReDoc
          <ExternalLink className="h-3 w-3 text-muted-foreground" />
        </a>
        <a
          href="/openapi.json"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm font-medium hover:bg-accent"
        >
          <FileJson className="h-4 w-4 text-primary" />
          openapi.json
          <ExternalLink className="h-3 w-3 text-muted-foreground" />
        </a>
      </div>

      {surfaces.map((surface) => {
        const Icon = surface.icon;
        return (
          <section key={surface.title} className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="rounded-md bg-primary/15 p-2 text-primary">
                <Icon className="h-4 w-4" />
              </div>
              <div className="min-w-0">
                <h2 className="text-base font-semibold tracking-tight">{surface.title}</h2>
                <p className="mt-0.5 text-sm text-muted-foreground">{surface.blurb}</p>
              </div>
            </div>

            <Card>
              <CardContent className="p-0">
                <ul className="divide-y divide-border">
                  {surface.endpoints.map((ep) => (
                    <li key={ep.method + ep.path} className="flex items-center gap-3 px-5 py-3">
                      <MethodBadge method={ep.method} />
                      <code className="font-mono text-sm">{ep.path}</code>
                      <span className="text-xs text-muted-foreground">{ep.description}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle>{surface.example.label}</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="overflow-x-auto rounded-md bg-muted/40 p-4 text-xs leading-relaxed">
                  <code>{surface.example.code}</code>
                </pre>
              </CardContent>
            </Card>
          </section>
        );
      })}

      <Separator />

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-muted-foreground">Admin and observability</h2>
        <Card>
          <CardContent className="p-0">
            <ul className="divide-y divide-border">
              {adminEndpoints.map((ep) => (
                <li key={ep.method + ep.path} className="flex items-center gap-3 px-5 py-3">
                  <MethodBadge method={ep.method} />
                  <code className="font-mono text-sm">{ep.path}</code>
                  <span className="text-xs text-muted-foreground">{ep.description}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

function MethodBadge({ method }: { method: Endpoint["method"] }) {
  const variant =
    method === "GET" ? "outline" : method === "WS" ? "warning" : "default";
  return (
    <Badge variant={variant} className="w-12 justify-center font-mono">
      {method}
    </Badge>
  );
}
