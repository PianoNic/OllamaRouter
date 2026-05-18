import { useEffect, useRef, useState } from "react";
import type { ConnectionStatus, DashboardSnapshot } from "@/types";

const MAX_BACKOFF_MS = 15_000;

export function useDashboard() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchInitial = async () => {
      try {
        const res = await fetch("/dashboard");
        if (!res.ok) return;
        const data = (await res.json()) as DashboardSnapshot;
        if (!cancelled) setSnapshot(data);
      } catch {
        // Will retry via WS connect.
      }
    };

    const connect = () => {
      if (cancelled) return;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${window.location.host}/ws/dashboard`;
      setStatus("connecting");
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("open");
        reconnectAttempts.current = 0;
      };
      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as DashboardSnapshot;
          setSnapshot(payload);
        } catch {
          // ignore malformed
        }
      };
      ws.onerror = () => {
        ws.close();
      };
      ws.onclose = () => {
        if (cancelled) return;
        setStatus("closed");
        const attempt = ++reconnectAttempts.current;
        const delay = Math.min(1000 * 2 ** (attempt - 1), MAX_BACKOFF_MS);
        reconnectTimer.current = window.setTimeout(connect, delay);
      };
    };

    void fetchInitial();
    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer.current !== null) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, []);

  return { snapshot, status };
}
