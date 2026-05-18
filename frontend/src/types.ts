export interface AccountSnapshot {
  name: string;
  display_name: string | null;
  requests_made: number;
  tokens_input: number;
  tokens_output: number;
  tool_calls: number;
  is_rate_limited: boolean;
  usage_percent: number;
  consecutive_errors: number;
  last_error: string | null;
  last_rate_limit: string | null;
  uptime_seconds: number;
}

export interface DashboardSummary {
  total_accounts: number;
  healthy_accounts: number;
  rate_limited_accounts: number;
  overall_health: "healthy" | "degraded" | "limited";
  total_requests: number;
  total_errors: number;
  tokens_input: number;
  tokens_output: number;
  tool_calls: number;
  estimated_capacity: string;
  rate_limit_per_account: string;
  estimated_total_capacity: string;
}

export interface DashboardSnapshot {
  timestamp: string;
  summary: DashboardSummary;
  accounts: AccountSnapshot[];
}

export type ConnectionStatus = "connecting" | "open" | "closed";
