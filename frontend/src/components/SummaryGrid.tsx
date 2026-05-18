import {
  Activity,
  ArrowDownToLine,
  ArrowUpToLine,
  CircleAlert,
  CircleCheck,
  Gauge,
  Server,
  Wrench,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber } from "@/lib/utils";
import type { DashboardSummary } from "@/types";

interface Props {
  summary: DashboardSummary;
}

export function SummaryGrid({ summary }: Props) {
  const tiles = [
    {
      label: "Accounts",
      value: `${summary.healthy_accounts} / ${summary.total_accounts}`,
      hint: summary.estimated_capacity,
      icon: Server,
      tone: "default" as const,
    },
    {
      label: "Healthy",
      value: formatNumber(summary.healthy_accounts),
      hint: summary.overall_health,
      icon: CircleCheck,
      tone: summary.overall_health === "healthy" ? ("success" as const) : ("warning" as const),
    },
    {
      label: "Rate limited",
      value: formatNumber(summary.rate_limited_accounts),
      hint: summary.estimated_total_capacity,
      icon: CircleAlert,
      tone: summary.rate_limited_accounts > 0 ? ("destructive" as const) : ("muted" as const),
    },
    {
      label: "Total requests",
      value: formatNumber(summary.total_requests),
      hint: `${formatNumber(summary.total_errors)} errors`,
      icon: Activity,
      tone: "default" as const,
    },
    {
      label: "Input tokens",
      value: formatNumber(summary.tokens_input),
      hint: summary.rate_limit_per_account,
      icon: ArrowUpToLine,
      tone: "default" as const,
    },
    {
      label: "Output tokens",
      value: formatNumber(summary.tokens_output),
      hint: "estimated",
      icon: ArrowDownToLine,
      tone: "default" as const,
    },
    {
      label: "Tool calls",
      value: formatNumber(summary.tool_calls),
      hint: "across all accounts",
      icon: Wrench,
      tone: "default" as const,
    },
    {
      label: "Capacity",
      value: summary.estimated_total_capacity,
      hint: summary.estimated_capacity,
      icon: Gauge,
      tone: "default" as const,
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      {tiles.map((tile) => {
        const Icon = tile.icon;
        const iconTone =
          tile.tone === "success"
            ? "text-success"
            : tile.tone === "warning"
              ? "text-warning"
              : tile.tone === "destructive"
                ? "text-destructive"
                : tile.tone === "muted"
                  ? "text-muted-foreground"
                  : "text-primary";
        return (
          <Card key={tile.label}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle>{tile.label}</CardTitle>
              <Icon className={`h-4 w-4 ${iconTone}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-semibold tracking-tight">{tile.value}</div>
              <p className="mt-1 text-xs text-muted-foreground capitalize">{tile.hint}</p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
