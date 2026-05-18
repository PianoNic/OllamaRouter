import {
  ArrowDownToLine,
  ArrowUpToLine,
  CircleAlert,
  CircleCheck,
  Clock,
  Hash,
  TriangleAlert,
  Wrench,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { formatDuration, formatNumber } from "@/lib/utils";
import type { AccountSnapshot } from "@/types";

interface Props {
  account: AccountSnapshot;
}

export function AccountCard({ account }: Props) {
  const limited = account.is_rate_limited;
  const progressTone = limited ? "destructive" : account.usage_percent > 70 ? "warning" : "default";

  const rows: { label: string; value: string; icon: typeof Hash }[] = [
    { label: "Requests", value: formatNumber(account.requests_made), icon: Hash },
    { label: "Input", value: formatNumber(account.tokens_input), icon: ArrowUpToLine },
    { label: "Output", value: formatNumber(account.tokens_output), icon: ArrowDownToLine },
    { label: "Tools", value: formatNumber(account.tool_calls), icon: Wrench },
    { label: "Errors", value: formatNumber(account.consecutive_errors), icon: TriangleAlert },
    { label: "Uptime", value: formatDuration(account.uptime_seconds), icon: Clock },
  ];

  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate font-medium tracking-tight">
              {account.display_name ?? account.name}
            </div>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {account.display_name ? account.name : `usage ${account.usage_percent.toFixed(1)}%`}
            </p>
          </div>
          {limited ? (
            <Badge variant="destructive">
              <CircleAlert className="h-3 w-3" />
              rate limited
            </Badge>
          ) : (
            <Badge variant="success">
              <CircleCheck className="h-3 w-3" />
              healthy
            </Badge>
          )}
        </div>

        <div className="mt-4">
          <Progress value={account.usage_percent} tone={progressTone} />
        </div>

        <dl className="mt-5 grid grid-cols-3 gap-y-3 text-sm">
          {rows.map(({ label, value, icon: Icon }) => (
            <div key={label} className="flex flex-col">
              <dt className="flex items-center gap-1 text-xs text-muted-foreground">
                <Icon className="h-3 w-3" />
                {label}
              </dt>
              <dd className="mt-0.5 font-medium tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>

        {account.last_error && account.last_error !== "rate_limit" ? (
          <div className="mt-4 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
            <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" />
            <span className="break-all">{account.last_error}</span>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
