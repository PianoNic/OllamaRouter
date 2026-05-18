import { useEffect, useState } from "react";
import { AccountCard } from "@/components/AccountCard";
import { Docs } from "@/components/Docs";
import { SummaryGrid } from "@/components/SummaryGrid";
import { Separator } from "@/components/ui/separator";
import { useDashboard } from "@/hooks/useDashboard";
import { cn } from "@/lib/utils";

type Tab = "overview" | "docs";

function readTabFromHash(): Tab {
  return window.location.hash === "#docs" ? "docs" : "overview";
}

export default function App() {
  const { snapshot } = useDashboard();
  const [tab, setTab] = useState<Tab>(readTabFromHash);

  useEffect(() => {
    const onHash = () => setTab(readTabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const setTo = (next: Tab) => {
    window.location.hash = next === "overview" ? "" : "#docs";
    setTab(next);
  };

  return (
    <div className="min-h-full">
      <header className="border-b border-border bg-card/30 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-end justify-between gap-6 px-6 py-4">
          <div>
            <h1 className="text-base font-semibold tracking-tight">Ollama Router</h1>
            <p className="text-xs text-muted-foreground">
              account metrics and rate-limit status
            </p>
          </div>
          <nav className="flex gap-1 text-sm">
            <TabButton active={tab === "overview"} onClick={() => setTo("overview")}>
              Overview
            </TabButton>
            <TabButton active={tab === "docs"} onClick={() => setTo("docs")}>
              Docs
            </TabButton>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8 space-y-8">
        {tab === "overview" ? (
          snapshot ? (
            <>
              <SummaryGrid summary={snapshot.summary} />
              <Separator />
              <section className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <h2 className="text-sm font-medium text-muted-foreground">Accounts</h2>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {snapshot.accounts.length} configured
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {snapshot.accounts.map((account) => (
                    <AccountCard key={account.name} account={account} />
                  ))}
                </div>
              </section>
            </>
          ) : (
            <SkeletonState />
          )
        ) : (
          <Docs />
        )}
      </main>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "rounded-md px-3 py-1.5 transition-colors",
        active
          ? "bg-primary/15 text-primary"
          : "text-muted-foreground hover:bg-accent hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

function SkeletonState() {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="h-24 animate-pulse rounded-lg border border-border bg-card"
        />
      ))}
    </div>
  );
}
