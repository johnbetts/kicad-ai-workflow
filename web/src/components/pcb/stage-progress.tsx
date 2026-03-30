"use client";

import { cn } from "@/lib/utils";
import { STAGES, type Stage } from "@/lib/design-tokens";
import { Check, X, Circle, Loader2, Ban } from "lucide-react";

export interface StageStatus {
  stage: Stage;
  state: "pending" | "running" | "passed" | "failed" | "blocked";
}

export interface StageProgressProps {
  stages: StageStatus[];
  className?: string;
}

const stateConfig: Record<
  StageStatus["state"],
  { icon: typeof Check; color: string; bgColor: string; lineColor: string; label: string }
> = {
  pending: {
    icon: Circle,
    color: "text-gray-500",
    bgColor: "bg-gray-500/10 border-gray-500/30",
    lineColor: "bg-gray-500/30",
    label: "Pending",
  },
  running: {
    icon: Loader2,
    color: "text-blue-500",
    bgColor: "bg-blue-500/10 border-blue-500/30",
    lineColor: "bg-blue-500/50",
    label: "Running",
  },
  passed: {
    icon: Check,
    color: "text-green-500",
    bgColor: "bg-green-500/10 border-green-500/30",
    lineColor: "bg-green-500",
    label: "Passed",
  },
  failed: {
    icon: X,
    color: "text-red-500",
    bgColor: "bg-red-500/10 border-red-500/30",
    lineColor: "bg-red-500",
    label: "Failed",
  },
  blocked: {
    icon: Ban,
    color: "text-yellow-500",
    bgColor: "bg-yellow-500/10 border-yellow-500/30",
    lineColor: "bg-yellow-500/30",
    label: "Blocked",
  },
};

/** Capitalize first letter of a stage name for display */
function stageLabel(stage: string): string {
  return stage.charAt(0).toUpperCase() + stage.slice(1);
}

export function StageProgress({ stages, className }: StageProgressProps) {
  // Default to all STAGES as pending if nothing provided
  const resolved: StageStatus[] =
    stages.length > 0
      ? stages
      : STAGES.map((s) => ({ stage: s, state: "pending" as const }));

  return (
    <nav
      className={cn("flex items-center gap-0", className)}
      aria-label="Pipeline stages"
    >
      {resolved.map((entry, idx) => {
        const config = stateConfig[entry.state];
        const Icon = config.icon;
        const isLast = idx === resolved.length - 1;

        return (
          <div key={entry.stage} className="flex items-center">
            {/* Stage node */}
            <div
              className="flex flex-col items-center gap-1"
              role="listitem"
              aria-label={`${stageLabel(entry.stage)}: ${config.label}`}
              aria-current={entry.state === "running" ? "step" : undefined}
            >
              <div
                className={cn(
                  "flex h-8 w-8 items-center justify-center rounded-full border",
                  config.bgColor
                )}
              >
                <Icon
                  className={cn(
                    "h-4 w-4",
                    config.color,
                    entry.state === "running" && "animate-spin"
                  )}
                  aria-hidden="true"
                />
              </div>
              <span
                className={cn(
                  "text-[10px] font-medium",
                  entry.state === "running"
                    ? "text-[var(--foreground)]"
                    : "text-[var(--muted-foreground)]"
                )}
              >
                {stageLabel(entry.stage)}
              </span>
            </div>

            {/* Connecting line */}
            {!isLast && (
              <div
                className={cn("mx-1 h-0.5 w-8 rounded-full", config.lineColor)}
                aria-hidden="true"
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
