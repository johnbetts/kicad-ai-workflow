"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  AlertCircle,
  AlertTriangle,
  Info,
  ShieldAlert,
  Wrench,
  EyeOff,
  HelpCircle,
} from "lucide-react";

export interface Finding {
  id: string;
  persona: "fab" | "ee";
  severity: "critical" | "major" | "minor" | "warning" | "info";
  ref: string;
  description: string;
}

export interface FindingsListProps {
  findings: Finding[];
  onFix?: (id: string) => void;
  onIgnore?: (id: string) => void;
  className?: string;
}

const severityConfig: Record<
  Finding["severity"],
  { icon: typeof AlertCircle; color: string; bgColor: string; label: string }
> = {
  critical: {
    icon: ShieldAlert,
    color: "text-red-500",
    bgColor: "bg-red-500/10 border-red-500/20",
    label: "Critical",
  },
  major: {
    icon: AlertCircle,
    color: "text-orange-500",
    bgColor: "bg-orange-500/10 border-orange-500/20",
    label: "Major",
  },
  warning: {
    icon: AlertTriangle,
    color: "text-yellow-500",
    bgColor: "bg-yellow-500/10 border-yellow-500/20",
    label: "Warning",
  },
  minor: {
    icon: AlertTriangle,
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/5 border-yellow-500/15",
    label: "Minor",
  },
  info: {
    icon: Info,
    color: "text-blue-400",
    bgColor: "bg-blue-500/5 border-blue-500/15",
    label: "Info",
  },
};

const personaConfig: Record<
  Finding["persona"],
  { label: string; variant: "info" | "default" }
> = {
  fab: { label: "Fab", variant: "info" },
  ee: { label: "EE", variant: "default" },
};

/**
 * Detect consensus findings: same ref flagged by both Fab and EE personas.
 * Returns a Set of ref designators that appear in both persona groups.
 */
function findConsensusRefs(findings: Finding[]): Set<string> {
  const fabRefs = new Set<string>();
  const eeRefs = new Set<string>();
  for (const f of findings) {
    if (f.persona === "fab") fabRefs.add(f.ref);
    else eeRefs.add(f.ref);
  }
  const consensus = new Set<string>();
  for (const ref of fabRefs) {
    if (eeRefs.has(ref)) consensus.add(ref);
  }
  return consensus;
}

export function FindingsList({
  findings,
  onFix,
  onIgnore,
  className,
}: FindingsListProps) {
  const consensusRefs = findConsensusRefs(findings);

  if (findings.length === 0) {
    return (
      <div
        className={cn(
          "rounded-lg border border-[var(--border)] bg-[var(--card)] p-6 text-center text-sm text-[var(--muted-foreground)]",
          className
        )}
      >
        No review findings. Board looks clean.
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)} role="list" aria-label="Review findings">
      {findings.map((finding) => {
        const sev = severityConfig[finding.severity];
        const persona = personaConfig[finding.persona];
        const Icon = sev.icon;
        const isConsensus = consensusRefs.has(finding.ref);

        return (
          <div
            key={finding.id}
            className={cn(
              "rounded-lg border p-3 transition-colors",
              sev.bgColor
            )}
            role="listitem"
          >
            <div className="flex items-start gap-2.5">
              {/* Severity icon */}
              <Icon
                className={cn("h-4 w-4 mt-0.5 flex-shrink-0", sev.color)}
                aria-hidden="true"
              />

              {/* Content */}
              <div className="flex-1 min-w-0">
                {/* Header row: badges + ref */}
                <div className="flex items-center gap-1.5 flex-wrap">
                  <Badge
                    variant={persona.variant}
                    className="text-[10px] px-1.5 py-0"
                  >
                    {persona.label}
                  </Badge>
                  {isConsensus && (
                    <Badge variant="warning" className="text-[10px] px-1.5 py-0">
                      Consensus
                    </Badge>
                  )}
                  <code className="rounded bg-[var(--muted)] px-1.5 py-0.5 text-[11px] font-mono text-[var(--foreground)]">
                    {finding.ref}
                  </code>
                  <span className={cn("text-[10px] font-medium", sev.color)}>
                    {sev.label}
                  </span>
                </div>

                {/* Description */}
                <p className="mt-1 text-sm text-[var(--foreground)] leading-snug">
                  {finding.description}
                </p>

                {/* Action buttons */}
                <div className="mt-2 flex items-center gap-1.5">
                  {onFix && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 px-2 text-[11px] gap-1"
                      onClick={() => onFix(finding.id)}
                    >
                      <Wrench className="h-3 w-3" aria-hidden="true" />
                      Fix It
                    </Button>
                  )}
                  {onIgnore && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 px-2 text-[11px] gap-1 text-[var(--muted-foreground)]"
                      onClick={() => onIgnore(finding.id)}
                    >
                      <EyeOff className="h-3 w-3" aria-hidden="true" />
                      Ignore
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 px-2 text-[11px] gap-1 text-[var(--muted-foreground)]"
                  >
                    <HelpCircle className="h-3 w-3" aria-hidden="true" />
                    Why?
                  </Button>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
