"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { gradeColors } from "@/lib/design-tokens";
import { ChevronDown, ChevronUp, TrendingUp, TrendingDown, Minus } from "lucide-react";

export interface ScoreDimension {
  name: string;
  score: number;
  weight: number;
}

export interface ScoreCardProps {
  /** Letter grade: A, B, C, D, or F */
  grade: string;
  /** Numeric score between 0.0 and 1.0 */
  score: number;
  /** Optional per-dimension breakdown */
  breakdown?: ScoreDimension[];
  /** Previous score for trend display */
  previousScore?: number;
  className?: string;
}

function scoreBarColor(score: number): string {
  if (score >= 0.9) return "bg-green-500";
  if (score >= 0.75) return "bg-lime-500";
  if (score >= 0.6) return "bg-yellow-500";
  if (score >= 0.4) return "bg-orange-500";
  return "bg-red-500";
}

export function ScoreCard({
  grade,
  score,
  breakdown,
  previousScore,
  className,
}: ScoreCardProps) {
  const [expanded, setExpanded] = useState(false);
  const trend = previousScore != null ? score - previousScore : null;

  return (
    <div
      className={cn(
        "rounded-lg border border-[var(--border)] bg-[var(--card)] p-4",
        className
      )}
      role="region"
      aria-label={`Quality score: grade ${grade}, ${(score * 100).toFixed(1)} percent`}
    >
      {/* Top row: grade letter + numeric score + trend */}
      <div className="flex items-center gap-4">
        <span
          className={cn(
            "text-4xl font-bold leading-none",
            gradeColors[grade] || "text-gray-500"
          )}
          aria-label={`Grade ${grade}`}
        >
          {grade}
        </span>
        <div className="flex flex-col">
          <span className="text-2xl font-semibold tabular-nums">
            {(score * 100).toFixed(1)}%
          </span>
          {trend != null && (
            <span
              className={cn(
                "flex items-center gap-1 text-xs",
                trend > 0.001
                  ? "text-green-500"
                  : trend < -0.001
                    ? "text-red-500"
                    : "text-[var(--muted-foreground)]"
              )}
              aria-label={`Trend: ${trend > 0 ? "up" : trend < 0 ? "down" : "unchanged"} ${Math.abs(trend * 100).toFixed(1)} percent`}
            >
              {trend > 0.001 ? (
                <TrendingUp className="h-3 w-3" aria-hidden="true" />
              ) : trend < -0.001 ? (
                <TrendingDown className="h-3 w-3" aria-hidden="true" />
              ) : (
                <Minus className="h-3 w-3" aria-hidden="true" />
              )}
              {trend > 0 ? "+" : ""}
              {(trend * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {/* Expandable dimension breakdown */}
      {breakdown && breakdown.length > 0 && (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-3 flex items-center gap-1 text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
            aria-expanded={expanded}
            aria-controls="score-breakdown"
          >
            {expanded ? (
              <ChevronUp className="h-3 w-3" aria-hidden="true" />
            ) : (
              <ChevronDown className="h-3 w-3" aria-hidden="true" />
            )}
            {expanded ? "Hide" : "Show"} breakdown ({breakdown.length} dimensions)
          </button>

          {expanded && (
            <div id="score-breakdown" className="mt-2 space-y-1.5" role="list">
              {breakdown.map((dim) => (
                <div
                  key={dim.name}
                  className="flex items-center gap-2 text-xs"
                  role="listitem"
                >
                  <span
                    className="w-32 text-[var(--muted-foreground)] truncate"
                    title={dim.name}
                  >
                    {dim.name}
                  </span>
                  <div
                    className="flex-1 h-1.5 bg-[var(--muted)] rounded-full overflow-hidden"
                    role="progressbar"
                    aria-valuenow={Math.round(dim.score * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`${dim.name}: ${(dim.score * 100).toFixed(0)}%`}
                  >
                    <div
                      className={cn("h-full rounded-full transition-all", scoreBarColor(dim.score))}
                      style={{ width: `${dim.score * 100}%` }}
                    />
                  </div>
                  <span className="w-10 text-right tabular-nums">
                    {(dim.score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
