import { cn } from "@/lib/utils";
import { AlertTriangle, AlertCircle, Info, Wrench } from "lucide-react";
import type { PipelineError } from "@/lib/api";

export interface ErrorBannerProps {
  error: PipelineError;
  /** Optional callback when the user clicks the "Fix It" button */
  onFix?: () => void;
  className?: string;
}

const severityConfig: Record<
  string,
  { icon: typeof AlertCircle; bg: string; border: string; text: string; label: string }
> = {
  fatal: {
    icon: AlertCircle,
    bg: "bg-red-500/10",
    border: "border-red-500/30",
    text: "text-red-500",
    label: "Fatal Error",
  },
  recoverable: {
    icon: AlertTriangle,
    bg: "bg-orange-500/10",
    border: "border-orange-500/30",
    text: "text-orange-500",
    label: "Recoverable Error",
  },
  warning: {
    icon: Info,
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/30",
    text: "text-yellow-500",
    label: "Warning",
  },
};

export function ErrorBanner({ error, onFix, className }: ErrorBannerProps) {
  const config = severityConfig[error.severity] || severityConfig.warning;
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "rounded-lg border p-4",
        config.bg,
        config.border,
        className
      )}
      role="alert"
      aria-label={`${config.label}: ${error.message}`}
    >
      <div className="flex gap-3">
        {/* Icon */}
        <div className="flex-shrink-0 pt-0.5">
          <Icon className={cn("h-5 w-5", config.text)} aria-hidden="true" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 space-y-1">
          {/* Header: code + severity */}
          <div className="flex items-center gap-2">
            {error.code && (
              <code className="rounded bg-[var(--muted)] px-1.5 py-0.5 text-xs font-mono">
                {error.code}
              </code>
            )}
            <span className={cn("text-xs font-medium", config.text)}>
              {config.label}
            </span>
            {error.stage && (
              <span className="text-xs text-[var(--muted-foreground)]">
                in {error.stage}
              </span>
            )}
          </div>

          {/* Message */}
          <p className="text-sm font-medium text-[var(--foreground)]">
            {error.message}
          </p>

          {/* Cause */}
          {error.cause && (
            <p className="text-xs text-[var(--muted-foreground)]">
              Cause: {error.cause}
            </p>
          )}

          {/* Fix suggestion */}
          {error.fix && (
            <div className="flex items-start gap-2 mt-2 rounded bg-[var(--muted)] px-3 py-2">
              <Wrench
                className="h-3.5 w-3.5 mt-0.5 text-[var(--muted-foreground)] flex-shrink-0"
                aria-hidden="true"
              />
              <div className="flex-1 min-w-0">
                <p className="text-xs text-[var(--muted-foreground)]">
                  Suggested fix
                </p>
                <p className="text-xs text-[var(--foreground)]">{error.fix}</p>
              </div>
              {onFix && (
                <button
                  onClick={onFix}
                  className="flex-shrink-0 rounded bg-[var(--primary)] px-2.5 py-1 text-xs font-medium text-white hover:bg-blue-600 transition-colors"
                  aria-label={`Apply fix: ${error.fix}`}
                >
                  Fix It
                </button>
              )}
            </div>
          )}

          {/* Reference link */}
          {error.ref && (
            <p className="text-[10px] text-[var(--muted-foreground)]">
              Ref: {error.ref}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
