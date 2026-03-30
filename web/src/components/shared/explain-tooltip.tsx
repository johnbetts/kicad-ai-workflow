"use client";

import { useState, useRef, useEffect, useCallback, type ReactNode } from "react";
import { HelpCircle, X } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ExplainTooltipProps {
  /** Short prompt text shown next to the "?" icon */
  question: string;
  /** Longer educational explanation shown in the popover */
  explanation: string;
  children: ReactNode;
  className?: string;
}

export function ExplainTooltip({
  question,
  explanation,
  children,
  className,
}: ExplainTooltipProps) {
  const [open, setOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const tooltipId = useRef(
    `explain-${Math.random().toString(36).slice(2, 9)}`
  ).current;

  // Close on outside click
  const handleClickOutside = useCallback(
    (e: Event) => {
      if (
        open &&
        popoverRef.current &&
        !popoverRef.current.contains(e.target as Node) &&
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    },
    [open]
  );

  // Close on Escape
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (open && e.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    },
    [open]
  );

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleClickOutside, handleKeyDown]);

  return (
    <span className={cn("inline-flex items-center gap-1", className)}>
      {children}
      <span className="relative inline-flex">
        <button
          ref={triggerRef}
          onClick={() => setOpen(!open)}
          className="inline-flex items-center justify-center rounded-full p-0.5 text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--muted)] transition-colors"
          aria-expanded={open}
          aria-controls={tooltipId}
          aria-label={question}
          title={question}
        >
          <HelpCircle className="h-3.5 w-3.5" aria-hidden="true" />
        </button>

        {open && (
          <div
            ref={popoverRef}
            id={tooltipId}
            role="tooltip"
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50 w-64 rounded-lg border border-[var(--border)] bg-[var(--card)] p-3 shadow-lg"
          >
            {/* Arrow */}
            <div
              className="absolute top-full left-1/2 -translate-x-1/2 -mt-px w-0 h-0 border-x-[6px] border-x-transparent border-t-[6px] border-t-[var(--border)]"
              aria-hidden="true"
            />

            {/* Header */}
            <div className="flex items-start justify-between gap-2 mb-1.5">
              <span className="text-xs font-semibold text-[var(--foreground)]">
                {question}
              </span>
              <button
                onClick={() => setOpen(false)}
                className="flex-shrink-0 rounded p-0.5 text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
                aria-label="Close explanation"
              >
                <X className="h-3 w-3" />
              </button>
            </div>

            {/* Body */}
            <p className="text-xs leading-relaxed text-[var(--muted-foreground)]">
              {explanation}
            </p>
          </div>
        )}
      </span>
    </span>
  );
}
