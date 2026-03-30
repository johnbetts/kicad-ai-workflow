"use client";

import { cn } from "@/lib/utils";
import { Check, Circle } from "lucide-react";

export interface ChecklistItem {
  label: string;
  checked: boolean;
  detail?: string;
}

export interface ReviewChecklistProps {
  items: ChecklistItem[];
  className?: string;
}

export function ReviewChecklist({ items, className }: ReviewChecklistProps) {
  if (items.length === 0) return null;

  return (
    <div
      className={cn("space-y-1", className)}
      role="list"
      aria-label="Review checklist"
    >
      {items.map((item, idx) => (
        <div
          key={idx}
          className={cn(
            "flex items-start gap-2.5 rounded-md px-3 py-2 transition-colors",
            item.checked
              ? "bg-green-500/5"
              : "bg-transparent hover:bg-[var(--muted)]"
          )}
          role="listitem"
        >
          {/* Icon */}
          {item.checked ? (
            <Check
              className="h-4 w-4 mt-0.5 flex-shrink-0 text-green-500"
              aria-hidden="true"
            />
          ) : (
            <Circle
              className="h-4 w-4 mt-0.5 flex-shrink-0 text-gray-500"
              aria-hidden="true"
            />
          )}

          {/* Text */}
          <div className="flex-1 min-w-0">
            <span
              className={cn(
                "text-sm",
                item.checked
                  ? "text-[var(--muted-foreground)] line-through"
                  : "text-[var(--foreground)]"
              )}
            >
              {item.label}
            </span>
            {item.detail && (
              <p className="mt-0.5 text-xs text-[var(--muted-foreground)]">
                {item.detail}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
