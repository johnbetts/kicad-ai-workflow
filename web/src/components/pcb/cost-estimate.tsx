import { cn } from "@/lib/utils";
import { DollarSign, ExternalLink } from "lucide-react";

export interface CostBreakdown {
  fab: string;
  pcb_cost: number;
  assembly_cost: number;
  quantity: number;
  total: number;
}

export interface CostEstimateProps {
  costs: CostBreakdown[];
  className?: string;
}

function formatCurrency(amount: number): string {
  return `$${amount.toFixed(2)}`;
}

export function CostEstimate({ costs, className }: CostEstimateProps) {
  if (costs.length === 0) {
    return (
      <div
        className={cn(
          "rounded-lg border border-[var(--border)] bg-[var(--card)] p-4 text-sm text-[var(--muted-foreground)]",
          className
        )}
      >
        No cost estimates available.
      </div>
    );
  }

  // Find the cheapest option by total
  const minTotal = Math.min(...costs.map((c) => c.total));

  return (
    <div
      className={cn(
        "rounded-lg border border-[var(--border)] bg-[var(--card)] overflow-hidden",
        className
      )}
    >
      <div className="flex items-center gap-2 px-4 py-3 border-b border-[var(--border)]">
        <DollarSign className="h-4 w-4 text-[var(--muted-foreground)]" aria-hidden="true" />
        <span className="text-sm font-medium">Cost Estimates</span>
      </div>

      <table className="w-full text-sm" role="table">
        <thead>
          <tr className="border-b border-[var(--border)] text-[var(--muted-foreground)]">
            <th className="px-4 py-2 text-left font-medium" scope="col">Fab</th>
            <th className="px-4 py-2 text-right font-medium" scope="col">PCB</th>
            <th className="px-4 py-2 text-right font-medium" scope="col">Assembly</th>
            <th className="px-4 py-2 text-right font-medium" scope="col">Qty</th>
            <th className="px-4 py-2 text-right font-medium" scope="col">Total</th>
            <th className="px-4 py-2 text-right font-medium" scope="col">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {costs.map((cost) => {
            const isCheapest = cost.total === minTotal && costs.length > 1;
            return (
              <tr
                key={cost.fab}
                className={cn(
                  "border-b border-[var(--border)] last:border-b-0",
                  isCheapest && "bg-green-500/5"
                )}
              >
                <td className="px-4 py-2 font-medium">
                  {cost.fab}
                  {isCheapest && (
                    <span className="ml-2 inline-flex items-center rounded-full bg-green-500/15 text-green-500 border border-green-500/30 px-1.5 py-0.5 text-[10px] font-semibold">
                      Best
                    </span>
                  )}
                </td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {formatCurrency(cost.pcb_cost)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {formatCurrency(cost.assembly_cost)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums">{cost.quantity}</td>
                <td className="px-4 py-2 text-right tabular-nums font-semibold">
                  {formatCurrency(cost.total)}
                </td>
                <td className="px-4 py-2 text-right">
                  {cost.fab === "JLCPCB" && (
                    <a
                      href="https://cart.jlcpcb.com/quote"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-[var(--primary)] hover:underline"
                      aria-label={`Order from ${cost.fab}`}
                    >
                      Order
                      <ExternalLink className="h-3 w-3" aria-hidden="true" />
                    </a>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
