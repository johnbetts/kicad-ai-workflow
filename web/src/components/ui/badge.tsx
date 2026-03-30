import { type HTMLAttributes } from "react";
import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-[var(--primary)] text-white",
        success:
          "border-transparent bg-green-500/15 text-green-500 border-green-500/30",
        warning:
          "border-transparent bg-yellow-500/15 text-yellow-500 border-yellow-500/30",
        destructive:
          "border-transparent bg-red-500/15 text-red-500 border-red-500/30",
        info:
          "border-transparent bg-blue-500/15 text-blue-500 border-blue-500/30",
        outline:
          "border-[var(--border)] text-[var(--foreground)]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  );
}

export { Badge, badgeVariants };
