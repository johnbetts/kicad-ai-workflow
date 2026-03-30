"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Cpu, Search, LayoutGrid, Plus, BookOpen, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";

const NAV_LINKS = [
  { href: "/", label: "Dashboard", icon: LayoutGrid },
  { href: "/parts", label: "Parts Search", icon: Search },
  { href: "/library", label: "Library", icon: BookOpen },
  { href: "/admin/components", label: "Admin", icon: Shield },
] as const;

export function NavHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-gray-950">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
        {/* Left: Logo */}
        <Link
          href="/"
          className="flex items-center gap-2 text-white font-semibold tracking-tight"
        >
          <Cpu className="h-5 w-5 text-blue-500" aria-hidden="true" />
          <span className="hidden sm:inline">KiCad AI Pipeline</span>
        </Link>

        {/* Center: Nav links */}
        <nav className="flex items-center gap-1" aria-label="Main navigation">
          {NAV_LINKS.map(({ href, label, icon: Icon }) => {
            const isActive =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-white/10 text-white"
                    : "text-gray-400 hover:bg-white/5 hover:text-gray-200"
                )}
                aria-current={isActive ? "page" : undefined}
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Right: New Project button */}
        <Link href="/new">
          <Button size="sm" className="gap-1.5">
            <Plus className="h-4 w-4" aria-hidden="true" />
            <span className="hidden sm:inline">New Project</span>
          </Button>
        </Link>
      </div>
    </header>
  );
}
