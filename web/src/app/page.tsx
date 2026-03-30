"use client";

import { useState, useEffect } from "react";
import { NavHeader } from "@/components/layout/nav-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { listBoards, type BoardSummary } from "@/lib/api";
import { gradeColors, gradeBgColors } from "@/lib/design-tokens";
import { Plus, ArrowRight, Zap, Shield, Package, Loader2 } from "lucide-react";
import Link from "next/link";

function gradeVariant(
  grade: string
): "success" | "warning" | "destructive" | "info" | "outline" {
  if (grade === "A") return "success";
  if (grade === "B") return "success";
  if (grade === "C") return "warning";
  if (grade === "D") return "warning";
  if (grade === "F") return "destructive";
  return "outline";
}

const FEATURES = [
  {
    icon: Zap,
    title: "AI Placement",
    desc: "Hierarchical EE-aware component placement with subcircuit detection and voltage isolation.",
  },
  {
    icon: Shield,
    title: "Validation Gates",
    desc: "DRC, electrical, manufacturing, and integrity checks catch errors before fabrication.",
  },
  {
    icon: Package,
    title: "One-Click Export",
    desc: "Gerbers, BOM, and CPL files ready for JLCPCB assembly with parts verification.",
  },
] as const;

export default function DashboardPage() {
  const [boards, setBoards] = useState<BoardSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listBoards()
      .then((data) => {
        if (!cancelled) {
          setBoards(data);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load boards");
          setBoards([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const avgScore =
    boards.length > 0
      ? boards.reduce((sum, b) => sum + (b.score ?? 0), 0) / boards.length
      : 0;
  const productionReady = boards.filter(
    (b) => b.stage === "production" && b.grade === "A"
  ).length;

  return (
    <>
      <NavHeader />

      <main className="mx-auto max-w-7xl flex-1 px-4 py-8">
        {/* Hero */}
        <section className="mb-10 text-center">
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
            From idea to manufactured PCB
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-[var(--muted-foreground)]">
            Describe your hardware project in plain language and get
            production-ready KiCad files, validated for JLCPCB assembly.
          </p>
          <Link href="/new" className="mt-6 inline-block">
            <Button size="lg" className="gap-2">
              <Plus className="h-4 w-4" aria-hidden="true" />
              Start New Project
            </Button>
          </Link>
        </section>

        {/* Stats row */}
        {!loading && boards.length > 0 && (
          <section className="mb-8 grid grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-3xl font-bold tabular-nums">{boards.length}</p>
                <p className="text-xs text-[var(--muted-foreground)]">Total Boards</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-3xl font-bold tabular-nums">
                  {(avgScore * 100).toFixed(0)}%
                </p>
                <p className="text-xs text-[var(--muted-foreground)]">Average Score</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <p className="text-3xl font-bold tabular-nums">{productionReady}</p>
                <p className="text-xs text-[var(--muted-foreground)]">Production Ready</p>
              </CardContent>
            </Card>
          </section>
        )}

        {/* Board grid */}
        <section className="mb-12">
          {loading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-6 w-6 animate-spin text-[var(--muted-foreground)]" />
              <span className="ml-2 text-sm text-[var(--muted-foreground)]">
                Loading projects...
              </span>
            </div>
          )}

          {error && (
            <Card className="border-yellow-500/30 bg-yellow-500/5">
              <CardContent className="p-4 text-center text-sm text-yellow-400">
                Could not connect to API. Start the backend to see projects.
              </CardContent>
            </Card>
          )}

          {!loading && !error && boards.length === 0 && (
            <div className="rounded-lg border border-dashed border-[var(--border)] py-16 text-center">
              <p className="text-lg text-[var(--muted-foreground)]">No projects yet</p>
              <Link href="/new" className="mt-4 inline-block">
                <Button size="lg" className="gap-2">
                  <Plus className="h-4 w-4" aria-hidden="true" />
                  Create Your First Board
                </Button>
              </Link>
            </div>
          )}

          {!loading && !error && boards.length > 0 && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {boards.map((board) => (
                <Link key={board.name} href={`/project/${board.name}`}>
                  <Card className="group cursor-pointer transition-colors hover:border-[var(--primary)]">
                    {/* Board thumbnail */}
                    <div className="relative h-32 w-full overflow-hidden rounded-t-lg bg-[var(--muted)]">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`/api/boards/${board.name}/${board.name}_3d_iso.png`}
                        alt={`${board.name} 3D preview`}
                        className="h-full w-full object-contain"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = "none";
                        }}
                      />
                    </div>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="truncate">{board.name}</CardTitle>
                        <Badge variant={gradeVariant(board.grade)}>
                          {board.grade}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-[var(--muted-foreground)]">
                          Stage: {board.stage}
                        </span>
                        {board.score != null && (
                          <span
                            className={
                              gradeColors[board.grade] ?? "text-gray-400"
                            }
                          >
                            {(board.score * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      <div className="mt-2 flex items-center gap-1 text-xs text-[var(--muted-foreground)] opacity-0 transition-opacity group-hover:opacity-100">
                        Open project
                        <ArrowRight className="h-3 w-3" aria-hidden="true" />
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          )}
        </section>

        {/* Feature highlights */}
        <section>
          <h2 className="mb-4 text-xl font-semibold">Key Features</h2>
          <div className="grid gap-4 sm:grid-cols-3">
            {FEATURES.map((feat) => {
              const Icon = feat.icon;
              return (
                <Card key={feat.title}>
                  <CardContent className="p-5">
                    <Icon
                      className="mb-3 h-8 w-8 text-blue-500"
                      aria-hidden="true"
                    />
                    <h3 className="font-semibold">{feat.title}</h3>
                    <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                      {feat.desc}
                    </p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </section>
      </main>
    </>
  );
}
