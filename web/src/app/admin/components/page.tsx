"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { NavHeader } from "@/components/layout/nav-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FootprintPreview } from "@/components/pcb/footprint-preview";
import { AIChat } from "@/components/chat/ai-chat";
import {
  listComponents,
  getComponentDetail,
  reviewComponent,
  triggerVerification,
  type ComponentSummary,
  type ComponentDetail,
  type CheckResult,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  Circle,
  AlertTriangle,
  Loader2,
  RotateCcw,
  Flag,
  ChevronRight,
  Info,
  Wrench,
  Zap,
  ImageOff,
  ExternalLink,
  FileText,
} from "lucide-react";

// --- Status helpers ---

type VerificationStatus = "verified" | "failed" | "pending" | "unknown";

function statusIcon(status: VerificationStatus) {
  switch (status) {
    case "verified":
      return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    case "failed":
      return <XCircle className="h-4 w-4 text-red-500" />;
    case "pending":
      return <Circle className="h-4 w-4 text-gray-400" />;
    default:
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
  }
}

function statusBadgeVariant(
  status: string
): "success" | "destructive" | "warning" | "outline" {
  if (status === "verified") return "success";
  if (status === "failed") return "destructive";
  if (status === "pending") return "warning";
  return "outline";
}

function severityColor(severity: string): string {
  switch (severity) {
    case "critical":
      return "text-red-500";
    case "major":
      return "text-orange-500";
    case "minor":
      return "text-yellow-500";
    case "info":
      return "text-blue-400";
    default:
      return "text-gray-400";
  }
}

function severityBadgeVariant(
  severity: string
): "destructive" | "warning" | "info" | "outline" {
  if (severity === "critical") return "destructive";
  if (severity === "major" || severity === "minor") return "warning";
  if (severity === "info") return "info";
  return "outline";
}

// --- Filter types ---

type FilterStatus = "all" | VerificationStatus;

// --- Dialog component ---

function Dialog({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md rounded-lg border border-[var(--border)] bg-[var(--card)] p-6 shadow-xl">
        <h3 className="mb-4 text-lg font-semibold">{title}</h3>
        {children}
        <div className="mt-4 flex justify-end">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}

// --- Main page ---

export default function ComponentReviewPage() {
  // Data state
  const [components, setComponents] = useState<ComponentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ComponentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [filter, setFilter] = useState<FilterStatus>("all");
  const [search, setSearch] = useState("");

  // Dialog state
  const [reworkOpen, setReworkOpen] = useState(false);
  const [flagOpen, setFlagOpen] = useState(false);
  const [dialogNotes, setDialogNotes] = useState("");
  const [dialogIssue, setDialogIssue] = useState("");

  // Image tab state
  const [imageTab, setImageTab] = useState<string>("");

  // Refs
  const listRef = useRef<HTMLDivElement>(null);

  // Load component list
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listComponents()
      .then((data) => {
        if (!cancelled) {
          setComponents(data);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Failed to load components"
          );
          setComponents([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Filtered components
  const filtered = components.filter((c) => {
    if (filter !== "all" && c.verification_status !== filter) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        c.component_id.toLowerCase().includes(q) ||
        c.description.toLowerCase().includes(q) ||
        c.footprint_id.toLowerCase().includes(q)
      );
    }
    return true;
  });

  // Load detail when selection changes
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    getComponentDetail(selectedId)
      .then((data) => {
        if (!cancelled) {
          setDetail(data);
          // Set default image tab
          const keys = Object.keys(data.images);
          if (keys.length > 0 && !keys.includes(imageTab)) {
            setImageTab(keys[0]);
          }
        }
      })
      .catch(() => {
        if (!cancelled) setDetail(null);
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // imageTab intentionally excluded to avoid refetch loop
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  // Stats
  const stats = {
    verified: components.filter((c) => c.verification_status === "verified")
      .length,
    failed: components.filter((c) => c.verification_status === "failed").length,
    pending: components.filter((c) => c.verification_status === "pending")
      .length,
    total: components.length,
  };

  // Navigate to next unreviewed
  const advanceToNextUnreviewed = useCallback(() => {
    const unreviewed = filtered.find(
      (c) =>
        c.verification_status !== "verified" && c.component_id !== selectedId
    );
    if (unreviewed) {
      setSelectedId(unreviewed.component_id);
    }
  }, [filtered, selectedId]);

  // Action handlers
  const handleApprove = useCallback(async () => {
    if (!selectedId) return;
    setActionLoading(true);
    try {
      const result = await reviewComponent(selectedId, "approve");
      // Update local state
      setComponents((prev) =>
        prev.map((c) =>
          c.component_id === selectedId
            ? {
                ...c,
                verification_status:
                  result.verification_status as VerificationStatus,
              }
            : c
        )
      );
      setDetail((prev) =>
        prev
          ? { ...prev, verification_status: result.verification_status }
          : prev
      );
      advanceToNextUnreviewed();
    } catch {
      // Error handling — could add toast later
    } finally {
      setActionLoading(false);
    }
  }, [selectedId, advanceToNextUnreviewed]);

  const handleRework = useCallback(async () => {
    if (!selectedId) return;
    setActionLoading(true);
    try {
      const result = await reviewComponent(
        selectedId,
        "rework",
        dialogNotes
      );
      setComponents((prev) =>
        prev.map((c) =>
          c.component_id === selectedId
            ? {
                ...c,
                verification_status:
                  result.verification_status as VerificationStatus,
              }
            : c
        )
      );
      setDetail((prev) =>
        prev
          ? { ...prev, verification_status: result.verification_status }
          : prev
      );
      setReworkOpen(false);
      setDialogNotes("");
      advanceToNextUnreviewed();
    } catch {
      // Error handling
    } finally {
      setActionLoading(false);
    }
  }, [selectedId, dialogNotes, advanceToNextUnreviewed]);

  const handleFlagIssue = useCallback(async () => {
    if (!selectedId) return;
    setActionLoading(true);
    try {
      await reviewComponent(
        selectedId,
        "flag_issue",
        "",
        dialogIssue
      );
      // Reload detail to reflect new issue
      const updated = await getComponentDetail(selectedId);
      setDetail(updated);
      setComponents((prev) =>
        prev.map((c) =>
          c.component_id === selectedId
            ? { ...c, known_issues_count: updated.known_issues.length }
            : c
        )
      );
      setFlagOpen(false);
      setDialogIssue("");
    } catch {
      // Error handling
    } finally {
      setActionLoading(false);
    }
  }, [selectedId, dialogIssue]);

  const handleReVerify = useCallback(async () => {
    if (!selectedId) return;
    setActionLoading(true);
    try {
      const result = await triggerVerification(selectedId);
      // Reload detail with new checks
      const updated = await getComponentDetail(selectedId);
      setDetail(updated);
      setComponents((prev) =>
        prev.map((c) =>
          c.component_id === selectedId
            ? {
                ...c,
                verification_status: result.passed
                  ? ("verified" as VerificationStatus)
                  : ("failed" as VerificationStatus),
              }
            : c
        )
      );
    } catch {
      // Error handling
    } finally {
      setActionLoading(false);
    }
  }, [selectedId]);

  // Keyboard navigation
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Skip if typing in an input/textarea
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const currentIndex = filtered.findIndex(
          (c) => c.component_id === selectedId
        );
        let nextIndex: number;
        if (e.key === "ArrowUp") {
          nextIndex = currentIndex <= 0 ? filtered.length - 1 : currentIndex - 1;
        } else {
          nextIndex = currentIndex >= filtered.length - 1 ? 0 : currentIndex + 1;
        }
        if (filtered[nextIndex]) {
          setSelectedId(filtered[nextIndex].component_id);
        }
      }

      if (e.key === "Enter" && selectedId && !actionLoading) {
        e.preventDefault();
        handleApprove();
      }

      if (
        (e.key === "r" || e.key === "R") &&
        selectedId &&
        !actionLoading
      ) {
        e.preventDefault();
        setReworkOpen(true);
      }

      if (
        (e.key === "f" || e.key === "F") &&
        selectedId &&
        !actionLoading
      ) {
        e.preventDefault();
        setFlagOpen(true);
      }

      if (
        (e.key === "n" || e.key === "N") &&
        !actionLoading
      ) {
        e.preventDefault();
        advanceToNextUnreviewed();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    filtered,
    selectedId,
    actionLoading,
    handleApprove,
    advanceToNextUnreviewed,
  ]);

  return (
    <>
      <NavHeader />

      <main className="mx-auto max-w-7xl px-4 py-6">
        {/* Page header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold tracking-tight">
            Admin: Component Review
          </h1>
          <p className="mt-1 text-sm text-[var(--muted-foreground)]">
            Review and certify components one at a time. Use arrow keys to
            navigate, Enter to approve, R for rework, F to flag.
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <Card className="mb-6 border-yellow-500/30 bg-yellow-500/5">
            <CardContent className="p-4 text-center text-sm text-yellow-400">
              {error}. Start the backend to load components.
            </CardContent>
          </Card>
        )}

        {/* Loading state */}
        {loading && (
          <div className="flex items-center justify-center py-24">
            <Loader2 className="h-6 w-6 animate-spin text-[var(--muted-foreground)]" />
            <span className="ml-2 text-sm text-[var(--muted-foreground)]">
              Loading components...
            </span>
          </div>
        )}

        {/* Main layout */}
        {!loading && !error && (
          <div className="flex gap-6" style={{ minHeight: "calc(100vh - 200px)" }}>
            {/* Left: Component list */}
            <div className="w-[300px] flex-shrink-0">
              <Card className="flex h-full flex-col">
                {/* Filters */}
                <div className="border-b border-[var(--border)] p-3 space-y-2">
                  <select
                    value={filter}
                    onChange={(e) => setFilter(e.target.value as FilterStatus)}
                    className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
                  >
                    <option value="all">All ({stats.total})</option>
                    <option value="verified">
                      Verified ({stats.verified})
                    </option>
                    <option value="failed">Failed ({stats.failed})</option>
                    <option value="pending">Pending ({stats.pending})</option>
                  </select>
                  <input
                    type="text"
                    placeholder="Search components..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1.5 text-sm placeholder:text-[var(--muted-foreground)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
                  />
                </div>

                {/* Scrollable list */}
                <div
                  ref={listRef}
                  className="flex-1 overflow-y-auto"
                  role="listbox"
                  aria-label="Component list"
                >
                  {filtered.length === 0 && (
                    <div className="p-4 text-center text-sm text-[var(--muted-foreground)]">
                      No components match filters.
                    </div>
                  )}
                  {filtered.map((comp) => {
                    const isSelected = comp.component_id === selectedId;
                    return (
                      <button
                        key={comp.component_id}
                        onClick={() => setSelectedId(comp.component_id)}
                        role="option"
                        aria-selected={isSelected}
                        className={cn(
                          "flex w-full items-center gap-2 border-b border-[var(--border)] px-3 py-2.5 text-left text-sm transition-colors hover:bg-[var(--muted)]",
                          isSelected &&
                            "border-l-2 border-l-blue-500 bg-blue-500/10"
                        )}
                      >
                        {statusIcon(comp.verification_status)}
                        <div className="min-w-0 flex-1">
                          <div className="truncate font-medium">
                            {comp.component_id}
                          </div>
                          <div className="truncate text-xs text-[var(--muted-foreground)]">
                            {comp.description}
                          </div>
                        </div>
                        {comp.known_issues_count > 0 && (
                          <span className="flex-shrink-0 rounded-full bg-red-500/15 px-1.5 py-0.5 text-[10px] font-medium text-red-500">
                            {comp.known_issues_count}
                          </span>
                        )}
                        {isSelected && (
                          <ChevronRight className="h-3 w-3 flex-shrink-0 text-blue-500" />
                        )}
                      </button>
                    );
                  })}
                </div>

                {/* Stats bar */}
                <div className="border-t border-[var(--border)] px-3 py-2 text-xs text-[var(--muted-foreground)]">
                  <div>
                    {stats.verified}/{stats.total} verified
                  </div>
                  {stats.failed > 0 && (
                    <div className="text-red-400">{stats.failed} failed</div>
                  )}
                  {stats.pending > 0 && (
                    <div className="text-yellow-400">
                      {stats.pending} pending
                    </div>
                  )}
                </div>
              </Card>
            </div>

            {/* Right: Detail panel */}
            <div className="flex-1 min-w-0">
              {!selectedId && (
                <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-[var(--border)] p-12">
                  <div className="text-center">
                    <Info className="mx-auto mb-3 h-8 w-8 text-[var(--muted-foreground)]" />
                    <p className="text-lg font-medium">
                      Select a component to begin review
                    </p>
                    <p className="mt-2 text-sm text-[var(--muted-foreground)]">
                      Use up/down keys to navigate, Enter to approve, R for rework.
                    </p>
                  </div>
                </div>
              )}

              {selectedId && detailLoading && (
                <div className="flex h-full items-center justify-center">
                  <Loader2 className="h-6 w-6 animate-spin text-[var(--muted-foreground)]" />
                  <span className="ml-2 text-sm text-[var(--muted-foreground)]">
                    Loading component detail...
                  </span>
                </div>
              )}

              {selectedId && !detailLoading && detail && (
                <div className="space-y-4 overflow-y-auto" style={{ maxHeight: "calc(100vh - 200px)" }}>
                  {/* Header */}
                  <Card>
                    <CardContent className="flex items-start gap-4 p-4">
                      <FootprintPreview
                        packageType={detail.footprint_id}
                        width={80}
                        height={80}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <h2 className="text-xl font-bold">
                            {detail.component_id}
                          </h2>
                          <Badge
                            variant={statusBadgeVariant(
                              detail.verification_status
                            )}
                          >
                            {detail.verification_status}
                          </Badge>
                        </div>
                        <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                          {detail.description}
                        </p>
                        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--muted-foreground)]">
                          <span>
                            Footprint: <strong>{detail.footprint_id}</strong>
                          </span>
                          <span>
                            Pads: <strong>{detail.expected_pads}</strong> (
                            {detail.expected_pad_type})
                          </span>
                          <span>
                            Body: <strong>{detail.body_width_mm}</strong> x{" "}
                            <strong>{detail.body_height_mm}</strong> mm
                          </span>
                          {detail.last_verified_commit && (
                            <span>
                              Last verified:{" "}
                              <code className="rounded bg-[var(--muted)] px-1">
                                {detail.last_verified_commit.slice(0, 7)}
                              </code>
                            </span>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Full Metadata */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Component Metadata</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
                        <div>
                          <span className="text-[var(--muted-foreground)]">Category</span>
                          <p className="font-medium">{detail.package_category || "—"}</p>
                        </div>
                        <div>
                          <span className="text-[var(--muted-foreground)]">Pad Type</span>
                          <p className="font-medium">{detail.expected_pad_type}</p>
                        </div>
                        <div>
                          <span className="text-[var(--muted-foreground)]">3D Model Rotation (Z)</span>
                          <p className="font-medium font-technical">{detail.model_rotation_z}°</p>
                        </div>
                        <div>
                          <span className="text-[var(--muted-foreground)]">Max XY Offset</span>
                          <p className="font-medium font-technical">{detail.model_offset_xy_max_mm} mm</p>
                        </div>
                        <div>
                          <span className="text-[var(--muted-foreground)]">KiCad Footprint</span>
                          <p className="font-medium font-technical text-[10px]">{detail.kicad_footprint_lib || "—"}</p>
                        </div>
                        <div>
                          <span className="text-[var(--muted-foreground)]">3D Model</span>
                          <p className="font-medium font-technical text-[10px] truncate" title={detail.model_3d_path}>{detail.model_3d_path || "—"}</p>
                        </div>
                      </div>

                      {/* Links */}
                      <div className="mt-3 flex flex-wrap gap-2">
                        {detail.lcsc_url && (
                          <a
                            href={detail.lcsc_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 rounded bg-blue-500/10 px-2 py-1 text-[10px] font-medium text-blue-400 hover:bg-blue-500/20"
                          >
                            <ExternalLink className="h-3 w-3" />
                            LCSC
                          </a>
                        )}
                        {detail.datasheet_url && (
                          <a
                            href={detail.datasheet_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 rounded bg-green-500/10 px-2 py-1 text-[10px] font-medium text-green-400 hover:bg-green-500/20"
                          >
                            <FileText className="h-3 w-3" />
                            Datasheet
                          </a>
                        )}
                        <a
                          href={`https://www.snapeda.com/search/?q=${encodeURIComponent(detail.footprint_id)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 rounded bg-purple-500/10 px-2 py-1 text-[10px] font-medium text-purple-400 hover:bg-purple-500/20"
                        >
                          <ExternalLink className="h-3 w-3" />
                          SnapEDA
                        </a>
                      </div>

                      {/* Pins table */}
                      {detail.pins && detail.pins.length > 0 && (
                        <div className="mt-3">
                          <p className="mb-1 text-xs font-medium text-[var(--muted-foreground)]">
                            Pin Map ({detail.pins.length} pins)
                          </p>
                          <div className="max-h-[120px] overflow-y-auto rounded border border-[var(--border)]">
                            <table className="w-full text-[10px]">
                              <thead className="bg-[var(--muted)] sticky top-0">
                                <tr>
                                  <th className="px-2 py-1 text-left">#</th>
                                  <th className="px-2 py-1 text-left">Name</th>
                                  <th className="px-2 py-1 text-left">Type</th>
                                </tr>
                              </thead>
                              <tbody>
                                {detail.pins.map((pin, i) => (
                                  <tr key={i} className="border-t border-[var(--border)]">
                                    <td className="px-2 py-0.5 font-technical">{pin.number}</td>
                                    <td className="px-2 py-0.5">{pin.name || "—"}</td>
                                    <td className="px-2 py-0.5 text-[var(--muted-foreground)]">{pin.type || "—"}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Evidence images */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Evidence Images</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {Object.keys(detail.images).length === 0 ? (
                        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-[var(--border)] py-8 text-[var(--muted-foreground)]">
                          <ImageOff className="mb-2 h-8 w-8" />
                          <p className="text-sm">No evidence images.</p>
                          <p className="text-xs">
                            Click Re-verify to generate.
                          </p>
                        </div>
                      ) : (
                        <>
                          {/* Tab bar */}
                          <div className="mb-3 flex gap-1 border-b border-[var(--border)]">
                            {Object.keys(detail.images).map((key) => (
                              <button
                                key={key}
                                onClick={() => setImageTab(key)}
                                className={cn(
                                  "px-3 py-1.5 text-xs font-medium transition-colors",
                                  imageTab === key
                                    ? "border-b-2 border-blue-500 text-white"
                                    : "text-[var(--muted-foreground)] hover:text-white"
                                )}
                              >
                                {key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                              </button>
                            ))}
                          </div>
                          {/* Image display */}
                          {imageTab && detail.images[imageTab] && (
                            <div className="flex justify-center rounded-lg bg-[var(--muted)] p-2">
                              {/* eslint-disable-next-line @next/next/no-img-element */}
                              <img
                                src={detail.images[imageTab]}
                                alt={`${detail.component_id} ${imageTab}`}
                                className="max-h-64 rounded object-contain"
                              />
                            </div>
                          )}
                        </>
                      )}
                    </CardContent>
                  </Card>

                  {/* Structural checks */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">
                        Structural Checks
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {detail.checks.length === 0 ? (
                        <p className="text-sm text-[var(--muted-foreground)]">
                          No checks run yet. Click Re-verify.
                        </p>
                      ) : (
                        <ul className="space-y-1.5">
                          {detail.checks.map((check: CheckResult) => (
                            <li
                              key={check.name}
                              className="flex items-center gap-2 text-sm"
                            >
                              {check.passed ? (
                                <CheckCircle2 className="h-4 w-4 flex-shrink-0 text-green-500" />
                              ) : (
                                <XCircle
                                  className={cn(
                                    "h-4 w-4 flex-shrink-0",
                                    severityColor(check.severity)
                                  )}
                                />
                              )}
                              <span className="font-mono text-xs">
                                {check.name}
                              </span>
                              <span className="text-[var(--muted-foreground)]">
                                {check.detail}
                              </span>
                              {!check.passed && (
                                <Badge
                                  variant={severityBadgeVariant(
                                    check.severity
                                  )}
                                  className="ml-auto text-[10px]"
                                >
                                  {check.severity}
                                </Badge>
                              )}
                            </li>
                          ))}
                        </ul>
                      )}
                    </CardContent>
                  </Card>

                  {/* Dual-persona review summary */}
                  {detail.checks.length > 0 && (
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">
                          Dual-Persona Review
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-1.5 text-sm">
                        <div className="flex items-center gap-2">
                          <Wrench className="h-4 w-4 text-orange-400" />
                          <span className="font-medium text-orange-400">
                            Fab:
                          </span>
                          <span className="text-[var(--muted-foreground)]">
                            {detail.checks.every((c) => c.passed)
                              ? "Body alignment and pad geometry OK"
                              : `${detail.checks.filter((c) => !c.passed).length} issue(s) found`}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-yellow-400" />
                          <span className="font-medium text-yellow-400">
                            EE:
                          </span>
                          <span className="text-[var(--muted-foreground)]">
                            {detail.checks.every((c) => c.passed)
                              ? "Pad count and type correct"
                              : `Review structural check failures`}
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Known issues */}
                  {detail.known_issues.length > 0 && (
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">
                          Known Issues ({detail.known_issues.length})
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-2">
                          {detail.known_issues.map((issue, i) => (
                            <li
                              key={i}
                              className="flex items-start gap-2 text-sm"
                            >
                              <Badge
                                variant={severityBadgeVariant(issue.severity)}
                                className="mt-0.5 text-[10px]"
                              >
                                {issue.severity}
                              </Badge>
                              <div>
                                <p>{issue.description}</p>
                                {issue.date && (
                                  <p className="text-xs text-[var(--muted-foreground)]">
                                    {issue.date}
                                  </p>
                                )}
                              </div>
                              <Badge
                                variant="outline"
                                className="ml-auto text-[10px]"
                              >
                                {issue.status}
                              </Badge>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}

                  {/* Verified fixes (collapsible) */}
                  {detail.verified_fixes.length > 0 && (
                    <details className="group">
                      <summary className="cursor-pointer rounded-lg border border-[var(--border)] bg-[var(--card)] px-4 py-2 text-sm font-medium">
                        Verified Fixes ({detail.verified_fixes.length})
                      </summary>
                      <Card className="mt-1">
                        <CardContent className="p-4">
                          <ul className="space-y-1.5">
                            {detail.verified_fixes.map((fix, i) => (
                              <li
                                key={i}
                                className="flex items-center gap-2 text-sm"
                              >
                                <CheckCircle2 className="h-3.5 w-3.5 flex-shrink-0 text-green-500" />
                                <span>{fix.description}</span>
                                {fix.date && (
                                  <span className="ml-auto text-xs text-[var(--muted-foreground)]">
                                    {fix.date}
                                  </span>
                                )}
                              </li>
                            ))}
                          </ul>
                        </CardContent>
                      </Card>
                    </details>
                  )}

                  {/* Actions */}
                  <Card>
                    <CardContent className="flex flex-wrap gap-2 p-4">
                      <Button
                        variant="success"
                        onClick={handleApprove}
                        disabled={actionLoading}
                        loading={actionLoading}
                        className="gap-1.5"
                      >
                        <CheckCircle2 className="h-4 w-4" />
                        Approve
                      </Button>
                      <Button
                        variant="destructive"
                        onClick={() => setReworkOpen(true)}
                        disabled={actionLoading}
                        className="gap-1.5"
                      >
                        <XCircle className="h-4 w-4" />
                        Send for Rework
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => setFlagOpen(true)}
                        disabled={actionLoading}
                        className="gap-1.5 border-orange-500/30 text-orange-400 hover:bg-orange-500/10"
                      >
                        <Flag className="h-4 w-4" />
                        Flag Issue
                      </Button>
                      <Button
                        variant="outline"
                        onClick={handleReVerify}
                        disabled={actionLoading}
                        loading={actionLoading}
                        className="gap-1.5"
                      >
                        <RotateCcw className="h-4 w-4" />
                        Re-verify
                      </Button>
                    </CardContent>
                  </Card>

                  {/* AI Chat */}
                  <AIChat
                    compact={false}
                    welcomeMessage={`I'm reviewing ${detail.component_id} (${detail.description}). What would you like to know about this component?`}
                    systemContext={[
                      `Component: ${detail.component_id}`,
                      `Footprint: ${detail.footprint_id}`,
                      `Pads: ${detail.expected_pads} (${detail.expected_pad_type})`,
                      `Body: ${detail.body_width_mm}x${detail.body_height_mm}mm`,
                      `Status: ${detail.verification_status}`,
                      `Checks: ${detail.checks.map((c) => `${c.name}=${c.passed ? "OK" : "FAIL"}`).join(", ")}`,
                      `Known issues: ${detail.known_issues.map((i) => i.description).join("; ") || "none"}`,
                    ].join("\n")}
                    placeholder="Ask about this component..."
                    className="h-[350px]"
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Rework dialog */}
      <Dialog
        open={reworkOpen}
        onClose={() => setReworkOpen(false)}
        title="Send for Rework"
      >
        <p className="mb-3 text-sm text-[var(--muted-foreground)]">
          Describe what needs to be fixed for {selectedId}:
        </p>
        <textarea
          value={dialogNotes}
          onChange={(e) => setDialogNotes(e.target.value)}
          rows={4}
          placeholder="Describe rework requirements..."
          className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm placeholder:text-[var(--muted-foreground)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
          autoFocus
        />
        <div className="mt-4 flex justify-end gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setReworkOpen(false)}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            size="sm"
            onClick={handleRework}
            disabled={actionLoading || !dialogNotes.trim()}
            loading={actionLoading}
          >
            Send for Rework
          </Button>
        </div>
      </Dialog>

      {/* Flag issue dialog */}
      <Dialog
        open={flagOpen}
        onClose={() => setFlagOpen(false)}
        title="Flag Issue"
      >
        <p className="mb-3 text-sm text-[var(--muted-foreground)]">
          Describe the issue with {selectedId}:
        </p>
        <textarea
          value={dialogIssue}
          onChange={(e) => setDialogIssue(e.target.value)}
          rows={4}
          placeholder="Describe the issue..."
          className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm placeholder:text-[var(--muted-foreground)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
          autoFocus
        />
        <div className="mt-4 flex justify-end gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setFlagOpen(false)}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={handleFlagIssue}
            disabled={actionLoading || !dialogIssue.trim()}
            loading={actionLoading}
            className="border-orange-500/30 bg-orange-500/15 text-orange-400 hover:bg-orange-500/25"
          >
            Flag Issue
          </Button>
        </div>
      </Dialog>
    </>
  );
}
