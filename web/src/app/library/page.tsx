"use client";

import { useState, useCallback, useEffect, useRef, useMemo, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { NavHeader } from "@/components/layout/nav-header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FootprintPreview } from "@/components/pcb/footprint-preview";
import { ComponentDetail } from "@/components/library/component-detail";
import {
  searchParts,
  listComponents,
  listBoards,
  getBoardDetail,
  type PartResult,
  type ComponentSummary,
  type BoardSummary,
  type BoardDetail,
} from "@/lib/api";
import {
  Search,
  ExternalLink,
  Package,
  Check,
  Loader2,
  SlidersHorizontal,
  FileDown,
  ArrowUpDown,
  ChevronDown,
  Eye,
  Plus,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types & Constants
// ---------------------------------------------------------------------------

type Category =
  | "all"
  | "resistors"
  | "capacitors"
  | "inductors"
  | "ics"
  | "connectors"
  | "diodes"
  | "leds"
  | "transistors"
  | "crystals"
  | "modules";

type PackageFilter =
  | "all"
  | "0402"
  | "0603"
  | "0805"
  | "1206"
  | "SOT-23"
  | "SOIC-8"
  | "QFP"
  | "QFN"
  | "Through-Hole";

type SortMode = "relevance" | "price_asc" | "price_desc" | "stock_desc";

const CATEGORIES: Array<{ value: Category; label: string }> = [
  { value: "all", label: "All Categories" },
  { value: "resistors", label: "Resistors" },
  { value: "capacitors", label: "Capacitors" },
  { value: "inductors", label: "Inductors" },
  { value: "ics", label: "ICs" },
  { value: "connectors", label: "Connectors" },
  { value: "diodes", label: "Diodes" },
  { value: "leds", label: "LEDs" },
  { value: "transistors", label: "Transistors" },
  { value: "crystals", label: "Crystals" },
  { value: "modules", label: "Modules" },
];

const PACKAGES: Array<{ value: PackageFilter; label: string }> = [
  { value: "all", label: "All Packages" },
  { value: "0402", label: "0402" },
  { value: "0603", label: "0603" },
  { value: "0805", label: "0805" },
  { value: "1206", label: "1206" },
  { value: "SOT-23", label: "SOT-23" },
  { value: "SOIC-8", label: "SOIC-8" },
  { value: "QFP", label: "QFP" },
  { value: "QFN", label: "QFN" },
  { value: "Through-Hole", label: "Through-Hole" },
];

const SORT_OPTIONS: Array<{ value: SortMode; label: string }> = [
  { value: "relevance", label: "Relevance" },
  { value: "price_asc", label: "Price Low \u2192 High" },
  { value: "price_desc", label: "Price High \u2192 Low" },
  { value: "stock_desc", label: "Stock High \u2192 Low" },
];

const CATEGORY_KEYWORDS: Record<Category, string[]> = {
  all: [],
  resistors: ["resistor", "res", "ohm", "\u03A9"],
  capacitors: ["capacitor", "cap", "farad", "pf", "nf", "uf"],
  inductors: ["inductor", "ferrite", "choke", "henry"],
  ics: ["ic", "mcu", "adc", "dac", "op-amp", "regulator", "driver"],
  connectors: ["connector", "header", "terminal", "jack", "socket", "plug"],
  diodes: ["diode", "zener", "schottky", "tvs"],
  leds: ["led", "light"],
  transistors: ["mosfet", "transistor", "bjt", "fet", "jfet"],
  crystals: ["crystal", "oscillator", "resonator"],
  modules: ["module", "esp32", "wifi", "bluetooth"],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function stockColor(stock: number): string {
  if (stock >= 10000) return "text-green-400";
  if (stock >= 1000) return "text-yellow-400";
  return "text-red-400";
}

function stockDot(stock: number): string {
  if (stock >= 10000) return "bg-green-500";
  if (stock >= 1000) return "bg-yellow-500";
  return "bg-red-500";
}

function matchesCategory(part: PartResult, category: Category): boolean {
  if (category === "all") return true;
  const kws = CATEGORY_KEYWORDS[category];
  const text = `${part.description} ${part.mfr} ${part.package}`.toLowerCase();
  return kws.some((kw) => text.includes(kw));
}

function matchesPackage(part: PartResult, pkg: PackageFilter): boolean {
  if (pkg === "all") return true;
  const p = part.package.toLowerCase();
  const f = pkg.toLowerCase();
  if (f === "through-hole") return p.includes("dip") || p.includes("th") || p.includes("through");
  return p.includes(f);
}

function sortResults(results: PartResult[], mode: SortMode): PartResult[] {
  if (mode === "relevance") return results;
  const sorted = [...results];
  switch (mode) {
    case "price_asc":
      sorted.sort((a, b) => a.price - b.price);
      break;
    case "price_desc":
      sorted.sort((a, b) => b.price - a.price);
      break;
    case "stock_desc":
      sorted.sort((a, b) => b.stock - a.stock);
      break;
  }
  return sorted;
}

// ---------------------------------------------------------------------------
// Dropdown component
// ---------------------------------------------------------------------------

function Dropdown<T extends string>({
  value,
  onChange,
  options,
  icon,
}: {
  value: T;
  onChange: (v: T) => void;
  options: Array<{ value: T; label: string }>;
  icon?: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const selected = options.find((o) => o.value === value);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-1.5 text-sm text-[var(--foreground)] hover:border-gray-500 transition-colors"
      >
        {icon}
        <span className="max-w-[120px] truncate">{selected?.label}</span>
        <ChevronDown className="h-3.5 w-3.5 text-[var(--muted-foreground)]" aria-hidden="true" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-30 mt-1 max-h-60 min-w-[160px] overflow-y-auto rounded-lg border border-[var(--border)] bg-gray-900 py-1 shadow-xl">
          {options.map((opt) => (
            <button
              key={opt.value}
              onClick={() => {
                onChange(opt.value);
                setOpen(false);
              }}
              className={
                "flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm transition-colors " +
                (opt.value === value
                  ? "bg-blue-500/10 text-blue-400"
                  : "text-[var(--foreground)] hover:bg-white/5")
              }
            >
              {opt.value === value && <Check className="h-3 w-3" />}
              <span className={opt.value === value ? "" : "ml-5"}>{opt.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component Card
// ---------------------------------------------------------------------------

function ComponentCard({
  part,
  onViewDetail,
}: {
  part: PartResult;
  onViewDetail: (p: PartResult) => void;
}) {
  return (
    <Card className="group flex flex-col transition-colors hover:border-gray-500">
      <CardContent className="flex flex-1 flex-col p-4">
        {/* Top row: package preview + badges */}
        <div className="mb-3 flex items-start gap-3">
          <div className="shrink-0 rounded-md border border-[var(--border)] bg-gray-900 p-1.5">
            <FootprintPreview packageType={part.package} width={48} height={48} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <a
                href={`https://www.lcsc.com/product-detail/${part.lcsc}.html`}
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-sm font-bold text-blue-400 hover:text-blue-300 hover:underline"
              >
                {part.lcsc}
              </a>
              {part.basic ? (
                <Badge variant="success" className="text-[10px]">
                  Basic
                </Badge>
              ) : (
                <Badge variant="warning" className="text-[10px]">
                  Extended
                </Badge>
              )}
            </div>
            <p className="mt-0.5 text-xs text-[var(--muted-foreground)] truncate">{part.mfr}</p>
          </div>
        </div>

        {/* Description */}
        <p className="mb-auto text-sm font-medium leading-snug line-clamp-2">{part.description}</p>

        {/* Package + Price row */}
        <div className="mt-3 flex items-center justify-between">
          <Badge variant="outline" className="text-[10px] font-mono">
            {part.package}
          </Badge>
          <span className="text-sm font-semibold tabular-nums">${part.price.toFixed(4)}/pc</span>
        </div>

        {/* Stock row */}
        <div className="mt-2 flex items-center justify-between text-xs">
          <div className="flex items-center gap-1.5">
            <span className={`inline-block h-2 w-2 rounded-full ${stockDot(part.stock)}`} />
            <span className={stockColor(part.stock)}>
              {part.stock.toLocaleString()} in stock
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="mt-3 flex items-center gap-1.5 border-t border-[var(--border)] pt-3">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1 px-2 text-xs"
            onClick={() => onViewDetail(part)}
          >
            <Eye className="h-3 w-3" aria-hidden="true" />
            Details
          </Button>
          <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs">
            <Plus className="h-3 w-3" aria-hidden="true" />
            Add
          </Button>
          <a
            href={`https://jlcpcb.com/partdetail/${part.lcsc}`}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-auto"
          >
            <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs">
              <FileDown className="h-3 w-3" aria-hidden="true" />
              Datasheet
            </Button>
          </a>
          <a
            href={`https://www.lcsc.com/product-detail/${part.lcsc}.html`}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Button variant="ghost" size="sm" className="h-7 px-1.5">
              <ExternalLink className="h-3 w-3" aria-hidden="true" />
            </Button>
          </a>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function LibraryPage() {
  return (
    <Suspense fallback={<div className="flex h-screen items-center justify-center"><span className="text-sm text-gray-500">Loading...</span></div>}>
      <LibraryPageInner />
    </Suspense>
  );
}

function LibraryPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Initialize from URL params
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [category, setCategory] = useState<Category>(
    (searchParams.get("cat") as Category) || "all"
  );
  const [packageFilter, setPackageFilter] = useState<PackageFilter>(
    (searchParams.get("pkg") as PackageFilter) || "all"
  );
  const [inStock, setInStock] = useState(searchParams.get("stock") !== "0");
  const [basicOnly, setBasicOnly] = useState(searchParams.get("basic") === "1");
  const [sortMode, setSortMode] = useState<SortMode>(
    (searchParams.get("sort") as SortMode) || "relevance"
  );

  const [rawResults, setRawResults] = useState<PartResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const [detailPart, setDetailPart] = useState<PartResult | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync filters to URL
  useEffect(() => {
    const params = new URLSearchParams();
    if (query) params.set("q", query);
    if (category !== "all") params.set("cat", category);
    if (packageFilter !== "all") params.set("pkg", packageFilter);
    if (!inStock) params.set("stock", "0");
    if (basicOnly) params.set("basic", "1");
    if (sortMode !== "relevance") params.set("sort", sortMode);
    const qs = params.toString();
    const newUrl = qs ? `/library?${qs}` : "/library";
    router.replace(newUrl, { scroll: false });
  }, [query, category, packageFilter, inStock, basicOnly, sortMode, router]);

  // Build the actual search query with category prefix
  const effectiveQuery = useMemo(() => {
    if (category === "all") return query;
    const catLabel = CATEGORIES.find((c) => c.value === category)?.label.toLowerCase() || "";
    if (query.toLowerCase().includes(catLabel)) return query;
    return `${catLabel} ${query}`.trim();
  }, [query, category]);

  const doSearch = useCallback(
    async (q: string) => {
      if (q.trim().length < 2) {
        setRawResults([]);
        setTotal(0);
        setSearched(false);
        return;
      }

      setLoading(true);
      setError(null);
      setSearched(true);

      try {
        const resp = await searchParts(q.trim(), {
          basicOnly,
          inStock,
          limit: 60,
        });
        setRawResults(resp.results);
        setTotal(resp.total);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Search failed");
        setRawResults([]);
        setTotal(0);
      } finally {
        setLoading(false);
      }
    },
    [basicOnly, inStock]
  );

  // Apply client-side filters (category, package, sort)
  const filteredResults = useMemo(() => {
    let results = rawResults;
    results = results.filter((r) => matchesCategory(r, category));
    results = results.filter((r) => matchesPackage(r, packageFilter));
    return sortResults(results, sortMode);
  }, [rawResults, category, packageFilter, sortMode]);

  // Debounced search on query/effectiveQuery change
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      doSearch(effectiveQuery);
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [effectiveQuery, doSearch]);

  // Re-search when server-side filters change
  useEffect(() => {
    if (effectiveQuery.trim().length >= 2) {
      doSearch(effectiveQuery);
    }
  }, [basicOnly, inStock]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        doSearch(effectiveQuery);
      }
    },
    [effectiveQuery, doSearch]
  );

  // --- Library tab state ---
  type LibTab = "framework" | "board" | "jlcpcb";
  const [libTab, setLibTab] = useState<LibTab>("framework");
  const [fwComponents, setFwComponents] = useState<ComponentSummary[]>([]);
  const [fwLoading, setFwLoading] = useState(false);
  const [fwSearch, setFwSearch] = useState("");
  const [fwFilter, setFwFilter] = useState<string>("all");
  const [boards, setBoards] = useState<BoardSummary[]>([]);
  const [selectedBoard, setSelectedBoard] = useState<string>("");
  const [boardDetail, setBoardDetail] = useState<BoardDetail | null>(null);
  const [boardLoading, setBoardLoading] = useState(false);

  // Load framework components on first switch to that tab
  useEffect(() => {
    if (libTab === "framework" && fwComponents.length === 0 && !fwLoading) {
      setFwLoading(true);
      listComponents()
        .then(setFwComponents)
        .catch(() => {})
        .finally(() => setFwLoading(false));
    }
  }, [libTab, fwComponents.length, fwLoading]);

  // Load boards list on first switch to board tab
  useEffect(() => {
    if (libTab === "board" && boards.length === 0) {
      listBoards().then(setBoards).catch(() => {});
    }
  }, [libTab, boards.length]);

  // Load board detail when a board is selected
  useEffect(() => {
    if (selectedBoard) {
      setBoardLoading(true);
      getBoardDetail(selectedBoard)
        .then(setBoardDetail)
        .catch(() => setBoardDetail(null))
        .finally(() => setBoardLoading(false));
    }
  }, [selectedBoard]);

  // Filtered framework components
  const filteredFw = useMemo(() => {
    let list = fwComponents;
    if (fwFilter !== "all") {
      list = list.filter((c) => c.verification_status === fwFilter);
    }
    if (fwSearch) {
      const q = fwSearch.toLowerCase();
      list = list.filter(
        (c) =>
          c.component_id.toLowerCase().includes(q) ||
          c.description.toLowerCase().includes(q) ||
          c.footprint_id.toLowerCase().includes(q)
      );
    }
    return list;
  }, [fwComponents, fwFilter, fwSearch]);

  return (
    <>
      <NavHeader />

      <main className="mx-auto max-w-7xl px-4 py-6">
        {/* Header + Tabs */}
        <div className="mb-4">
          <h1 className="text-2xl font-bold">Component Library</h1>
          <div className="mt-3 flex items-center gap-1 rounded-lg bg-[var(--muted)] p-0.5 w-fit">
            {([
              { key: "framework" as LibTab, label: "Framework Components", count: fwComponents.length },
              { key: "board" as LibTab, label: "Board Components", count: boardDetail?.components?.length },
              { key: "jlcpcb" as LibTab, label: "JLCPCB Catalog", count: undefined },
            ]).map(({ key, label, count }) => (
              <button
                key={key}
                onClick={() => setLibTab(key)}
                className={
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors " +
                  (libTab === key
                    ? "bg-[var(--card)] text-[var(--foreground)] shadow-sm"
                    : "text-[var(--muted-foreground)] hover:text-[var(--foreground)]")
                }
              >
                {label}
                {count != null && (
                  <span className="ml-1.5 rounded-full bg-[var(--border)] px-1.5 py-0.5 text-[10px]">
                    {count}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* --- FRAMEWORK COMPONENTS TAB --- */}
        {libTab === "framework" && (
          <div>
            {/* Search + filter */}
            <div className="mb-4 flex items-center gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--muted-foreground)]" />
                <input
                  type="text"
                  value={fwSearch}
                  onChange={(e) => setFwSearch(e.target.value)}
                  placeholder="Search components by ID, footprint, or description..."
                  className="w-full rounded-lg border border-[var(--border)] bg-[var(--background)] py-2 pl-10 pr-4 text-sm placeholder:text-[var(--muted-foreground)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
                />
              </div>
              <select
                value={fwFilter}
                onChange={(e) => setFwFilter(e.target.value)}
                className="rounded-lg border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"
              >
                <option value="all">All Status</option>
                <option value="verified">Verified</option>
                <option value="failed">Failed</option>
                <option value="pending">Pending</option>
              </select>
            </div>

            {fwLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-5 w-5 animate-spin text-[var(--muted-foreground)]" />
                <span className="ml-2 text-sm text-[var(--muted-foreground)]">Loading components...</span>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {filteredFw.map((comp) => (
                  <Card
                    key={comp.component_id}
                    className="group cursor-pointer transition-colors hover:border-gray-500"
                    onClick={() => router.push(`/admin/components?select=${comp.component_id}`)}
                  >
                    <CardContent className="flex items-start gap-3 p-4">
                      <div className="shrink-0 rounded-md border border-[var(--border)] bg-gray-900 p-1.5">
                        <FootprintPreview packageType={comp.footprint_id} width={48} height={48} />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm font-bold">{comp.component_id}</span>
                          <Badge
                            variant={comp.verification_status === "verified" ? "success" : comp.verification_status === "failed" ? "destructive" : "outline"}
                            className="text-[10px]"
                          >
                            {comp.verification_status}
                          </Badge>
                        </div>
                        <p className="mt-0.5 text-xs text-[var(--muted-foreground)] line-clamp-2">
                          {comp.description}
                        </p>
                        <div className="mt-1.5 flex items-center gap-3 text-[10px] text-[var(--muted-foreground)]">
                          <span>{comp.expected_pads} pads</span>
                          <span>{comp.footprint_id}</span>
                          {comp.known_issues_count > 0 && (
                            <span className="text-orange-400">{comp.known_issues_count} issues</span>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
            {!fwLoading && filteredFw.length === 0 && (
              <div className="flex flex-col items-center py-12 text-[var(--muted-foreground)]">
                <Package className="mb-2 h-8 w-8" />
                <p className="text-sm">No components match your search.</p>
              </div>
            )}
          </div>
        )}

        {/* --- BOARD COMPONENTS TAB --- */}
        {libTab === "board" && (
          <div>
            {/* Board selector */}
            <div className="mb-4 flex items-center gap-3">
              <select
                value={selectedBoard}
                onChange={(e) => setSelectedBoard(e.target.value)}
                className="rounded-lg border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"
              >
                <option value="">Select a board...</option>
                {boards
                  .filter((b) => b.has_evidence || b.name.startsWith("train_") || b.name.startsWith("eval_"))
                  .map((b) => (
                    <option key={b.name} value={b.name}>
                      {b.name} {b.grade ? `[${b.grade}]` : ""}
                    </option>
                  ))}
              </select>
              {boardDetail && (
                <span className="text-xs text-[var(--muted-foreground)]">
                  {boardDetail.components?.length || 0} components on this board
                </span>
              )}
            </div>

            {boardLoading && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-5 w-5 animate-spin text-[var(--muted-foreground)]" />
                <span className="ml-2 text-sm text-[var(--muted-foreground)]">Loading board components...</span>
              </div>
            )}

            {!boardLoading && !selectedBoard && (
              <div className="flex flex-col items-center py-12 text-[var(--muted-foreground)]">
                <Package className="mb-2 h-8 w-8" />
                <p className="text-sm">Select a board to see its components.</p>
              </div>
            )}

            {!boardLoading && boardDetail && boardDetail.components && (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {boardDetail.components.map((comp, i) => (
                  <Card key={`${comp.ref}-${i}`} className="transition-colors hover:border-gray-500">
                    <CardContent className="flex items-start gap-3 p-4">
                      <div className="shrink-0 rounded-md border border-[var(--border)] bg-gray-900 p-1.5">
                        <FootprintPreview
                          packageType={String(comp.type) === "passive" ? "0805" : String(comp.type) === "ic" ? "SOIC" : String(comp.type) === "connector" ? "Conn" : "0805"}
                          width={48}
                          height={48}
                        />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm font-bold">{String(comp.ref)}</span>
                          <Badge variant="outline" className="text-[10px]">{String(comp.type)}</Badge>
                        </div>
                        <div className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-0.5 text-[10px] text-[var(--muted-foreground)] font-technical">
                          <span>X: {Number(comp.x).toFixed(1)} mm</span>
                          <span>Y: {Number(comp.y).toFixed(1)} mm</span>
                          <span>W: {Number(comp.width).toFixed(1)} mm</span>
                          <span>H: {Number(comp.height).toFixed(1)} mm</span>
                          {Number(comp.rotation) !== 0 && (
                            <span className="col-span-2">Rotation: {Number(comp.rotation)}°</span>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}

        {/* --- JLCPCB CATALOG TAB --- */}
        {libTab === "jlcpcb" && (<div>

        {/* Search bar */}
        <div className="mb-4">
          <div className="relative">
            <Search
              className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-[var(--muted-foreground)]"
              aria-hidden="true"
            />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search components by name, value, LCSC#, or description..."
              className="w-full rounded-xl border border-[var(--border)] bg-[var(--card)] py-3.5 pl-12 pr-4 text-base text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:border-[var(--primary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]/20"
              aria-label="Search components"
            />
            {loading && (
              <Loader2
                className="absolute right-4 top-1/2 h-5 w-5 -translate-y-1/2 animate-spin text-[var(--muted-foreground)]"
                aria-hidden="true"
              />
            )}
          </div>
        </div>

        {/* Filter bar */}
        <div className="mb-6 flex flex-wrap items-center gap-2">
          <Dropdown
            value={category}
            onChange={setCategory}
            options={CATEGORIES}
            icon={<SlidersHorizontal className="h-3.5 w-3.5 text-[var(--muted-foreground)]" />}
          />

          <Dropdown
            value={packageFilter}
            onChange={setPackageFilter}
            options={PACKAGES}
            icon={<Package className="h-3.5 w-3.5 text-[var(--muted-foreground)]" />}
          />

          <button
            onClick={() => setInStock(!inStock)}
            className={
              "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors " +
              (inStock
                ? "border-blue-500/50 bg-blue-500/10 text-blue-400"
                : "border-[var(--border)] text-[var(--muted-foreground)] hover:border-gray-500")
            }
          >
            {inStock && <Check className="h-3.5 w-3.5" aria-hidden="true" />}
            In Stock Only
          </button>

          <button
            onClick={() => setBasicOnly(!basicOnly)}
            className={
              "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors " +
              (basicOnly
                ? "border-green-500/50 bg-green-500/10 text-green-400"
                : "border-[var(--border)] text-[var(--muted-foreground)] hover:border-gray-500")
            }
            title="Basic parts have no $3 setup fee on JLCPCB"
          >
            {basicOnly && <Check className="h-3.5 w-3.5" aria-hidden="true" />}
            Basic Parts Only
            {!basicOnly && (
              <span className="ml-1 rounded bg-green-500/15 px-1.5 py-0.5 text-[10px] text-green-400">
                No setup fee
              </span>
            )}
          </button>

          <div className="ml-auto">
            <Dropdown
              value={sortMode}
              onChange={setSortMode}
              options={SORT_OPTIONS}
              icon={<ArrowUpDown className="h-3.5 w-3.5 text-[var(--muted-foreground)]" />}
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <Card className="mb-6 border-yellow-500/30 bg-yellow-500/5">
            <CardContent className="p-4 text-center text-sm text-yellow-400">
              {error.includes("API")
                ? "Could not connect to the parts API. Make sure the backend is running."
                : error}
            </CardContent>
          </Card>
        )}

        {/* Empty state: before search */}
        {!searched && !loading && (
          <div className="py-20 text-center text-[var(--muted-foreground)]">
            <Search className="mx-auto mb-4 h-12 w-12 opacity-30" aria-hidden="true" />
            <p className="text-base font-medium">Search the JLCPCB Component Library</p>
            <p className="mt-1 text-sm">
              Type at least 2 characters to search. Try &ldquo;100nF 0805&rdquo;,
              &ldquo;ESP32&rdquo;, or &ldquo;AMS1117&rdquo;.
            </p>
            <div className="mx-auto mt-6 flex max-w-md flex-wrap justify-center gap-2">
              {["100nF 0805", "ESP32-S3", "AMS1117-3.3", "10K resistor", "1N4148", "USB-C connector"].map(
                (suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setQuery(suggestion)}
                    className="rounded-full border border-[var(--border)] px-3 py-1 text-xs text-[var(--foreground)] hover:border-blue-500/50 hover:bg-blue-500/5 transition-colors"
                  >
                    {suggestion}
                  </button>
                )
              )}
            </div>
          </div>
        )}

        {/* Empty state: no results */}
        {!loading && searched && filteredResults.length === 0 && !error && (
          <div className="py-20 text-center text-[var(--muted-foreground)]">
            <Package className="mx-auto mb-4 h-12 w-12 opacity-30" aria-hidden="true" />
            <p className="text-base font-medium">No components found</p>
            <p className="mt-1 text-sm">
              {rawResults.length > 0
                ? "Try adjusting your category or package filter."
                : "Try a different search term or broaden your filters."}
            </p>
          </div>
        )}

        {/* Results */}
        {filteredResults.length > 0 && (
          <>
            <div className="mb-4 flex items-center justify-between">
              <p className="text-sm text-[var(--muted-foreground)]">
                Showing <span className="font-medium text-[var(--foreground)]">{filteredResults.length}</span>
                {rawResults.length !== filteredResults.length && (
                  <> of {rawResults.length} loaded</>
                )}
                {total > rawResults.length && (
                  <> ({total.toLocaleString()} total matches)</>
                )}
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {filteredResults.map((part) => (
                <ComponentCard
                  key={part.lcsc}
                  part={part}
                  onViewDetail={setDetailPart}
                />
              ))}
            </div>
          </>
        )}
        </div>)}
      </main>

      {/* Detail panel */}
      {detailPart && (
        <ComponentDetail
          part={detailPart}
          open={!!detailPart}
          onClose={() => setDetailPart(null)}
        />
      )}
    </>
  );
}
