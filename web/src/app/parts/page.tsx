"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { NavHeader } from "@/components/layout/nav-header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { searchParts, type PartResult } from "@/lib/api";
import { Search, ExternalLink, Package, Check, Loader2 } from "lucide-react";

type PackageFilter = "any" | "0402" | "0603" | "0805" | "1206";

export default function PartsSearchPage() {
  const [query, setQuery] = useState("");
  const [basicOnly, setBasicOnly] = useState(false);
  const [inStock, setInStock] = useState(true);
  const [packageFilter, setPackageFilter] = useState<PackageFilter>("any");
  const [results, setResults] = useState<PartResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const doSearch = useCallback(
    async (q: string) => {
      if (q.trim().length < 2) {
        setResults([]);
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
          limit: 50,
        });
        // Client-side package filter (API may not support it directly)
        const filtered =
          packageFilter === "any"
            ? resp.results
            : resp.results.filter((r) =>
                r.package.toLowerCase().includes(packageFilter)
              );
        setResults(filtered);
        setTotal(resp.total);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Search failed");
        setResults([]);
        setTotal(0);
      } finally {
        setLoading(false);
      }
    },
    [basicOnly, inStock, packageFilter]
  );

  // Debounced search on query change
  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => {
      doSearch(query);
    }, 300);
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [query, doSearch]);

  // Re-search when filters change (if there is a query)
  useEffect(() => {
    if (query.trim().length >= 2) {
      doSearch(query);
    }
  }, [basicOnly, inStock, packageFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        doSearch(query);
      }
    },
    [query, doSearch]
  );

  return (
    <>
      <NavHeader />

      <main className="mx-auto max-w-5xl flex-1 px-4 py-8">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-bold">JLCPCB Parts Search</h1>
          <p className="mt-1 text-sm text-[var(--muted-foreground)]">
            Search 7M+ components. Basic parts have no setup fee.
          </p>
        </div>

        {/* Search bar */}
        <div className="mb-4">
          <div className="relative">
            <Search
              className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--muted-foreground)]"
              aria-hidden="true"
            />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search parts... e.g. '100nF 0805', 'ESP32', 'AMS1117'"
              className="w-full rounded-lg border border-[var(--border)] bg-[var(--card)] py-3 pl-10 pr-4 text-sm text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:border-[var(--primary)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
              aria-label="Search parts"
            />
            {loading && (
              <Loader2
                className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-[var(--muted-foreground)]"
                aria-hidden="true"
              />
            )}
          </div>
        </div>

        {/* Filters */}
        <div className="mb-6 flex flex-wrap items-center gap-3">
          <button
            onClick={() => setBasicOnly(!basicOnly)}
            className={
              "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors " +
              (basicOnly
                ? "border-green-500/50 bg-green-500/10 text-green-400"
                : "border-[var(--border)] text-[var(--muted-foreground)] hover:border-gray-500")
            }
          >
            {basicOnly && <Check className="h-3.5 w-3.5" aria-hidden="true" />}
            Basic Only
          </button>

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
            In Stock
          </button>

          <span className="text-xs text-[var(--muted-foreground)]">Package:</span>
          {(["any", "0402", "0603", "0805", "1206"] as PackageFilter[]).map(
            (pkg) => (
              <button
                key={pkg}
                onClick={() => setPackageFilter(pkg)}
                className={
                  "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors " +
                  (packageFilter === pkg
                    ? "border-[var(--primary)] bg-blue-500/10 text-[var(--foreground)]"
                    : "border-[var(--border)] text-[var(--muted-foreground)] hover:border-gray-500")
                }
              >
                {pkg === "any" ? "Any" : pkg}
              </button>
            )
          )}
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

        {/* Results */}
        {!loading && searched && results.length === 0 && !error && (
          <div className="py-16 text-center text-[var(--muted-foreground)]">
            <Package className="mx-auto mb-3 h-10 w-10 opacity-40" aria-hidden="true" />
            <p className="text-sm">No parts found. Try a different search term.</p>
          </div>
        )}

        {results.length > 0 && (
          <>
            <p className="mb-3 text-xs text-[var(--muted-foreground)]">
              Showing {results.length} of {total.toLocaleString()} results
            </p>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {results.map((part) => (
                <Card key={part.lcsc} className="transition-colors hover:border-gray-500">
                  <CardContent className="p-4">
                    <div className="mb-2 flex items-start justify-between gap-2">
                      <span className="font-mono text-xs font-bold text-blue-400">
                        {part.lcsc}
                      </span>
                      {part.basic ? (
                        <Badge variant="success" className="text-[10px] shrink-0">
                          Basic
                        </Badge>
                      ) : (
                        <Badge variant="warning" className="text-[10px] shrink-0">
                          Extended
                        </Badge>
                      )}
                    </div>
                    <p className="mb-1 text-sm font-medium leading-snug line-clamp-2">
                      {part.description}
                    </p>
                    <p className="text-xs text-[var(--muted-foreground)]">
                      {part.mfr}
                    </p>
                    <div className="mt-3 flex items-center justify-between text-xs">
                      <span className="text-[var(--muted-foreground)]">
                        {part.package}
                      </span>
                      <span className="tabular-nums">
                        ${part.price.toFixed(4)}
                      </span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-xs text-[var(--muted-foreground)]">
                      <span>
                        Stock: {part.stock.toLocaleString()}
                      </span>
                      <div className="flex gap-2">
                        <a
                          href={`https://jlcpcb.com/partdetail/${part.lcsc}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300"
                        >
                          <ExternalLink className="h-3 w-3" aria-hidden="true" />
                          Datasheet
                        </a>
                        <Button variant="ghost" size="sm" className="h-auto px-2 py-0.5 text-xs">
                          Add to BOM
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </>
        )}

        {/* Empty state before search */}
        {!searched && !loading && (
          <div className="py-16 text-center text-[var(--muted-foreground)]">
            <Search className="mx-auto mb-3 h-10 w-10 opacity-40" aria-hidden="true" />
            <p className="text-sm">
              Type at least 2 characters to search the JLCPCB parts catalog.
            </p>
          </div>
        )}
      </main>
    </>
  );
}
