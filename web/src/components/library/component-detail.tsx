"use client";

import { type PartResult } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FootprintPreview } from "@/components/pcb/footprint-preview";
import { X, ExternalLink, FileDown, Plus, Package, Zap, Thermometer, Ruler } from "lucide-react";
import { useEffect, useCallback } from "react";

export interface ComponentDetailProps {
  part: PartResult;
  open: boolean;
  onClose: () => void;
}

/** Mock specifications derived from part description and package. */
function deriveSpecs(part: PartResult) {
  const desc = part.description.toLowerCase();
  const specs: Array<{ label: string; value: string; icon: typeof Package }> = [];

  specs.push({ label: "Package", value: part.package, icon: Ruler });

  // Extract value from description
  const valueMatch = part.description.match(/^([\d.]+\s*[kMGTpnumKR]*[FHR\u03A9])/i);
  if (valueMatch) {
    specs.push({ label: "Value", value: valueMatch[1], icon: Zap });
  }

  // Tolerance
  const tolMatch = desc.match(/[+-]?([\d.]+)%/);
  if (tolMatch) {
    specs.push({ label: "Tolerance", value: `\u00B1${tolMatch[1]}%`, icon: Zap });
  }

  // Voltage rating
  const voltMatch = desc.match(/(\d+)\s*v\b/i);
  if (voltMatch) {
    specs.push({ label: "Voltage Rating", value: `${voltMatch[1]}V`, icon: Zap });
  }

  // Temperature
  specs.push({ label: "Temp Range", value: "-40 to +85 C", icon: Thermometer });

  // Dielectric (for caps)
  const dielMatch = desc.match(/\b(X7R|X5R|C0G|NP0|Y5V)\b/i);
  if (dielMatch) {
    specs.push({ label: "Dielectric", value: dielMatch[1].toUpperCase(), icon: Package });
  }

  return specs;
}

/** Mock price breaks based on unit price. */
function priceBreaks(unitPrice: number) {
  return [
    { qty: 1, price: unitPrice },
    { qty: 10, price: unitPrice * 0.95 },
    { qty: 100, price: unitPrice * 0.85 },
    { qty: 1000, price: unitPrice * 0.7 },
    { qty: 5000, price: unitPrice * 0.6 },
  ];
}

/** Mock alternatives. */
function mockAlternatives(part: PartResult): Array<{ lcsc: string; description: string; price: number }> {
  // Generate 2-3 plausible alternatives based on description keywords
  const descs = [
    part.description.replace(/0805/, "0603"),
    part.description.replace(/\d+%/, "5%"),
    part.description.replace(/50V/, "25V"),
  ].filter((d) => d !== part.description);
  return descs.slice(0, 3).map((d, i) => ({
    lcsc: `C${parseInt(part.lcsc.replace("C", ""), 10) + i + 100}`,
    description: d,
    price: part.price * (0.8 + Math.random() * 0.4),
  }));
}

function stockColor(stock: number): string {
  if (stock >= 10000) return "text-green-400";
  if (stock >= 1000) return "text-yellow-400";
  return "text-red-400";
}

export function ComponentDetail({ part, open, onClose }: ComponentDetailProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose]
  );

  useEffect(() => {
    if (open) {
      document.addEventListener("keydown", handleKeyDown);
      return () => document.removeEventListener("keydown", handleKeyDown);
    }
  }, [open, handleKeyDown]);

  if (!open) return null;

  const specs = deriveSpecs(part);
  const breaks = priceBreaks(part.price);
  const alts = mockAlternatives(part);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-over panel */}
      <div className="fixed inset-y-0 right-0 z-50 w-full max-w-lg overflow-y-auto border-l border-[var(--border)] bg-gray-950 shadow-2xl sm:max-w-xl">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-start justify-between border-b border-[var(--border)] bg-gray-950/95 px-6 py-4 backdrop-blur">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-mono text-lg font-bold text-blue-400">{part.lcsc}</span>
              {part.basic ? (
                <Badge variant="success">Basic</Badge>
              ) : (
                <Badge variant="warning">Extended</Badge>
              )}
            </div>
            <p className="mt-1 text-sm text-[var(--muted-foreground)]">{part.mfr}</p>
            <p className="mt-0.5 text-sm font-medium">{part.description}</p>
          </div>
          <button
            onClick={onClose}
            className="ml-4 shrink-0 rounded-md p-1 text-[var(--muted-foreground)] hover:bg-white/10 hover:text-white"
            aria-label="Close detail panel"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-6 px-6 py-6">
          {/* Specifications */}
          <section>
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Specifications
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {specs.map((spec) => (
                <div
                  key={spec.label}
                  className="flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-2"
                >
                  <spec.icon className="h-3.5 w-3.5 shrink-0 text-[var(--muted-foreground)]" aria-hidden="true" />
                  <div className="min-w-0">
                    <p className="text-[10px] uppercase tracking-wide text-[var(--muted-foreground)]">{spec.label}</p>
                    <p className="truncate text-sm font-medium">{spec.value}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Footprint Preview */}
          <section>
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Footprint
            </h3>
            <div className="flex items-center justify-center rounded-lg border border-[var(--border)] bg-gray-900 p-4">
              <FootprintPreview packageType={part.package} width={160} height={160} />
            </div>
            <p className="mt-2 text-center text-xs text-[var(--muted-foreground)]">
              Package: {part.package}
            </p>
          </section>

          {/* 3D Preview (simplified) */}
          <section>
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              3D Preview
            </h3>
            <div className="flex items-center justify-center rounded-lg border border-[var(--border)] bg-gray-900 p-4">
              <Package3DPreview packageType={part.package} />
            </div>
          </section>

          {/* Pricing */}
          <section>
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Pricing
            </h3>
            <div className="overflow-hidden rounded-lg border border-[var(--border)]">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)] bg-[var(--card)]">
                    <th className="px-4 py-2 text-left font-medium text-[var(--muted-foreground)]">Qty</th>
                    <th className="px-4 py-2 text-right font-medium text-[var(--muted-foreground)]">Unit Price</th>
                    <th className="px-4 py-2 text-right font-medium text-[var(--muted-foreground)]">Subtotal</th>
                  </tr>
                </thead>
                <tbody>
                  {breaks.map((b) => (
                    <tr key={b.qty} className="border-b border-[var(--border)] last:border-0">
                      <td className="px-4 py-2 tabular-nums">{b.qty.toLocaleString()}+</td>
                      <td className="px-4 py-2 text-right tabular-nums">${b.price.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right tabular-nums">${(b.qty * b.price).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {!part.basic && (
              <p className="mt-2 text-xs text-yellow-400">
                Extended part: $3.00 setup fee per unique part in JLCPCB assembly.
              </p>
            )}
          </section>

          {/* Availability */}
          <section>
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Availability
            </h3>
            <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] px-4 py-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-[var(--muted-foreground)]">JLCPCB Stock</span>
                <span className={`text-sm font-semibold tabular-nums ${stockColor(part.stock)}`}>
                  {part.stock.toLocaleString()} pcs
                </span>
              </div>
              <div className="mt-2 flex items-center justify-between">
                <span className="text-sm text-[var(--muted-foreground)]">Assembly</span>
                <Badge variant={part.basic ? "success" : "info"}>
                  {part.basic ? "SMT Basic" : "SMT Extended"}
                </Badge>
              </div>
              <div className="mt-2 flex items-center justify-between">
                <span className="text-sm text-[var(--muted-foreground)]">Min Order</span>
                <span className="text-sm">1 pc</span>
              </div>
            </div>
          </section>

          {/* Alternatives */}
          {alts.length > 0 && (
            <section>
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                Similar Parts
              </h3>
              <div className="space-y-2">
                {alts.map((alt) => (
                  <div
                    key={alt.lcsc}
                    className="flex items-center justify-between rounded-lg border border-[var(--border)] bg-[var(--card)] px-4 py-2.5"
                  >
                    <div className="min-w-0">
                      <span className="font-mono text-xs font-bold text-blue-400">{alt.lcsc}</span>
                      <p className="truncate text-xs text-[var(--muted-foreground)]">{alt.description}</p>
                    </div>
                    <span className="ml-3 shrink-0 text-xs tabular-nums">${alt.price.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Actions */}
          <div className="flex flex-wrap gap-2 border-t border-[var(--border)] pt-6">
            <Button className="gap-1.5">
              <Plus className="h-4 w-4" aria-hidden="true" />
              Add to Project
            </Button>
            <a
              href={`https://www.lcsc.com/product-detail/${part.lcsc}.html`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button variant="outline" className="gap-1.5">
                <ExternalLink className="h-4 w-4" aria-hidden="true" />
                Open on LCSC
              </Button>
            </a>
            <a
              href={`https://jlcpcb.com/partdetail/${part.lcsc}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button variant="outline" className="gap-1.5">
                <FileDown className="h-4 w-4" aria-hidden="true" />
                Datasheet
              </Button>
            </a>
          </div>
        </div>
      </div>
    </>
  );
}

/** Simplified 3D box preview using CSS transforms. */
function Package3DPreview({ packageType }: { packageType: string }) {
  const pkg = packageType.toUpperCase();
  // Rough dimensions for the 3D box (w, h, d in px)
  let w = 40;
  let h = 20;
  let d = 10;
  let color = "#1a1a1a";

  if (/0402/.test(pkg)) { w = 20; h = 12; d = 6; color = "#2a2016"; }
  else if (/0603/.test(pkg)) { w = 28; h = 16; d = 8; color = "#2a2016"; }
  else if (/0805/.test(pkg)) { w = 36; h = 22; d = 10; color = "#2a2016"; }
  else if (/1206/.test(pkg)) { w = 44; h = 26; d = 12; color = "#2a2016"; }
  else if (/SOT/.test(pkg)) { w = 30; h = 24; d = 14; color = "#111"; }
  else if (/SOIC|SOP|TSSOP/.test(pkg)) { w = 50; h = 30; d = 12; color = "#111"; }
  else if (/QFP/.test(pkg)) { w = 60; h = 60; d = 10; color = "#111"; }
  else if (/QFN|DFN/.test(pkg)) { w = 50; h = 50; d = 6; color = "#111"; }

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        style={{
          width: w,
          height: h,
          perspective: "200px",
          transformStyle: "preserve-3d",
        }}
      >
        <div
          style={{
            width: w,
            height: h,
            backgroundColor: color,
            border: "1px solid #444",
            borderRadius: 2,
            transform: `rotateX(-20deg) rotateY(-25deg) translateZ(${d / 2}px)`,
            boxShadow: `${d / 2}px ${d / 2}px 0 rgba(0,0,0,0.3)`,
          }}
        />
      </div>
      <span className="text-[10px] text-[var(--muted-foreground)]">
        {packageType}
      </span>
    </div>
  );
}
