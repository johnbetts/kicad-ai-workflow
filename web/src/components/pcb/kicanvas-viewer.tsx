"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

interface KiCanvasViewerProps {
  /** URL to a .kicad_pcb or .kicad_sch file */
  src: string;
  /** Controls mode */
  controls?: "none" | "basic" | "full";
  className?: string;
}

/**
 * Native KiCad file viewer using the KiCanvas web component.
 * Renders .kicad_pcb and .kicad_sch files interactively in the browser
 * with real copper layers, traces, pads, silkscreen, and component outlines.
 */
export function KiCanvasViewer({
  src,
  controls = "basic",
  className,
}: KiCanvasViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!src) return;

    // Load KiCanvas script if not already loaded
    const scriptId = "kicanvas-script";
    let script = document.getElementById(scriptId) as HTMLScriptElement | null;

    if (!script) {
      script = document.createElement("script");
      script.id = scriptId;
      script.type = "module";
      script.src = "/vendor/kicanvas.js";
      script.onerror = () => setError("Failed to load KiCanvas viewer");
      document.head.appendChild(script);
    }

    // Wait for the custom element to be defined
    const waitForElement = async () => {
      try {
        // Wait up to 5 seconds for kicanvas-embed to be defined
        const timeout = 5000;
        const start = Date.now();
        while (!customElements.get("kicanvas-embed")) {
          if (Date.now() - start > timeout) {
            setError("KiCanvas component did not load");
            return;
          }
          await new Promise((r) => setTimeout(r, 100));
        }
        setLoaded(true);
      } catch {
        setError("Failed to initialize KiCanvas");
      }
    };

    waitForElement();
  }, [src]);

  useEffect(() => {
    if (!loaded || !containerRef.current || !src) return;

    // Clear previous content
    containerRef.current.innerHTML = "";

    // Create the kicanvas-embed element
    const embed = document.createElement("kicanvas-embed");
    embed.setAttribute("src", src);
    embed.setAttribute("controls", controls);
    embed.style.width = "100%";
    embed.style.height = "100%";
    embed.style.display = "block";

    containerRef.current.appendChild(embed);

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = "";
      }
    };
  }, [loaded, src, controls]);

  if (error) {
    return (
      <div className={cn("flex items-center justify-center bg-[var(--background)] text-[var(--muted-foreground)]", className)}>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  if (!loaded) {
    return (
      <div className={cn("flex items-center justify-center bg-[var(--background)]", className)}>
        <Loader2 className="h-5 w-5 animate-spin text-[var(--muted-foreground)]" />
        <span className="ml-2 text-sm text-[var(--muted-foreground)]">Loading native viewer...</span>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn("bg-[var(--background)]", className)}
    />
  );
}
