"use client";

import {
  useState,
  useRef,
  useCallback,
  useEffect,
  type MouseEvent,
  type WheelEvent,
} from "react";
import { cn } from "@/lib/utils";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Minimize2,
} from "lucide-react";

export type ViewMode = "pcb-native" | "sch-native" | "2d" | "3d-top" | "3d-iso" | "3d-back" | "3d-interactive";

const VIEW_LABELS: Record<ViewMode, string> = {
  "pcb-native": "PCB",
  "sch-native": "Schematic",
  "2d": "2D Render",
  "3d-top": "3D Top",
  "3d-iso": "3D Iso",
  "3d-back": "3D Back",
  "3d-interactive": "3D Model",
};

// --- Zoom tuning ---
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 8;
// Smaller step = smoother scroll. 0.08 means ~8% per tick.
const WHEEL_ZOOM_FACTOR = 0.08;
const BUTTON_ZOOM_STEP = 0.2;

export interface BoardViewerProps {
  /** Map of view mode to image URL. Missing modes are shown as disabled tabs. */
  images: Partial<Record<ViewMode, string>>;
  /** Board dimensions in mm — used for the interactive 3D view */
  boardSize?: { width: number; height: number; origin_x?: number; origin_y?: number };
  /** Component positions for the interactive 3D view */
  components?: Array<{
    ref: string;
    x: number;
    y: number;
    width: number;
    height: number;
    rotation: number;
    type: "ic" | "passive" | "connector" | "other";
  }>;
  /** URL to .kicad_pcb file for native rendering */
  pcbFileUrl?: string;
  /** URL to .kicad_sch file for native schematic rendering */
  schFileUrl?: string;
  /** Callback when a component is clicked (ref designator) */
  onComponentClick?: (ref: string) => void;
  /** Initial view mode (defaults to first available) */
  defaultView?: ViewMode;
  className?: string;
}

export function BoardViewer({
  images,
  boardSize,
  components,
  pcbFileUrl,
  schFileUrl,
  onComponentClick,
  defaultView,
  className,
}: BoardViewerProps) {
  // Include native views when file URLs are available
  const has3DInteractive = !!boardSize;
  const effectiveImages: Partial<Record<ViewMode, string>> = {
    ...(pcbFileUrl ? { "pcb-native": pcbFileUrl } : {}),
    ...(schFileUrl ? { "sch-native": schFileUrl } : {}),
    ...images,
    ...(has3DInteractive ? { "3d-interactive": "__interactive__" } : {}),
  };

  const availableModes = (Object.keys(VIEW_LABELS) as ViewMode[]).filter(
    (m) => effectiveImages[m],
  );
  const [activeView, setActiveView] = useState<ViewMode>(
    defaultView && effectiveImages[defaultView]
      ? defaultView
      : availableModes[0] || "2d",
  );
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [fitScale, setFitScale] = useState(1);
  const dragStart = useRef({ x: 0, y: 0 });
  const panStart = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const currentSrc =
    activeView === "3d-interactive" ? null : images[activeView];

  // --- Fit-to-view: scale image to fit container (width or height) ---
  const computeFitScale = useCallback(() => {
    if (!viewportRef.current || !imgRef.current) return;
    const vw = viewportRef.current.clientWidth;
    const vh = viewportRef.current.clientHeight;
    const iw = imgRef.current.naturalWidth;
    const ih = imgRef.current.naturalHeight;
    if (iw === 0 || ih === 0) return;
    // Fit so the entire board is visible
    const fit = Math.min(vw / iw, vh / ih);
    setFitScale(fit);
    setZoom(fit);
    setPan({ x: 0, y: 0 });
  }, []);

  const resetTransform = useCallback(() => {
    computeFitScale();
  }, [computeFitScale]);

  // Reset when view mode changes
  useEffect(() => {
    // Small delay to let image load
    const timer = setTimeout(resetTransform, 100);
    return () => clearTimeout(timer);
  }, [activeView, resetTransform]);

  // Also fit when image loads
  const handleImageLoad = useCallback(() => {
    computeFitScale();
  }, [computeFitScale]);

  // --- Smooth wheel zoom ---
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      // Proportional zoom: multiply by factor, not add constant
      const direction = e.deltaY > 0 ? -1 : 1;
      setZoom((z) => {
        const next = z * (1 + direction * WHEEL_ZOOM_FACTOR);
        return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next));
      });
    },
    [],
  );

  // --- Pan (with drag threshold so clicks pass through to hotspots) ---
  const mouseDownPos = useRef({ x: 0, y: 0 });
  const isDragStarted = useRef(false);
  const DRAG_THRESHOLD = 4; // px before drag starts

  const handleMouseDown = useCallback(
    (e: MouseEvent) => {
      if (e.button !== 0) return;
      mouseDownPos.current = { x: e.clientX, y: e.clientY };
      isDragStarted.current = false;
      dragStart.current = { x: e.clientX, y: e.clientY };
      panStart.current = { ...pan };
    },
    [pan],
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (mouseDownPos.current.x === 0 && mouseDownPos.current.y === 0) return;
      const dx = e.clientX - mouseDownPos.current.x;
      const dy = e.clientY - mouseDownPos.current.y;
      // Only start dragging after threshold
      if (!isDragStarted.current) {
        if (Math.abs(dx) < DRAG_THRESHOLD && Math.abs(dy) < DRAG_THRESHOLD) return;
        isDragStarted.current = true;
        setDragging(true);
      }
      if (!isDragStarted.current) return;
      setPan({
        x: panStart.current.x + (e.clientX - dragStart.current.x),
        y: panStart.current.y + (e.clientY - dragStart.current.y),
      });
    },
    [],
  );

  const handleMouseUp = useCallback(() => {
    setDragging(false);
    isDragStarted.current = false;
    mouseDownPos.current = { x: 0, y: 0 };
  }, []);

  const handleDoubleClick = useCallback(() => {
    resetTransform();
  }, [resetTransform]);

  // --- Click-to-select component (coordinate-based, no SVG dependency) ---
  const handleViewportClick = useCallback(
    (e: MouseEvent) => {
      if (isDragStarted.current) return;
      if (!onComponentClick || !components || !boardSize || !imgRef.current) return;

      const iw = imgRef.current.naturalWidth;
      const ih = imgRef.current.naturalHeight;
      if (iw === 0 || ih === 0) return;

      // Get click position relative to viewport
      const rect = viewportRef.current?.getBoundingClientRect();
      if (!rect) return;
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      // Convert screen coords to image coords (reverse the pan+zoom transform)
      const imgX = (clickX - pan.x) / zoom;
      const imgY = (clickY - pan.y) / zoom;

      // Convert image coords to board mm coords
      const padFrac = 0.05;
      const bpx0 = iw * padFrac;
      const bpy0 = ih * padFrac;
      const bpw = iw * (1 - 2 * padFrac);
      const bph = ih * (1 - 2 * padFrac);
      const originX = boardSize.origin_x ?? 0;
      const originY = boardSize.origin_y ?? 0;

      const boardX = originX + ((imgX - bpx0) / bpw) * boardSize.width;
      const boardY = originY + ((imgY - bpy0) / bph) * boardSize.height;

      // Find which component was clicked (check all, pick closest)
      let bestRef: string | null = null;
      let bestDist = Infinity;
      for (const comp of components) {
        const hw = comp.width / 2;
        const hh = comp.height / 2;
        if (
          boardX >= comp.x - hw &&
          boardX <= comp.x + hw &&
          boardY >= comp.y - hh &&
          boardY <= comp.y + hh
        ) {
          const dist = Math.hypot(boardX - comp.x, boardY - comp.y);
          if (dist < bestDist) {
            bestDist = dist;
            bestRef = comp.ref;
          }
        }
      }

      if (bestRef) {
        onComponentClick(bestRef);
      }
    },
    [onComponentClick, components, boardSize, pan, zoom],
  );

  const [isFullscreen, setIsFullscreen] = useState(false);
  const handleFullscreen = useCallback(() => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
        setIsFullscreen(false);
      } else {
        containerRef.current.requestFullscreen();
        setIsFullscreen(true);
      }
    }
  }, []);

  // Listen for fullscreen changes (Escape key)
  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  // Refit when entering/exiting fullscreen
  useEffect(() => {
    const timer = setTimeout(computeFitScale, 200);
    return () => clearTimeout(timer);
  }, [isFullscreen, computeFitScale]);

  const showInteractive = activeView === "3d-interactive";
  const showNative = activeView === "pcb-native" || activeView === "sch-native";

  return (
    <div
      ref={containerRef}
      className={cn(
        "flex flex-col bg-[var(--card)] overflow-hidden",
        className,
      )}
    >
      {/* Tab bar */}
      <div className="flex items-center justify-between border-b border-[var(--border)] px-2">
        <div className="flex" role="tablist" aria-label="Board view mode">
          {(Object.keys(VIEW_LABELS) as ViewMode[]).map((mode) => {
            const available = !!effectiveImages[mode];
            return (
              <button
                key={mode}
                role="tab"
                aria-selected={mode === activeView}
                aria-disabled={!available}
                disabled={!available}
                onClick={() => available && setActiveView(mode)}
                className={cn(
                  "px-3 py-2 text-xs font-medium transition-colors border-b-2",
                  mode === activeView
                    ? "border-[var(--primary)] text-[var(--foreground)]"
                    : "border-transparent text-[var(--muted-foreground)] hover:text-[var(--foreground)]",
                  !available && "opacity-30 cursor-not-allowed",
                )}
              >
                {VIEW_LABELS[mode]}
              </button>
            );
          })}
        </div>

        {/* Component selector + Toolbar */}
        <div className="flex items-center gap-2 py-1">
          {/* Component dropdown — always visible when components exist */}
          {components && components.length > 0 && onComponentClick && (
            <select
              onChange={(e) => {
                if (e.target.value) onComponentClick(e.target.value);
              }}
              defaultValue=""
              className="h-7 rounded border border-[var(--border)] bg-[var(--background)] px-1.5 text-[10px] text-[var(--foreground)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
            >
              <option value="" disabled>Select component...</option>
              {components.map((comp) => (
                <option key={comp.ref} value={comp.ref}>
                  {comp.ref} ({String(comp.type)})
                </option>
              ))}
            </select>
          )}

        {!showInteractive && !showNative && (
          <div className="flex items-center gap-1">
            <button
              onClick={() =>
                setZoom((z) => Math.min(MAX_ZOOM, z + BUTTON_ZOOM_STEP))
              }
              className="p-1.5 rounded hover:bg-[var(--muted)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
              aria-label="Zoom in"
              title="Zoom in"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
            <span className="text-[10px] tabular-nums text-[var(--muted-foreground)] w-10 text-center font-technical">
              {(zoom * 100).toFixed(0)}%
            </span>
            <button
              onClick={() =>
                setZoom((z) => Math.max(MIN_ZOOM, z - BUTTON_ZOOM_STEP))
              }
              className="p-1.5 rounded hover:bg-[var(--muted)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
              aria-label="Zoom out"
              title="Zoom out"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <div className="w-px h-4 bg-[var(--border)] mx-1" />
            <button
              onClick={resetTransform}
              className="p-1.5 rounded hover:bg-[var(--muted)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
              aria-label="Fit to view"
              title="Fit to view (double-click)"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
            <button
              onClick={handleFullscreen}
              className="p-1.5 rounded hover:bg-[var(--muted)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
              aria-label="Toggle fullscreen"
              title="Fullscreen"
            >
              {isFullscreen ? (
                <Minimize2 className="h-4 w-4" />
              ) : (
                <Maximize2 className="h-4 w-4" />
              )}
            </button>
          </div>
        )}
        </div>
      </div>

      {/* Viewport */}
      {showNative ? (
        <NativeKiCanvasView
          src={effectiveImages[activeView] || ""}
          controls="full"
          onComponentClick={onComponentClick}
          components={components}
        />
      ) : showInteractive ? (
        <Interactive3DView
          boardSize={boardSize!}
          components={components || []}
          onComponentClick={onComponentClick}
        />
      ) : (
        <div
          ref={viewportRef}
          className={cn(
            "relative flex-1 min-h-[200px] overflow-hidden bg-[var(--background)]",
            dragging ? "cursor-grabbing" : "cursor-default",
          )}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={handleViewportClick}
          onDoubleClick={handleDoubleClick}
          role="img"
          aria-label={`Board ${VIEW_LABELS[activeView]} view`}
        >
          {currentSrc ? (
            <div
              className="absolute top-0 left-0"
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "top left",
                transition: dragging ? "none" : "transform 0.1s ease-out",
              }}
            >
              <img
                ref={imgRef}
                src={currentSrc}
                alt={`PCB board - ${VIEW_LABELS[activeView]} view`}
                className="block select-none max-w-none"
                draggable={false}
                onLoad={handleImageLoad}
              />
              {/* SVG hotspots overlaid directly on the image */}
              {activeView === "2d" && components && components.length > 0 && boardSize && imgRef.current && imgRef.current.naturalWidth > 0 && (
                <ComponentHotspotsSVG
                  components={components}
                  boardSize={boardSize}
                  imageWidth={imgRef.current.naturalWidth}
                  imageHeight={imgRef.current.naturalHeight}
                  onComponentClick={onComponentClick}
                />
              )}
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-[var(--muted-foreground)] text-sm">
              No image available for {VIEW_LABELS[activeView]} view
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================
// Native KiCanvas viewer for PCB and Schematic files
// ============================================================

function NativeKiCanvasView({
  src,
  controls = "basic",
  onComponentClick,
  components,
}: {
  src: string;
  controls?: "none" | "basic" | "full";
  onComponentClick?: (ref: string) => void;
  components?: Array<{ ref: string; x: number; y: number; width: number; height: number; rotation: number; type: string }>;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Use refs for callbacks to avoid re-creating the embed when they change
  const onClickRef = useRef(onComponentClick);
  onClickRef.current = onComponentClick;

  useEffect(() => {
    const scriptId = "kicanvas-script";
    if (!document.getElementById(scriptId)) {
      const script = document.createElement("script");
      script.id = scriptId;
      script.type = "module";
      script.src = "/vendor/kicanvas.js";
      script.onerror = () => setError("Failed to load KiCanvas");
      document.head.appendChild(script);
    }

    let cancelled = false;
    (async () => {
      const t0 = Date.now();
      while (!customElements.get("kicanvas-embed") && Date.now() - t0 < 8000) {
        await new Promise((r) => setTimeout(r, 150));
        if (cancelled) return;
      }
      if (!cancelled) {
        if (customElements.get("kicanvas-embed")) {
          setReady(true);
        } else {
          setError("KiCanvas did not initialize");
        }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!ready || !containerRef.current || !src) return;
    const el = containerRef.current;
    el.innerHTML = "";
    const embed = document.createElement("kicanvas-embed");
    embed.setAttribute("src", src);
    embed.setAttribute("controls", controls);
    // "objects" zooms to fit the board content
    embed.setAttribute("zoom", "objects");
    embed.style.width = "100%";
    embed.style.height = "100%";
    embed.style.display = "block";
    el.appendChild(embed);

    // After KiCanvas loads, zoom to board extents (Edge.Cuts)
    // KiCanvas has viewer.zoom_to_board() which uses the Edge.Cuts layer
    const zoomInterval = setInterval(() => {
      try {
        const app = embed.shadowRoot?.querySelector("kc-board-app") as unknown as Record<string, unknown> | null;
        const viewer = app?.viewer as Record<string, unknown> | undefined;
        if (viewer?.zoom_to_board && typeof viewer.zoom_to_board === "function") {
          (viewer.zoom_to_board as () => void)();
          clearInterval(zoomInterval);
        }
      } catch {
        // Not ready yet
      }
    }, 500);
    // Stop trying after 10 seconds
    setTimeout(() => clearInterval(zoomInterval), 10000);

    // Poll KiCanvas viewer's selected state every 300ms
    // Path: embed.shadowRoot → kc-board-app.viewer.selected
    let lastRef = "";
    const poll = setInterval(() => {
      if (!onClickRef.current) return;
      try {
        const app = embed.shadowRoot?.querySelector("kc-board-app") as unknown as Record<string, unknown> | null;
        if (!app?.viewer) return;
        const viewer = app.viewer as Record<string, unknown>;
        const selected = viewer.selected;
        if (!selected) return;

        // Walk the selected object tree looking for a reference designator
        const findRef = (obj: unknown, depth: number): string | null => {
          if (depth > 5 || !obj || typeof obj !== "object") return null;
          const o = obj as Record<string, unknown>;
          // Check known property names
          for (const key of ["reference", "Reference", "ref"]) {
            const val = o[key];
            if (typeof val === "string" && /^[A-Z]+\d/.test(val)) return val;
          }
          // Check properties sub-object (KiCad stores Reference in properties)
          if (o.properties && typeof o.properties === "object") {
            const props = o.properties as Record<string, unknown>;
            if (typeof props.Reference === "string" && /^[A-Z]+\d/.test(props.Reference)) {
              return props.Reference;
            }
          }
          // Recurse into sub-objects
          for (const v of Object.values(o)) {
            const found = findRef(v, depth + 1);
            if (found) return found;
          }
          return null;
        };

        const items = Array.isArray(selected) ? selected : [selected];
        for (const item of items) {
          const ref = findRef(item, 0);
          if (ref && ref !== lastRef) {
            lastRef = ref;
            onClickRef.current?.(ref);
            break;
          }
        }
      } catch {
        // KiCanvas internal structure may vary — silently ignore
      }
    }, 300);

    return () => {
      clearInterval(poll);
      clearInterval(zoomInterval);
      el.innerHTML = "";
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps -- onComponentClick via ref to prevent embed recreation
  }, [ready, src, controls]);

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-[var(--muted-foreground)]">
        {error}
      </div>
    );
  }
  if (!ready) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="h-4 w-4 border-2 border-[var(--primary)] border-t-transparent rounded-full animate-spin" />
        <span className="ml-2 text-sm text-[var(--muted-foreground)]">Loading KiCanvas...</span>
      </div>
    );
  }
  return <div ref={containerRef} className="flex-1 min-h-[200px]" />;
}

// ============================================================
// SVG component hotspots — overlaid directly on the board image
// ============================================================

interface ComponentHotspotsSVGProps {
  components: Array<{
    ref: string;
    x: number;
    y: number;
    width: number;
    height: number;
    rotation: number;
    type: "ic" | "passive" | "connector" | "other";
  }>;
  boardSize: { width: number; height: number; origin_x?: number; origin_y?: number };
  imageWidth: number;
  imageHeight: number;
  onComponentClick?: (ref: string) => void;
}

function ComponentHotspotsSVG({
  components,
  boardSize,
  imageWidth,
  imageHeight,
  onComponentClick,
}: ComponentHotspotsSVGProps) {
  const [hovered, setHovered] = useState<string | null>(null);

  if (imageWidth === 0 || imageHeight === 0) return null;
  if (boardSize.width === 0 || boardSize.height === 0) return null;

  // kicad-image-gen renders the board with some padding around it.
  // The board outline occupies the central portion of the image.
  // Estimate: ~5% padding on each side (the renderer adds a margin).
  const padFrac = 0.05;
  const bpx0 = imageWidth * padFrac;
  const bpy0 = imageHeight * padFrac;
  const bpw = imageWidth * (1 - 2 * padFrac);
  const bph = imageHeight * (1 - 2 * padFrac);

  // Scale: pixels per mm
  const sx = bpw / boardSize.width;
  const sy = bph / boardSize.height;

  // Board origin in PCB coordinates (from Edge.Cuts outline)
  const originX = boardSize.origin_x ?? 0;
  const originY = boardSize.origin_y ?? 0;

  return (
    <svg
      width={imageWidth}
      height={imageHeight}
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      className="absolute top-0 left-0"
      style={{ cursor: "default" }}
    >
      {components.map((comp) => {
        // Convert PCB mm coordinates to image pixel coordinates
        // comp.x/y are absolute PCB coords; originX/Y is the board outline min corner
        const cx = bpx0 + (comp.x - originX) * sx;
        const cy = bpy0 + (comp.y - originY) * sy;
        const w = Math.max(comp.width * sx, 14);
        const h = Math.max(comp.height * sy, 14);
        const isHovered = hovered === comp.ref;

        return (
          <g key={comp.ref}>
            {/* Clickable rect */}
            <rect
              x={cx - w / 2}
              y={cy - h / 2}
              width={w}
              height={h}
              rx={2}
              fill={isHovered ? "rgba(59,130,246,0.2)" : "rgba(0,0,0,0.001)"}
              stroke={isHovered ? "rgba(59,130,246,0.8)" : "none"}
              strokeWidth={isHovered ? 2 : 0}
              style={{ cursor: "pointer", pointerEvents: "all" }}
              onMouseEnter={() => setHovered(comp.ref)}
              onMouseLeave={() => setHovered(null)}
              onMouseDown={(e) => {
                e.stopPropagation();
                e.preventDefault();
              }}
              onClick={(e) => {
                e.stopPropagation();
                e.preventDefault();
                onComponentClick?.(comp.ref);
              }}
            />
            {/* Ref label — only visible on hover */}
            <text
              x={cx}
              y={cy - h / 2 - 4}
              textAnchor="middle"
              fill={isHovered ? "#3B82F6" : "transparent"}
              fontSize={10}
              fontWeight="bold"
              fontFamily="monospace"
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {comp.ref}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ============================================================
// Interactive 3D board view using Three.js
// ============================================================

interface Interactive3DProps {
  boardSize: { width: number; height: number };
  components: Array<{
    ref: string;
    x: number;
    y: number;
    width: number;
    height: number;
    rotation: number;
    type: "ic" | "passive" | "connector" | "other";
  }>;
  onComponentClick?: (ref: string) => void;
}

function Interactive3DView({ boardSize, components, onComponentClick }: Interactive3DProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [loaded, setLoaded] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;

    async function init() {
      // Dynamic import to avoid SSR issues
      const THREE = await import("three");
      const { OrbitControls } = await import(
        "three/examples/jsm/controls/OrbitControls.js"
      );

      if (disposed || !mountRef.current) return;

      const container = mountRef.current;
      const w = container.clientWidth;
      const h = container.clientHeight;

      // Scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1a1a1a);

      // Camera
      const aspect = w / h;
      const camDist = Math.max(boardSize.width, boardSize.height) * 1.2;
      const camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
      camera.position.set(camDist * 0.6, camDist * 0.8, camDist * 0.6);
      camera.lookAt(0, 0, 0);

      // Renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(w, h);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      container.appendChild(renderer.domElement);

      // Controls
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;
      controls.minDistance = 10;
      controls.maxDistance = camDist * 3;
      controls.target.set(0, 0, 0);

      // Lighting
      const ambient = new THREE.AmbientLight(0xffffff, 0.5);
      scene.add(ambient);
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(camDist * 0.5, camDist, camDist * 0.3);
      dirLight.castShadow = true;
      scene.add(dirLight);
      const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3);
      fillLight.position.set(-camDist * 0.5, camDist * 0.5, -camDist * 0.3);
      scene.add(fillLight);

      // Grid helper
      const gridSize = Math.max(boardSize.width, boardSize.height) * 1.5;
      const grid = new THREE.GridHelper(gridSize, 20, 0x333333, 0x222222);
      grid.position.y = -0.85;
      scene.add(grid);

      // --- PCB Board ---
      const boardThickness = 1.6;
      const boardGeo = new THREE.BoxGeometry(
        boardSize.width,
        boardThickness,
        boardSize.height,
      );
      const boardMat = new THREE.MeshStandardMaterial({
        color: 0x006400, // Soldermask green
        roughness: 0.6,
        metalness: 0.1,
      });
      const boardMesh = new THREE.Mesh(boardGeo, boardMat);
      boardMesh.receiveShadow = true;
      scene.add(boardMesh);

      // Copper layer (top) — slightly above board
      const copperGeo = new THREE.BoxGeometry(
        boardSize.width - 1,
        0.035,
        boardSize.height - 1,
      );
      const copperMat = new THREE.MeshStandardMaterial({
        color: 0xb87333, // Copper
        roughness: 0.3,
        metalness: 0.8,
      });
      const copperMesh = new THREE.Mesh(copperGeo, copperMat);
      copperMesh.position.y = boardThickness / 2 + 0.02;
      scene.add(copperMesh);

      // Component type colors
      const typeColors: Record<string, number> = {
        ic: 0x222222,
        passive: 0x444444,
        connector: 0x336699,
        other: 0x555555,
      };

      // Component height by type
      const typeHeights: Record<string, number> = {
        ic: 2.0,
        passive: 1.0,
        connector: 5.0,
        other: 1.5,
      };

      // --- Components ---
      const raycaster = new THREE.Raycaster();
      const mouse = new THREE.Vector2();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const componentMeshes: Array<{ mesh: any; ref: string }> = [];

      for (const comp of components) {
        const ch = typeHeights[comp.type] || 1.5;
        const geo = new THREE.BoxGeometry(comp.width, ch, comp.height);
        const mat = new THREE.MeshStandardMaterial({
          color: typeColors[comp.type] || 0x555555,
          roughness: 0.5,
          metalness: 0.2,
        });
        const mesh = new THREE.Mesh(geo, mat);

        // Position: PCB coords to Three.js (x → x, y → z, centered)
        const px = comp.x - boardSize.width / 2;
        const pz = comp.y - boardSize.height / 2;
        mesh.position.set(px, boardThickness / 2 + ch / 2, pz);
        mesh.rotation.y = THREE.MathUtils.degToRad(-comp.rotation);
        mesh.castShadow = true;

        scene.add(mesh);
        componentMeshes.push({ mesh, ref: comp.ref });
      }

      // --- Hover detection ---
      function onPointerMove(event: PointerEvent) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      }
      renderer.domElement.addEventListener("pointermove", onPointerMove);

      // --- Click detection ---
      function onPointerClick(event: PointerEvent) {
        const rect = renderer.domElement.getBoundingClientRect();
        const clickMouse = new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          -((event.clientY - rect.top) / rect.height) * 2 + 1,
        );
        const clickRay = new THREE.Raycaster();
        clickRay.setFromCamera(clickMouse, camera);
        const hits = clickRay.intersectObjects(componentMeshes.map(c => c.mesh));
        if (hits.length > 0) {
          const hit = componentMeshes.find(c => c.mesh === hits[0].object);
          if (hit && onComponentClick) {
            onComponentClick(hit.ref);
          }
        }
      }
      renderer.domElement.addEventListener("click", onPointerClick);

      // --- Animation loop ---
      let animFrame: number;
      function animate() {
        if (disposed) return;
        animFrame = requestAnimationFrame(animate);
        controls.update();

        // Hover raycast
        raycaster.setFromCamera(mouse, camera);
        const meshes = componentMeshes.map((c) => c.mesh);
        const intersects = raycaster.intersectObjects(meshes);

        // Reset all colors
        for (const c of componentMeshes) {
          const baseMat = c.mesh.material as any;
          baseMat.emissive.setHex(0x000000);
        }

        if (intersects.length > 0) {
          const hit = componentMeshes.find(
            (c) => c.mesh === intersects[0].object,
          );
          if (hit) {
            const mat = hit.mesh.material as any;
            mat.emissive.setHex(0x333366);
            // Update hovered ref via closure
            if (mountRef.current) {
              mountRef.current.dataset.hoveredRef = hit.ref;
              const event = new CustomEvent("componenthover", {
                detail: hit.ref,
              });
              mountRef.current.dispatchEvent(event);
            }
          }
        }

        renderer.render(scene, camera);
      }
      animate();

      // --- Resize handler ---
      const resizeObserver = new ResizeObserver(() => {
        if (disposed || !container) return;
        const nw = container.clientWidth;
        const nh = container.clientHeight;
        camera.aspect = nw / nh;
        camera.updateProjectionMatrix();
        renderer.setSize(nw, nh);
      });
      resizeObserver.observe(container);

      setLoaded(true);

      // Cleanup
      return () => {
        disposed = true;
        cancelAnimationFrame(animFrame);
        resizeObserver.disconnect();
        renderer.domElement.removeEventListener("pointermove", onPointerMove);
        renderer.domElement.removeEventListener("click", onPointerClick);
        renderer.dispose();
        if (container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      };
    }

    const cleanup = init();
    return () => {
      disposed = true;
      cleanup.then((fn) => fn?.());
    };
  }, [boardSize, components]);

  // Listen for hover events from Three.js
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const handler = (e: Event) => {
      setHovered((e as CustomEvent).detail);
    };
    el.addEventListener("componenthover", handler);
    return () => el.removeEventListener("componenthover", handler);
  }, []);

  return (
    <div className="relative flex-1 min-h-[200px]">
      <div ref={mountRef} className="absolute inset-0" />
      {!loaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-[var(--background)]">
          <div className="flex items-center gap-2 text-sm text-[var(--muted-foreground)]">
            <div className="h-4 w-4 border-2 border-[var(--primary)] border-t-transparent rounded-full animate-spin" />
            Loading 3D view...
          </div>
        </div>
      )}
      {hovered && (
        <div className="absolute top-3 left-3 bg-[var(--card)] border border-[var(--border)] rounded px-2 py-1 text-xs font-technical">
          {hovered}
        </div>
      )}
      <div className="absolute bottom-3 left-3 text-[10px] text-[var(--muted-foreground)]">
        Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan
      </div>
    </div>
  );
}
