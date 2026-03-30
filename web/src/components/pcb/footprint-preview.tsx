"use client";

interface FootprintPreviewProps {
  packageType: string;
  width?: number;
  height?: number;
  className?: string;
}

const PAD_COLOR = "#B87333";
const SOLDERMASK_BG = "#006400";
const SILKSCREEN_COLOR = "#FFFFFF";
const PAD_STROKE = "#8B5A2B";

interface PadDef {
  x: number;
  y: number;
  w: number;
  h: number;
}

function drawPads(pads: PadDef[], viewBox: string, width: number, height: number, silkOutline?: { x: number; y: number; w: number; h: number }, className?: string) {
  return (
    <svg
      viewBox={viewBox}
      width={width}
      height={height}
      className={className}
      role="img"
      aria-label="Footprint pad layout"
    >
      <rect x={viewBox.split(" ")[0]} y={viewBox.split(" ")[1]} width={viewBox.split(" ")[2]} height={viewBox.split(" ")[3]} fill={SOLDERMASK_BG} rx="2" />
      {silkOutline && (
        <rect
          x={silkOutline.x}
          y={silkOutline.y}
          width={silkOutline.w}
          height={silkOutline.h}
          fill="none"
          stroke={SILKSCREEN_COLOR}
          strokeWidth="0.8"
          strokeDasharray="2 1"
          rx="1"
        />
      )}
      {pads.map((pad, i) => (
        <rect
          key={i}
          x={pad.x}
          y={pad.y}
          width={pad.w}
          height={pad.h}
          fill={PAD_COLOR}
          stroke={PAD_STROKE}
          strokeWidth="0.3"
          rx="0.5"
        />
      ))}
      {/* Pin 1 marker */}
      {pads.length > 0 && (
        <circle
          cx={pads[0].x + pads[0].w / 2}
          cy={pads[0].y - 2}
          r="1"
          fill={SILKSCREEN_COLOR}
        />
      )}
    </svg>
  );
}

function twoPadPackage(padW: number, padH: number, gap: number, bodyW: number, bodyH: number, viewW: number, viewH: number, width: number, height: number, className?: string) {
  const cx = viewW / 2;
  const cy = viewH / 2;
  const pads: PadDef[] = [
    { x: cx - gap / 2 - padW, y: cy - padH / 2, w: padW, h: padH },
    { x: cx + gap / 2, y: cy - padH / 2, w: padW, h: padH },
  ];
  const silk = { x: cx - bodyW / 2, y: cy - bodyH / 2, w: bodyW, h: bodyH };
  return drawPads(pads, `0 0 ${viewW} ${viewH}`, width, height, silk, className);
}

function render0402(width: number, height: number, className?: string) {
  return twoPadPackage(2.5, 3, 2, 5, 3.5, 12, 10, width, height, className);
}

function render0603(width: number, height: number, className?: string) {
  return twoPadPackage(3, 4, 3, 7, 4.5, 16, 12, width, height, className);
}

function render0805(width: number, height: number, className?: string) {
  return twoPadPackage(3.5, 5, 4, 8, 5.5, 18, 14, width, height, className);
}

function render1206(width: number, height: number, className?: string) {
  return twoPadPackage(4, 6, 5, 10, 7, 22, 16, width, height, className);
}

function renderSOT23(width: number, height: number, className?: string) {
  const viewW = 20;
  const viewH = 20;
  const padW = 3;
  const padH = 2;
  const pads: PadDef[] = [
    { x: 3, y: 14, w: padW, h: padH },   // pin 1 bottom-left
    { x: 14, y: 14, w: padW, h: padH },   // pin 2 bottom-right
    { x: 8.5, y: 4, w: padW, h: padH },   // pin 3 top-center
  ];
  const silk = { x: 4, y: 5, w: 12, h: 10 };
  return drawPads(pads, `0 0 ${viewW} ${viewH}`, width, height, silk, className);
}

function renderSOIC8(width: number, height: number, className?: string) {
  const viewW = 30;
  const viewH = 30;
  const padW = 3;
  const padH = 1.8;
  const pads: PadDef[] = [];
  // Left side pins 1-4
  for (let i = 0; i < 4; i++) {
    pads.push({ x: 3, y: 6 + i * 5, w: padW, h: padH });
  }
  // Right side pins 5-8
  for (let i = 3; i >= 0; i--) {
    pads.push({ x: 24, y: 6 + i * 5, w: padW, h: padH });
  }
  const silk = { x: 7, y: 4, w: 16, h: 22 };
  return drawPads(pads, `0 0 ${viewW} ${viewH}`, width, height, silk, className);
}

function renderQFP(pinCount: number, width: number, height: number, className?: string) {
  const pinsPerSide = Math.max(Math.floor(pinCount / 4), 3);
  const viewSize = 10 + pinsPerSide * 3;
  const padLen = 2.5;
  const padThick = 1.2;
  const pads: PadDef[] = [];
  const margin = 4;
  const bodyStart = margin + padLen + 1;
  const bodySize = viewSize - 2 * bodyStart;
  const spacing = bodySize / (pinsPerSide + 1);

  // Left side
  for (let i = 0; i < pinsPerSide; i++) {
    pads.push({ x: margin, y: bodyStart + spacing * (i + 0.5), w: padLen, h: padThick });
  }
  // Bottom side
  for (let i = 0; i < pinsPerSide; i++) {
    pads.push({ x: bodyStart + spacing * (i + 0.5), y: viewSize - margin - padThick, w: padThick, h: padLen });
  }
  // Right side
  for (let i = pinsPerSide - 1; i >= 0; i--) {
    pads.push({ x: viewSize - margin - padLen, y: bodyStart + spacing * (i + 0.5), w: padLen, h: padThick });
  }
  // Top side
  for (let i = pinsPerSide - 1; i >= 0; i--) {
    pads.push({ x: bodyStart + spacing * (i + 0.5), y: margin, w: padThick, h: padLen });
  }

  const silk = { x: bodyStart, y: bodyStart, w: bodySize, h: bodySize };
  return drawPads(pads, `0 0 ${viewSize} ${viewSize}`, width, height, silk, className);
}

function renderQFN(pinCount: number, width: number, height: number, className?: string) {
  // QFN is like QFP but pads are under the body edge + exposed pad
  const pinsPerSide = Math.max(Math.floor(pinCount / 4), 3);
  const viewSize = 10 + pinsPerSide * 3;
  const padLen = 2;
  const padThick = 1;
  const pads: PadDef[] = [];
  const bodyStart = 5;
  const bodySize = viewSize - 10;
  const spacing = bodySize / (pinsPerSide + 1);

  // Left side
  for (let i = 0; i < pinsPerSide; i++) {
    pads.push({ x: bodyStart - padLen / 2, y: bodyStart + spacing * (i + 0.5), w: padLen, h: padThick });
  }
  // Bottom
  for (let i = 0; i < pinsPerSide; i++) {
    pads.push({ x: bodyStart + spacing * (i + 0.5), y: bodyStart + bodySize - padThick / 2, w: padThick, h: padLen });
  }
  // Right
  for (let i = pinsPerSide - 1; i >= 0; i--) {
    pads.push({ x: bodyStart + bodySize - padLen / 2, y: bodyStart + spacing * (i + 0.5), w: padLen, h: padThick });
  }
  // Top
  for (let i = pinsPerSide - 1; i >= 0; i--) {
    pads.push({ x: bodyStart + spacing * (i + 0.5), y: bodyStart - padThick / 2, w: padThick, h: padLen });
  }
  // Exposed center pad
  const epSize = bodySize * 0.5;
  const epStart = bodyStart + (bodySize - epSize) / 2;
  pads.push({ x: epStart, y: epStart, w: epSize, h: epSize });

  const silk = { x: bodyStart, y: bodyStart, w: bodySize, h: bodySize };
  return drawPads(pads, `0 0 ${viewSize} ${viewSize}`, width, height, silk, className);
}

function renderConnector(pinCount: number, width: number, height: number, className?: string) {
  const pins = Math.max(pinCount, 2);
  const padDia = 3;
  const pitch = 5;
  const viewW = pins * pitch + 6;
  const viewH = 16;
  const pads: PadDef[] = [];
  for (let i = 0; i < pins; i++) {
    pads.push({ x: 3 + i * pitch + (pitch - padDia) / 2, y: (viewH - padDia) / 2, w: padDia, h: padDia });
  }
  const silk = { x: 2, y: 2, w: viewW - 4, h: viewH - 4 };
  return drawPads(pads, `0 0 ${viewW} ${viewH}`, width, height, silk, className);
}

function renderUnknown(width: number, height: number, className?: string) {
  return (
    <svg viewBox="0 0 40 40" width={width} height={height} className={className} role="img" aria-label="Unknown package footprint">
      <rect x="0" y="0" width="40" height="40" fill={SOLDERMASK_BG} rx="2" />
      <rect x="8" y="8" width="24" height="24" fill="none" stroke={SILKSCREEN_COLOR} strokeWidth="0.8" strokeDasharray="3 2" rx="2" />
      <text x="20" y="22" textAnchor="middle" fill={SILKSCREEN_COLOR} fontSize="8" fontFamily="monospace">?</text>
    </svg>
  );
}

function parsePackageType(packageType: string): { type: string; pinCount: number } {
  const s = packageType.toUpperCase().replace(/[_-]/g, "");
  if (/^0402/.test(s)) return { type: "0402", pinCount: 2 };
  if (/^0603/.test(s)) return { type: "0603", pinCount: 2 };
  if (/^0805/.test(s)) return { type: "0805", pinCount: 2 };
  if (/^1206/.test(s)) return { type: "1206", pinCount: 2 };
  if (/^SOT\s*23/.test(s) || /^SOT23/.test(s)) return { type: "SOT-23", pinCount: 3 };
  if (/^SOIC\s*(\d+)?/.test(s)) {
    const m = s.match(/(\d+)/);
    return { type: "SOIC", pinCount: m ? parseInt(m[1], 10) || 8 : 8 };
  }
  if (/^QFP/.test(s)) {
    const m = s.match(/(\d+)/);
    return { type: "QFP", pinCount: m ? parseInt(m[1], 10) || 48 : 48 };
  }
  if (/^QFN/.test(s) || /^DFN/.test(s)) {
    const m = s.match(/(\d+)/);
    return { type: "QFN", pinCount: m ? parseInt(m[1], 10) || 24 : 24 };
  }
  if (/^(CONN|HDR|PIN|JST|MOLEX)/.test(s) || /THROUGH.?HOLE/.test(s)) {
    const m = s.match(/(\d+)/);
    return { type: "CONN", pinCount: m ? parseInt(m[1], 10) || 4 : 4 };
  }
  if (/^SOP/.test(s) || /^TSSOP/.test(s) || /^MSOP/.test(s) || /^SSOP/.test(s)) {
    const m = s.match(/(\d+)/);
    return { type: "SOIC", pinCount: m ? parseInt(m[1], 10) || 8 : 8 };
  }
  return { type: "unknown", pinCount: 0 };
}

export function FootprintPreview({ packageType, width = 120, height = 120, className }: FootprintPreviewProps) {
  const { type, pinCount } = parsePackageType(packageType);

  switch (type) {
    case "0402":
      return render0402(width, height, className);
    case "0603":
      return render0603(width, height, className);
    case "0805":
      return render0805(width, height, className);
    case "1206":
      return render1206(width, height, className);
    case "SOT-23":
      return renderSOT23(width, height, className);
    case "SOIC":
      return renderSOIC8(width, height, className);
    case "QFP":
      return renderQFP(pinCount, width, height, className);
    case "QFN":
      return renderQFN(pinCount, width, height, className);
    case "CONN":
      return renderConnector(pinCount, width, height, className);
    default:
      return renderUnknown(width, height, className);
  }
}
