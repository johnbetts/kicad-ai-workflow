"use client";

import { useState, useEffect, useMemo } from "react";
import { useParams } from "next/navigation";
import { NavHeader } from "@/components/layout/nav-header";
import { BoardViewer, type ViewMode } from "@/components/pcb/board-viewer";
import { ScoreCard } from "@/components/pcb/score-card";
import { StageProgress } from "@/components/pcb/stage-progress";
import { FindingsList, type Finding } from "@/components/pcb/findings-list";
import { AIChat } from "@/components/chat/ai-chat";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Stage } from "@/lib/design-tokens";
import { getBoardDetail, type BoardDetail, type BoardComponent } from "@/lib/api";
import {
  CheckCircle2,
  XCircle,
  Download,
  ExternalLink,
  ShieldCheck,
  Loader2,
  Maximize2,
  Minimize2,
  FileText,
  Wrench,
  Zap,
  Info,
  X,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const MOCK_FINDINGS: Finding[] = [
  { id: "1", persona: "fab", severity: "minor", ref: "C3", description: "Silkscreen overlaps C4 pad area" },
  { id: "2", persona: "ee", severity: "warning", ref: "U2", description: "Decoupling cap 4.2mm from VCC pin (target <3mm)" },
  { id: "3", persona: "fab", severity: "info", ref: "J1", description: "Connector footprint could use wider pads for hand-soldering" },
  { id: "4", persona: "ee", severity: "minor", ref: "R5", description: "Pull-up on SDA line may be too strong (2.2k, recommend 4.7k)" },
  { id: "5", persona: "fab", severity: "warning", ref: "U1", description: "Decoupling loop area for C3 exceeds 10mm^2 target" },
];

const MOCK_SCORE = {
  grade: "B", score: 0.82,
  breakdown: [
    { name: "Collisions", score: 1.0, weight: 0.12 },
    { name: "Voltage Isolation", score: 0.85, weight: 0.12 },
    { name: "Decoupling Proximity", score: 0.72, weight: 0.08 },
    { name: "Connector Edge", score: 0.90, weight: 0.08 },
    { name: "Group Cohesion", score: 0.78, weight: 0.04 },
    { name: "MCU Peripheral", score: 0.88, weight: 0.08 },
  ],
};

const MOCK_STAGES: Array<{ stage: Stage; state: "passed" | "blocked" | "pending" | "running" | "failed" }> = [
  { stage: "requirements", state: "passed" },
  { stage: "schematic", state: "passed" },
  { stage: "pcb", state: "passed" },
  { stage: "validation", state: "blocked" },
  { stage: "production", state: "pending" },
];

const MOCK_REQUIREMENTS = {
  project: "train_mcu_core",
  description: "ESP32-S3 MCU core board with WiFi/BLE, USB-C, and GPIO breakout",
  features: [
    { name: "MCU", components: ["U1 (ESP32-S3-WROOM-1)"], status: "placed" },
    { name: "Power", components: ["U2 (AMS1117-3.3)", "C1, C2 (decoupling)"], status: "placed" },
    { name: "USB", components: ["J1 (USB-C)", "R1, R2 (CC resistors)"], status: "placed" },
    { name: "Reset/Boot", components: ["SW1, SW2 (buttons)", "R3, R4 (pull-ups)"], status: "placed" },
    { name: "Crystal", components: ["Y1 (32.768kHz)", "C5, C6 (load caps)"], status: "placed" },
  ],
  constraints: [
    "Board size: 70 x 50 mm",
    "2-layer FR4, 1.6mm",
    "JLCPCB assembly compatible",
    "USB-C on board edge",
    "Antenna keepout zone required",
  ],
};

// Per-component placement constraints (would come from requirements API)
const MOCK_CONSTRAINTS: Record<string, Array<{ type: string; description: string; met: boolean }>> = {
  C1: [
    { type: "proximity", description: "Within 3mm of U1 pin VCC", met: true },
    { type: "group", description: "Member of Power group", met: true },
  ],
  C2: [
    { type: "proximity", description: "Within 3mm of U1 pin 3V3", met: false },
    { type: "group", description: "Member of Power group", met: true },
  ],
  U1: [
    { type: "placement", description: "Center of board, antenna towards edge", met: true },
    { type: "keepout", description: "Antenna keepout zone: 10mm clear area", met: true },
    { type: "group", description: "MCU group anchor component", met: true },
  ],
  U2: [
    { type: "proximity", description: "Within 15mm of U1", met: true },
    { type: "boundary", description: "Power domain boundary component", met: true },
  ],
  J1: [
    { type: "edge", description: "USB-C connector flush with board edge", met: true },
    { type: "orientation", description: "Mating face towards nearest edge", met: true },
  ],
  R1: [
    { type: "proximity", description: "Within 5mm of J1 CC1 pin", met: true },
    { type: "group", description: "Member of USB group", met: true },
  ],
  R5: [
    { type: "proximity", description: "Within 10mm of U1 SDA pin", met: false },
    { type: "group", description: "Member of I2C bus group", met: true },
  ],
  SW1: [
    { type: "edge", description: "Accessible from board edge", met: true },
    { type: "proximity", description: "Within 15mm of U1 EN pin", met: true },
  ],
  Y1: [
    { type: "proximity", description: "Within 5mm of U1 XTAL pins", met: true },
    { type: "group", description: "Crystal oscillator subcircuit", met: true },
  ],
};

// ---------------------------------------------------------------------------
// Right panel tab type
// ---------------------------------------------------------------------------

type RightTab = "specs" | "fab" | "ee" | "component";

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function ProjectPage() {
  const params = useParams();
  const name = typeof params.name === "string" ? params.name : "";
  const projectName = decodeURIComponent(name);

  const [boardDetail, setBoardDetail] = useState<BoardDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [rightTab, setRightTab] = useState<RightTab>("specs");
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    if (!name) return;
    setLoading(true);
    getBoardDetail(name)
      .then(setBoardDetail)
      .catch(() => setBoardDetail(null))
      .finally(() => setLoading(false));
  }, [name]);

  const images = boardDetail?.images && Object.keys(boardDetail.images).length > 0
    ? boardDetail.images : {};
  const grade = boardDetail?.score?.grade ?? MOCK_SCORE.grade;
  const score = boardDetail?.score?.overall_score ?? MOCK_SCORE.score;
  const breakdown = boardDetail?.score?.breakdown
    ? Object.entries(boardDetail.score.breakdown).map(([n, v]) => ({ name: n, score: v as number, weight: 0.08 }))
    : MOCK_SCORE.breakdown;

  const fabFindings = useMemo(() => MOCK_FINDINGS.filter(f => f.persona === "fab"), []);
  const eeFindings = useMemo(() => MOCK_FINDINGS.filter(f => f.persona === "ee"), []);

  // Find component detail from board data
  const selectedComp = useMemo(() => {
    if (!selectedComponent || !boardDetail?.components) return null;
    return boardDetail.components.find(c => String(c.ref) === selectedComponent) ?? null;
  }, [selectedComponent, boardDetail]);

  // When a component is selected, auto-switch to component tab
  const handleComponentClick = (ref: string) => {
    setSelectedComponent(ref);
    setRightTab("component");
  };

  const chatContext = `Board: ${projectName}, Grade: ${grade}, Score: ${(score * 100).toFixed(0)}%. ` +
    `Components: ${boardDetail?.components?.length ?? 0}. ` +
    `Findings: ${MOCK_FINDINGS.map(f => `${f.ref}: ${f.description}`).join("; ")}`;

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-[var(--background)]">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--muted-foreground)]" />
      </div>
    );
  }

  if (fullscreen) {
    return (
      <div className="fixed inset-0 z-50 bg-black">
        <Button size="icon" variant="ghost" className="absolute top-2 right-2 z-10 text-white" onClick={() => setFullscreen(false)}>
          <Minimize2 className="h-5 w-5" />
        </Button>
        <BoardViewer
          images={images}
          boardSize={boardDetail?.board_size ?? undefined}
          components={boardDetail?.components}
          pcbFileUrl={boardDetail?.pcb_file_url || undefined}
          schFileUrl={boardDetail?.sch_file_url || undefined}
          className="h-screen"
        />
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[var(--background)]">
      {/* Top bar: nav + project info */}
      <NavHeader />
      <div className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--card)] px-3 py-1">
        <div className="flex items-center gap-2">
          <h1 className="text-sm font-semibold">{projectName}</h1>
          <Badge variant={grade <= "B" ? "success" : "outline"} className="text-[10px]">{grade} {(score * 100).toFixed(0)}%</Badge>
          <StageProgress stages={MOCK_STAGES} className="ml-1" />
        </div>
        <div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" className="h-6 text-[10px] gap-1" onClick={() => setFullscreen(true)}><Maximize2 className="h-3 w-3" />Full</Button>
          <Button size="sm" variant="outline" className="h-6 text-[10px] gap-1"><Download className="h-3 w-3" />Gerbers</Button>
          <Button size="sm" className="h-6 text-[10px] gap-1"><ExternalLink className="h-3 w-3" />JLCPCB</Button>
          <Button size="sm" variant="success" className="h-6 text-[10px] gap-1"><CheckCircle2 className="h-3 w-3" />Approve</Button>
        </div>
      </div>

      {/* Main area: Board (left) | Info panel (right) | AI chat (bottom-right) */}
      <div className="flex flex-1 overflow-hidden">

        {/* LEFT: Board viewer — fills available space */}
        <div className="flex-1 min-w-0 relative">
          <BoardViewer
            images={images}
            boardSize={boardDetail?.board_size ?? undefined}
            components={boardDetail?.components}
            pcbFileUrl={boardDetail?.pcb_file_url || undefined}
            schFileUrl={boardDetail?.sch_file_url || undefined}
            onComponentClick={handleComponentClick}
            className="h-full"
          />
        </div>

        {/* RIGHT: Stacked info panel (top) + AI chat (bottom) */}
        <div className="flex w-[360px] flex-shrink-0 flex-col border-l border-[var(--border)]">

          {/* Tab bar */}
          <div className="flex border-b border-[var(--border)] bg-[var(--card)]">
            {([
              { key: "specs" as RightTab, label: "Specs", icon: FileText },
              { key: "fab" as RightTab, label: `Fab (${fabFindings.length})`, icon: Wrench },
              { key: "ee" as RightTab, label: `EE (${eeFindings.length})`, icon: Zap },
              { key: "component" as RightTab, label: selectedComponent ?? "Part", icon: Info },
            ]).map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setRightTab(key)}
                className={cn(
                  "flex-1 py-1.5 text-[10px] font-medium text-center transition-colors border-b-2 flex items-center justify-center gap-1",
                  rightTab === key
                    ? "border-[var(--primary)] text-[var(--foreground)]"
                    : "border-transparent text-[var(--muted-foreground)] hover:text-[var(--foreground)]",
                  key === "component" && !selectedComponent && "opacity-40",
                )}
                disabled={key === "component" && !selectedComponent}
              >
                <Icon className="h-3 w-3" />
                {label}
              </button>
            ))}
          </div>

          {/* Tab content — scrollable, takes ~55% of right panel */}
          <div className="flex-1 overflow-y-auto min-h-0" style={{ maxHeight: "calc(55vh - 80px)" }}>

            {/* SPECS TAB */}
            {rightTab === "specs" && (
              <div className="p-3 space-y-3 text-xs">
                <ScoreCard grade={grade} score={score} breakdown={breakdown} />
                <div>
                  <h3 className="font-medium mb-1.5 text-[var(--muted-foreground)]">Design Requirements</h3>
                  <p className="text-[var(--foreground)] mb-2">{MOCK_REQUIREMENTS.description}</p>
                  <div className="space-y-1">
                    {MOCK_REQUIREMENTS.features.map(f => (
                      <div key={f.name} className="flex items-start gap-2 rounded bg-[var(--muted)] px-2 py-1.5">
                        <CheckCircle2 className="h-3 w-3 mt-0.5 text-green-500 shrink-0" />
                        <div>
                          <span className="font-medium">{f.name}</span>
                          <span className="text-[var(--muted-foreground)] ml-1">{f.components.join(", ")}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="font-medium mb-1.5 text-[var(--muted-foreground)]">Constraints</h3>
                  <ul className="space-y-0.5">
                    {MOCK_REQUIREMENTS.constraints.map((c, i) => (
                      <li key={i} className="text-[var(--foreground)] flex items-center gap-1.5">
                        <span className="w-1 h-1 rounded-full bg-[var(--muted-foreground)] shrink-0" />
                        {c}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {/* FAB REVIEW TAB */}
            {rightTab === "fab" && (
              <div className="p-3 space-y-2">
                <div className="flex items-center gap-2 mb-2">
                  <Wrench className="h-4 w-4 text-blue-400" />
                  <span className="text-xs font-medium">Fabricator Review</span>
                  <Badge variant="outline" className="text-[10px] ml-auto">{fabFindings.length} findings</Badge>
                </div>
                <FindingsList
                  findings={fabFindings}
                  onFix={(id) => console.log("Fix:", id)}
                  onIgnore={(id) => console.log("Ignore:", id)}
                />
              </div>
            )}

            {/* EE REVIEW TAB */}
            {rightTab === "ee" && (
              <div className="p-3 space-y-2">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="h-4 w-4 text-purple-400" />
                  <span className="text-xs font-medium">Electrical Engineer Review</span>
                  <Badge variant="outline" className="text-[10px] ml-auto">{eeFindings.length} findings</Badge>
                </div>
                <FindingsList
                  findings={eeFindings}
                  onFix={(id) => console.log("Fix:", id)}
                  onIgnore={(id) => console.log("Ignore:", id)}
                />
              </div>
            )}

            {/* COMPONENT DETAIL TAB */}
            {rightTab === "component" && selectedComponent && (
              <div className="p-3 space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm font-bold">{selectedComponent}</span>
                  <Button size="sm" variant="ghost" className="h-5 w-5 p-0" onClick={() => { setSelectedComponent(null); setRightTab("specs"); }}>
                    <X className="h-3 w-3" />
                  </Button>
                </div>

                {selectedComp ? (
                  <>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[var(--muted-foreground)]">
                      <span>Type: <strong className="text-[var(--foreground)]">{String(selectedComp.type)}</strong></span>
                      <span>Rotation: <strong className="text-[var(--foreground)]">{Number(selectedComp.rotation)}°</strong></span>
                      <span>Position: <strong className="text-[var(--foreground)] font-technical">{Number(selectedComp.x).toFixed(1)}, {Number(selectedComp.y).toFixed(1)} mm</strong></span>
                      <span>Size: <strong className="text-[var(--foreground)] font-technical">{Number(selectedComp.width).toFixed(1)} x {Number(selectedComp.height).toFixed(1)} mm</strong></span>
                    </div>

                    {/* Component crop image if available */}
                    {boardDetail?.crops?.filter(c => c.ref === selectedComponent).map(crop => (
                      <img
                        key={crop.url}
                        src={crop.url}
                        alt={`${selectedComponent} crop`}
                        className="w-full rounded border border-[var(--border)]"
                      />
                    ))}

                    {/* Findings for this component */}
                    {(() => {
                      const compFindings = MOCK_FINDINGS.filter(f => f.ref === selectedComponent);
                      if (compFindings.length === 0) return (
                        <p className="text-[var(--muted-foreground)] italic">No review findings for this component.</p>
                      );
                      return (
                        <div>
                          <h4 className="font-medium text-[var(--muted-foreground)] mb-1">Findings for {selectedComponent}</h4>
                          <FindingsList findings={compFindings} onFix={(id) => console.log("Fix:", id)} />
                        </div>
                      );
                    })()}

                    {/* Placement constraints */}
                    {(() => {
                      const constraints = MOCK_CONSTRAINTS[selectedComponent];
                      if (!constraints || constraints.length === 0) return null;
                      return (
                        <div>
                          <h4 className="font-medium text-[var(--muted-foreground)] mb-1.5">Placement Constraints</h4>
                          <div className="space-y-1">
                            {constraints.map((c, i) => (
                              <div key={i} className={cn(
                                "flex items-start gap-2 rounded px-2 py-1.5 text-[11px]",
                                c.met ? "bg-green-500/5" : "bg-red-500/10",
                              )}>
                                {c.met ? (
                                  <CheckCircle2 className="h-3 w-3 mt-0.5 text-green-500 shrink-0" />
                                ) : (
                                  <XCircle className="h-3 w-3 mt-0.5 text-red-500 shrink-0" />
                                )}
                                <div>
                                  <Badge variant="outline" className="text-[8px] mr-1 px-1 py-0">{c.type}</Badge>
                                  <span className={c.met ? "text-[var(--foreground)]" : "text-red-400"}>
                                    {c.description}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })()}

                    {/* Links */}
                    <div className="flex gap-2 pt-1">
                      <a
                        href={`https://www.snapeda.com/search/?q=${encodeURIComponent(selectedComponent)}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-[10px] text-blue-400 hover:underline"
                      >
                        <ExternalLink className="h-3 w-3" /> SnapEDA
                      </a>
                      <button
                        onClick={() => setRightTab("fab")}
                        className="text-[10px] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
                      >
                        View all fab findings →
                      </button>
                    </div>
                  </>
                ) : (
                  <p className="text-[var(--muted-foreground)]">Component data not available for {selectedComponent}.</p>
                )}
              </div>
            )}

            {rightTab === "component" && !selectedComponent && (
              <div className="flex flex-col items-center justify-center py-8 text-[var(--muted-foreground)]">
                <Info className="h-6 w-6 mb-2" />
                <p className="text-xs">Click a component on the board to see its details.</p>
              </div>
            )}
          </div>

          {/* AI Chat — always visible, bottom of right panel */}
          <div className="border-t border-[var(--border)] flex-shrink-0" style={{ height: "calc(45vh - 40px)" }}>
            <AIChat
              compact
              systemContext={chatContext}
              placeholder="Ask about the board, request fixes..."
              welcomeMessage={`Reviewing **${projectName}** (${grade}). ${MOCK_FINDINGS.length} findings from fab/EE review. What would you like to work on?`}
              className="h-full border-none rounded-none"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
