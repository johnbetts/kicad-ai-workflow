"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { NavHeader } from "@/components/layout/nav-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StageProgress, type StageStatus } from "@/components/pcb/stage-progress";
import { ScoreCard } from "@/components/pcb/score-card";
import { AIChat } from "@/components/chat/ai-chat";
import { cn } from "@/lib/utils";
import { STAGES } from "@/lib/design-tokens";
import {
  ArrowLeft,
  ArrowRight,
  Sparkles,
  Upload,
  FileText,
  Cpu,
  Thermometer,
  Zap,
  ToggleLeft,
  Usb,
  Radio,
  Battery,
  Volume2,
  Lightbulb,
  Check,
  Download,
  ExternalLink,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type WizardStep = 1 | 2 | 3 | 4;
type InputMode = "describe" | "template" | "upload";

interface Template {
  id: string;
  name: string;
  icon: keyof typeof ICON_MAP;
  components: number;
  cost: string;
  desc: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ICON_MAP = {
  Cpu,
  Thermometer,
  Zap,
  ToggleLeft,
  Usb,
  Radio,
  Battery,
  Volume2,
  Lightbulb,
} as const;

const TEMPLATES: Template[] = [
  {
    id: "esp32-devkit",
    name: "ESP32 DevKit",
    icon: "Cpu",
    components: 15,
    cost: "$3.50",
    desc: "WiFi/BLE development board with USB-C",
  },
  {
    id: "sensor-board",
    name: "Sensor Board",
    icon: "Thermometer",
    components: 12,
    cost: "$2.80",
    desc: "I2C sensor breakout (temp, humidity, light)",
  },
  {
    id: "motor-driver",
    name: "Motor Driver",
    icon: "Zap",
    components: 18,
    cost: "$4.20",
    desc: "Dual H-bridge MOSFET motor controller",
  },
  {
    id: "relay-controller",
    name: "Relay Controller",
    icon: "ToggleLeft",
    components: 25,
    cost: "$5.60",
    desc: "4-channel relay outputs with MCU",
  },
  {
    id: "usb-hub",
    name: "USB Hub",
    icon: "Usb",
    components: 10,
    cost: "$3.10",
    desc: "4-port USB 2.0 hub with ESD protection",
  },
  {
    id: "lora-node",
    name: "LoRa Node",
    icon: "Radio",
    components: 14,
    cost: "$4.80",
    desc: "Long-range wireless sensor node",
  },
  {
    id: "power-supply",
    name: "Power Supply",
    icon: "Battery",
    components: 20,
    cost: "$3.90",
    desc: "24V to 5V to 3.3V buck converter chain",
  },
  {
    id: "audio-amp",
    name: "Audio Amp",
    icon: "Volume2",
    components: 16,
    cost: "$2.50",
    desc: "Class-D audio amplifier with filters",
  },
  {
    id: "led-driver",
    name: "LED Driver",
    icon: "Lightbulb",
    components: 12,
    cost: "$2.20",
    desc: "WS2812B addressable LED controller",
  },
];

const STEP_LABELS = ["Describe", "Configure", "Parts", "Generate"] as const;

const PLACEHOLDER_PARTS = [
  { ref: "U1", value: "ESP32-S3-WROOM-1", pkg: "Module", lcsc: "C2913202", price: 2.85, stock: 4520, basic: false },
  { ref: "U2", value: "AMS1117-3.3", pkg: "SOT-223", lcsc: "C6186", price: 0.04, stock: 98000, basic: true },
  { ref: "C1", value: "100nF", pkg: "0805", lcsc: "C49678", price: 0.003, stock: 500000, basic: true },
  { ref: "C2", value: "10uF", pkg: "0805", lcsc: "C15850", price: 0.01, stock: 320000, basic: true },
  { ref: "R1", value: "10k", pkg: "0805", lcsc: "C17414", price: 0.002, stock: 680000, basic: true },
  { ref: "R2", value: "5.1k", pkg: "0805", lcsc: "C27834", price: 0.002, stock: 450000, basic: true },
  { ref: "J1", value: "USB-C", pkg: "SMD", lcsc: "C168688", price: 0.28, stock: 15000, basic: false },
  { ref: "L1", value: "4.7uH", pkg: "0805", lcsc: "C76753", price: 0.02, stock: 92000, basic: true },
] as const;

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StepIndicator({
  currentStep,
  onStepClick,
}: {
  currentStep: WizardStep;
  onStepClick: (step: WizardStep) => void;
}) {
  return (
    <nav className="mb-8 flex items-center justify-center gap-0" aria-label="Wizard steps">
      {STEP_LABELS.map((label, idx) => {
        const stepNum = (idx + 1) as WizardStep;
        const isActive = stepNum === currentStep;
        const isPast = stepNum < currentStep;
        const isLast = idx === STEP_LABELS.length - 1;

        return (
          <div key={label} className="flex items-center">
            <button
              onClick={() => {
                if (isPast) onStepClick(stepNum);
              }}
              disabled={!isPast && !isActive}
              className={cn(
                "flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                isActive && "bg-[var(--primary)] text-white",
                isPast && "cursor-pointer bg-green-500/15 text-green-500 hover:bg-green-500/25",
                !isActive && !isPast && "text-gray-500 cursor-default"
              )}
              aria-current={isActive ? "step" : undefined}
            >
              <span
                className={cn(
                  "flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold",
                  isActive && "bg-white/20",
                  isPast && "bg-green-500/20",
                  !isActive && !isPast && "bg-gray-500/20"
                )}
              >
                {isPast ? (
                  <Check className="h-3.5 w-3.5" aria-hidden="true" />
                ) : (
                  stepNum
                )}
              </span>
              <span className="hidden sm:inline">{label}</span>
            </button>
            {!isLast && (
              <div
                className={cn(
                  "mx-2 h-0.5 w-8 rounded-full",
                  isPast ? "bg-green-500" : "bg-gray-700"
                )}
                aria-hidden="true"
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}

function TemplateCard({
  template,
  selected,
  onSelect,
}: {
  template: Template;
  selected: boolean;
  onSelect: () => void;
}) {
  const Icon = ICON_MAP[template.icon];
  return (
    <button
      onClick={onSelect}
      className={cn(
        "rounded-lg border p-4 text-left transition-all",
        selected
          ? "border-[var(--primary)] bg-blue-500/10 ring-1 ring-[var(--primary)]"
          : "border-[var(--border)] bg-[var(--card)] hover:border-gray-500"
      )}
    >
      <div className="mb-2 flex items-center gap-2">
        <Icon className="h-5 w-5 text-blue-500" aria-hidden="true" />
        <span className="font-semibold text-sm">{template.name}</span>
      </div>
      <p className="text-xs text-[var(--muted-foreground)] leading-relaxed">
        {template.desc}
      </p>
      <div className="mt-2 flex items-center gap-3 text-xs text-[var(--muted-foreground)]">
        <span>{template.components} parts</span>
        <span>{template.cost}</span>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main wizard
// ---------------------------------------------------------------------------

export default function NewProjectPage() {
  const router = useRouter();
  const [step, setStep] = useState<WizardStep>(1);
  const [inputMode, setInputMode] = useState<InputMode>("template");
  const [description, setDescription] = useState("");
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);

  // Step 2 config
  const [boardName, setBoardName] = useState("my-board");
  const [boardWidth, setBoardWidth] = useState(80);
  const [boardHeight, setBoardHeight] = useState(50);
  const [layerCount, setLayerCount] = useState<2 | 4>(2);
  const [targetFab, setTargetFab] = useState("jlcpcb");
  const [packagePref, setPackagePref] = useState("0805");

  // Step 4 generate
  const [generating, setGenerating] = useState(false);
  const [generationDone, setGenerationDone] = useState(false);
  const [stageStatuses, setStageStatuses] = useState<StageStatus[]>(
    STAGES.map((s) => ({ stage: s, state: "pending" as const }))
  );

  const canProceedStep1 =
    (inputMode === "describe" && description.trim().length > 10) ||
    (inputMode === "template" && selectedTemplate !== null) ||
    (inputMode === "upload" && uploadedFile !== null);

  const canProceedStep2 = boardName.trim().length > 0;

  const handleNext = useCallback(() => {
    if (step < 4) {
      setStep((s) => (s + 1) as WizardStep);
    }
  }, [step]);

  const handleBack = useCallback(() => {
    if (step > 1) {
      setStep((s) => (s - 1) as WizardStep);
    }
  }, [step]);

  const handleStepClick = useCallback((target: WizardStep) => {
    setStep(target);
  }, []);

  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        setUploadedFile(file.name);
      }
    },
    []
  );

  const handleGenerate = useCallback(() => {
    setGenerating(true);
    setGenerationDone(false);

    // Simulate stage-by-stage generation
    const stageNames = [...STAGES];
    let idx = 0;

    const advance = () => {
      if (idx < stageNames.length) {
        setStageStatuses((prev) =>
          prev.map((s, i) => {
            if (i === idx) return { ...s, state: "running" as const };
            if (i < idx) return { ...s, state: "passed" as const };
            return s;
          })
        );

        idx++;
        setTimeout(() => {
          setStageStatuses((prev) =>
            prev.map((s, i) => {
              if (i < idx) return { ...s, state: "passed" as const };
              return s;
            })
          );
          if (idx < stageNames.length) {
            setTimeout(advance, 400);
          } else {
            setGenerating(false);
            setGenerationDone(true);
          }
        }, 800 + Math.random() * 600);
      }
    };

    setTimeout(advance, 300);
  }, []);

  // Auto-start generation when entering step 4
  const handleGoToGenerate = useCallback(() => {
    setStep(4);
    // Delay slightly so the step renders first
    setTimeout(() => {
      handleGenerate();
    }, 100);
  }, [handleGenerate]);

  const totalCost = PLACEHOLDER_PARTS.reduce((sum, p) => sum + p.price, 0);
  const allInStock = PLACEHOLDER_PARTS.every((p) => p.stock > 0);
  const basicCount = PLACEHOLDER_PARTS.filter((p) => p.basic).length;

  return (
    <>
      <NavHeader />

      <main className="mx-auto max-w-5xl flex-1 px-4 py-8">
        <StepIndicator currentStep={step} onStepClick={handleStepClick} />

        {/* ----- STEP 1: Describe ----- */}
        {step === 1 && (
          <div className="space-y-6">
            <div className="text-center">
              <h1 className="text-2xl font-bold">Describe Your Project</h1>
              <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                Choose how you want to get started
              </p>
            </div>

            {/* Input mode tabs */}
            <div className="flex items-center justify-center gap-1 rounded-lg bg-[var(--muted)] p-1">
              {(
                [
                  { mode: "template" as InputMode, label: "Templates", icon: Sparkles },
                  { mode: "describe" as InputMode, label: "Describe", icon: FileText },
                  { mode: "upload" as InputMode, label: "Upload", icon: Upload },
                ] as const
              ).map(({ mode, label, icon: Icon }) => (
                <button
                  key={mode}
                  onClick={() => setInputMode(mode)}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-colors",
                    inputMode === mode
                      ? "bg-[var(--card)] text-[var(--foreground)] shadow-sm"
                      : "text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
                  )}
                >
                  <Icon className="h-4 w-4" aria-hidden="true" />
                  {label}
                </button>
              ))}
            </div>

            {/* Template gallery */}
            {inputMode === "template" && (
              <div className="grid gap-3 sm:grid-cols-3">
                {TEMPLATES.map((tmpl) => (
                  <TemplateCard
                    key={tmpl.id}
                    template={tmpl}
                    selected={selectedTemplate === tmpl.id}
                    onSelect={() => setSelectedTemplate(tmpl.id)}
                  />
                ))}
              </div>
            )}

            {/* Free-text description */}
            {inputMode === "describe" && (
              <div>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="ESP32 board with 4 relay outputs, analog sensor inputs, and Ethernet connectivity for industrial monitoring..."
                  rows={6}
                  className="w-full rounded-lg border border-[var(--border)] bg-[var(--card)] p-4 text-sm text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:border-[var(--primary)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                />
                <p className="mt-1 text-xs text-[var(--muted-foreground)]">
                  Be specific about interfaces, sensors, power, and connectors.
                  {description.trim().length > 0 && description.trim().length <= 10 && (
                    <span className="ml-1 text-yellow-500">
                      Please provide a more detailed description.
                    </span>
                  )}
                </p>
              </div>
            )}

            {/* File upload */}
            {inputMode === "upload" && (
              <div className="flex flex-col items-center gap-3">
                <label
                  htmlFor="req-upload"
                  className="flex w-full cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed border-[var(--border)] py-12 text-[var(--muted-foreground)] transition-colors hover:border-[var(--primary)] hover:text-[var(--foreground)]"
                >
                  <Upload className="h-8 w-8" aria-hidden="true" />
                  <span className="text-sm font-medium">
                    {uploadedFile ?? "Drop requirements.json here or click to browse"}
                  </span>
                  <span className="text-xs">JSON format</span>
                  <input
                    id="req-upload"
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                </label>
              </div>
            )}

            {/* Nav */}
            <div className="flex justify-end">
              <Button onClick={handleNext} disabled={!canProceedStep1} className="gap-1.5">
                Next
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Button>
            </div>
          </div>
        )}

        {/* ----- STEP 2: Configure ----- */}
        {step === 2 && (
          <div className="space-y-6">
            <div className="text-center">
              <h1 className="text-2xl font-bold">Configure Board</h1>
              <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                Set board parameters. Defaults work for most projects.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Config form */}
              <Card>
                <CardContent className="space-y-5 p-6">
                  {/* Board name */}
                  <div>
                    <label
                      htmlFor="board-name"
                      className="mb-1 block text-sm font-medium"
                    >
                      Board Name
                    </label>
                    <input
                      id="board-name"
                      type="text"
                      value={boardName}
                      onChange={(e) => setBoardName(e.target.value)}
                      className="w-full rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-sm text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                    />
                  </div>

                  {/* Board size */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label
                        htmlFor="board-width"
                        className="mb-1 block text-sm font-medium"
                      >
                        Width (mm)
                      </label>
                      <input
                        id="board-width"
                        type="number"
                        min={10}
                        max={400}
                        value={boardWidth}
                        onChange={(e) => setBoardWidth(Number(e.target.value))}
                        className="w-full rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-sm text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="board-height"
                        className="mb-1 block text-sm font-medium"
                      >
                        Height (mm)
                      </label>
                      <input
                        id="board-height"
                        type="number"
                        min={10}
                        max={400}
                        value={boardHeight}
                        onChange={(e) => setBoardHeight(Number(e.target.value))}
                        className="w-full rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-sm text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                      />
                    </div>
                  </div>

                  {/* Layer count */}
                  <div>
                    <label className="mb-1 block text-sm font-medium">Layers</label>
                    <div className="flex gap-2">
                      {([2, 4] as const).map((n) => (
                        <button
                          key={n}
                          onClick={() => setLayerCount(n)}
                          className={cn(
                            "rounded-md border px-4 py-2 text-sm font-medium transition-colors",
                            layerCount === n
                              ? "border-[var(--primary)] bg-blue-500/10 text-[var(--foreground)]"
                              : "border-[var(--border)] text-[var(--muted-foreground)] hover:border-gray-500"
                          )}
                        >
                          {n}-layer
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Target fab */}
                  <div>
                    <label
                      htmlFor="target-fab"
                      className="mb-1 block text-sm font-medium"
                    >
                      Target Fabricator
                    </label>
                    <select
                      id="target-fab"
                      value={targetFab}
                      onChange={(e) => setTargetFab(e.target.value)}
                      className="w-full rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-sm text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                    >
                      <option value="jlcpcb">JLCPCB</option>
                      <option value="pcbway">PCBWay</option>
                      <option value="oshpark">OSHPark</option>
                    </select>
                  </div>

                  {/* Package preference */}
                  <div>
                    <label className="mb-1 block text-sm font-medium">
                      Preferred Package Size
                    </label>
                    <div className="flex gap-2">
                      {["0402", "0603", "0805", "1206"].map((pkg) => (
                        <button
                          key={pkg}
                          onClick={() => setPackagePref(pkg)}
                          className={cn(
                            "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                            packagePref === pkg
                              ? "border-[var(--primary)] bg-blue-500/10 text-[var(--foreground)]"
                              : "border-[var(--border)] text-[var(--muted-foreground)] hover:border-gray-500"
                          )}
                        >
                          {pkg}
                        </button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Right: AI Chat for requirement clarification */}
              <AIChat
                systemContext={`New project wizard. Template: ${selectedTemplate ?? "custom"}. Description: ${description}`}
                placeholder="Ask about board configuration..."
                welcomeMessage="I'll help you refine your project requirements. Based on your description, I have a few questions to make sure we get the design right. Feel free to ask about board sizing, layer count, component packages, or fabrication options."
              />
            </div>

            {/* Nav */}
            <div className="flex justify-between">
              <Button variant="outline" onClick={handleBack} className="gap-1.5">
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                Back
              </Button>
              <Button onClick={handleNext} disabled={!canProceedStep2} className="gap-1.5">
                Next
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Button>
            </div>
          </div>
        )}

        {/* ----- STEP 3: Parts Review ----- */}
        {step === 3 && (
          <div className="space-y-6">
            <div className="text-center">
              <h1 className="text-2xl font-bold">Review Parts</h1>
              <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                Verify components are available and in stock at JLCPCB.
              </p>
            </div>

            {/* Summary bar */}
            <div className="flex items-center justify-between rounded-lg border border-[var(--border)] bg-[var(--card)] p-3">
              <div className="flex items-center gap-4 text-sm">
                <span>{PLACEHOLDER_PARTS.length} components</span>
                <span className="text-[var(--muted-foreground)]">|</span>
                <span>${totalCost.toFixed(2)} total</span>
                <span className="text-[var(--muted-foreground)]">|</span>
                <span>{basicCount} basic parts</span>
              </div>
              {allInStock ? (
                <Badge variant="success">All In Stock</Badge>
              ) : (
                <Badge variant="warning">Stock Issues</Badge>
              )}
            </div>

            {/* Parts table */}
            <Card>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[var(--border)] text-left text-xs text-[var(--muted-foreground)]">
                        <th className="px-4 py-3 font-medium">Ref</th>
                        <th className="px-4 py-3 font-medium">Value</th>
                        <th className="px-4 py-3 font-medium">Package</th>
                        <th className="px-4 py-3 font-medium">LCSC</th>
                        <th className="px-4 py-3 font-medium text-right">Price</th>
                        <th className="px-4 py-3 font-medium text-right">Stock</th>
                        <th className="px-4 py-3 font-medium">Type</th>
                        <th className="px-4 py-3 font-medium" />
                      </tr>
                    </thead>
                    <tbody>
                      {PLACEHOLDER_PARTS.map((part) => (
                        <tr
                          key={part.ref}
                          className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--muted)]"
                        >
                          <td className="px-4 py-3 font-mono font-medium">{part.ref}</td>
                          <td className="px-4 py-3">{part.value}</td>
                          <td className="px-4 py-3 text-[var(--muted-foreground)]">
                            {part.pkg}
                          </td>
                          <td className="px-4 py-3 font-mono text-xs">{part.lcsc}</td>
                          <td className="px-4 py-3 text-right tabular-nums">
                            ${part.price.toFixed(3)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums">
                            {part.stock.toLocaleString()}
                          </td>
                          <td className="px-4 py-3">
                            {part.basic ? (
                              <Badge variant="success" className="text-[10px]">
                                Basic
                              </Badge>
                            ) : (
                              <Badge variant="warning" className="text-[10px]">
                                Extended
                              </Badge>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <Button variant="ghost" size="sm" className="text-xs">
                              Swap
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Nav */}
            <div className="flex justify-between">
              <Button variant="outline" onClick={handleBack} className="gap-1.5">
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                Back
              </Button>
              <Button onClick={handleGoToGenerate} className="gap-1.5">
                <Sparkles className="h-4 w-4" aria-hidden="true" />
                Generate Board
              </Button>
            </div>
          </div>
        )}

        {/* ----- STEP 4: Generate ----- */}
        {step === 4 && (
          <div className="space-y-6">
            <div className="text-center">
              <h1 className="text-2xl font-bold">
                {generationDone ? "Board Generated" : "Generating Your Board..."}
              </h1>
              <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                {generationDone
                  ? "Your PCB design is ready for review."
                  : "Running the AI pipeline. This takes a moment."}
              </p>
            </div>

            {/* Stage progress */}
            <div className="flex justify-center">
              <StageProgress stages={stageStatuses} />
            </div>

            {/* Result card */}
            {generationDone && (
              <div className="space-y-4">
                <ScoreCard grade="B" score={0.82} />

                <Card>
                  <CardContent className="p-6 text-center">
                    <div className="mx-auto mb-4 flex h-40 items-center justify-center rounded-lg border border-dashed border-[var(--border)] bg-[var(--muted)]">
                      <span className="text-sm text-[var(--muted-foreground)]">
                        Board preview will render here
                      </span>
                    </div>

                    <div className="flex flex-wrap items-center justify-center gap-3">
                      <Button
                        onClick={() => router.push(`/project/${boardName}`)}
                        className="gap-1.5"
                      >
                        View Project
                        <ArrowRight className="h-4 w-4" aria-hidden="true" />
                      </Button>
                      <Button variant="outline" className="gap-1.5">
                        <Download className="h-4 w-4" aria-hidden="true" />
                        Download KiCad Files
                      </Button>
                      <Button variant="outline" className="gap-1.5">
                        <ExternalLink className="h-4 w-4" aria-hidden="true" />
                        Order from JLCPCB
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Back button (only while generating or before) */}
            {!generationDone && !generating && (
              <div className="flex justify-start">
                <Button variant="outline" onClick={handleBack} className="gap-1.5">
                  <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                  Back
                </Button>
              </div>
            )}
          </div>
        )}
      </main>
    </>
  );
}
