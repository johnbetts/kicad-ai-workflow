export const gradeColors: Record<string, string> = {
  A: "text-green-500",
  B: "text-lime-500",
  C: "text-yellow-500",
  D: "text-orange-500",
  F: "text-red-500",
};

export const gradeBgColors: Record<string, string> = {
  A: "bg-green-500/10 border-green-500/30",
  B: "bg-lime-500/10 border-lime-500/30",
  C: "bg-yellow-500/10 border-yellow-500/30",
  D: "bg-orange-500/10 border-orange-500/30",
  F: "bg-red-500/10 border-red-500/30",
};

export const severityColors: Record<string, string> = {
  fatal: "text-red-500 bg-red-500/10",
  recoverable: "text-orange-500 bg-orange-500/10",
  warning: "text-yellow-500 bg-yellow-500/10",
};

export const stageIcons: Record<string, string> = {
  requirements: "FileText",
  schematic: "GitBranch",
  pcb: "Cpu",
  validation: "ShieldCheck",
  production: "Package",
};

export const STAGES = [
  "requirements",
  "schematic",
  "pcb",
  "validation",
  "production",
] as const;

export type Stage = (typeof STAGES)[number];
