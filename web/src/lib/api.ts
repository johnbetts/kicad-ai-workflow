const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export interface PipelineError {
  code: string;
  message: string;
  cause: string;
  fix: string;
  severity: string;
  ref: string;
  stage: string;
}

export interface StageOutcome {
  stage: string;
  success: boolean;
  artifacts: string[];
  errors: PipelineError[];
  warnings: string[];
  duration_secs: number;
  score_overall: number | null;
  score_grade: string | null;
}

export interface PipelineResult {
  outcomes: StageOutcome[];
  overall_success: boolean;
  board_path: string | null;
  production_path: string | null;
}

export interface BoardSummary {
  name: string;
  grade: string;
  score: number | null;
  stage: string;
  has_evidence: boolean;
}

export interface PartResult {
  lcsc: string;
  mfr: string;
  description: string;
  package: string;
  stock: number;
  price: number;
  basic: boolean;
}

export interface PartSearchResponse {
  query: string;
  results: PartResult[];
  total: number;
}

export interface BoardComponent {
  ref: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  type: "ic" | "passive" | "connector" | "other";
}

export interface BoardDetail {
  name: string;
  images: Record<string, string>;
  crops: Array<{ ref: string; url: string; filename: string }>;
  score: { grade: string; overall_score: number; breakdown: Record<string, number> } | null;
  files: string[];
  pcb_file_url: string;
  sch_file_url: string;
  has_pcb: boolean;
  has_schematic: boolean;
  has_requirements: boolean;
  board_size: { width: number; height: number; origin_x?: number; origin_y?: number } | null;
  components: BoardComponent[];
}

// --- API Functions ---

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export async function getHealth(): Promise<{ status: string; version: string }> {
  return fetchJSON("/api/health");
}

export async function listBoards(): Promise<BoardSummary[]> {
  return fetchJSON("/api/evidence/boards");
}

export async function getBoardDetail(boardName: string): Promise<BoardDetail> {
  return fetchJSON(`/api/evidence/board/${boardName}`);
}

export async function searchParts(
  query: string,
  opts?: { basicOnly?: boolean; inStock?: boolean; limit?: number }
): Promise<PartSearchResponse> {
  const params = new URLSearchParams({ q: query });
  if (opts?.basicOnly) params.set("basic_only", "true");
  if (opts?.inStock) params.set("in_stock", "true");
  if (opts?.limit) params.set("limit", String(opts.limit));
  return fetchJSON(`/api/parts/search?${params}`);
}

export async function runStage(
  stage: string,
  requirementsJson: string,
  opts?: { boardName?: string; placementMode?: string }
): Promise<StageOutcome> {
  const params = new URLSearchParams({ requirements_json: requirementsJson });
  return fetchJSON(`/api/pipeline/run/${stage}?${params}`, {
    method: "POST",
    body: JSON.stringify({
      board_name: opts?.boardName || "board",
      placement_mode: opts?.placementMode || "grouped",
    }),
  });
}

export async function runFullPipeline(
  requirementsJson: string,
  opts?: { boardName?: string }
): Promise<PipelineResult> {
  const params = new URLSearchParams({ requirements_json: requirementsJson });
  return fetchJSON(`/api/pipeline/run-full?${params}`, {
    method: "POST",
    body: JSON.stringify({
      board_name: opts?.boardName || "board",
    }),
  });
}

// --- Component Review Types ---

export interface ComponentSummary {
  component_id: string;
  ref: string;
  value: string;
  footprint_id: string;
  description: string;
  verification_status: "verified" | "failed" | "pending" | "unknown";
  expected_pads: number;
  known_issues_count: number;
}

export interface CheckResult {
  name: string;
  passed: boolean;
  detail: string;
  severity: "critical" | "major" | "minor" | "info";
}

export interface KnownIssue {
  description: string;
  severity: string;
  status: string;
  date?: string;
}

export interface ComponentPin {
  number: string;
  name: string;
  type: string;
}

export interface ComponentDetail {
  component_id: string;
  ref: string;
  value: string;
  footprint_id: string;
  description: string;
  verification_status: string;
  last_verified_commit: string;
  expected_pads: number;
  expected_pad_type: string;
  body_width_mm: number;
  body_height_mm: number;
  model_rotation_z: number;
  model_offset_xy_max_mm: number;
  pins: ComponentPin[];
  known_issues: KnownIssue[];
  verified_fixes: Array<{ description: string; date?: string }>;
  checks: CheckResult[];
  images: Record<string, string>;
  lcsc_url: string;
  datasheet_url: string;
  kicad_footprint_lib: string;
  model_3d_path: string;
  package_category: string;
}

export async function listComponents(): Promise<ComponentSummary[]> {
  return fetchJSON("/api/components/list");
}

export async function getComponentDetail(id: string): Promise<ComponentDetail> {
  return fetchJSON(`/api/components/detail/${id}`);
}

export async function reviewComponent(
  id: string,
  action: "approve" | "rework" | "flag_issue",
  notes: string = "",
  issueDescription: string = ""
): Promise<{ status: string; verification_status: string }> {
  return fetchJSON(`/api/components/review/${id}`, {
    method: "POST",
    body: JSON.stringify({ action, notes, issue_description: issueDescription }),
  });
}

export async function triggerVerification(id: string): Promise<{
  component_id: string;
  passed: boolean;
  checks: CheckResult[];
}> {
  return fetchJSON(`/api/components/verify/${id}`, { method: "POST" });
}
