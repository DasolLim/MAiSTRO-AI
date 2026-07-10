// `??` would accept an empty string, which is exactly how this was set in Vercel's
// Production environment: every request then resolved to a relative path, and the
// deployment answered its own /generate/options with a 404 HTML page. Treat blank
// as unset.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://localhost:8000";

export type JobState = "pending" | "running" | "done" | "error";

export interface JobStatus<TResult = unknown> {
  id: string;
  state: JobState;
  progress: Record<string, unknown>;
  log: string[];
  result: TResult | null;
  error: string | null;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!res.ok) {
    // FastAPI puts the human-readable reason in `detail`; surface that rather
    // than the raw body so the UI can show "train a second model first".
    const body = await res.text();
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {
      /* not JSON — use the raw text */
    }
    throw new ApiError(detail, res.status);
  }

  return res.json() as Promise<T>;
}

export function getJob<TResult = unknown>(jobId: string): Promise<JobStatus<TResult>> {
  return request(`/jobs/${jobId}`);
}

/* -------------------------------------------------------------- dataset */

export interface DatasetStats {
  midi_file_count: number;
  notes_prepared: boolean;
}

export function getDatasetStats(): Promise<DatasetStats> {
  return request("/dataset/stats");
}

export async function uploadMidiFiles(files: File[]): Promise<{ saved: string[]; count: number }> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await fetch(`${API_BASE_URL}/dataset/upload`, { method: "POST", body: form });
  if (!res.ok) throw new ApiError(await res.text(), res.status);
  return res.json();
}

export function prepareDataset(force = false): Promise<{ job_id: string }> {
  return request(`/dataset/prepare?force=${force}`, { method: "POST" });
}

/* ---------------------------------------------------------------- train */

export function startTraining(
  arch: string,
  epochs: number,
  batchSize: number,
): Promise<{ job_id: string }> {
  return request("/train/start", {
    method: "POST",
    body: JSON.stringify({ arch, epochs, batch_size: batchSize }),
  });
}

export interface TrainingHistory {
  arch: string;
  epochs: number;
  batch_size: number;
  history: { loss: number[]; val_loss?: number[] };
}

export function getTrainingHistory(arch: string): Promise<TrainingHistory> {
  return request(`/train/history/${arch}`);
}

/* ------------------------------------------------------------- generate */

export interface ArchOption {
  key: string;
  label: string;
  description: string;
  trained: boolean;
}

export interface MoodOption {
  key: string;
  label: string;
  description: string;
  temperature: number;
  scale: string;
}

export interface GenerateOptions {
  architectures: ArchOption[];
  default_architecture: string;
  sampling: {
    temperature: { default: number; min: number; max: number; step: number; greedy_below: number };
    top_p: number;
    top_k: number;
  };
  scales: string[];
  keys: string[];
  moods: MoodOption[];
}

export function getGenerateOptions(): Promise<GenerateOptions> {
  return request("/generate/options");
}

export interface GenerateParams {
  filename?: string;
  arch?: string;
  n_notes?: number;
  temperature?: number | null;
  top_k?: number;
  top_p?: number | null;
  key?: string | null;
  scale?: string;
  mood?: string | null;
  tempo_bpm?: number;
  seed?: number | null;
}

export interface GenerationMetrics {
  note_count: number;
  unique_token_ratio: number;
  repetition_rate: number;
  rest_fraction: number;
  distinct_pitch_classes: number;
  mean_pitch: number | null;
  pitch_range: number | null;
  notes_per_quarter: number | null;
  pitch_class_kl?: number;
}

export interface GenerateResult {
  filename: string;
  metrics: GenerationMetrics;
  config: Required<GenerateParams>;
}

export function startGeneration(params: GenerateParams = {}): Promise<{ job_id: string }> {
  return request("/generate", { method: "POST", body: JSON.stringify(params) });
}

/* ---------------------------------------------------------------- arena */

export interface ArenaClipRef {
  clip_id: string;
  midi_filename: string;
}

export interface ArenaPair {
  pair_id: string;
  a: ArenaClipRef;
  b: ArenaClipRef;
  config: Required<GenerateParams>;
}

export interface LeaderboardRow {
  arch: string;
  label: string;
  description: string;
  trained: boolean;
  elo: number;
  games: number;
  wins: number;
  losses: number;
  ties: number;
  win_rate: number | null;
}

export interface VoteReveal {
  pair_id: string;
  outcome: "a" | "b" | "tie";
  reveal: {
    a: { arch: string; metrics: GenerationMetrics };
    b: { arch: string; metrics: GenerationMetrics };
  };
  leaderboard: LeaderboardRow[];
}

export function startArenaPair(params: Partial<GenerateParams> = {}): Promise<{ job_id: string }> {
  return request("/arena/pair", { method: "POST", body: JSON.stringify(params) });
}

export function castVote(
  pairId: string,
  outcome: "a" | "b" | "tie",
  note?: string,
): Promise<VoteReveal> {
  return request("/arena/vote", {
    method: "POST",
    body: JSON.stringify({ pair_id: pairId, outcome, note }),
  });
}

export function getLeaderboard(): Promise<{
  leaderboard: LeaderboardRow[];
  matchups: [string, string][];
  recent_votes: unknown[];
}> {
  return request("/arena/leaderboard");
}

/* ------------------------------------------------------------- external */

export interface MusicGenStatus {
  available: boolean;
  reason: string | null;
  models: { name: string; description: string }[];
  default_model: string;
  max_duration: number;
  prompt_ideas: string[];
}

export function getMusicGenStatus(): Promise<MusicGenStatus> {
  return request("/external/musicgen/status");
}

export function startMusicGen(params: {
  prompt: string;
  duration: number;
  model_name?: string;
  temperature?: number;
}): Promise<{ job_id: string }> {
  return request("/external/musicgen/generate", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/* ---------------------------------------------------------------- audio */

export interface LibraryTrack {
  name: string;
  midi_filename: string;
  wav_filename: string | null;
  created_at: number;
  config: Required<GenerateParams> | null;
  metrics: GenerationMetrics | null;
}

export function getLibrary(): Promise<{ tracks: LibraryTrack[] }> {
  return request("/audio/library");
}

export function renderAudio(
  filename: string,
  mode: "generation" | "collaboration" = "generation",
): Promise<{ job_id: string }> {
  return request("/audio/render", { method: "POST", body: JSON.stringify({ filename, mode }) });
}

/** Filenames may contain a subdirectory (`arena/…`), so encode segments, not slashes. */
export function audioFileUrl(filename: string): string {
  const path = filename.split("/").map(encodeURIComponent).join("/");
  return `${API_BASE_URL}/audio/file/${path}`;
}

/* ----------------------------------------------------------------- jobs */

/** Resolve once a job reaches a terminal state; rejects if the job errored. */
export async function waitForJob<TResult = unknown>(
  jobId: string,
  { intervalMs = 1200, onUpdate }: { intervalMs?: number; onUpdate?: (job: JobStatus<TResult>) => void } = {},
): Promise<TResult> {
  for (;;) {
    const job = await getJob<TResult>(jobId);
    onUpdate?.(job);

    if (job.state === "error") throw new Error(job.error ?? "job failed");
    if (job.state === "done") return job.result as TResult;

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
