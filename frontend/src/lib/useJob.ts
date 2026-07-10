"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  generateSync,
  getCapabilities,
  getDatasetStats,
  getGenerateOptions,
  getJob,
  getLeaderboard,
  getLibrary,
  getMusicGenStatus,
  prepareDataset,
  renderAudio,
  startArenaPair,
  startGeneration,
  startMusicGen,
  startTraining,
  waitForJob,
  type ArenaPair,
  type Capabilities,
  type GenerateParams,
  type GenerateResult,
  type GenerationMetrics,
  type JobStatus,
  type SyncGenerateResult,
} from "./api";

const POLL_INTERVAL_MS = 1200;

/**
 * Poll one background job until it settles.
 *
 * `refetchInterval` returns false once the job is terminal, which stops the timer
 * without an effect or a cleanup function. The job log streams in as `data.log`,
 * so a caller gets progress for free.
 */
export function useJob<TResult = unknown>(jobId: string | null) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob<TResult>(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state === "done" || state === "error" ? false : POLL_INTERVAL_MS;
    },
    // Every poll is a new snapshot; caching one would freeze the log.
    staleTime: 0,
    gcTime: 5 * 60_000,
  });
}

/** A job-backed action: start it, then watch it. Returns both halves to the caller. */
function useJobAction<TResult, TVariables>(
  start: (vars: TVariables) => Promise<{ job_id: string }>,
  onDone?: (result: TResult) => void,
) {
  const [jobId, setJobId] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: start,
    onMutate: () => setJobId(null),
    onSuccess: ({ job_id }) => setJobId(job_id),
  });

  const job = useJob<TResult>(jobId);

  // Fire onDone exactly once per job, from an effect rather than during render.
  const settledJobId = useRef<string | null>(null);
  useEffect(() => {
    if (!jobId || job.data?.state !== "done" || settledJobId.current === jobId) return;
    settledJobId.current = jobId;
    onDone?.(job.data.result as TResult);
  }, [jobId, job.data, onDone]);

  const isRunning =
    mutation.isPending ||
    (job.data ? job.data.state === "pending" || job.data.state === "running" : jobId !== null);

  return {
    start: mutation.mutate,
    reset: () => setJobId(null),
    job: job.data as JobStatus<TResult> | undefined,
    isRunning,
    // A failed POST (503 from MusicGen, 409 from the arena) never produces a job.
    error: mutation.error ?? (job.data?.state === "error" ? new Error(job.data.error ?? "") : null),
  };
}

/* --------------------------------------------------------------- queries */

/**
 * Which backend are we talking to? Answered once and cached for the session.
 *
 * A backend that predates this endpoint, or one that is simply down, is treated as
 * "local with everything on" — the individual pages already handle their own
 * failures, and guessing "serverless" would wrongly hide features.
 */
export function useCapabilities() {
  return useQuery({
    queryKey: ["capabilities"],
    queryFn: getCapabilities,
    staleTime: Infinity,
    retry: false,
  });
}

/** True only when we know the backend is the stateless deployment. */
export function useFeature(name: keyof Capabilities["features"]): {
  available: boolean;
  serverless: boolean;
  reason: string | null;
} {
  const { data } = useCapabilities();
  if (!data) return { available: true, serverless: false, reason: null };
  return {
    available: data.features[name],
    serverless: data.mode === "serverless",
    reason: data.reason,
  };
}

export function useDatasetStats() {
  return useQuery({ queryKey: ["dataset", "stats"], queryFn: getDatasetStats, staleTime: 0 });
}

export function useGenerateOptions() {
  return useQuery({
    queryKey: ["generate", "options"],
    queryFn: getGenerateOptions,
    staleTime: 5 * 60_000,
  });
}

export function useLibrary() {
  return useQuery({ queryKey: ["library"], queryFn: getLibrary, staleTime: 0 });
}

export function useLeaderboard() {
  return useQuery({ queryKey: ["arena", "leaderboard"], queryFn: getLeaderboard });
}

export function useMusicGenStatus(enabled = true) {
  return useQuery({
    queryKey: ["external", "musicgen", "status"],
    queryFn: getMusicGenStatus,
    staleTime: 60_000,
    // The serverless deployment has no /external routes at all; asking would 404.
    enabled,
  });
}

/* ------------------------------------------------------------- mutations */

/** One composition, however the backend chose to hand it over. */
export interface GeneratedPiece {
  /** Local backend: the piece was written to disk and is served by filename. */
  filename?: string;
  /** Serverless: no disk, so the MIDI comes back inline. */
  midiBase64?: string;
  metrics: GenerationMetrics;
  config: Required<GenerateParams>;
}

/**
 * Generate a piece, hiding which transport the backend offers.
 *
 * The local backend runs generation as a background job and streams a log. The
 * serverless function has nowhere to keep a job between the POST that starts it and
 * the GET that polls it, so it answers in one request. Both hooks are always called
 * — only one of them is ever started.
 */
export function useGenerate() {
  const { data: capabilities } = useCapabilities();
  const serverless = capabilities?.mode === "serverless";

  const invalidateLibrary = useInvalidateLibrary();
  // Refresh the library the moment a composition lands, so the recent-tracks
  // list under the button is never one generation behind.
  const job = useJobAction<GenerateResult, GenerateParams>(startGeneration, invalidateLibrary);
  const sync = useMutation<SyncGenerateResult, Error, GenerateParams>({ mutationFn: generateSync });

  const jobResult = job.job?.state === "done" ? job.job.result : null;
  const piece: GeneratedPiece | null = serverless
    ? sync.data
      ? { midiBase64: sync.data.midi_base64, metrics: sync.data.metrics, config: sync.data.config }
      : null
    : jobResult
      ? { filename: jobResult.filename, metrics: jobResult.metrics, config: jobResult.config }
      : null;

  return {
    start: (params: GenerateParams) => (serverless ? sync.mutate(params) : job.start(params)),
    isRunning: serverless ? sync.isPending : job.isRunning,
    error: serverless ? sync.error : job.error,
    // The serverless call is a single request, so there is no progress to stream.
    log: serverless ? [] : (job.job?.log ?? []),
    started: serverless ? sync.isPending || sync.isSuccess || sync.isError : job.job !== undefined,
    piece,
  };
}

export interface PrepareResult {
  midi_file_count: number;
  note_count: number;
  vocab_size: number;
}

export function usePrepareDataset() {
  const queryClient = useQueryClient();
  const onDone = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["dataset", "stats"] });
  }, [queryClient]);

  return useJobAction<PrepareResult, boolean>((force) => prepareDataset(force), onDone);
}

export interface TrainResult {
  arch: string;
  checkpoint_dir: string;
  n_vocab: number;
  sequences: number;
  history: { loss: number[]; val_loss?: number[] };
}

export function useTrain() {
  const queryClient = useQueryClient();
  // A newly trained architecture changes which models can enter the arena and
  // which appear as trained in the generate dropdown.
  const onDone = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["generate", "options"] });
    queryClient.invalidateQueries({ queryKey: ["arena", "leaderboard"] });
  }, [queryClient]);

  return useJobAction<TrainResult, { arch: string; epochs: number; batchSize: number }>(
    ({ arch, epochs, batchSize }) => startTraining(arch, epochs, batchSize),
    onDone,
  );
}

export function useArenaPair() {
  return useJobAction<ArenaPair, Partial<GenerateParams>>(startArenaPair);
}

export function useMusicGen() {
  return useJobAction<
    { wav_filename: string; prompt: string; model_name: string; duration: number },
    { prompt: string; duration: number; model_name?: string; temperature?: number }
  >(startMusicGen);
}

/** Render a MIDI file to audio and resolve only when the wav exists. */
export function useRenderAudio() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (filename: string) => {
      const { job_id } = await renderAudio(filename);
      return waitForJob<{ wav_filename: string }>(job_id);
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["library"] }),
  });
}

export function useInvalidateLibrary() {
  const queryClient = useQueryClient();
  return useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["library"] });
  }, [queryClient]);
}
