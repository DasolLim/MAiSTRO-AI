"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getDatasetStats,
  pollJob,
  prepareDataset,
  uploadMidiFiles,
  type DatasetStats,
  type JobStatus,
} from "@/lib/api";
import { StaffDivider } from "@/components/StaffDivider";
import { JobLog } from "@/components/JobLog";
import { ProgressBar } from "@/components/ProgressBar";

interface PrepareResult {
  midi_file_count: number;
  note_count: number;
  vocab_size: number;
}

export default function DatasetPage() {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [job, setJob] = useState<JobStatus<PrepareResult> | null>(null);
  const stopPolling = useRef<() => void>(() => {});
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshStats = useCallback(() => {
    getDatasetStats().then(setStats).catch(() => setStats(null));
  }, []);

  useEffect(() => {
    refreshStats();
    return () => stopPolling.current();
  }, [refreshStats]);

  const handleFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const midiFiles = Array.from(fileList).filter((f) => /\.midi?$/i.test(f.name));
    if (midiFiles.length === 0) return;

    setUploading(true);
    try {
      await uploadMidiFiles(midiFiles);
      refreshStats();
    } finally {
      setUploading(false);
    }
  };

  const handlePrepare = () => {
    prepareDataset().then(({ job_id }) => {
      stopPolling.current = pollJob<PrepareResult>(job_id, (update) => {
        setJob(update);
        if (update.state === "done") refreshStats();
      });
    });
  };

  const isRunning = job?.state === "running" || job?.state === "pending";

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        I. Prepare
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground">The dataset</h1>
      <p className="mt-4 max-w-[65ch] text-muted-foreground">
        Upload a folder of <span className="text-foreground">.mid</span> files. They are parsed
        into a single sequence of notes, chords and rests that the model will train on.
      </p>

      <StaffDivider className="mt-10" />

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        className={`mt-8 flex cursor-pointer flex-col items-center gap-3 border py-16 text-center transition-colors ${
          dragOver ? "border-brass text-brass" : "border-border text-muted-foreground hover:border-brass/60"
        }`}
      >
        <span className="font-display text-xl italic">
          {uploading ? "Uploading…" : "Drop MIDI files here"}
        </span>
        <span className="text-xs tracking-wide uppercase">or click to choose files</span>
        <input
          ref={fileInputRef}
          type="file"
          accept=".mid,.midi"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      <div className="mt-8 flex items-baseline justify-between gap-6">
        <p className="text-sm text-muted-foreground">
          {stats ? (
            <>
              <span className="text-foreground">{stats.midi_file_count}</span> MIDI file
              {stats.midi_file_count === 1 ? "" : "s"} in the dataset ·{" "}
              {stats.notes_prepared ? "notes already prepared" : "notes not yet prepared"}
            </>
          ) : (
            "Loading dataset status…"
          )}
        </p>
        <button
          onClick={handlePrepare}
          disabled={isRunning || !stats || stats.midi_file_count === 0}
          className="shrink-0 cursor-pointer border border-brass px-5 py-2.5 text-sm font-medium text-brass transition-colors hover:bg-brass hover:text-brass-foreground disabled:cursor-not-allowed disabled:border-border disabled:text-muted-foreground disabled:hover:bg-transparent"
        >
          {isRunning ? "Preparing…" : "Prepare dataset"}
        </button>
      </div>

      {job && (
        <div className="mt-6">
          {isRunning && <ProgressBar fraction={0.5} />}
          {job.state === "done" && job.result && (
            <p className="text-sm text-verdigris">
              Extracted {job.result.note_count.toLocaleString()} notes across{" "}
              {job.result.midi_file_count} files ({job.result.vocab_size.toLocaleString()} unique
              tokens).
            </p>
          )}
          {job.state === "error" && (
            <p className="text-sm text-destructive">{job.error}</p>
          )}
          <JobLog lines={job.log} />
        </div>
      )}
    </div>
  );
}
