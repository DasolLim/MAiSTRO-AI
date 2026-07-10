"use client";

import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { uploadMidiFiles } from "@/lib/api";
import { useDatasetStats, usePrepareDataset } from "@/lib/useJob";
import { StaffDivider } from "@/components/StaffDivider";
import { JobLog } from "@/components/JobLog";
import { ProgressBar } from "@/components/ProgressBar";

export default function DatasetPage() {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: stats, refetch: refreshStats } = useDatasetStats();
  const prepare = usePrepareDataset();

  const upload = useMutation({
    mutationFn: uploadMidiFiles,
    onSuccess: () => refreshStats(),
  });

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const midiFiles = Array.from(fileList).filter((f) => /\.midi?$/i.test(f.name));
    if (midiFiles.length > 0) upload.mutate(midiFiles);
  };

  const job = prepare.job;
  const isRunning = prepare.isRunning;
  const uploading = upload.isPending;

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
          onClick={() => prepare.start(false)}
          disabled={isRunning || !stats || stats.midi_file_count === 0}
          className="shrink-0 cursor-pointer border border-brass px-5 py-2.5 text-sm font-medium text-brass transition-colors hover:bg-brass hover:text-brass-foreground disabled:cursor-not-allowed disabled:border-border disabled:text-muted-foreground disabled:hover:bg-transparent"
        >
          {isRunning ? "Preparing…" : "Prepare dataset"}
        </button>
      </div>

      {upload.error && <p className="mt-4 text-sm text-destructive">{upload.error.message}</p>}

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
