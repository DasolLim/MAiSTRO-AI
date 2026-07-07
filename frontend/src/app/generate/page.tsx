"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { getLibrary, pollJob, startGeneration, type JobStatus, type LibraryTrack } from "@/lib/api";
import { StaffDivider } from "@/components/StaffDivider";
import { JobLog } from "@/components/JobLog";
import { TrackRow } from "@/components/TrackRow";

interface GenerateResult {
  filename: string;
}

export default function GeneratePage() {
  const [job, setJob] = useState<JobStatus<GenerateResult> | null>(null);
  const [recentTracks, setRecentTracks] = useState<LibraryTrack[]>([]);
  const stopPolling = useRef<() => void>(() => {});

  const isRunning = job?.state === "running" || job?.state === "pending";

  const refreshLibrary = () => {
    getLibrary().then(({ tracks }) => setRecentTracks(tracks.slice(0, 3)));
  };

  useEffect(() => {
    refreshLibrary();
    return () => stopPolling.current();
  }, []);

  const handleGenerate = () => {
    stopPolling.current();
    startGeneration().then(({ job_id }) => {
      stopPolling.current = pollJob<GenerateResult>(job_id, (update) => {
        setJob(update);
        if (update.state === "done") refreshLibrary();
      });
    });
  };

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        III. Generate
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground">A new composition</h1>
      <p className="mt-4 max-w-[65ch] text-muted-foreground">
        Sample a fresh sequence from the trained model, seeded from a random passage in the
        dataset, and write it out as a MIDI file.
      </p>

      <StaffDivider className="mt-10" />

      <div className="mt-8">
        <button
          onClick={handleGenerate}
          disabled={isRunning}
          className="cursor-pointer border border-brass px-5 py-2.5 text-sm font-medium text-brass transition-colors hover:bg-brass hover:text-brass-foreground disabled:cursor-not-allowed disabled:border-border disabled:text-muted-foreground disabled:hover:bg-transparent"
        >
          {isRunning ? "Composing…" : "Generate composition"}
        </button>

        {job && (
          <div className="mt-6">
            {job.state === "done" && job.result && (
              <p className="text-sm text-verdigris">
                Saved <span className="font-display italic">{job.result.filename}</span>. Render
                it to audio from the{" "}
                <Link href="/library" className="text-brass underline underline-offset-2">
                  listening room
                </Link>
                .
              </p>
            )}
            {job.state === "error" && <p className="text-sm text-destructive">{job.error}</p>}
            <JobLog lines={job.log} />
          </div>
        )}
      </div>

      {recentTracks.length > 0 && (
        <div className="mt-16">
          <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
            Recent
          </p>
          <StaffDivider className="mt-4" />
          <ul className="divide-y divide-border">
            {recentTracks.map((track) => (
              <TrackRow key={track.midi_filename} track={track} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
