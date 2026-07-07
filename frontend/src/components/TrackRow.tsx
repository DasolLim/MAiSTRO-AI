"use client";

import { useRef, useState } from "react";
import { audioFileUrl, pollJob, renderAudio, type LibraryTrack } from "@/lib/api";

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
}

export function TrackRow({ track, onRendered }: { track: LibraryTrack; onRendered?: () => void }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [rendering, setRendering] = useState(false);

  if (!track.wav_filename) {
    const handleRender = () => {
      setRendering(true);
      renderAudio(track.midi_filename).then(({ job_id }) => {
        pollJob(job_id, (job) => {
          if (job.state === "done") {
            setRendering(false);
            onRendered?.();
          } else if (job.state === "error") {
            setRendering(false);
          }
        });
      });
    };

    return (
      <li className="flex items-center justify-between py-4 text-sm">
        <span className="font-display italic text-foreground">{track.name}</span>
        <button
          onClick={handleRender}
          disabled={rendering}
          className="cursor-pointer text-xs tracking-wide text-brass uppercase underline underline-offset-2 disabled:cursor-not-allowed disabled:text-muted-foreground"
        >
          {rendering ? "Rendering…" : "Render to audio"}
        </button>
      </li>
    );
  }

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) audio.pause();
    else audio.play();
  };

  return (
    <li className="flex items-center gap-4 py-4">
      <button
        onClick={togglePlay}
        aria-label={playing ? `Pause ${track.name}` : `Play ${track.name}`}
        className="flex h-11 w-11 shrink-0 cursor-pointer items-center justify-center rounded-full border border-border text-foreground transition-colors hover:border-brass hover:text-brass"
      >
        {playing ? (
          <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true">
            <rect x="3" y="2" width="3.5" height="12" />
            <rect x="9.5" y="2" width="3.5" height="12" />
          </svg>
        ) : (
          <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true">
            <path d="M4 2.5v11l10-5.5z" />
          </svg>
        )}
      </button>

      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-4">
          <span className="truncate font-display text-lg italic text-foreground">{track.name}</span>
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
        <div className="mt-2 h-px w-full bg-border">
          <div
            className="h-px origin-left bg-brass transition-transform duration-150 ease-linear"
            style={{ transform: `scaleX(${duration ? currentTime / duration : 0})`, width: "100%" }}
          />
        </div>
      </div>

      <audio
        ref={audioRef}
        src={audioFileUrl(track.wav_filename)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
      />
    </li>
  );
}
