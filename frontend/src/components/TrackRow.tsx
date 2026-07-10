"use client";

import { useState } from "react";
import { MidiPlayer } from "@/components/MidiPlayer";
import { audioFileUrl, type LibraryTrack } from "@/lib/api";
import { useRenderAudio } from "@/lib/useJob";

/**
 * One composition in the listening room.
 *
 * MIDI plays immediately, synthesised in the browser — no soundfont, no round
 * trip. Rendering to WAV through Fluidsynth is now optional, for when you want
 * the sampled piano rather than the synth, or a file to hand to someone else.
 */
export function TrackRow({ track }: { track: LibraryTrack }) {
  const [expanded, setExpanded] = useState(false);
  const render = useRenderAudio();

  return (
    <li className="py-5">
      <MidiPlayer filename={track.midi_filename} label={track.name} showRoll={expanded} />

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 pl-15">
        <button
          onClick={() => setExpanded((open) => !open)}
          className="cursor-pointer text-xs tracking-wide text-muted-foreground uppercase underline underline-offset-2 hover:text-brass"
        >
          {expanded ? "Hide piano roll" : "Show piano roll"}
        </button>

        {track.config && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {track.config.arch?.replace(/_/g, " + ")} · T{track.config.temperature?.toFixed(2)}
            {track.config.key && ` · ${track.config.key} ${track.config.scale?.replace(/_/g, " ")}`}
            {track.config.seed != null && ` · seed ${track.config.seed}`}
          </span>
        )}

        {track.metrics && (
          <span className="text-xs text-muted-foreground tabular-nums">
            repetition {track.metrics.repetition_rate.toFixed(3)}
          </span>
        )}

        {track.wav_filename ? (
          <a
            href={audioFileUrl(track.wav_filename)}
            download
            className="text-xs tracking-wide text-brass uppercase underline underline-offset-2"
          >
            WAV
          </a>
        ) : (
          <button
            onClick={() => render.mutate(track.midi_filename)}
            disabled={render.isPending}
            className="cursor-pointer text-xs tracking-wide text-muted-foreground uppercase underline underline-offset-2 hover:text-brass disabled:cursor-not-allowed"
          >
            {render.isPending ? "Rendering…" : "Render sampled piano"}
          </button>
        )}
      </div>

      {render.error && <p className="mt-2 pl-15 text-xs text-destructive">{render.error.message}</p>}
    </li>
  );
}
