"use client";

import { useEffect, useState } from "react";
import { audioFileUrl } from "@/lib/api";
import { NotePlayer } from "./NotePlayer";
import type { RollNote } from "./PianoRoll";

/**
 * Plays a generated `.mid` straight from the API: no server-side rendering step,
 * no Fluidsynth, no soundfont install. The file is parsed with @tonejs/midi and
 * synthesised by Tone.js, which also yields the note timings the piano roll draws.
 */
interface MidiPlayerProps {
  /** Output-relative filename, e.g. `generated_output.mid` or `arena/ab12-a.mid`. */
  filename: string;
  label?: string;
  height?: number;
  showRoll?: boolean;
}

/** A load outcome, tagged with the filename it belongs to. */
type Loaded =
  | { filename: string; ok: true; notes: RollNote[]; duration: number }
  | { filename: string; ok: false; message: string };

export function MidiPlayer({ filename, label, height = 180, showRoll = true }: MidiPlayerProps) {
  // Tagging the result with its filename means "loading" is derived from a prop
  // mismatch rather than from a setState in the effect body, which would cascade
  // an extra render on every mount.
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        // @tonejs/midi is client-only; keep it out of the server bundle.
        const { Midi } = await import("@tonejs/midi");
        const midi = await Midi.fromUrl(audioFileUrl(filename));
        if (cancelled) return;

        const notes = midi.tracks.flatMap((track) =>
          track.notes.map((note) => ({
            time: note.time,
            duration: note.duration,
            midi: note.midi,
            velocity: note.velocity,
          })),
        );

        setLoaded(
          notes.length > 0
            ? { filename, ok: true, notes, duration: midi.duration }
            : { filename, ok: false, message: "This MIDI file contains no notes." },
        );
      } catch (cause) {
        if (cancelled) return;
        setLoaded({
          filename,
          ok: false,
          message: cause instanceof Error ? cause.message : "Could not load the MIDI file.",
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [filename]);

  if (loaded?.filename !== filename) {
    return <p className="text-sm text-muted-foreground">Loading {label ?? filename}…</p>;
  }
  if (!loaded.ok) return <p className="text-sm text-destructive">{loaded.message}</p>;

  return (
    <NotePlayer
      notes={loaded.notes}
      duration={loaded.duration}
      label={label ?? filename}
      height={height}
      showRoll={showRoll}
      action={
        <a
          href={audioFileUrl(filename)}
          download
          className="shrink-0 text-xs tracking-wide text-brass uppercase underline underline-offset-2"
        >
          MIDI
        </a>
      }
    />
  );
}
