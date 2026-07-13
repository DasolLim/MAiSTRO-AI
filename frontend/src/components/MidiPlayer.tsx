"use client";

import { useEffect, useState } from "react";
import { audioFileUrl } from "@/lib/api";
import { NotePlayer } from "./NotePlayer";
import type { RollNote } from "./PianoRoll";

/**
 * Plays a generated MIDI: no server-side rendering step, no Fluidsynth, no soundfont
 * install. The file is parsed with @tonejs/midi and synthesised by Tone.js, which
 * also yields the note timings the piano roll draws.
 *
 * The MIDI arrives one of two ways. The local backend writes it to disk and serves it
 * by name. The deployed serverless function has no writable disk, so it returns the
 * bytes inline as base64. Either way the player only ever sees notes.
 */
interface MidiPlayerProps {
  /** Output-relative filename, e.g. `generated_output.mid` or `arena/ab12-a.mid`. */
  filename?: string;
  /** Base64 MIDI, as returned by the serverless `/generate/sync`. */
  midiBase64?: string;
  label?: string;
  height?: number;
  showRoll?: boolean;
}

type Loaded =
  | { source: string; ok: true; notes: RollNote[]; duration: number }
  | { source: string; ok: false; message: string };

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

export function MidiPlayer({
  filename,
  midiBase64,
  label,
  height = 180,
  showRoll = true,
}: MidiPlayerProps) {
  // Identifies which MIDI the loaded state belongs to, so "loading" is derived from a
  // prop mismatch rather than from a setState in the effect body.
  const source = filename ?? (midiBase64 ? `inline:${midiBase64.length}` : "none");
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        // @tonejs/midi is client-only; keep it out of the server bundle.
        const { Midi } = await import("@tonejs/midi");
        const midi = midiBase64
          ? new Midi(base64ToArrayBuffer(midiBase64))
          : await Midi.fromUrl(audioFileUrl(filename as string));
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
            ? { source, ok: true, notes, duration: midi.duration }
            : { source, ok: false, message: "This MIDI file contains no notes." },
        );
      } catch (cause) {
        if (cancelled) return;
        setLoaded({
          source,
          ok: false,
          message: cause instanceof Error ? cause.message : "Could not load the MIDI file.",
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [source, filename, midiBase64]);

  if (loaded?.source !== source) {
    return <p className="text-sm text-muted-foreground">Loading {label ?? filename}…</p>;
  }
  if (!loaded.ok) return <p className="text-sm text-destructive">{loaded.message}</p>;

  const downloadHref = midiBase64
    ? `data:audio/midi;base64,${midiBase64}`
    : audioFileUrl(filename as string);

  return (
    <NotePlayer
      notes={loaded.notes}
      duration={loaded.duration}
      label={label ?? filename ?? "Composition"}
      height={height}
      showRoll={showRoll}
      action={
        <a
          href={downloadHref}
          download={filename ?? "maistro.mid"}
          className="shrink-0 text-xs tracking-wide text-brass uppercase underline underline-offset-2"
        >
          MIDI
        </a>
      }
    />
  );
}
