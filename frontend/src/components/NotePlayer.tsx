"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { PianoRoll, type RollNote } from "./PianoRoll";

/**
 * Synthesises a list of notes in the browser with Tone.js and draws a piano roll.
 *
 * Tone owns a single global transport and AudioContext, so only one player can
 * sound at a time. `activePlayer` enforces that explicitly rather than leaving two
 * instances to fight over the transport — which matters most in the arena, where
 * starting take B must silence take A.
 */

let activePlayer: { id: object; stop: () => void } | null = null;

interface NotePlayerProps {
  notes: RollNote[];
  duration: number;
  label: string;
  subtitle?: string;
  /** Rendered to the right of the title, e.g. a MIDI download link. */
  action?: React.ReactNode;
  height?: number;
  showRoll?: boolean;
}

export function NotePlayer({
  notes,
  duration,
  label,
  subtitle,
  action,
  height = 180,
  showRoll = true,
}: NotePlayerProps) {
  const [playing, setPlaying] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  // The transport is read every animation frame by the piano roll; a ref keeps
  // that off the React render path entirely.
  const transportRef = useRef<{ seconds: number } | null>(null);
  const teardownRef = useRef<(() => void) | null>(null);
  // Stable identity for this instance, so `stop` can tell whether *it* is the
  // player currently holding the global transport without referencing itself.
  const instanceId = useRef({});

  const stop = useCallback(() => {
    teardownRef.current?.();
    teardownRef.current = null;
    transportRef.current = null;
    if (activePlayer?.id === instanceId.current) activePlayer = null;
    setElapsed(0);
    setPlaying(false);
  }, []);

  // Never leave a synth running behind a navigation.
  useEffect(() => () => teardownRef.current?.(), []);

  const play = useCallback(async () => {
    activePlayer?.stop();

    // Tone touches `window` and `AudioContext` at import time, so it is pulled in
    // on the client only, never during the server render.
    const Tone = await import("tone");
    // Browsers only let an AudioContext start from a user gesture.
    await Tone.start();

    const synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "triangle" },
      envelope: { attack: 0.005, decay: 0.35, sustain: 0.15, release: 1.1 },
    }).toDestination();
    // Dense chords stack well past the default voice limit; without this the tail
    // of a chord is dropped silently.
    synth.maxPolyphony = 32;
    synth.volume.value = -8;

    const transport = Tone.getTransport();
    transport.stop();
    transport.cancel();
    transport.seconds = 0;

    // Tone.Part schedules each event at its own `time` field, which RollNote has.
    const part = new Tone.Part<RollNote>((time, note) => {
      synth.triggerAttackRelease(
        Tone.Frequency(note.midi, "midi").toNote(),
        note.duration,
        time,
        note.velocity,
      );
    }, notes).start(0);

    // Tear down after the release tail, and off the audio callback: disposing the
    // transport from inside one of its own scheduled events is asking for trouble.
    transport.scheduleOnce(() => setTimeout(stop, 0), duration + 1.2);

    teardownRef.current = () => {
      part.dispose();
      synth.dispose();
      transport.stop();
      transport.cancel();
    };
    transportRef.current = transport;
    activePlayer = { id: instanceId.current, stop };

    transport.start();
    setPlaying(true);
  }, [notes, duration, stop]);

  const getPlayhead = useCallback(() => transportRef.current?.seconds ?? 0, []);

  // The numeric readout only needs ~10fps; the playhead itself is drawn on the
  // canvas at full frame rate without touching React state.
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setElapsed(transportRef.current?.seconds ?? 0), 100);
    return () => clearInterval(id);
  }, [playing]);

  return (
    <div>
      <div className="flex items-center gap-4">
        <button
          onClick={playing ? stop : play}
          disabled={notes.length === 0}
          aria-label={playing ? `Stop ${label}` : `Play ${label}`}
          className="flex h-11 w-11 shrink-0 cursor-pointer items-center justify-center rounded-full border border-border text-foreground transition-colors hover:border-brass hover:text-brass disabled:cursor-not-allowed disabled:text-muted-foreground"
        >
          {playing ? (
            <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <rect x="3" y="3" width="10" height="10" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <path d="M4 2.5v11l10-5.5z" />
            </svg>
          )}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-4">
            <span className="truncate font-display text-lg italic text-foreground">{label}</span>
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
              {formatTime(elapsed)} / {formatTime(duration)}
            </span>
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {subtitle ?? `${notes.length} notes · synthesised in your browser`}
          </p>
        </div>

        {action}
      </div>

      {showRoll && notes.length > 0 && (
        <div className="mt-4 border border-border">
          <PianoRoll notes={notes} getPlayhead={getPlayhead} duration={duration} height={height} />
        </div>
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${rest}`;
}
