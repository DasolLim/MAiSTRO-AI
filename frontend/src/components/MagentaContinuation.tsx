"use client";

import { useState } from "react";
import { Button, Field, Select, Slider } from "./Controls";
import { NotePlayer } from "./NotePlayer";
import type { RollNote } from "./PianoRoll";

/**
 * Melody continuation running entirely in the browser via Magenta.js + TensorFlow.js.
 *
 * Nothing here touches MAiSTRO's backend. The MelodyRNN checkpoint is fetched from
 * Google's public bucket, the inference runs on the user's GPU through WebGL, and
 * the result is played by the same Tone.js synth as everything else. It is the
 * counterweight to the server-side LSTM: same task, opposite deployment model —
 * no Python process, no cold start, no GPU bill, but a first-load cost of a few MB
 * of weights and whatever the visitor's laptop can manage.
 *
 * Magenta.js is unmaintained (last release 2022) and pins an old TensorFlow.js, so
 * it is imported dynamically and its failures are contained to this component.
 */

const CHECKPOINT =
  "https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/melody_rnn";

const STEPS_PER_QUARTER = 4;
const QPM = 120;

interface Motif {
  key: string;
  label: string;
  /** [pitch, startStep, endStep] triples over a 4/4 bar at 4 steps per quarter. */
  notes: [number, number, number][];
}

const MOTIFS: Motif[] = [
  {
    key: "ascending",
    label: "Ascending C major",
    notes: [
      [60, 0, 2],
      [62, 2, 4],
      [64, 4, 6],
      [67, 6, 8],
    ],
  },
  {
    key: "fanfare",
    label: "Fanfare",
    notes: [
      [60, 0, 1],
      [64, 1, 2],
      [67, 2, 4],
      [72, 4, 8],
    ],
  },
  {
    key: "lament",
    label: "Descending lament",
    notes: [
      [69, 0, 2],
      [67, 2, 4],
      [65, 4, 6],
      [64, 6, 8],
    ],
  },
];

type Status = "idle" | "loading-model" | "generating" | "ready" | "error";

export function MagentaContinuation() {
  const [motifKey, setMotifKey] = useState(MOTIFS[0].key);
  const [temperature, setTemperature] = useState(1.1);
  const [bars, setBars] = useState(4);

  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<RollNote[]>([]);
  const [duration, setDuration] = useState(0);

  const motif = MOTIFS.find((m) => m.key === motifKey) ?? MOTIFS[0];

  const generate = async () => {
    setStatus("loading-model");
    setError(null);

    try {
      // Loaded on demand: pulling in TensorFlow.js on every page view would cost
      // every visitor several megabytes for a feature most never open.
      const mm = await import("@magenta/music/es6");

      const rnn = new mm.MusicRNN(CHECKPOINT);
      await rnn.initialize();

      setStatus("generating");

      const seed = {
        notes: motif.notes.map(([pitch, quantizedStartStep, quantizedEndStep]) => ({
          pitch,
          quantizedStartStep,
          quantizedEndStep,
        })),
        quantizationInfo: { stepsPerQuarter: STEPS_PER_QUARTER },
        totalQuantizedSteps: 8,
        tempos: [{ time: 0, qpm: QPM }],
      };

      const steps = bars * 4 * STEPS_PER_QUARTER;
      const continuation = await rnn.continueSequence(
        seed as never,
        steps,
        temperature,
      );

      // The model returns quantized steps; unquantizing maps them onto seconds so
      // the same Tone.js player and piano roll can consume them unchanged.
      const merged = mm.sequences.concatenate([seed as never, continuation]);
      const unquantized = mm.sequences.unquantizeSequence(merged, QPM);

      // Every field on a Magenta NoteSequence is optional in its protobuf types,
      // so drop anything that lacks the pitch and timing a note needs.
      const rollNotes: RollNote[] = (unquantized.notes ?? []).flatMap((note) => {
        const { pitch, startTime, endTime } = note;
        if (pitch == null || startTime == null || endTime == null) return [];
        return [{ time: startTime, duration: endTime - startTime, midi: pitch, velocity: 0.8 }];
      });

      rnn.dispose();

      if (rollNotes.length === 0) throw new Error("The model returned an empty sequence.");

      setNotes(rollNotes);
      setDuration(Math.max(...rollNotes.map((n) => n.time + n.duration)));
      setStatus("ready");
    } catch (cause) {
      setStatus("error");
      setError(
        cause instanceof Error
          ? `${cause.message} (Magenta loads its checkpoint from Google's servers — check your connection.)`
          : "Magenta failed to load.",
      );
    }
  };

  const busy = status === "loading-model" || status === "generating";

  return (
    <div>
      <div className="grid gap-8 sm:grid-cols-3">
        <Field label="Seed motif" hint="Four notes the model has to continue from.">
          <Select
            value={motifKey}
            onChange={setMotifKey}
            options={MOTIFS.map((m) => ({ value: m.key, label: m.label }))}
          />
        </Field>

        <Field label="Temperature" hint="MelodyRNN's own sampling dial.">
          <Slider value={temperature} onChange={setTemperature} min={0.5} max={1.8} step={0.05} />
        </Field>

        <Field label="Bars to add" hint="At 120bpm, 4/4.">
          <Slider value={bars} onChange={setBars} min={1} max={8} step={1} format={(v) => `${v}`} />
        </Field>
      </div>

      <div className="mt-8">
        <Button onClick={generate} disabled={busy}>
          {status === "loading-model"
            ? "Downloading checkpoint…"
            : status === "generating"
              ? "Running inference…"
              : "Continue the melody"}
        </Button>
      </div>

      {error && <p className="mt-6 text-sm text-destructive">{error}</p>}

      {status === "ready" && notes.length > 0 && (
        <div className="mt-8">
          <NotePlayer
            notes={notes}
            duration={duration}
            label="MelodyRNN continuation"
            subtitle={`${notes.length} notes · inference ran on your device, not the server`}
            height={150}
          />
        </div>
      )}
    </div>
  );
}
