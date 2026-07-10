"use client";

import { useState } from "react";
import { JobLog } from "@/components/JobLog";
import { Button, Field, MetricGrid, Select, Slider } from "@/components/Controls";
import { MidiPlayer } from "@/components/MidiPlayer";
import { StaffDivider } from "@/components/StaffDivider";
import { useGenerate, useGenerateOptions } from "@/lib/useJob";

const NO_KEY = "__none__";
const NO_MOOD = "__none__";

export default function GeneratePage() {
  const { data: options, isLoading, error: optionsError } = useGenerateOptions();
  const generation = useGenerate();

  const [arch, setArch] = useState<string | null>(null);
  const [mood, setMood] = useState(NO_MOOD);
  const [tonic, setTonic] = useState(NO_KEY);
  const [scale, setScale] = useState("major");
  const [tempo, setTempo] = useState(96);
  const [notes, setNotes] = useState(300);
  const [temperature, setTemperature] = useState<number | null>(null);

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading the model registry…</p>;
  if (optionsError || !options) {
    return (
      <p className="text-sm text-destructive">
        Could not reach the API. Is the backend running on port 8000?
      </p>
    );
  }

  const sampling = options.sampling;
  const selectedArch = arch ?? options.default_architecture;
  // A mood preset supplies its own temperature; the slider only overrides it once
  // the user actually moves it.
  const activeMood = options.moods.find((m) => m.key === mood);
  const effectiveTemperature = temperature ?? activeMood?.temperature ?? sampling.temperature.default;
  const isGreedy = effectiveTemperature < sampling.temperature.greedy_below;

  const result = generation.job?.state === "done" ? generation.job.result : null;

  const handleGenerate = () => {
    generation.start({
      arch: selectedArch,
      n_notes: notes,
      temperature: effectiveTemperature,
      key: tonic === NO_KEY ? null : tonic,
      scale: tonic === NO_KEY ? "chromatic" : scale,
      mood: mood === NO_MOOD ? null : mood,
      tempo_bpm: tempo,
    });
  };

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        III. Generate
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground">A new composition</h1>
      <p className="mt-4 max-w-[65ch] text-muted-foreground">
        Sample a fresh sequence from a trained model. Temperature decides how far the decoder
        strays from its favourite note; key and mood bias it without retraining.
      </p>

      <StaffDivider className="mt-10" />

      <div className="mt-8 grid gap-8 sm:grid-cols-2">
        <Field
          label="Model"
          hint={options.architectures.find((a) => a.key === selectedArch)?.description}
        >
          <Select
            value={selectedArch}
            onChange={setArch}
            options={options.architectures.map((a) => ({
              value: a.key,
              label: a.trained ? a.label : `${a.label} — not trained`,
            }))}
          />
        </Field>

        <Field
          label="Creativity (temperature)"
          hint={
            isGreedy
              ? "Below 1.30 this model is so confident that sampling collapses to greedy decoding — the same passage every time."
              : `Sampling from the top ${sampling.top_k} notes, nucleus ${sampling.top_p}.`
          }
        >
          <Slider
            value={effectiveTemperature}
            onChange={setTemperature}
            min={sampling.temperature.min}
            max={sampling.temperature.max}
            step={sampling.temperature.step}
          />
        </Field>

        <Field label="Mood" hint={activeMood?.description ?? "No preset — the dials below are yours."}>
          <Select
            value={mood}
            onChange={(next) => {
              setMood(next);
              setTemperature(null); // let the new preset's temperature take effect
              const preset = options.moods.find((m) => m.key === next);
              if (preset) setScale(preset.scale);
            }}
            options={[
              { value: NO_MOOD, label: "None" },
              ...options.moods.map((m) => ({ value: m.key, label: m.label })),
            ]}
          />
        </Field>

        <Field label="Tempo" hint="Written into the MIDI file as a metronome mark.">
          <Slider
            value={tempo}
            onChange={setTempo}
            min={40}
            max={180}
            step={1}
            format={(v) => `${v}`}
          />
        </Field>

        <Field label="Key" hint="Biases the decoder toward notes in this scale.">
          <div className="flex gap-3">
            <Select
              value={tonic}
              onChange={setTonic}
              options={[
                { value: NO_KEY, label: "Any" },
                ...options.keys.map((k) => ({ value: k, label: k })),
              ]}
            />
            <Select
              value={scale}
              onChange={setScale}
              disabled={tonic === NO_KEY}
              options={options.scales
                .filter((s) => s !== "chromatic")
                .map((s) => ({ value: s, label: s.replace(/_/g, " ") }))}
            />
          </div>
        </Field>

        <Field label="Length" hint="Notes, chords and rests to sample.">
          <Slider value={notes} onChange={setNotes} min={32} max={600} step={4} format={(v) => `${v}`} />
        </Field>
      </div>

      <div className="mt-10">
        <Button onClick={handleGenerate} disabled={generation.isRunning}>
          {generation.isRunning ? "Composing…" : "Generate composition"}
        </Button>
      </div>

      {generation.error && (
        <p className="mt-6 text-sm text-destructive">{generation.error.message}</p>
      )}

      {generation.job && (
        <div className="mt-8">
          {result && (
            <>
              <MidiPlayer filename={result.filename} label={result.filename} />
              <MetricGrid
                metrics={{
                  "Repetition": result.metrics.repetition_rate.toFixed(3),
                  "Distinct pitches": result.metrics.distinct_pitch_classes,
                  "Corpus KL": result.metrics.pitch_class_kl?.toFixed(3) ?? null,
                  "Seed": result.config.seed ?? null,
                }}
              />
              <p className="mt-4 max-w-[65ch] text-xs leading-relaxed text-muted-foreground">
                Repetition is the share of 4-note windows the decoder has already played —
                lower is more inventive. Corpus KL compares this piece&apos;s pitch-class
                distribution against the training set; near zero means it sounds like the corpus.
                The seed reproduces this exact take.
              </p>
            </>
          )}
          <JobLog lines={generation.job.log} />
        </div>
      )}
    </div>
  );
}
