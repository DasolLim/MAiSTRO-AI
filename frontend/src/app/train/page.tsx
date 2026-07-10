"use client";

import { useState } from "react";
import { Field, Select } from "@/components/Controls";
import { useGenerateOptions, useTrain } from "@/lib/useJob";
import { StaffDivider } from "@/components/StaffDivider";
import { JobLog } from "@/components/JobLog";
import { ProgressBar } from "@/components/ProgressBar";

interface LossReading {
  label: string;
  note: string;
}

/** Loss is a more reliable signal than epoch count, since it scales with dataset size. */
function readLoss(loss: number): LossReading {
  if (loss > 0.6) {
    return {
      label: "finding a pulse",
      note: "expect long held notes and little variation, the model hasn't learned note-to-note transitions yet",
    };
  }
  if (loss > 0.4) {
    return {
      label: "chords before rhythm",
      note: "harmony usually arrives before timing does, watch for uniform note durations",
    };
  }
  if (loss > 0.2) {
    return {
      label: "phrasing emerging",
      note: "the target range, structure and variation in balance",
    };
  }
  return {
    label: "past the sweet spot",
    note: "a very low loss is as consistent with memorizing the dataset as with mastering it, verify before trusting",
  };
}

const READING_STAGES = [
  {
    range: "loss > 0.6",
    title: "Finding a pulse",
    description:
      "The safest bet the model has before it understands sequence at all: one note, or a narrow cluster, held far longer than anything in the dataset, with only the occasional stray token breaking the silence. Not a bug, just the starting line.",
  },
  {
    range: "0.4 – 0.6",
    title: "Chords before rhythm",
    description:
      "Harmony tends to arrive before timing does. Expect the full pitch range in use and dense vertical stacks, the model has learned which notes co-occur, but note durations stay suspiciously uniform and the texture repeats in blocky columns.",
  },
  {
    range: "0.2 – 0.4",
    title: "Phrasing emerges",
    description:
      "Duration starts to vary, the texture thins toward a single line with harmony underneath, and a melodic contour appears across bars. Usually the first checkpoint worth actually generating from and listening to.",
  },
  {
    range: "< 0.2",
    title: "Past the sweet spot",
    description:
      "Don't assume lower is better. A loss this low is exactly as consistent with genuine mastery as with quoting the dataset outright. Trust your ears less here, and your comparisons more.",
  },
] as const;

export default function TrainPage() {
  const [arch, setArch] = useState<string | null>(null);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(64);

  const { data: options } = useGenerateOptions();
  const training = useTrain();

  const job = training.job;
  const isRunning = training.isRunning;
  const selectedArch = arch ?? options?.default_architecture ?? "lstm_attention";
  const archSpec = options?.architectures.find((a) => a.key === selectedArch);

  const progress = job?.progress as
    | { epoch?: number; total_epochs?: number; loss?: number; val_loss?: number | null }
    | undefined;
  const fraction = progress?.epoch && progress?.total_epochs ? progress.epoch / progress.total_epochs : 0;
  // Validation loss is the honest signal: training loss keeps falling long after
  // the model has started memorising 200 MIDI files.
  const judgedLoss = progress?.val_loss ?? progress?.loss;

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        II. Train
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground">Rehearsal</h1>
      <p className="mt-4 max-w-[65ch] text-muted-foreground">
        Fit one of the three architectures on the prepared notes. A tenth of the corpus is held
        out for validation, and checkpoints are saved every five epochs into{" "}
        <code className="text-foreground">checkpoints/{selectedArch}/</code>, so models never
        overwrite each other and any two can meet in the arena.
      </p>

      <StaffDivider className="mt-10" />

      {options && (
        <div className="mt-8 max-w-md">
          <Field label="Architecture" hint={archSpec?.description}>
            <Select
              value={selectedArch}
              onChange={setArch}
              disabled={isRunning}
              options={options.architectures.map((a) => ({
                value: a.key,
                label: a.trained ? `${a.label} — trained` : a.label,
              }))}
            />
          </Field>
        </div>
      )}

      <div className="mt-8 flex flex-wrap items-end gap-8">
        <label className="flex flex-col gap-1.5">
          <span className="text-xs tracking-wide text-muted-foreground uppercase">Epochs</span>
          <input
            type="number"
            min={1}
            step={1}
            inputMode="numeric"
            value={epochs}
            disabled={isRunning}
            onChange={(e) => setEpochs(Math.max(1, Math.round(Number(e.target.value) || 0)))}
            className="w-24 border border-border bg-transparent px-3 py-2 text-foreground tabular-nums focus:border-brass focus:outline-none"
          />
        </label>
        <label className="flex flex-col gap-1.5">
          <span className="text-xs tracking-wide text-muted-foreground uppercase">Batch size</span>
          <input
            type="number"
            min={1}
            step={1}
            inputMode="numeric"
            value={batchSize}
            disabled={isRunning}
            onChange={(e) => setBatchSize(Math.max(1, Math.round(Number(e.target.value) || 0)))}
            className="w-24 border border-border bg-transparent px-3 py-2 text-foreground tabular-nums focus:border-brass focus:outline-none"
          />
        </label>
        <button
          onClick={() => training.start({ arch: selectedArch, epochs, batchSize })}
          disabled={isRunning}
          className="cursor-pointer border border-brass px-5 py-2.5 text-sm font-medium text-brass transition-colors hover:bg-brass hover:text-brass-foreground disabled:cursor-not-allowed disabled:border-border disabled:text-muted-foreground disabled:hover:bg-transparent"
        >
          {isRunning ? "Training…" : "Start training"}
        </button>
      </div>

      {training.error && <p className="mt-6 text-sm text-destructive">{training.error.message}</p>}

      {job && (
        <div className="mt-10">
          {isRunning && progress?.epoch && (
            <>
              <div className="flex items-baseline justify-between text-sm">
                <span className="font-display text-2xl text-foreground tabular-nums">
                  Epoch {progress.epoch} / {progress.total_epochs}
                </span>
                <span className="text-muted-foreground tabular-nums">
                  loss {progress.loss?.toFixed(4)}
                  {progress.val_loss != null && ` · val ${progress.val_loss.toFixed(4)}`}
                </span>
              </div>
              <div className="mt-3">
                <ProgressBar fraction={fraction} />
              </div>
              {judgedLoss !== undefined && (
                <p className="mt-3 text-sm text-muted-foreground">
                  <span className="font-display italic text-brass">{readLoss(judgedLoss).label}</span>
                  {", "}
                  {readLoss(judgedLoss).note}
                </p>
              )}
            </>
          )}
          {job.state === "done" && job.result && (
            <p className="text-sm text-verdigris">
              Training complete. {job.result.arch} checkpoints saved to {job.result.checkpoint_dir}.
            </p>
          )}
          {job.state === "error" && <p className="text-sm text-destructive">{job.error}</p>}
          <JobLog lines={job.log} />
        </div>
      )}

      <div className="mt-20">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          Field notes
        </p>
        <h2 className="mt-3 font-display text-2xl text-foreground">Reading the rehearsal</h2>
        <p className="mt-3 max-w-[65ch] text-sm text-muted-foreground">
          Loss is a more reliable marker than epoch count, since it scales with your dataset,
          a run of 200 files reaches a given loss in a different number of epochs than a run of
          20. Here is the arc most training runs follow.
        </p>

        <div className="mt-8">
          <StaffDivider />
          {READING_STAGES.map((stage) => (
            <div key={stage.title}>
              <div className="flex items-baseline gap-6 py-6">
                <span className="w-24 shrink-0 text-xs tracking-wide text-muted-foreground tabular-nums uppercase">
                  {stage.range}
                </span>
                <div className="min-w-0 flex-1">
                  <h3 className="font-display text-xl italic text-foreground">{stage.title}</h3>
                  <p className="mt-1.5 max-w-[60ch] text-sm text-muted-foreground">
                    {stage.description}
                  </p>
                </div>
              </div>
              <StaffDivider />
            </div>
          ))}
        </div>

        <div className="mt-10">
          <h3 className="font-display text-xl text-foreground">Picking a checkpoint</h3>
          <ul className="mt-4 space-y-4">
            <li className="max-w-[65ch] text-sm text-muted-foreground">
              <span className="text-foreground">Sample more than once.</span> One generated clip
              is a noisy read on a checkpoint, generate a handful before judging a run by ear.
            </li>
            <li className="max-w-[65ch] text-sm text-muted-foreground">
              <span className="text-foreground">Watch for verbatim quoting.</span> If a late
              checkpoint reproduces long unbroken runs from the dataset, that is overfitting, no
              matter how good the loss number looks.
            </li>
            <li className="max-w-[65ch] text-sm text-muted-foreground">
              <span className="text-foreground">Check the distribution, not just the ear.</span>{" "}
              The note-distribution comparison in{" "}
              <code className="text-foreground">backend/maistro/overfit_check.py</code> scores a
              generated piece against every dataset track by cosine similarity, above ~0.8 means
              it is closer to copying than composing, 0.5–0.7 is the range you want.
            </li>
            <li className="max-w-[65ch] text-sm text-muted-foreground">
              <span className="text-foreground">When in doubt, go earlier.</span> Between two
              checkpoints that sound about the same, the earlier one is the safer bet, it has
              had less opportunity to memorize anything.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
