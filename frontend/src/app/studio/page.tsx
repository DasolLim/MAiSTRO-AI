"use client";

import { useState } from "react";
import { Button, Field, Slider } from "@/components/Controls";
import { JobLog } from "@/components/JobLog";
import { MagentaContinuation } from "@/components/MagentaContinuation";
import { StaffDivider } from "@/components/StaffDivider";
import { audioFileUrl } from "@/lib/api";
import { useFeature, useMusicGen, useMusicGenStatus } from "@/lib/useJob";

export default function StudioPage() {
  // Magenta runs in the browser and is always available. MusicGen needs torch —
  // 511MB — so the free-tier deployment does not carry it.
  const musicgenAvailable = useFeature("musicgen").available;
  const { data: status, isLoading } = useMusicGenStatus(musicgenAvailable);
  const musicgen = useMusicGen();

  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState(12);

  const result = musicgen.job?.state === "done" ? musicgen.job.result : null;

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        Compose
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground">Beyond the piano</h1>
      <p className="mt-4 max-w-[68ch] text-muted-foreground">
        MAiSTRO&apos;s own model composes <em>notes</em>, and a soundfont turns those notes into
        sound. That is exactly the right tool for classical piano and exactly the wrong one for
        music that lives in timbre — a synth pad, a drum break, a guitar tone. Nothing in a
        piano-note vocabulary can express &ldquo;dusty Rhodes through a tape delay.&rdquo; These
        two models can.
      </p>

      <StaffDivider className="mt-10" />

      {/* -------------------------------------------------- MusicGen */}
      <section className="mt-10">
        <h2 className="font-display text-2xl text-foreground">Text to audio — Meta MusicGen</h2>
        <p className="mt-3 max-w-[65ch] text-sm leading-relaxed text-muted-foreground">
          A transformer over the tokens of an audio codec, not over notes. It predicts the next
          frame of compressed <em>waveform</em>, so it can render any genre it heard in training,
          at the cost of never producing a score you can edit. Runs on the Python backend.
        </p>

        {!musicgenAvailable && (
          <div className="mt-6 border border-border bg-surface p-6">
            <p className="text-sm text-foreground">MusicGen is not part of this deployment.</p>
            <p className="mt-2 max-w-[60ch] text-sm text-muted-foreground">
              It needs torch, which is 511MB — the whole serverless function is capped at 500MB,
              and the model weights are another 2GB on top. Run the backend locally and MusicGen
              works. Magenta, below, runs in your browser and needs nothing.
            </p>
          </div>
        )}

        {musicgenAvailable && isLoading && (
          <p className="mt-6 text-sm text-muted-foreground">Checking the audio stack…</p>
        )}

        {musicgenAvailable && status && !status.available && (
          <div className="mt-6 border border-border bg-surface p-6">
            <p className="text-sm text-foreground">MusicGen is not installed.</p>
            <p className="mt-2 max-w-[60ch] text-sm text-muted-foreground">{status.reason}</p>
            <code className="mt-4 block overflow-x-auto bg-bg px-4 py-3 text-xs text-brass">
              pip install -r requirements-external.txt
            </code>
          </div>
        )}

        {musicgenAvailable && status?.available && (
          <>
            <div className="mt-8 grid gap-8 sm:grid-cols-2">
              <Field label="Prompt" hint="Describe instruments, texture and feel — not notes.">
                <textarea
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  rows={3}
                  placeholder={status.prompt_ideas[0]}
                  className="w-full resize-none border border-border bg-surface px-3 py-2 text-sm text-foreground transition-colors placeholder:text-muted-foreground hover:border-brass focus:border-brass focus:outline-none"
                />
              </Field>

              {/* Measured on musicgen-small, CPU: ~3.5x slower than real time. */}
              <Field
                label="Duration"
                hint={`Up to ${status.max_duration}s. On CPU expect roughly 3s of compute per second of audio.`}
              >
                <Slider
                  value={duration}
                  onChange={setDuration}
                  min={1}
                  max={status.max_duration}
                  step={1}
                  format={(v) => `${v}s`}
                />
              </Field>
            </div>

            <div className="mt-6 flex flex-wrap gap-2">
              {status.prompt_ideas.map((idea) => (
                <button
                  key={idea}
                  onClick={() => setPrompt(idea)}
                  className="cursor-pointer border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-brass hover:text-brass"
                >
                  {idea}
                </button>
              ))}
            </div>

            <div className="mt-8">
              <Button
                onClick={() => musicgen.start({ prompt, duration })}
                disabled={musicgen.isRunning || prompt.trim().length < 3}
              >
                {musicgen.isRunning ? "Rendering audio…" : "Generate audio"}
              </Button>
            </div>
          </>
        )}

        {musicgen.error && <p className="mt-6 text-sm text-destructive">{musicgen.error.message}</p>}

        {musicgen.job && (
          <div className="mt-8">
            {result && (
              <div>
                <p className="font-display text-lg italic text-foreground">
                  &ldquo;{result.prompt}&rdquo;
                </p>
                {/* A waveform, not a score — the native <audio> element is the right player. */}
                <audio
                  controls
                  src={audioFileUrl(result.wav_filename)}
                  className="mt-3 w-full"
                  aria-label={`MusicGen output for ${result.prompt}`}
                />
                <p className="mt-2 text-xs text-muted-foreground">
                  {result.model_name} · {result.duration}s
                </p>
              </div>
            )}
            <JobLog lines={musicgen.job.log} />
          </div>
        )}
      </section>

      <StaffDivider className="mt-16" />

      {/* -------------------------------------------------- Magenta */}
      <section className="mt-10">
        <h2 className="font-display text-2xl text-foreground">
          In the browser — Google Magenta MelodyRNN
        </h2>
        <p className="mt-3 max-w-[65ch] text-sm leading-relaxed text-muted-foreground">
          The same job as MAiSTRO&apos;s LSTM — continue a melody, one note at a time — but the
          weights download to your browser and the inference runs on your GPU through
          TensorFlow.js. No Python process, no cold start, no server bill. The trade is a few
          megabytes on first use and whatever your laptop can manage.
        </p>

        <div className="mt-8">
          <MagentaContinuation />
        </div>
      </section>
    </div>
  );
}
