# What's new in MAiSTRO

Six additions, in the order they matter. Every number below was measured on the
trained `lstm_attention` checkpoint against the 194,441-note dataset — none are
estimates.

---

## 1. The decoder samples instead of guessing

**Before:** `generate.py` picked `argmax(prediction)` at every step. A deterministic
decoder can only ever walk one path out of a given context, so the moment a
100-note window repeated, the music looped forever.

**Now:** the next note is *sampled* from the model's distribution, with three dials —
temperature, top-k, and nucleus (top-p). Temperature is exposed in the UI as a
**creativity slider**.

The interesting part was calibrating it. This model is far more confident than a
text language model:

```
mean max probability   0.95        mean entropy   0.21 nats
```

The argmax token already holds more probability mass than any sane nucleus
threshold. So the usual defaults (`T=0.8, top_p=0.9`) do **nothing** — they
truncate to a single token and silently reproduce greedy decoding. Sweeping
temperature over 25 real next-token distributions (vocabulary 3,388):

| Temperature | max prob | perplexity | nucleus @ 0.95 |
|---|---|---|---|
| 1.0 | 0.96 | 1.1 | 1 token |
| 1.5 | 0.94 | 1.4 | 12 tokens |
| 1.8 | 0.87 | 3.7 | 335 tokens |
| 2.0 | 0.77 | 12.7 | 1,713 tokens |
| 2.5 | 0.41 | 239.0 | 2,899 tokens |

Below ~1.3 the decoder is greedy; above ~2.1 it samples from a flat tail of
noise. **The usable band is 1.4–2.0**, and that is exactly the range the slider
spans. Defaults: `T=1.5, top_p=0.98, top_k=40`.

What it buys, over 160 generated notes:

| Setting | Repetition rate ↓ | Corpus KL ↓ |
|---|---|---|
| Greedy (the old default) | 0.357 | 0.784 |
| `T=1.0, top_p=0.92` | 0.357 | 0.772 |
| `T=1.5` (new default) | **0.140** | **0.446** |
| `T=1.9` | **0.051** | **0.232** |

*Repetition rate* is the share of 4-note windows the decoder has already played.
*Corpus KL* compares the generated pitch-class histogram against the training
set's — lower means it sounds more like the corpus it learned from.

Note the second row: `T=1.0, top_p=0.92` is byte-for-byte identical to greedy.
That is the trap this calibration exists to avoid.

> `backend/maistro/sampling.py`

---

## 2. Steer the music without retraining

You can now ask for **a key, a scale, a tempo, and a mood**.

The model has no key or mood *input* — it takes a window of past notes and nothing
else, and retraining a conditional model needs labelled data we don't have. So
conditioning happens at **decode time**: every vocabulary token gets an additive
score in log-probability space, and that vector is added to the logits before
sampling. Same mechanism as OpenAI's `logit_bias`, and it composes cleanly with
temperature.

- **Key + scale** — out-of-scale notes are penalised proportionally to how much of
  the chord falls outside the key. Soft, not banned: the model keeps its passing tones.
- **Register** — a quadratic pull toward a target pitch centre.
- **Density** — biases note durations and how often the model rests.
- **Seed selection** — the starting 100-note window is chosen from candidates that
  are *already* in the requested key, so generation doesn't have to fight its way there.

Measured, on 160 notes asked for C major:

| | Fraction of notes in key |
|---|---|
| Unconditioned | 0.72 |
| `key=C, scale=major` | **0.96** |

Five mood presets (`serene`, `melancholic`, `triumphant`, `turbulent`, `playful`)
bundle a temperature, scale, register and density into one choice. Their
temperatures come from the calibration table above, not from intuition.

> `backend/maistro/theory.py`, `backend/maistro/conditioning.py`

---

## 3. Three architectures, one dataset

The model zoo is now a registry. All three share a tokenizer, a dataset and a decoder,
so comparing them compares *architectures* rather than pipelines.

| Model | Parameters | How it reads the window |
|---|---|---|
| `lstm` | 4.9M | Two stacked LSTMs. The textbook baseline. |
| `lstm_attention` | **178.8M** | Bidirectional LSTM + self-attention. What MAiSTRO shipped with. |
| `transformer` | **3.9M** | 4-layer causal decoder, learned token + position embeddings. |

That parameter count is the headline. The original network ends in
`Flatten() → Dense(3388)` over a 100×512 sequence — a **175M-parameter** output layer
alone. The transformer gets comparable capacity with **46× fewer parameters** by
never materialising that matrix.

The two families also read the input differently, which the registry declares:
the LSTMs take one normalised float per step (`token_id / n_vocab` — which wastes
the vocabulary's structure, since token 41 is not musically "between" 40 and 42),
while the transformer embeds each token id properly.

Checkpoints now live in `checkpoints/<arch>/`, so models never overwrite each other.

> `backend/maistro/architectures.py`

---

## 4. The arena: which model actually sounds better?

Validation loss tells you which model fits the corpus. It does not tell you which
one a person would rather listen to — cross-entropy punishes a beautiful passing
tone the corpus happened not to contain.

So: two architectures compose the **same brief** — identical length, key,
temperature, and random seed, the architecture is the only variable — and you hear
them blind. You vote. Elo ratings update.

- Elo (K=24, start 1500) because votes arrive as *pairwise comparisons*, listeners
  disagree, and a new architecture can be added without re-running earlier matchups.
- Which model is presented as "A" is randomised, so a listener who always plays the
  left one first can't systematically favour a model.
- Blinding is enforced on the server: the architecture is stripped from the payload,
  kept out of the streamed job log, and the config sidecar is not written next to
  arena clips (every file under `output/` is reachable via `GET /audio/file`).
- Every vote is persisted to `data/arena.json` with both takes' objective metrics.

Alongside the human vote, `POST /arena/objective` scores every trained architecture
on the reproducible metrics — cheap, no listener required.

**This needs two trained models.** Only `lstm_attention` ships trained; train the
transformer from the Train page first. At 3.9M parameters it is far quicker.

> `backend/maistro/arena.py`, `backend/maistro/metrics.py`

---

## 5. The frontend is now a real client

**React Query** owns every server interaction. The backend hands back a job id
immediately for anything slow, so job state is really *server state on a timer* —
which is what `refetchInterval` exists for. Polling stops by itself when a job
reaches a terminal state. Loading, error, and streaming-log states are handled
per-page, including a backend that isn't running.

**Tone.js + @tonejs/midi** play generated MIDI directly in the browser: the `.mid`
is parsed client-side and synthesised live. No Fluidsynth, no soundfont install —
which removes the single most painful setup step in the old README. Rendering to
WAV through Fluidsynth is still there, now as an *option* for the sampled piano.

**A canvas piano roll** draws every note (x = time, y = pitch) with a playhead
tracking the transport. The notes are painted once into an offscreen canvas and
blitted per frame, so a 300-note piece doesn't cost 300 style recalculations every
animation frame.

> `frontend/src/lib/useJob.ts`, `frontend/src/components/NotePlayer.tsx`, `PianoRoll.tsx`

---

## 6. Studio: music the note vocabulary can't express

MAiSTRO composes **notes**, and a soundfont turns them into sound. That is the right
tool for classical piano and the wrong one for music that lives in *timbre*. Nothing
in a piano-note vocabulary can express "dusty Rhodes through a tape delay."

Two models that can, deliberately chosen to sit at opposite ends of the deployment
spectrum:

**Meta MusicGen** (server, Python) — a transformer over the tokens of an audio codec.
It predicts the next frame of compressed *waveform*, so it renders any genre it heard
in training, at the cost of never producing an editable score. Optional install
(`pip install -r requirements-external.txt`); the endpoints return a clear 503 with
install instructions when it's absent.

**Google Magenta MelodyRNN** (browser, TensorFlow.js) — the *same task* as MAiSTRO's
LSTM, continue a melody one note at a time, but the weights download to the browser
and inference runs on the visitor's GPU. No Python process, no cold start, no server
bill. The trade is a few megabytes on first use.

> `backend/maistro/external/musicgen.py`, `frontend/src/components/MagentaContinuation.tsx`

---

## Bugs fixed along the way

These weren't on the plan; they turned up while testing the above.

**The shipped weights couldn't be loaded.** `weights-epoch035-0.2775.keras` was saved
by Keras 3.8, but the project pins Keras 2.10 (`keras_self_attention` has no Keras 3
build). `load_weights()` choked on the Keras 3 zip archive. Rather than retrain, the
weight arrays are now read out of the archive and assigned layer by layer, every
assignment shape-checked so a mis-mapping fails loudly instead of generating noise.
→ `backend/maistro/weights_compat.py`

**Every generated note was a quarter note.** The old writer built its stream with
`Stream(list)`, whose `append()` recomputes offsets from each element's duration — and
it never *set* a duration. So the model's predicted rhythms were discarded at the last
step, before they reached the MIDI file. Durations are now honoured and offsets
inserted explicitly.

**Training reported the wrong loss.** There was no validation split, so the number on
screen was training loss — which keeps falling long after a model has started
memorising 200 MIDI files. A contiguous 10% tail is now held out (contiguous, not
random: the windows overlap by one note, so a random split would leak almost every
validation window into training).

**Path traversal in the file endpoint.** Filenames now carry a subdirectory
(`arena/…`, `external/…`), so `GET /audio/file/{filename}` could no longer be a plain
join. Resolved paths are checked against the output root; `../../etc/passwd` returns
400 in both raw and percent-encoded forms.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /generate/options` | Architectures, moods, scales, and the calibrated slider bounds |
| `POST /generate` | Generate with `arch`, `temperature`, `key`, `scale`, `mood`, `tempo_bpm`, `seed` |
| `POST /train/start` | Train any architecture; reports `val_loss` per epoch |
| `GET /train/history/{arch}` | Loss curves from the last run |
| `POST /arena/pair` | Two blinded takes from two architectures |
| `POST /arena/vote` | Vote, reveal, update Elo |
| `GET /arena/leaderboard` | Ratings, W/L/D, recent votes |
| `POST /arena/objective` | Score architectures on reproducible metrics |
| `GET /external/musicgen/status` | Whether the optional audio stack is installed |
| `POST /external/musicgen/generate` | Text → audio |
| `GET /audio/file/{path}` | Serves `.mid` and `.wav`, traversal-guarded |

Every generation writes a `.json` sidecar next to its `.mid` recording the exact
config and RNG seed, so **any piece in the library can be reproduced exactly**.

---

## Honest caveats

- **The transformer ships untrained.** Its architecture, training path, and arena
  integration are tested end to end, but no one has run the epochs yet. The
  comparison in §3 is a parameter count, not a quality claim.
- **Conditioning is steering, not learned control.** A strong bias fights the model
  rather than collaborating with it. The penalties are deliberately soft (a few nats).
- **Magenta.js is unmaintained** (last release 2022) and pins an old TensorFlow.js;
  `npm audit` flags its dependency tree. It is dynamically imported so its failures
  stay contained to the one component that uses it.
- **The arena's Elo is only meaningful after a few dozen votes.** One listener cannot
  invert a rating, but they can wobble it.
