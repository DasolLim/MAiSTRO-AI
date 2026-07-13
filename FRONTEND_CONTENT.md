# Frontend content inventory

All descriptive and explanatory prose from the MAiSTRO frontend (`frontend/src/app/`), extracted
verbatim and organized by page. This is source content, not documentation of code — pulled from
[`layout.tsx`](frontend/src/app/layout.tsx), [`page.tsx`](frontend/src/app/page.tsx) (home),
[`story/page.tsx`](frontend/src/app/story/page.tsx), and
[`how-it-works/page.tsx`](frontend/src/app/how-it-works/page.tsx).

---

## Site-wide

**Meta description** (every page's `<head>`):

> An LSTM that composes classical piano, and the tools to steer and judge it.

**Footer** (every page):

> MAiSTRO · Western Cyber Society
> Trained on Mozart, Beethoven and Chopin. MIT licensed.

**Nav structure** (from `SiteNav.tsx`):

| Group | Summary | Pages |
|---|---|---|
| Compose | Use a trained model | Generate — *Sample a new piece, steered by key and mood*; Listening room — *Everything composed so far*; Arena — *Judge two models blind*; Beyond the piano — *MusicGen and Magenta, for other genres* |
| Model | Build a new one | Prepare data — *Turn MIDI files into a note vocabulary*; Train — *Fit an LSTM or a transformer* |
| Project | Read about it | The story — *Why eight students built a composer*; How it works — *The engineering, measured* |

---

## Home page (`/`)

### Hero

> Western Cyber Society
>
> **A model that composes. Now it takes direction.**
>
> MAiSTRO is an LSTM trained on Mozart, Beethoven and Chopin. It could always write music. What it
> could not do was write it *differently* twice, take a request, or tell you whether it had
> improved.

CTAs: "Compose something" → `/generate` · "How it works" → `/how-it-works`

### Origin

> **Eight students, one LSTM**
>
> AI was writing essays and painting pictures, and music composition sat strangely untouched. A
> team of eight set out to see whether a network could learn the shape of classical piano from the
> notation itself.
>
> It worked. A bidirectional LSTM with an attention layer, trained on 200 MIDI files, produced
> pieces you would happily leave playing. That model is still here, still the default, and the
> story of building it is worth reading first.
>
> What follows is what happened when it stopped being a model and started being a tool.

### Six problems, six answers

> Each of these began as something that annoyed someone using it.

| Title | Problem | Change |
|---|---|---|
| The decoder was never asked | It always played the likeliest next note, so it looped. | Temperature, top-k and nucleus sampling, calibrated against this model's own confidence rather than borrowed from text generation. |
| You couldn't ask for anything | Generation started from a random passage. If you disliked it, you re-rolled. | Ask for a key, a scale, a tempo, a mood. The bias is applied as the note is chosen, so nothing needed retraining. |
| Loss cannot hear | A lower validation loss does not mean a better piece of music. | Two models compose the same brief and you judge them blind. Elo ratings, the same system chess uses, settle the argument. |
| One model, no comparison | The original network spends 175M of its parameters on a single output layer. | A plain LSTM and a 3.9M-parameter transformer now share the dataset and decoder, so the architecture is the only variable. |
| Hearing it meant installing Fluidsynth | The model emits MIDI. Turning that into sound needed a soundfont and a C library. | Tone.js synthesises the notes in your browser, and a canvas piano roll draws them as they play. |
| A piano cannot play every genre | No arrangement of piano notes means "dusty Rhodes through a tape delay". | MusicGen generates audio from a text prompt; Magenta continues a melody entirely in your browser. |

### What it bought

> Every figure comes from running the trained model, averaged across four random seeds.

| Value | Label | Note |
|---|---|---|
| 0.95 | Mean top probability | Why textbook sampling defaults do nothing here |
| 0.151 → 0.024 | Repetition rate | Greedy against temperature 1.9, over 4 seeds |
| 0.79 → 0.93 | Notes in the asked-for key | Unconditioned against C major |
| 46× | Fewer parameters | Transformer against the original network |

### Closing CTA

> **The model is trained and waiting**
>
> Pick a key, pick a mood, move the creativity dial, and hear what a network trained on three dead
> composers does with the request.

---

## Story page (`/story`)

> An LSTM composer trained on Mozart, Beethoven, and Chopin, built by a team of eight at Western
> Cyber Society. The story so far, in the order we lived it.

### Motivation

> Outside of engineering, music has always been a huge part of my life, listening to it, picking
> apart how a track is built, appreciating the work that goes into composing something that
> actually moves people. I've always been fascinated by how music artists compose, layering
> melody, rhythm, and emotion into something that connects with listeners in a way that's hard to
> put into words. That fascination made me wonder how I could bring my engineering skills into
> that world instead of just enjoying it from the outside.
>
> At the same time, AI was exploding everywhere: essays written in seconds, images that made you
> question what was even real anymore. It felt like AI was closing in on doing anything a human
> could do. But one space seemed strangely untouched: music composition. That gap felt like the
> perfect intersection of two things I wanted to explore, could AI genuinely create music the way
> a person could, and could I build something that gave engineers a real way to support musicians
> instead of sidelining them?

### The plan

> We decided to start simple: build a machine learning model that could learn from classical piano
> music and generate its own compositions. This meant finding a suitable dataset in a format the
> model could understand, selecting the right model architecture, running inference to generate
> new sequences, and finally converting those outputs back into audible music.

### Data collection & preprocessing

> For the dataset, we gathered a collection of piano pieces by Mozart, Beethoven, and Chopin since
> they are easily recognizable, which made checking for overfitting much more straightforward.
> Initially, we thought we would need to convert audio recordings of these compositions into a
> format the model could understand. However, because these pieces are so well-known,
> computer-readable versions already exist online. These files, known as MIDI files, essentially
> represent digital sheet music, making them perfect for feeding into our model.

### The model

> When planning this project, Henry and I had a few contenders for models we could use to
> accomplish the task. Autoencoders can be used by learning compressed representations of musical
> patterns and then decoding them to create new compositions. Generative Adversarial Networks
> (GANs), consisting of two models in competition, one generating music and the other evaluating
> its realism, also presented a compelling option. Lastly, Recurrent Neural Networks (RNNs) work
> by learning patterns in sequences of musical notes and predicting the next note based on
> previously played ones. After doing research, we felt that RNNs most closely mimicked the way
> humans process and compose music, leading us to choose a specific variant known as the Long
> Short-Term Memory (LSTM) model for our implementation.
>
> You would undoubtedly ask, "Why not just use a regular RNN?" While RNNs and LSTMs essentially
> aim to accomplish the same task, standard RNNs have a much smaller memory capacity compared to
> LSTMs. For example, if I asked an RNN, "The sky is…", it would (hopefully) respond with "blue."
> But if I asked it, "Explain why the sky is blue," by the second sentence it would start
> rambling about how the sky is actually just a reflection of the ocean and that's why deserts
> don't exist at night. This is because of a common mathematical issue called the vanishing
> gradient problem, where the gradient of the loss function decays exponentially over time. This
> is fine if the model is outputting one-word answers, but worse if the answers are meant to be
> longer since the model will forget more information faster and faster as the length of the
> answer increases.
>
> LSTMs help solve this problem by allowing a larger memory of past inputs. An additional
> enhancement to these models is the attention mechanism. While RNNs and LSTMs use previous inputs
> to influence future outputs, they lack the ability to label what parts of previous inputs are
> important to a specific output. This is important in music generation, where the model must
> understand structural elements like verses and choruses to repeat or develop them appropriately.
> The attention mechanism allows the model to focus on these significant sections, improving its
> ability to generate musically structured compositions. Here's an example that helped me
> understand what the model was doing:
>
> *Unless you've seen this joke before, you probably fell for it (like I did… more times than I'd
> like to admit).*
>
> As humans, we don't read word-by-word. Instead, we read the most important parts and assume
> where the sentence is going from there. This is what attention is doing within our model:
> guiding the LSTM in what it needs to remember or forget as time goes on.
>
> Our final model had the following architecture:

**Final architecture (7 layers):**

1. **Bidirectional LSTM layer** — reads the note sequence both forward (start to end) and
   backward (end to start), then combines both readings into a single, better-informed
   representation.
2. **Dropout layer** — randomly ignores a portion of the network during training, used throughout
   the model to keep it from overfitting.
3. **Attention layer** — weighs which parts of the sequence actually matter to the next note, the
   way a person skims a sentence for the load-bearing words instead of reading every one equally.
4. **LSTM layer** — a second pass that folds the attention-weighted context back into the
   sequence.
5. **Dropout layer** — a second guard against overfitting before the final decision is made.
6. **Dense layer** — organizes every note and chord the model has learned into a single set of
   candidates.
7. **Softmax activation** — turns that set of candidates into probabilities, so the model can pick
   the most likely next note or chord.

### Results

> I'm done with the theory. Here's the model at four points along the same training run.

| Epoch | Loss | Caption |
|---|---|---|
| 1 | — | That definitely sounded like the first time someone tried playing the piano. Clearly, it hasn't learned much from the dataset yet. |
| 10 | — | The model has picked up some more complex chords and started developing patterns, but still struggles to vary its rhythm. |
| 20 | — | Noticeably more melodic coherence here, and we're finally seeing some variation in note lengths, although still limited. |
| 50 | 0.1534 | Arguably the best-sounding output we generated, though at a loss this low, there's a good chance some of it is memorized rather than composed. |

> By the way, the model only outputs MIDI files during generation, and to turn one into something
> audible we manually applied a piano soundfont. Generated pieces now play straight in the
> browser, no soundfont required. With the remaining Google Colab credits we had, we trained the
> model for a final 50 epochs, that's the last clip above. This is something I would genuinely
> listen to if I was trying to zone in during a study session, at a cafe, or any other situation
> where background music would fit in.

### Closing

> **Then we kept going**
>
> Everything above is where the project stood when the team finished it. Since then the decoder
> has been rebuilt, two more architectures joined the model, and the whole thing became something
> you can steer rather than re-roll.

---

## How it works page (`/how-it-works`)

> MAiSTRO started as a model that could compose. Making it a tool anyone could actually use meant
> fixing how it chooses notes, giving a person a way to steer it, and building an honest way to
> tell whether a change made the music better. Every number here was measured on the trained
> model, not estimated.

### Decoding — "The model was never the problem"

> The original decoder took the single most likely next note, every time. That sounds reasonable
> and it is quietly fatal: a deterministic decoder can only ever walk one path out of a given
> passage. The moment the model saw a context it had seen before, it played the same continuation,
> forever. What sounded like a model with nothing to say was really a model never being asked.
>
> The fix is to sample from the distribution rather than collapse it. Temperature controls how
> much: low values sharpen toward the model's favourite note, high values flatten toward surprise.
>
> Calibrating it was the interesting part, because this model is unusually sure of itself. Its
> next-note distribution has a mean top probability of 0.95 and an entropy of 0.21 nats — far
> sharper than a language model. The likeliest note already holds more probability than any
> sensible nucleus threshold, so the textbook defaults borrowed from text generation truncate to a
> single token and silently reproduce greedy decoding. The dial does nothing, and you would never
> know.

**Temperature calibration sweep:**

| Temperature | Top probability | Perplexity | Notes in the nucleus |
|---|---|---|---|
| 1.0 | 0.96 | 1.1 | 1 |
| 1.5 | 0.94 | 1.4 | 12 |
| 1.8 | 0.87 | 3.7 | 335 |
| 2.0 | 0.77 | 12.7 | 1,713 |
| 2.5 | 0.41 | 239.0 | 2,899 |

> Below about 1.3 the decoder is greedy. Above about 2.1 it is drawing from a flat tail of two
> thousand notes, which is noise. The usable band is 1.4 to 2.0, and that is the range the
> creativity slider spans — not an arbitrary 0 to 1.
>
> What that buys, averaged over four seeds at 160 notes each. Repetition is the share of four-note
> windows already played; corpus KL measures distance from the training set's pitch distribution.
> Lower is better in both.

| Setting | Repetition | Corpus KL | Unique notes |
|---|---|---|---|
| Greedy — the old default | 0.151 | 0.498 | 0.338 |
| T=1.0, top_p=0.92 | 0.177 | 0.503 | 0.331 |
| T=1.5 — the new default | 0.064 | 0.301 | 0.444 |
| T=1.9 | 0.024 | 0.147 | 0.570 |

> The second row is the point. Those are the sampling defaults you would copy from any
> text-generation tutorial, and they are indistinguishable from greedy decoding — the difference
> sits well inside the variation between seeds. The dial would have been connected to nothing.
>
> One honest caveat. Sampling diverges from the greedy path slowly: at 120 notes it changes only a
> fiftieth of the model's decisions, but by 300 notes it has changed most of them. Sampling
> matters more the longer the piece runs.

### Control — "Steering a model that has no steering wheel"

> The network takes a window of past notes and nothing else. It has no input for key, no input
> for mood, and training a model that did would need labelled data nobody has. So conditioning
> happens at the last possible moment instead: every note in the vocabulary gets a score, and that
> score is added to the model's own opinion just before a note is picked.
>
> Ask for C major and notes outside the scale get pushed down — in proportion to how far outside
> they are, so a chord with one foreign note is discouraged less than one with three. The push is
> deliberately gentle, a few nats. A heavier hand would win every argument with the model and
> flatten the music into scales.
>
> Mood presets bundle a temperature, a scale, a register and a note density. They exist because
> "melancholic" is easier to ask for than temperature 1.45, natural minor, centred on G3, favour
> long notes.

| | Notes in the requested key | Spread across seeds |
|---|---|---|
| Unconditioned | 0.786 | ±0.145 |
| Asked for C major | 0.930 | ±0.039 |

> The tightened spread matters as much as the raised average. Left alone, which key you get is
> luck of the seed. Asked, you get the key you asked for.

### Architecture — "Three models, one dataset"

> Comparing architectures is only meaningful when nothing else moves. All three share a dataset, a
> tokenizer and a decoder, so a difference in output is a difference in architecture rather than
> in plumbing.

| Model | Parameters | What it is |
|---|---|---|
| LSTM | 4.9M | Two stacked layers. The baseline. |
| LSTM + attention | 178.8M | The original MAiSTRO network. |
| Transformer | 3.9M | A small GPT-style causal decoder. |

> That middle row is worth staring at. The original network ends by flattening a 100×512 sequence
> into a single dense layer, which costs 175 million parameters on the output layer alone. The
> transformer reaches comparable capacity with 46× fewer, because it never builds that matrix.
> Whether it also sounds better is not a question a parameter count can answer — which is what the
> arena is for.

### Evaluation — "Loss is not taste"

> Validation loss tells you how well a model predicts the corpus. It does not tell you whether a
> person would want to listen to it. Cross-entropy punishes a beautiful passing tone for the crime
> of not appearing in Chopin.
>
> So the arena asks a person. Two architectures compose the same brief — same length, key,
> temperature and random seed, so the architecture is the only thing that differs — and you hear
> them without being told which is which. Elo ratings update from your vote, the same system chess
> uses, because votes arrive as pairwise comparisons and a new model can join without re-running
> every earlier matchup.
>
> Blinding is enforced on the server rather than in the interface: the model name is stripped from
> the response, kept out of the progress log, and the metadata file that would otherwise sit next
> to each clip is not written. It would have been reachable by guessing a URL.
>
> Two cheap statistics run alongside the human vote. Repetition is the share of four-note windows
> the decoder has already played. Corpus KL compares the piece's pitch-class distribution against
> the training set. Neither replaces listening. Both are reproducible, which listening is not.

### Reach — "Music the note vocabulary cannot express"

> MAiSTRO composes notes, and a piano renders them. That is exactly right for classical piano and
> exactly wrong for music that lives in timbre. There is no arrangement of piano notes that means
> "dusty Rhodes through a tape delay."
>
> MusicGen predicts compressed audio instead of notes, so it can render any genre it was trained
> on — at the cost of never producing a score you can edit. It runs on the Python backend and is
> an optional install. Magenta's MelodyRNN does the same job as MAiSTRO's LSTM, continuing a
> melody one note at a time, but downloads to your browser and runs on your own GPU. Together they
> bracket the question of where a model should live.

### Deployment — "Fitting a neural network into 40 megabytes"

> This site runs on a free tier. Vercel's Python runtime is 3.12 with a 500MB ceiling on a
> deployed function — and TensorFlow is 877MB unpacked, and needs Python 3.10 or older. The
> training backend cannot go there at any size.
>
> The way through is a distinction worth internalising: TensorFlow is only needed to train.
> Running a trained model forward is a dozen matrix multiplications over a few million numbers.
> Nothing about that requires a deep-learning framework.
>
> So the deployed API reimplements the transformer's forward pass in plain NumPy and reads its
> weights from a 7MB file. Writing MIDI moves from music21 (111MB) to mido (under 1MB). What ships
> is fastapi, numpy and mido.

| Component | Unpacked size |
|---|---|
| TensorFlow | 877 MB |
| music21 | 111 MB |
| fastapi + numpy + mido | 31 MB |
| Transformer weights, float16 | 7.0 MB |
| Vocabulary + seed corpus | 0.17 MB |
| Deployed total | ≈ 40 MB |

> A rewrite is only worth anything if it computes the same function. The NumPy path is checked
> against Keras on identical weights: the largest disagreement anywhere in the output distribution
> is 4.2 × 10⁻⁷, the two always pick the same next note, and a 300-note piece takes 9.9 seconds
> against a 300-second limit. The training notebook re-runs that comparison and refuses to export
> weights that fail it.
>
> Which model gets deployed follows from the arithmetic. The original LSTM keeps 97% of its 178.8M
> parameters in one output layer — 357MB even at half precision. The transformer's 3.9M parameters
> (256 dimensions, 4 heads, 4 layers) fit in 7MB. It is trained on a free Colab T4, where an epoch
> takes about two minutes instead of the seventeen it takes on a laptop.
>
> A serverless function is stateless with a read-only disk, so the deployment keeps what fits and
> is honest about the rest. Generation answers in one request, returning the MIDI inline rather
> than writing a file. Magenta was already running in your browser. The library, the arena,
> training and MusicGen need a filesystem, two trained models, hours, and 511MB of torch
> respectively — they stay local, and the interface says so rather than failing at you.

### Stack — "What it is built with, and why"

> Every dependency here earns its place by removing a specific problem.

| Tool | Role | Why |
|---|---|---|
| TensorFlow / Keras 2.10 | Trains and runs all three architectures. | Pinned to 2.10 because keras_self_attention, which the original model depends on, has no Keras 3 build. |
| music21 | Parses MIDI into a note vocabulary and writes it back out. | Handles chords, rests and fractional durations that raw MIDI libraries leave to you. |
| FastAPI | Wraps training and generation as background jobs. | Training blocks for minutes. A job id and a poll endpoint beat holding an HTTP request open. |
| Next.js 16 · React 19 | The interface you are reading. | Server components for the prose, client components for anything that touches audio. |
| TanStack Query | Owns server state, including the job-polling loop. | Job state is server state on a timer, which is exactly what refetchInterval is for. Polling stops by itself. |
| Tone.js · @tonejs/midi | Plays generated MIDI in your browser. | Removed the Fluidsynth and soundfont install that used to stand between a fresh clone and hearing anything. |
| Meta MusicGen | Text prompt to audio, on the Python backend. | Composes in timbre rather than notes, so it reaches genres a piano vocabulary cannot express. Optional install. |
| Google Magenta | Melody continuation, in the browser via TensorFlow.js. | The same task as MAiSTRO's LSTM with the opposite deployment model — no server, no cold start. |
| NumPy + mido | The entire deployed inference stack. | Together with FastAPI they come to 31MB, against TensorFlow's 877MB. It is what makes a free-tier deployment possible at all. |

### Honesty — "What this does not do"

> The transformer ships untrained. Its architecture, training path and arena integration are
> tested end to end, but nobody has run the epochs, so the comparison above is a parameter count
> and not a quality claim.
>
> Conditioning is steering, not learned control. A strong bias argues with the model rather than
> collaborating with it, which is why the penalties are small.
>
> The arena's ratings mean little until a few dozen votes are in. One listener cannot invert a
> leaderboard, but they can certainly wobble it.
