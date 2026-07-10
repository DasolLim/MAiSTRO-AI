import type { Metadata } from "next";
import Link from "next/link";
import { StaffDivider } from "@/components/StaffDivider";

export const metadata: Metadata = {
  title: "How it works — MAiSTRO",
  description:
    "The engineering behind MAiSTRO: calibrated sampling, decode-time conditioning, three architectures, and a blind listening arena.",
};

function Section({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-16 scroll-mt-24">
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        {eyebrow}
      </p>
      <h2 className="mt-3 font-display text-3xl text-foreground">{title}</h2>
      <div className="mt-5 max-w-[70ch] space-y-4 text-muted-foreground">{children}</div>
    </section>
  );
}

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="mt-8 overflow-x-auto">
      <table className="w-full min-w-120 text-sm">
        <thead>
          <tr className="border-b border-border text-left">
            {head.map((cell) => (
              <th
                key={cell}
                className="pb-3 text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase"
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((row) => (
            <tr key={String(row[0])}>
              {row.map((cell, index) => (
                <td
                  key={index}
                  className={`py-3 pr-6 ${
                    index === 0 ? "text-foreground" : "tabular-nums text-muted-foreground"
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Named so the tooling reads as an inventory, not a logo wall. */
const STACK = [
  {
    name: "TensorFlow / Keras 2.10",
    role: "Trains and runs all three architectures.",
    why: "Pinned to 2.10 because keras_self_attention, which the original model depends on, has no Keras 3 build.",
  },
  {
    name: "music21",
    role: "Parses MIDI into a note vocabulary and writes it back out.",
    why: "Handles chords, rests and fractional durations that raw MIDI libraries leave to you.",
  },
  {
    name: "FastAPI",
    role: "Wraps training and generation as background jobs.",
    why: "Training blocks for minutes. A job id and a poll endpoint beat holding an HTTP request open.",
  },
  {
    name: "Next.js 16 · React 19",
    role: "The interface you are reading.",
    why: "Server components for the prose, client components for anything that touches audio.",
  },
  {
    name: "TanStack Query",
    role: "Owns server state, including the job-polling loop.",
    why: "Job state is server state on a timer, which is exactly what refetchInterval is for. Polling stops by itself.",
  },
  {
    name: "Tone.js · @tonejs/midi",
    role: "Plays generated MIDI in your browser.",
    why: "Removed the Fluidsynth and soundfont install that used to stand between a fresh clone and hearing anything.",
  },
  {
    name: "Meta MusicGen",
    role: "Text prompt to audio, on the Python backend.",
    why: "Composes in timbre rather than notes, so it reaches genres a piano vocabulary cannot express. Optional install.",
  },
  {
    name: "Google Magenta",
    role: "Melody continuation, in the browser via TensorFlow.js.",
    why: "The same task as MAiSTRO's LSTM with the opposite deployment model — no server, no cold start.",
  },
  {
    name: "NumPy + mido",
    role: "The entire deployed inference stack.",
    why: "Together with FastAPI they come to 31MB, against TensorFlow's 877MB. It is what makes a free-tier deployment possible at all.",
  },
];

export default function HowItWorksPage() {
  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        Under the hood
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground sm:text-5xl">How it works</h1>
      <p className="mt-4 max-w-[68ch] text-muted-foreground">
        MAiSTRO started as a model that could compose. Making it a tool anyone could actually
        use meant fixing how it chooses notes, giving a person a way to steer it, and building
        an honest way to tell whether a change made the music better. Every number here was
        measured on the trained model, not estimated.
      </p>

      <StaffDivider className="mt-10" />

      <Section eyebrow="Decoding" title="The model was never the problem">
        <p>
          The original decoder took the single most likely next note, every time. That sounds
          reasonable and it is quietly fatal: a deterministic decoder can only ever walk one path
          out of a given passage. The moment the model saw a context it had seen before, it played
          the same continuation, forever. What sounded like a model with nothing to say was really
          a model never being asked.
        </p>
        <p>
          The fix is to sample from the distribution rather than collapse it. Temperature controls
          how much: low values sharpen toward the model&apos;s favourite note, high values flatten
          toward surprise.
        </p>
        <p>
          Calibrating it was the interesting part, because this model is unusually sure of itself.
          Its next-note distribution has a mean top probability of{" "}
          <span className="text-foreground">0.95</span> and an entropy of{" "}
          <span className="text-foreground">0.21 nats</span> — far sharper than a language model.
          The likeliest note already holds more probability than any sensible nucleus threshold, so
          the textbook defaults borrowed from text generation truncate to a single token and
          silently reproduce greedy decoding. The dial does nothing, and you would never know.
        </p>

        <Table
          head={["Temperature", "Top probability", "Perplexity", "Notes in the nucleus"]}
          rows={[
            ["1.0", "0.96", "1.1", "1"],
            ["1.5", "0.94", "1.4", "12"],
            ["1.8", "0.87", "3.7", "335"],
            ["2.0", "0.77", "12.7", "1,713"],
            ["2.5", "0.41", "239.0", "2,899"],
          ]}
        />

        <p className="mt-8">
          Below about 1.3 the decoder is greedy. Above about 2.1 it is drawing from a flat tail of
          two thousand notes, which is noise. The usable band is{" "}
          <span className="text-foreground">1.4 to 2.0</span>, and that is the range the creativity
          slider spans — not an arbitrary 0 to 1.
        </p>
        <p>
          What that buys, averaged over four seeds at 160 notes each. Repetition is the share of
          four-note windows already played; corpus KL measures distance from the training set&apos;s
          pitch distribution. Lower is better in both.
        </p>

        <Table
          head={["Setting", "Repetition", "Corpus KL", "Unique notes"]}
          rows={[
            ["Greedy — the old default", "0.151", "0.498", "0.338"],
            ["T=1.0, top_p=0.92", "0.177", "0.503", "0.331"],
            ["T=1.5 — the new default", "0.064", "0.301", "0.444"],
            ["T=1.9", "0.024", "0.147", "0.570"],
          ]}
        />

        <p className="mt-8">
          The second row is the point. Those are the sampling defaults you would copy from any
          text-generation tutorial, and they are{" "}
          <span className="text-foreground">indistinguishable from greedy decoding</span> — the
          difference sits well inside the variation between seeds. The dial would have been
          connected to nothing.
        </p>
        <p>
          One honest caveat. Sampling diverges from the greedy path slowly: at 120 notes it changes
          only a fiftieth of the model&apos;s decisions, but by 300 notes it has changed most of
          them. Sampling matters more the longer the piece runs.
        </p>
      </Section>

      <StaffDivider className="mt-16" />

      <Section eyebrow="Control" title="Steering a model that has no steering wheel">
        <p>
          The network takes a window of past notes and nothing else. It has no input for key, no
          input for mood, and training a model that did would need labelled data nobody has. So
          conditioning happens at the last possible moment instead: every note in the vocabulary
          gets a score, and that score is added to the model&apos;s own opinion just before a note
          is picked.
        </p>
        <p>
          Ask for C major and notes outside the scale get pushed down — in proportion to how far
          outside they are, so a chord with one foreign note is discouraged less than one with
          three. The push is deliberately gentle, a few nats. A heavier hand would win every
          argument with the model and flatten the music into scales.
        </p>
        <p>
          Mood presets bundle a temperature, a scale, a register and a note density. They exist
          because &ldquo;melancholic&rdquo; is easier to ask for than{" "}
          <span className="text-foreground">
            temperature 1.45, natural minor, centred on G3, favour long notes
          </span>
          .
        </p>
        <Table
          head={["", "Notes in the requested key", "Spread across seeds"]}
          rows={[
            ["Unconditioned", "0.786", "±0.145"],
            ["Asked for C major", "0.930", "±0.039"],
          ]}
        />
        <p className="mt-8">
          The tightened spread matters as much as the raised average. Left alone, which key you get
          is luck of the seed. Asked, you get the key you asked for.
        </p>
      </Section>

      <StaffDivider className="mt-16" />

      <Section eyebrow="Architecture" title="Three models, one dataset">
        <p>
          Comparing architectures is only meaningful when nothing else moves. All three share a
          dataset, a tokenizer and a decoder, so a difference in output is a difference in
          architecture rather than in plumbing.
        </p>
        <Table
          head={["Model", "Parameters", "What it is"]}
          rows={[
            ["LSTM", "4.9M", "Two stacked layers. The baseline."],
            ["LSTM + attention", "178.8M", "The original MAiSTRO network."],
            ["Transformer", "3.9M", "A small GPT-style causal decoder."],
          ]}
        />
        <p className="mt-8">
          That middle row is worth staring at. The original network ends by flattening a 100×512
          sequence into a single dense layer, which costs{" "}
          <span className="text-foreground">175 million parameters on the output layer alone</span>
          . The transformer reaches comparable capacity with 46× fewer, because it never builds
          that matrix. Whether it also sounds better is not a question a parameter count can
          answer — which is what the arena is for.
        </p>
      </Section>

      <StaffDivider className="mt-16" />

      <Section eyebrow="Evaluation" title="Loss is not taste">
        <p>
          Validation loss tells you how well a model predicts the corpus. It does not tell you
          whether a person would want to listen to it. Cross-entropy punishes a beautiful passing
          tone for the crime of not appearing in Chopin.
        </p>
        <p>
          So the{" "}
          <Link href="/arena" className="text-brass underline underline-offset-2">
            arena
          </Link>{" "}
          asks a person. Two architectures compose the same brief — same length, key, temperature
          and random seed, so the architecture is the only thing that differs — and you hear them
          without being told which is which. Elo ratings update from your vote, the same system
          chess uses, because votes arrive as pairwise comparisons and a new model can join without
          re-running every earlier matchup.
        </p>
        <p>
          Blinding is enforced on the server rather than in the interface: the model name is
          stripped from the response, kept out of the progress log, and the metadata file that
          would otherwise sit next to each clip is not written. It would have been reachable by
          guessing a URL.
        </p>
        <p>
          Two cheap statistics run alongside the human vote.{" "}
          <span className="text-foreground">Repetition</span> is the share of four-note windows the
          decoder has already played. <span className="text-foreground">Corpus KL</span> compares
          the piece&apos;s pitch-class distribution against the training set. Neither replaces
          listening. Both are reproducible, which listening is not.
        </p>
      </Section>

      <StaffDivider className="mt-16" />

      <Section eyebrow="Reach" title="Music the note vocabulary cannot express">
        <p>
          MAiSTRO composes notes, and a piano renders them. That is exactly right for classical
          piano and exactly wrong for music that lives in timbre. There is no arrangement of piano
          notes that means &ldquo;dusty Rhodes through a tape delay.&rdquo;
        </p>
        <p>
          <span className="text-foreground">MusicGen</span> predicts compressed audio instead of
          notes, so it can render any genre it was trained on — at the cost of never producing a
          score you can edit. It runs on the Python backend and is an optional install.{" "}
          <span className="text-foreground">Magenta&apos;s MelodyRNN</span> does the same job as
          MAiSTRO&apos;s LSTM, continuing a melody one note at a time, but downloads to your browser
          and runs on your own GPU. Together they bracket the question of where a model should
          live.
        </p>
      </Section>

      <StaffDivider className="mt-16" />

      <Section eyebrow="Deployment" title="Fitting a neural network into 40 megabytes">
        <p>
          This site runs on a free tier. Vercel&apos;s Python runtime is 3.12 with a 500MB ceiling
          on a deployed function — and TensorFlow is 877MB unpacked, and needs Python 3.10 or
          older. The training backend cannot go there at any size.
        </p>
        <p>
          The way through is a distinction worth internalising:{" "}
          <span className="text-foreground">TensorFlow is only needed to train</span>. Running a
          trained model forward is a dozen matrix multiplications over a few million numbers.
          Nothing about that requires a deep-learning framework.
        </p>
        <p>
          So the deployed API reimplements the transformer&apos;s forward pass in plain NumPy and
          reads its weights from a 7MB file. Writing MIDI moves from music21 (111MB) to mido
          (under 1MB). What ships is fastapi, numpy and mido.
        </p>

        <Table
          head={["Component", "Unpacked size"]}
          rows={[
            ["TensorFlow", "877 MB"],
            ["music21", "111 MB"],
            ["fastapi + numpy + mido", "31 MB"],
            ["Transformer weights, float16", "7.0 MB"],
            ["Vocabulary + seed corpus", "0.17 MB"],
            ["Deployed total", "≈ 40 MB"],
          ]}
        />

        <p className="mt-8">
          A rewrite is only worth anything if it computes the same function. The NumPy path is
          checked against Keras on identical weights: the largest disagreement anywhere in the
          output distribution is{" "}
          <span className="text-foreground tabular-nums">4.2 × 10⁻⁷</span>, the two always pick
          the same next note, and a 300-note piece takes{" "}
          <span className="text-foreground">9.9 seconds</span> against a 300-second limit. The
          training notebook re-runs that comparison and refuses to export weights that fail it.
        </p>
        <p>
          Which model gets deployed follows from the arithmetic. The original LSTM keeps 97% of
          its 178.8M parameters in one output layer — 357MB even at half precision. The
          transformer&apos;s 3.9M parameters (256 dimensions, 4 heads, 4 layers) fit in 7MB. It is
          trained on a free Colab T4, where an epoch takes about two minutes instead of the
          seventeen it takes on a laptop.
        </p>
        <p>
          A serverless function is stateless with a read-only disk, so the deployment keeps what
          fits and is honest about the rest. Generation answers in one request, returning the MIDI
          inline rather than writing a file. Magenta was already running in your browser. The
          library, the arena, training and MusicGen need a filesystem, two trained models, hours,
          and 511MB of torch respectively — they stay local, and the interface says so rather than
          failing at you.
        </p>
      </Section>

      <StaffDivider className="mt-16" />

      <Section eyebrow="Stack" title="What it is built with, and why">
        <p>Every dependency here earns its place by removing a specific problem.</p>
        <div className="mt-8">
          <StaffDivider />
          {STACK.map((tool) => (
            <div key={tool.name}>
              <div className="py-6">
                <h3 className="font-display text-xl text-foreground">{tool.name}</h3>
                <p className="mt-1.5 max-w-[62ch] text-sm text-foreground/80">{tool.role}</p>
                <p className="mt-1 max-w-[62ch] text-sm text-muted-foreground">{tool.why}</p>
              </div>
              <StaffDivider />
            </div>
          ))}
        </div>
      </Section>

      <Section eyebrow="Honesty" title="What this does not do">
        <p>
          The transformer ships untrained. Its architecture, training path and arena integration
          are tested end to end, but nobody has run the epochs, so the comparison above is a
          parameter count and not a quality claim.
        </p>
        <p>
          Conditioning is steering, not learned control. A strong bias argues with the model rather
          than collaborating with it, which is why the penalties are small.
        </p>
        <p>
          The arena&apos;s ratings mean little until a few dozen votes are in. One listener cannot
          invert a leaderboard, but they can certainly wobble it.
        </p>
      </Section>

      <div className="mt-16 mb-4 border border-border p-8">
        <h2 className="font-display text-2xl text-foreground">Enough reading</h2>
        <p className="mt-3 max-w-[60ch] text-muted-foreground">
          The model is trained and waiting. Pick a key, pick a mood, and hear what it does with
          them.
        </p>
        <Link
          href="/generate"
          className="mt-6 inline-block border border-brass px-5 py-2.5 text-sm font-medium text-brass transition-colors hover:bg-brass hover:text-brass-foreground"
        >
          Compose something
        </Link>
      </div>
    </div>
  );
}
