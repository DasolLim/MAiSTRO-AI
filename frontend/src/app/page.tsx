import Link from "next/link";
import { HeroRolls } from "@/components/HeroRolls";
import { StaffDivider } from "@/components/StaffDivider";

/**
 * The landing page's one job: make a visitor understand what MAiSTRO is and what
 * changed, then send them to /generate. Everything longer lives in /story (why it
 * was built) and /how-it-works (how it works, measured).
 */

const CHANGES = [
  {
    title: "The decoder was never asked",
    problem: "It always played the likeliest next note, so it looped.",
    change:
      "Temperature, top-k and nucleus sampling, calibrated against this model's own confidence rather than borrowed from text generation.",
    href: "/how-it-works",
  },
  {
    title: "You couldn't ask for anything",
    problem: "Generation started from a random passage. If you disliked it, you re-rolled.",
    change:
      "Ask for a key, a scale, a tempo, a mood. The bias is applied as the note is chosen, so nothing needed retraining.",
    href: "/generate",
  },
  {
    title: "Loss cannot hear",
    problem: "A lower validation loss does not mean a better piece of music.",
    change:
      "Two models compose the same brief and you judge them blind. Elo ratings, the same system chess uses, settle the argument.",
    href: "/arena",
  },
  {
    title: "One model, no comparison",
    problem: "The original network spends 175M of its parameters on a single output layer.",
    change:
      "A plain LSTM and a 3.9M-parameter transformer now share the dataset and decoder, so the architecture is the only variable.",
    href: "/train",
  },
  {
    title: "Hearing it meant installing Fluidsynth",
    problem: "The model emits MIDI. Turning that into sound needed a soundfont and a C library.",
    change:
      "Tone.js synthesises the notes in your browser, and a canvas piano roll draws them as they play.",
    href: "/library",
  },
  {
    title: "A piano cannot play every genre",
    problem: "No arrangement of piano notes means 'dusty Rhodes through a tape delay'.",
    change:
      "MusicGen generates audio from a text prompt; Magenta continues a melody entirely in your browser.",
    href: "/studio",
  },
];

const NUMBERS = [
  { value: "0.95", label: "Mean top probability", note: "Why textbook sampling defaults do nothing here" },
  { value: "0.151 → 0.024", label: "Repetition rate", note: "Greedy against temperature 1.9, over 4 seeds" },
  { value: "0.79 → 0.93", label: "Notes in the asked-for key", note: "Unconditioned against C major" },
  { value: "46×", label: "Fewer parameters", note: "Transformer against the original network" },
];

export default function Home() {
  return (
    <div>
      {/* Hero */}
      <section>
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          Western Cyber Society
        </p>
        <h1 className="mt-4 max-w-[18ch] font-display text-5xl leading-[1.05] text-foreground sm:text-6xl">
          A model that composes. Now it takes direction.
        </h1>
        <p className="mt-6 max-w-[64ch] text-lg leading-relaxed text-muted-foreground">
          MAiSTRO is an LSTM trained on Mozart, Beethoven and Chopin. It could always write
          music. What it could not do was write it <em className="text-foreground">differently</em>{" "}
          twice, take a request, or tell you whether it had improved.
        </p>

        <div className="mt-8 flex flex-wrap gap-4">
          <Link
            href="/generate"
            className="border border-brass px-5 py-2.5 text-sm font-medium text-brass transition-colors hover:bg-brass hover:text-brass-foreground"
          >
            Compose something
          </Link>
          <Link
            href="/how-it-works"
            className="border border-border px-5 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:border-brass hover:text-brass"
          >
            How it works
          </Link>
        </div>
      </section>

      {/* Signature: the argument, drawn */}
      <section className="mt-20">
        <HeroRolls />
      </section>

      <StaffDivider className="mt-20" />

      {/* Origin, briefly */}
      <section className="mt-16 grid gap-10 sm:grid-cols-[1fr_1.3fr]">
        <div>
          <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
            Where it started
          </p>
          <h2 className="mt-3 font-display text-3xl text-foreground">Eight students, one LSTM</h2>
        </div>
        <div className="max-w-[62ch] space-y-4 text-muted-foreground">
          <p>
            AI was writing essays and painting pictures, and music composition sat strangely
            untouched. A team of eight set out to see whether a network could learn the shape of
            classical piano from the notation itself.
          </p>
          <p>
            It worked. A bidirectional LSTM with an attention layer, trained on 200 MIDI files,
            produced pieces you would happily leave playing. That model is still here, still the
            default, and{" "}
            <Link href="/story" className="text-brass underline underline-offset-2">
              the story of building it
            </Link>{" "}
            is worth reading first.
          </p>
          <p className="text-foreground">
            What follows is what happened when it stopped being a model and started being a tool.
          </p>
        </div>
      </section>

      <StaffDivider className="mt-16" />

      {/* What changed */}
      <section className="mt-16">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          What changed
        </p>
        <h2 className="mt-3 font-display text-3xl text-foreground">Six problems, six answers</h2>
        <p className="mt-4 max-w-[62ch] text-muted-foreground">
          Each of these began as something that annoyed someone using it.
        </p>

        <div className="mt-10 grid gap-x-10 gap-y-1 sm:grid-cols-2">
          {CHANGES.map((item) => (
            <Link key={item.title} href={item.href} className="group block border-t border-border py-6">
              <h3 className="font-display text-xl text-foreground group-hover:text-brass">
                {item.title}
              </h3>
              <p className="mt-2 max-w-[46ch] text-sm text-muted-foreground italic">
                {item.problem}
              </p>
              <p className="mt-2.5 max-w-[46ch] text-sm leading-relaxed text-foreground/80">
                {item.change}
              </p>
            </Link>
          ))}
        </div>
      </section>

      <StaffDivider className="mt-16" />

      {/* Numbers */}
      <section className="mt-16">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          Measured, not estimated
        </p>
        <h2 className="mt-3 font-display text-3xl text-foreground">What it bought</h2>

        <dl className="mt-10 grid gap-x-8 gap-y-10 sm:grid-cols-2 lg:grid-cols-4">
          {NUMBERS.map((stat) => (
            <div key={stat.label}>
              <dd className="font-display text-3xl text-brass tabular-nums">{stat.value}</dd>
              <dt className="mt-2 text-sm text-foreground">{stat.label}</dt>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{stat.note}</p>
            </div>
          ))}
        </dl>

        <p className="mt-10 max-w-[62ch] text-sm text-muted-foreground">
          Every figure comes from running the trained model, averaged across four random seeds.
          The workings, including where the numbers are less flattering than they look, are in{" "}
          <Link href="/how-it-works" className="text-brass underline underline-offset-2">
            how it works
          </Link>
          .
        </p>
      </section>

      <StaffDivider className="mt-16" />

      {/* CTA */}
      <section className="mt-16 mb-4 border border-border p-8 sm:p-10">
        <h2 className="max-w-[20ch] font-display text-3xl text-foreground">
          The model is trained and waiting
        </h2>
        <p className="mt-4 max-w-[58ch] text-muted-foreground">
          Pick a key, pick a mood, move the creativity dial, and hear what a network trained on
          three dead composers does with the request.
        </p>
        <Link
          href="/generate"
          className="mt-7 inline-block border border-brass px-5 py-2.5 text-sm font-medium text-brass transition-colors hover:bg-brass hover:text-brass-foreground"
        >
          Compose something
        </Link>
      </section>
    </div>
  );
}
