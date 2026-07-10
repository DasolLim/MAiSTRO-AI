import { GREEDY_ROLL, SAMPLED_ROLL, type HeroRoll } from "@/lib/heroRolls";

/**
 * The page's thesis, drawn rather than described: one model, two decoders.
 *
 * Both rolls come from the same checkpoint, seed and 300-note budget. The left one
 * always takes the likeliest next note; the right one samples. They are real
 * outputs (see lib/heroRolls.ts), so the captions quote these exact excerpts
 * rather than a flattering average.
 */

const ROLL_HEIGHT = 132;

function Roll({ roll, tone }: { roll: HeroRoll; tone: "dim" | "bright" }) {
  const pitches = roll.notes.map((note) => note.m);
  const low = Math.min(...pitches) - 2;
  const span = Math.max(Math.max(...pitches) - low + 3, 12);
  const rowHeight = ROLL_HEIGHT / span;

  const color = tone === "bright" ? "var(--color-brass)" : "var(--color-muted-foreground)";

  return (
    <svg
      viewBox={`0 0 ${roll.span} ${ROLL_HEIGHT}`}
      preserveAspectRatio="none"
      className="block h-32 w-full sm:h-36"
      role="img"
      aria-label={`Piano roll of ${roll.notes.length} notes. Repetition rate ${roll.repetition}.`}
    >
      {roll.notes.map((note, index) => (
        <rect
          key={index}
          x={note.t}
          y={ROLL_HEIGHT - (note.m - low) * rowHeight - rowHeight}
          width={Math.max(note.d, 0.12)}
          height={Math.max(rowHeight - 0.6, 1)}
          fill={color}
          className="hero-note"
          style={
            {
              // The keyframes settle on --note-opacity, so the final opacity is
              // still per-roll; an `opacity` attribute would fight the animation.
              "--note-opacity": tone === "bright" ? 0.92 : 0.5,
              // Notes arrive left to right, at the pace the piece would be played.
              animationDelay: `${(note.t / roll.span) * 1.5}s`,
            } as React.CSSProperties
          }
        />
      ))}
    </svg>
  );
}

function Panel({
  roll,
  tone,
  title,
  caption,
}: {
  roll: HeroRoll;
  tone: "dim" | "bright";
  title: string;
  caption: string;
}) {
  return (
    <figure className="min-w-0">
      <figcaption className="flex items-baseline justify-between gap-3">
        <span
          className={`font-display text-lg italic ${tone === "bright" ? "text-brass" : "text-muted-foreground"}`}
        >
          {title}
        </span>
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
          {roll.distinctPitches} pitches
        </span>
      </figcaption>

      <div className="mt-3 border border-border bg-surface/40 p-2">
        <Roll roll={roll} tone={tone} />
      </div>

      <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
        <span className="text-foreground tabular-nums">
          repetition {roll.repetition.toFixed(3)}
        </span>{" "}
        · {caption}
      </p>
    </figure>
  );
}

export function HeroRolls() {
  return (
    <div>
      <div className="grid gap-8 sm:grid-cols-2 sm:gap-10">
        <Panel
          roll={GREEDY_ROLL}
          tone="dim"
          title="Always the likeliest note"
          caption="The decoder MAiSTRO shipped with. One path out of any passage, so it keeps returning to material it has already played."
        />
        <Panel
          roll={SAMPLED_ROLL}
          tone="bright"
          title="Sampled, temperature 1.9"
          caption="Same model, same seed, same 300 notes. Nine more pitches in play, and it stops circling."
        />
      </div>

      <p className="mt-8 max-w-[62ch] text-sm leading-relaxed text-muted-foreground">
        Nothing was retrained between these two. The only change is how the next note gets
        chosen — and by note 300 the two decoders disagree on 61% of the model&apos;s decisions.
      </p>
    </div>
  );
}
