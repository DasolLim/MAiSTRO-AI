"use client";

import { LocalOnly } from "@/components/LocalOnly";
import { StaffDivider } from "@/components/StaffDivider";
import { TrackRow } from "@/components/TrackRow";
import { useFeature, useLibrary } from "@/lib/useJob";

export default function LibraryPage() {
  const { available } = useFeature("library");
  const gate = !available ? (
    <LocalOnly
      feature={"The listening room"}
      what={"It lists every piece the model has composed and lets you play them back, tagged with the settings and seed that produced them."}
      why={"It needs a filesystem that survives between requests. The deployed API runs as a stateless function, so a piece it generates exists only in the response that returns it."}
    />
  ) : null;

  const { data, isLoading, error } = useLibrary();
  const tracks = data?.tracks;

  // Rendered after every hook above, so the hook order does not change when
  // capabilities load and `gate` flips from null to a panel.
  if (gate) return <div>{gate}</div>;

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        Compose
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground">The listening room</h1>
      <p className="mt-4 max-w-[65ch] text-muted-foreground">
        Everything the model has composed so far. Each piece plays straight from its MIDI,
        synthesised in your browser, and carries the settings and seed that produced it.
      </p>

      <StaffDivider className="mt-10" />

      {isLoading && <p className="mt-8 text-sm text-muted-foreground">Loading…</p>}
      {error && (
        <p className="mt-8 text-sm text-destructive">
          Could not reach the API. Is the backend running on port 8000?
        </p>
      )}
      {tracks?.length === 0 && (
        <p className="mt-8 text-sm text-muted-foreground">
          No compositions yet. Generate one from the Generate movement.
        </p>
      )}
      {tracks && tracks.length > 0 && (
        <ul className="mt-2 divide-y divide-border">
          {tracks.map((track) => (
            <TrackRow key={track.midi_filename} track={track} />
          ))}
        </ul>
      )}
    </div>
  );
}
