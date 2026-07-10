"use client";

import { StaffDivider } from "@/components/StaffDivider";
import { TrackRow } from "@/components/TrackRow";
import { useLibrary } from "@/lib/useJob";

export default function LibraryPage() {
  const { data, isLoading, error } = useLibrary();
  const tracks = data?.tracks;

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        IV. Listen
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
