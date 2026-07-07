"use client";

import { useEffect, useState } from "react";
import { getLibrary, type LibraryTrack } from "@/lib/api";
import { StaffDivider } from "@/components/StaffDivider";
import { TrackRow } from "@/components/TrackRow";

export default function LibraryPage() {
  const [tracks, setTracks] = useState<LibraryTrack[] | null>(null);

  const refresh = () => {
    getLibrary().then(({ tracks }) => setTracks(tracks));
  };

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        IV. Listen
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground">The listening room</h1>
      <p className="mt-4 max-w-[65ch] text-muted-foreground">
        Everything the model has composed so far. New compositions need to be rendered to
        audio once before they can be played back.
      </p>

      <StaffDivider className="mt-10" />

      {tracks === null && <p className="mt-8 text-sm text-muted-foreground">Loading…</p>}
      {tracks?.length === 0 && (
        <p className="mt-8 text-sm text-muted-foreground">
          No compositions yet. Generate one from the Generate movement.
        </p>
      )}
      {tracks && tracks.length > 0 && (
        <ul className="mt-2 divide-y divide-border">
          {tracks.map((track) => (
            <TrackRow key={track.midi_filename} track={track} onRendered={refresh} />
          ))}
        </ul>
      )}
    </div>
  );
}
