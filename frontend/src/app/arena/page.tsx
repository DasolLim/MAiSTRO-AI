"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button, MetricGrid } from "@/components/Controls";
import { JobLog } from "@/components/JobLog";
import { MidiPlayer } from "@/components/MidiPlayer";
import { StaffDivider } from "@/components/StaffDivider";
import { castVote, type VoteReveal } from "@/lib/api";
import { useArenaPair, useLeaderboard } from "@/lib/useJob";

type Side = "a" | "b" | "tie";

export default function ArenaPage() {
  const pairAction = useArenaPair();
  const leaderboard = useLeaderboard();
  const queryClient = useQueryClient();

  const [reveal, setReveal] = useState<VoteReveal | null>(null);

  const vote = useMutation({
    mutationFn: ({ pairId, outcome }: { pairId: string; outcome: Side }) =>
      castVote(pairId, outcome),
    onSuccess: (result) => {
      setReveal(result);
      queryClient.invalidateQueries({ queryKey: ["arena", "leaderboard"] });
    },
  });

  const pair = pairAction.job?.state === "done" ? pairAction.job.result : null;
  const rows = reveal?.leaderboard ?? leaderboard.data?.leaderboard ?? [];
  const trainedCount = rows.filter((row) => row.trained).length;

  const startNewPair = () => {
    setReveal(null);
    pairAction.start({ n_notes: 160 });
  };

  return (
    <div>
      <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
        Compose
      </p>
      <h1 className="mt-3 font-display text-4xl text-foreground">Which one sounds better?</h1>
      <p className="mt-4 max-w-[65ch] text-muted-foreground">
        Two architectures compose the same brief — identical length, key, temperature and random
        seed — and you hear them blind. Validation loss says which model fits the corpus. Only a
        listener says which one is worth hearing. Votes update an Elo rating for each model.
      </p>

      <StaffDivider className="mt-10" />

      {trainedCount < 2 && (
        <div className="mt-8 border border-border bg-surface p-6">
          <p className="text-sm text-foreground">
            The arena needs two trained models. {trainedCount} is trained so far.
          </p>
          <p className="mt-2 max-w-[60ch] text-sm text-muted-foreground">
            Train another from the Train movement — the transformer is the interesting opponent,
            and at ~3.9M parameters it trains far faster than the 179M-parameter LSTM + attention.
          </p>
        </div>
      )}

      {trainedCount >= 2 && (
        <div className="mt-8">
          <Button onClick={startNewPair} disabled={pairAction.isRunning}>
            {pairAction.isRunning ? "Composing both takes…" : pair ? "New pair" : "Start a listening test"}
          </Button>

          {pairAction.error && (
            <p className="mt-6 text-sm text-destructive">{pairAction.error.message}</p>
          )}
          {pairAction.isRunning && pairAction.job && <JobLog lines={pairAction.job.log} />}
        </div>
      )}

      {pair && (
        <div className="mt-12">
          <div className="grid gap-10 sm:grid-cols-2">
            {(["a", "b"] as const).map((side) => (
              <section key={side} className="border border-border p-5">
                <h2 className="font-display text-2xl text-foreground">
                  Take {side.toUpperCase()}
                  {reveal && (
                    <span className="ml-3 align-middle text-xs tracking-[0.15em] text-brass uppercase">
                      {reveal.reveal[side].arch.replace(/_/g, " + ")}
                    </span>
                  )}
                </h2>
                <div className="mt-4">
                  <MidiPlayer filename={pair[side].midi_filename} label={`Take ${side.toUpperCase()}`} height={150} />
                </div>
                {reveal && (
                  <MetricGrid
                    metrics={{
                      Repetition: reveal.reveal[side].metrics.repetition_rate.toFixed(3),
                      "Corpus KL": reveal.reveal[side].metrics.pitch_class_kl?.toFixed(3) ?? null,
                    }}
                  />
                )}
              </section>
            ))}
          </div>

          {!reveal ? (
            <div className="mt-8 flex flex-wrap items-center gap-4">
              <span className="text-sm text-muted-foreground">Your verdict:</span>
              {(
                [
                  ["a", "Take A"],
                  ["tie", "Too close to call"],
                  ["b", "Take B"],
                ] as const
              ).map(([outcome, label]) => (
                <Button
                  key={outcome}
                  variant={outcome === "tie" ? "ghost" : "primary"}
                  disabled={vote.isPending}
                  onClick={() => vote.mutate({ pairId: pair.pair_id, outcome })}
                >
                  {label}
                </Button>
              ))}
            </div>
          ) : (
            <p className="mt-8 text-sm text-verdigris">
              Recorded. {reveal.outcome === "tie" ? "A draw" : `Take ${reveal.outcome.toUpperCase()} wins`} —
              ratings updated below.
            </p>
          )}

          {vote.error && <p className="mt-4 text-sm text-destructive">{vote.error.message}</p>}
        </div>
      )}

      <div className="mt-16">
        <p className="text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
          Leaderboard
        </p>
        <StaffDivider className="mt-4" />

        <div className="overflow-x-auto">
          <table className="w-full min-w-[34rem] text-sm">
            <thead>
              <tr className="border-b border-border text-left">
                {["Model", "Elo", "Games", "W / L / D", "Win rate"].map((heading) => (
                  <th
                    key={heading}
                    className="pb-3 text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase"
                  >
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((row) => (
                <tr key={row.arch} className={row.trained ? "" : "opacity-45"}>
                  <td className="py-4 pr-4">
                    <span className="font-display text-lg italic text-foreground">{row.label}</span>
                    {!row.trained && (
                      <span className="ml-2 text-xs text-muted-foreground">untrained</span>
                    )}
                  </td>
                  <td className="py-4 pr-4 tabular-nums text-brass">{row.elo.toFixed(0)}</td>
                  <td className="py-4 pr-4 tabular-nums text-muted-foreground">{row.games}</td>
                  <td className="py-4 pr-4 tabular-nums text-muted-foreground">
                    {row.wins} / {row.losses} / {row.ties}
                  </td>
                  <td className="py-4 tabular-nums text-muted-foreground">
                    {row.win_rate === null ? "—" : `${(row.win_rate * 100).toFixed(0)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="mt-6 max-w-[65ch] text-xs leading-relaxed text-muted-foreground">
          Every model starts at 1500. A win moves the rating by up to 24 points, scaled by how
          surprising the result was — beating a higher-rated model is worth more. Ratings are only
          meaningful after a few dozen votes.
        </p>
      </div>
    </div>
  );
}
