"use client";

import { StaffDivider } from "./StaffDivider";

/**
 * Shown in place of a feature the deployed backend cannot offer.
 *
 * The free-tier deployment is a stateless function with a read-only disk: it can
 * generate a piece, and nothing else. Rather than let a visitor click Train and watch
 * a request fail, each such page explains what it does, why it is not here, and the
 * one command that brings it back.
 */
export function LocalOnly({
  feature,
  what,
  why,
}: {
  feature: string;
  what: string;
  why: string;
}) {
  return (
    <div className="mt-8 border border-border bg-surface p-8">
      <p className="text-xs font-medium tracking-[0.2em] text-brass uppercase">Runs locally</p>
      <h2 className="mt-3 font-display text-2xl text-foreground">{feature} is not on this site</h2>

      <p className="mt-4 max-w-[62ch] text-sm leading-relaxed text-muted-foreground">{what}</p>
      <p className="mt-3 max-w-[62ch] text-sm leading-relaxed text-muted-foreground">{why}</p>

      <StaffDivider className="mt-6" />

      <p className="mt-4 text-xs tracking-[0.12em] text-muted-foreground uppercase">
        To use it, run the full backend
      </p>
      <pre className="mt-3 overflow-x-auto bg-bg px-4 py-3 text-xs leading-relaxed text-brass">
        <code>
          {`git clone https://github.com/DasolLim/MAiSTRO-AI.git\ncd MAiSTRO-AI && pip install -r requirements.txt\nuvicorn api.main:app --port 8000`}
        </code>
      </pre>
      <p className="mt-3 text-xs text-muted-foreground">
        Then start the frontend with <code className="text-foreground">npm run dev</code> and every
        feature is available.
      </p>
    </div>
  );
}
