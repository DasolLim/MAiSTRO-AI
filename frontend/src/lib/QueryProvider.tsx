"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/**
 * React Query owns every server interaction: cache, retries, and the polling loop
 * that the job API needs. Long-running work (dataset prep, training, generation)
 * returns a job id immediately, so the UI's job state is really *server* state on
 * a timer — exactly what `refetchInterval` exists for. See useJob.ts.
 */
export function QueryProvider({ children }: { children: React.ReactNode }) {
  // Created in state, not at module scope: a module-level client would be shared
  // across requests on the server and leak one user's cache into another's.
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            refetchOnWindowFocus: false,
            // A backend that isn't running yet is the common failure here, and
            // retrying three times just delays the error message.
            retry: 1,
          },
        },
      }),
  );

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
