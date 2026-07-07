/** Thin progress rail; the fill animates via scaleX (not width) to avoid layout thrash. */
export function ProgressBar({ fraction, tone = "brass" }: { fraction: number; tone?: "brass" | "verdigris" }) {
  const clamped = Math.min(1, Math.max(0, fraction));
  const fillColor = tone === "brass" ? "bg-brass" : "bg-verdigris";

  return (
    <div className="h-px w-full bg-border" role="progressbar" aria-valuenow={Math.round(clamped * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div
        className={`h-px origin-left ${fillColor} transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)]`}
        style={{ transform: `scaleX(${clamped})`, width: "100%" }}
      />
    </div>
  );
}
