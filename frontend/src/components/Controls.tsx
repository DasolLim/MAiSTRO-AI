"use client";

/** Form primitives styled for the ink-and-brass theme. No third-party UI kit. */

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium tracking-[0.15em] text-muted-foreground uppercase">
        {label}
      </span>
      <div className="mt-2">{children}</div>
      {hint && <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{hint}</p>}
    </label>
  );
}

export function Select({
  value,
  onChange,
  options,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      className="w-full cursor-pointer border border-border bg-surface px-3 py-2 text-sm text-foreground transition-colors hover:border-brass focus:border-brass focus:outline-none disabled:cursor-not-allowed disabled:text-muted-foreground"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value} className="bg-surface">
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step,
  format = (v: number) => v.toFixed(2),
  disabled,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (value: number) => string;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-4">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-border accent-brass disabled:cursor-not-allowed"
      />
      <span className="w-12 shrink-0 text-right text-sm tabular-nums text-brass">
        {format(value)}
      </span>
    </div>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  type = "button",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost";
  type?: "button" | "submit";
}) {
  const styles =
    variant === "primary"
      ? "border-brass text-brass hover:bg-brass hover:text-brass-foreground"
      : "border-border text-muted-foreground hover:border-brass hover:text-brass";

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`cursor-pointer border px-5 py-2.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:border-border disabled:text-muted-foreground disabled:hover:bg-transparent ${styles}`}
    >
      {children}
    </button>
  );
}

/** Key/value strip used to show generation metrics under a result. */
export function MetricGrid({ metrics }: { metrics: Record<string, string | number | null> }) {
  const entries = Object.entries(metrics).filter(([, value]) => value !== null && value !== undefined);
  if (entries.length === 0) return null;

  return (
    <dl className="mt-6 grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-4">
      {entries.map(([label, value]) => (
        <div key={label}>
          <dt className="text-xs tracking-[0.12em] text-muted-foreground uppercase">{label}</dt>
          <dd className="mt-1 font-display text-xl tabular-nums text-foreground">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
