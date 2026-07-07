export function JobLog({ lines }: { lines: string[] }) {
  if (lines.length === 0) return null;

  return (
    <div className="mt-4 max-h-48 overflow-y-auto border-t border-border pt-3">
      <ul className="space-y-1">
        {lines.map((line, i) => (
          <li key={i} className="text-xs leading-relaxed text-muted-foreground tabular-nums">
            {line}
          </li>
        ))}
      </ul>
    </div>
  );
}
