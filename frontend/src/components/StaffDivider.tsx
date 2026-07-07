/** A five-line staff rule used as a section divider in place of a plain <hr>. */
export function StaffDivider({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 400 24"
      preserveAspectRatio="none"
      className={`h-6 w-full text-border ${className}`}
      aria-hidden="true"
    >
      {[4, 9, 12, 15, 20].map((y) => (
        <line key={y} x1="0" y1={y} x2="400" y2={y} stroke="currentColor" strokeWidth="1" />
      ))}
    </svg>
  );
}
