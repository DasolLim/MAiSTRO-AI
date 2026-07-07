"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const MOVEMENTS = [
  { numeral: "I", label: "Prepare", href: "/dataset" },
  { numeral: "II", label: "Train", href: "/train" },
  { numeral: "III", label: "Generate", href: "/generate" },
  { numeral: "IV", label: "Listen", href: "/library" },
] as const;

export function ProgramNav() {
  const pathname = usePathname();

  return (
    <header className="border-b border-border">
      <div className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-8 sm:flex-row sm:items-end sm:justify-between">
        <Link href="/" className="block">
          <span className="font-display text-3xl tracking-tight text-foreground">MAiSTRO</span>
          <span className="mt-1 block text-xs font-medium tracking-[0.2em] text-muted-foreground uppercase">
            AI classical composition
          </span>
        </Link>

        <nav aria-label="Program" className="flex gap-6 sm:gap-8">
          {MOVEMENTS.map((movement) => {
            const active = pathname?.startsWith(movement.href);
            return (
              <Link
                key={movement.href}
                href={movement.href}
                className="group flex flex-col items-start gap-1.5"
                aria-current={active ? "page" : undefined}
              >
                <span
                  className={`text-sm tracking-wide ${
                    active ? "text-foreground" : "text-muted-foreground group-hover:text-foreground"
                  }`}
                >
                  <span className="font-display italic">{movement.numeral}.</span> {movement.label}
                </span>
                <span
                  className={`h-px w-full transition-colors duration-200 ${
                    active ? "bg-brass" : "bg-transparent group-hover:bg-border"
                  }`}
                />
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
