"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

/**
 * Six flat links asked the visitor to work out the hierarchy themselves. These
 * three groups encode it: what you *do* with a trained model, what you do *to*
 * one, and what the project is.
 *
 * Only the Model group is numbered, because only it is a real sequence — you
 * cannot train before you prepare. Numbering the others would be decoration.
 */

interface NavItem {
  href: string;
  label: string;
  blurb: string;
  step?: string;
}

interface NavGroup {
  label: string;
  summary: string;
  items: NavItem[];
}

const GROUPS: NavGroup[] = [
  {
    label: "Compose",
    summary: "Use a trained model",
    items: [
      { href: "/generate", label: "Generate", blurb: "Sample a new piece, steered by key and mood" },
      { href: "/library", label: "Listening room", blurb: "Everything composed so far" },
      { href: "/arena", label: "Arena", blurb: "Judge two models blind" },
      { href: "/studio", label: "Beyond the piano", blurb: "MusicGen and Magenta, for other genres" },
    ],
  },
  {
    label: "Model",
    summary: "Build a new one",
    items: [
      { href: "/dataset", label: "Prepare data", blurb: "Turn MIDI files into a note vocabulary", step: "1" },
      { href: "/train", label: "Train", blurb: "Fit an LSTM or a transformer", step: "2" },
    ],
  },
  {
    label: "Project",
    summary: "Read about it",
    items: [
      { href: "/story", label: "The story", blurb: "Why eight students built a composer" },
      { href: "/how-it-works", label: "How it works", blurb: "The engineering, measured" },
    ],
  },
];

/** A menu is open only on the route it was opened from, so navigating closes it. */
type OpenPanel = { path: string; key: string } | null;

export function SiteNav() {
  const pathname = usePathname();
  const [panel, setPanel] = useState<OpenPanel>(null);
  const navRef = useRef<HTMLDivElement>(null);

  // Deriving "open" from the pathname closes every panel on navigation without an
  // effect that would setState on each route change and cascade a second render.
  const openKey = panel?.path === pathname ? panel.key : null;
  const openGroup = openKey === "mobile" ? null : openKey;
  const mobileOpen = openKey === "mobile";

  const open = (key: string) => setPanel({ path: pathname ?? "", key });
  const close = () => setPanel(null);

  useEffect(() => {
    if (!openKey) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    // Only the desktop dropdowns dismiss on an outside click. The mobile panel sits
    // outside navRef, so the same listener would close it on its own headings.
    const onPointerDown = (event: PointerEvent) => {
      if (openGroup && !navRef.current?.contains(event.target as Node)) close();
    };

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [openKey, openGroup]);

  const groupOf = (path: string) => GROUPS.find((g) => g.items.some((i) => path.startsWith(i.href)));
  const activeGroup = pathname ? groupOf(pathname)?.label : undefined;

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-bg/90 backdrop-blur-sm">
      <div
        ref={navRef}
        className="mx-auto flex max-w-5xl items-center justify-between gap-6 px-6 py-5"
      >
        <Link href="/" className="group block shrink-0">
          <span className="font-display text-2xl tracking-tight text-foreground">MAiSTRO</span>
          <span className="mt-0.5 block text-[0.65rem] font-medium tracking-[0.2em] text-muted-foreground uppercase">
            AI classical composition
          </span>
        </Link>

        {/* Desktop */}
        <nav aria-label="Main" className="hidden items-center gap-1 sm:flex">
          {GROUPS.map((group) => {
            const isOpen = openGroup === group.label;
            const isActive = activeGroup === group.label;

            return (
              <div key={group.label} className="relative">
                <button
                  onClick={() => (isOpen ? close() : open(group.label))}
                  aria-expanded={isOpen}
                  aria-haspopup="true"
                  className={`flex cursor-pointer items-center gap-1.5 px-3 py-2 text-sm transition-colors ${
                    isActive || isOpen ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {group.label}
                  <svg
                    viewBox="0 0 10 6"
                    aria-hidden="true"
                    className={`h-1.5 w-2.5 transition-transform duration-200 motion-reduce:transition-none ${
                      isOpen ? "rotate-180" : ""
                    }`}
                  >
                    <path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                </button>

                <span
                  className={`absolute inset-x-3 -bottom-px block h-px transition-colors ${
                    isActive ? "bg-brass" : "bg-transparent"
                  }`}
                />

                {isOpen && (
                  <div className="absolute right-0 top-full z-50 mt-3 w-80 border border-border bg-surface p-2 shadow-2xl shadow-black/50">
                    <p className="px-3 pt-2 pb-3 text-[0.65rem] tracking-[0.18em] text-muted-foreground uppercase">
                      {group.summary}
                    </p>
                    <ul>
                      {group.items.map((item) => (
                        <li key={item.href}>
                          <Link
                            href={item.href}
                            className={`flex gap-3 px-3 py-2.5 transition-colors hover:bg-surface-raised ${
                              pathname?.startsWith(item.href) ? "bg-surface-raised" : ""
                            }`}
                          >
                            {item.step && (
                              <span className="mt-0.5 font-display text-sm italic text-brass">
                                {item.step}
                              </span>
                            )}
                            <span className="min-w-0">
                              <span className="block text-sm text-foreground">{item.label}</span>
                              <span className="mt-0.5 block text-xs leading-snug text-muted-foreground">
                                {item.blurb}
                              </span>
                            </span>
                          </Link>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })}

          <Link
            href="/generate"
            className="ml-3 border border-brass px-4 py-2 text-sm font-medium text-brass transition-colors hover:bg-brass hover:text-brass-foreground"
          >
            Compose
          </Link>
        </nav>

        {/* Mobile: dropdowns are awkward on touch, so everything expands at once. */}
        <button
          onClick={() => (mobileOpen ? close() : open("mobile"))}
          aria-expanded={mobileOpen}
          aria-label={mobileOpen ? "Close menu" : "Open menu"}
          className="cursor-pointer border border-border p-2.5 text-foreground sm:hidden"
        >
          <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true">
            {mobileOpen ? (
              <path d="M3.5 2.5l10 10m0-10l-10 10" stroke="currentColor" strokeWidth="1.5" />
            ) : (
              <>
                <rect x="1" y="3" width="14" height="1.5" />
                <rect x="1" y="7.25" width="14" height="1.5" />
                <rect x="1" y="11.5" width="14" height="1.5" />
              </>
            )}
          </svg>
        </button>
      </div>

      {mobileOpen && (
        <nav aria-label="Main" className="border-t border-border px-6 pb-6 sm:hidden">
          {GROUPS.map((group) => (
            <div key={group.label} className="mt-6">
              <p className="text-[0.65rem] tracking-[0.18em] text-muted-foreground uppercase">
                {group.label} — {group.summary}
              </p>
              <ul className="mt-2 divide-y divide-border border-t border-border">
                {group.items.map((item) => (
                  <li key={item.href}>
                    <Link href={item.href} className="flex gap-3 py-3">
                      {item.step && (
                        <span className="font-display text-sm italic text-brass">{item.step}</span>
                      )}
                      <span>
                        <span className="block text-sm text-foreground">{item.label}</span>
                        <span className="mt-0.5 block text-xs text-muted-foreground">
                          {item.blurb}
                        </span>
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      )}
    </header>
  );
}
