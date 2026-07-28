"use client";

import { useEffect, useState } from "react";

const links = [
  { href: "#approach", label: "Approach" },
  { href: "#pipeline", label: "Pipeline" },
  { href: "#case-study", label: "Case Study" },
  { href: "#log", label: "Log" },
  { href: "#contact", label: "Contact" },
] as const;

export function Nav() {
  const [solid, setSolid] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setSolid(window.scrollY > window.innerHeight * 0.85);
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  return (
    <header
      className={[
        "fixed inset-x-0 top-0 z-50 transition-[background-color,border-color] duration-300",
        solid
          ? "border-b border-ink-800 bg-ink-900"
          : "border-b border-transparent bg-transparent",
      ].join(" ")}
    >
      <nav className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-5 md:h-16 md:px-8">
        <a
          href="#"
          className="shrink-0 font-mono text-sm tracking-[0.28em] text-paper-50"
        >
          JAZARI
        </a>

        <ul className="ml-auto hidden items-center gap-6 md:flex">
          {links.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className="text-sm text-paper-400 transition-colors hover:text-paper-50"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        <div className="ml-auto flex items-center gap-3 md:ml-0">
          <div className="flex max-w-[40vw] items-center gap-4 overflow-x-auto md:hidden">
            {links.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="shrink-0 text-xs text-paper-400"
              >
                {link.label}
              </a>
            ))}
          </div>

          <a
            href="#contact"
            className="shrink-0 rounded-sm bg-signal-hot px-3 py-2 font-mono text-xs tracking-wide text-ink-950 transition-opacity hover:opacity-90 md:px-4 md:text-sm"
          >
            Talk to the engineer
          </a>
        </div>
      </nav>
    </header>
  );
}
