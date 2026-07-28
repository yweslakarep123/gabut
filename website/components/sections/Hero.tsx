"use client";

import { motion, useReducedMotion } from "motion/react";
import { MeshCanvas } from "@/components/MeshCanvas";
import {
  revealAnimate,
  revealInitial,
  revealTransition,
} from "@/lib/motion";

export function Hero() {
  const reduced = useReducedMotion();

  const initial = reduced ? false : revealInitial;
  const animate = revealAnimate;

  return (
    <section className="relative flex min-h-screen items-center overflow-hidden">
      <MeshCanvas />

      <div className="relative z-10 mx-auto w-full max-w-6xl px-5 pb-20 pt-28 md:px-8 md:pb-24 md:pt-32">
        <motion.p
          initial={initial}
          animate={animate}
          transition={{ ...revealTransition, delay: 0.05 }}
          className="mb-5 font-mono text-[11px] tracking-[0.22em] text-paper-400 md:text-xs"
        >
          PHYSICS-VALIDATED AI FOR ROBOTICS HARDWARE
        </motion.p>

        <motion.h1
          initial={initial}
          animate={animate}
          transition={{ ...revealTransition, delay: 0.12 }}
          className="max-w-4xl text-[clamp(2rem,5vw,4.25rem)] font-medium leading-[1.08] tracking-tight text-paper-50"
        >
          We don&apos;t ship a prediction until it survives contact with the
          ground truth.
        </motion.h1>

        <motion.p
          initial={initial}
          animate={animate}
          transition={{ ...revealTransition, delay: 0.2 }}
          className="mt-6 max-w-2xl text-base leading-relaxed text-paper-400 md:text-lg"
        >
          A pipeline from raw CAD to trained surrogate model — geometry healing,
          validated finite-element simulation, and graph neural networks that
          predict structural and thermal behavior directly from geometry, in
          milliseconds.
        </motion.p>

        <motion.div
          initial={initial}
          animate={animate}
          transition={{ ...revealTransition, delay: 0.28 }}
          className="mt-10 flex flex-wrap items-center gap-6"
        >
          <a
            href="#case-study"
            className="inline-flex items-center rounded-sm bg-signal-hot px-5 py-3 font-mono text-sm tracking-wide text-ink-950 transition-opacity hover:opacity-90"
          >
            See the validation →
          </a>
          <a
            href="#pipeline"
            className="font-mono text-sm text-paper-400 underline-offset-4 transition-colors hover:text-paper-50 hover:underline"
          >
            Read the pipeline
          </a>
        </motion.div>
      </div>
    </section>
  );
}
