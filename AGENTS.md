# Jazari — Project Context for AI Agents

## What this company is
Jazari builds physics-validated AI for robotics hardware: a pipeline that takes raw
CAD (STEP files) through automated geometry healing, FEA-grade tetrahedral meshing,
closed-loop structural and thermal validation, and trained ML surrogate models that
predict stress and displacement fields directly from geometry — in milliseconds
instead of a multi-hour solver run.

The differentiator is not "AI for engineering" in the abstract. It's that every claim
is checked against ground truth before it ships: surrogate predictions are benchmarked
against an official, held-out test split (not a self-selected one); FEA setups are
validated against hand-derivable analytic solutions before being trusted on real
geometry; the thermal solver is validated against a closed-form fin equation.

## Audience
Robotics hardware engineers and technical leads evaluating whether ML surrogates can
shortcut their structural/thermal iteration loop. They are skeptical of AI-for-
engineering hype and will bounce off vague marketing language. Every claim on this
site should be a specific, checkable number, not an adjective.

## Tone rules
- Evidence before adjectives. "20/20 targets beat the naive baseline" beats "our model
  is highly accurate."
- Show the validation step, not just the result. Say what the ground truth was and
  what the error against it was.
- No unearned company language: no "our team" (it's one person unless stated
  otherwise), no customer logos, no pricing, no "trusted by."
- Confident, plain, technical. Short sentences. No exclamation points.

## Tech stack
- Next.js 14+, App Router, TypeScript strict mode
- Tailwind CSS for styling
- Motion (`motion/react`) for scroll reveals and number-counter animations
- Fonts: a geometric sans for display type (Geist Sans or Inter), a monospace for
  every number/metric/code-like element (JetBrains Mono or Geist Mono)
- Deploy target: Vercel
- No CMS needed at this size — content lives in typed data files under `/content`,
  not hardcoded in JSX, so copy can be edited without touching layout code
- Site lives in `website/` (monorepo alongside the Python pipeline)

## Design system

Color (dark base, one hot accent, one cold accent — echoing a stress-field colormap,
which is thematically exact for this project, not decorative):

```ts
// Prefer Tailwind v4 @theme in globals.css; fall back to tailwind.config.ts if v3
colors: {
  ink: {
    950: '#08090B',   // page background
    900: '#0F1113',   // section/card background
    800: '#1A1D21',   // borders, hairlines
  },
  paper: {
    50: '#F5F5F3',    // primary text
    400: '#9A9CA3',   // secondary text
  },
  signal: {
    hot: '#FF6A3D',    // primary accent — stress-field "high" color, used for CTAs and key numbers
    cold: '#3DD6FF',   // secondary accent — used sparingly, for contrast/annotation only
  },
}
```

Typography: large display sizes for section subheads (clamp(2rem, 5vw, 4.5rem)),
generous line-height on body copy, every metric/number rendered in the monospace face
regardless of where it appears in the page.

Motion: restrained and precise, not bouncy. Fade + 12px slide-up on scroll-into-view
(`whileInView`), 400–600ms duration, `ease: [0.16, 1, 0.3, 1]`. Animated
number counters count up once, on first view, never loop. No parallax gimmicks.

Layout: single column, generous vertical rhythm (each major section gets its own
100vh-ish breathing room on desktop), 1px hairline dividers between sections instead
of drop shadows or heavy borders, numbered section labels (01, 02, 03…) in the
monospace face as a small fixed/sticky element per section — this is the one direct
structural borrow from the reference genre and it's doing real work: it signals "this
is a sequence, read it in order," which suits a pipeline narrative.

Explicitly avoid: purple-to-blue gradient blobs, stock "AI brain/neural network"
imagery, glassmorphism cards, bouncy spring easing, generic SaaS bento grids, pricing
tables, testimonial carousels, fake logo walls.

## Content inventory — real, validated numbers only

Use these numbers exactly. Do not round up, do not invent additional metrics.

**Surrogate model (MeshGraphNet-style GNN on SimJEB benchmark, 381 real bracket
designs, official 3-way train/test splits):**
- Beats a naive polynomial baseline on **20 of 20** target quantities (5 fields ×
  4 load cases), on all 3 official held-out test splits
- Error is **7–33% lower** than the naive baseline depending on target
- Displacement fields (x/y/z/magnitude) and von Mises stress, per load case, all
  predicted from geometry alone

**Geometry healing (STEP → watertight solid):**
- Raw STEP imports arrive non-manifold; automated healing (BRep ShapeFix + sewing +
  micro-void removal) corrects them to watertight, genus-0 solids (Euler number = 2)
- Demonstrated on two independent robot-arm links (upper arm, shoulder housing)

**Structural FEA validation (CalculiX):**
- Reaction-moment check: analytic hand calculation **106.52 N·m** vs. FEA result
  **106.48 N·m** — **-0.04% error**
- Full cantilever load case (gravity + distal payload) validated before being trusted
  on any downstream geometry variant

**Thermal solver validation:**
- Closed-form analytic fin solution vs. numerical solver
- Naive setup: 0.30°C max error
- After fixing a boundary-condition classification bug: **0.03°C max error at 1mm
  mesh resolution**

**Automated dataset generation:**
- Parametric batch: 11 geometry variants generated, meshed, and solved automatically
- **11 of 11 succeeded** — zero mesh failures, zero reaction-check failures

**Volume mesh quality (example: robot upper-arm link):**
- ~38,800 nodes / ~192,000 tetrahedra
- Zero inverted elements
- Watertight, Euler number = 2

## What this site is not
Not a SaaS product with a signup flow. Not a funded startup (unless/until that's
literally true — don't imply funding, headcount, or customers that don't exist). Not
a resume. It's a technical showcase that argues, with numbers, that this pipeline
works.
