# Jazari Website — Cursor Build Brief

A "PhysicsX, but for robotics" site, built from your actual validated engineering work
(SimJEB surrogate model + UR5e digital-twin pipeline), not from invented company facts.

---

## 0. How to use this document

1. Read **Part A** and adjust anything that's wrong (name, tone, stack).
2. Save Part A as `AGENTS.md` at your project root — Cursor (and Claude Code, and most
   other agent tools) reads this automatically on every session, so you never have to
   re-paste context.
3. Open Cursor, start a fresh Composer/Agent chat, and run the prompts in **Part C** in
   order — one phase per prompt, committing to git between phases.
4. Use **Part B** as the copy/content source Cursor should pull from when it writes JSX.
5. Read **Part E** before you make this public.

---

## 1. What physicsx.ai actually does structurally (so we're translating the right thing)

I fetched the live site rather than going from memory. Structurally, it is **not** a
typical SaaS landing page — no pricing table, no feature-grid bento boxes, no gradient
blobs. It's closer to an editorial long-form page:

- Minimal top nav (logo + 5–6 text links + one persistent CTA), full-bleed background
  video behind the hero and behind most sections.
- A **numbered scroll sequence** (1 Mission → 2 Platform → 3 Impact/Industries →
  4 Team → 5 Newsroom), each section carrying one short eyebrow label, one bolded
  subhead, 2–3 sentences of real prose, and a single text-link CTA — not card grids.
- Credibility is carried by **specificity**, not adjectives: a named funding round, named
  industrial partners, two physical office addresses in the footer, a dated press feed.
- Restrained motion, restrained color, a lot of confidence in white space.

I can't reliably read exact hex values or font names off a text fetch, so I'm not going
to claim I've cloned their palette — Part A below is a **direction built for your
narrative** (physics validation, error percentages, closed-loop testing), not a copy of
theirs. The structural pattern (numbered editorial sections, credibility-by-specificity,
no fake feature grids) is what's worth borrowing; the words, wordmark, and video assets
are theirs and shouldn't appear on your site.

**The honest translation for you:** PhysicsX proves credibility with a $300M round and
named partners like Siemens and NVIDIA. You don't have that — but you have something
they'd respect just as much: a reaction-moment FEA check accurate to **-0.04%** against
hand calculation, a thermal solver validated to **0.03°C**, and a surrogate model that
beats a naive baseline on **20 out of 20** target quantities across three held-out
official test splits. The site should lead with *that*, not with startup theater.

---

## 2. Assumptions I'm making — correct these before you build

| Decision | Default I picked | Why |
|---|---|---|
| Name | **Jazari** | It's already your repo name. Ismail al-Jazari (12th c.) is the guy who literally invented programmable automata — it's a genuinely good, on-theme name, not a placeholder. **Check it isn't already trademarked in robotics/AI before you commit to it publicly.** |
| Framing | Ambitious technical portfolio, presented with startup-grade seriousness — not a fundraising pitch, not a plain résumé | Everything you've shared reads as one engineer's rigorous solo work. Overclaiming ("our team," "our customers") would be dishonest; underselling it as "my hobby project" would undersell genuinely rare, validated work. |
| Stack | Next.js 14+ (App Router), TypeScript, Tailwind CSS, Framer Motion | Modern default for this genre of site, and what Cursor scaffolds most reliably in 2026. Tell it otherwise if you already have a preferred stack. |
| Scope | Single-page scroll (like the reference) + one dedicated case-study page | Matches the reference pattern; a dedicated page is where the UR5e numbers get room to breathe. |

If any of these are wrong, just edit Part A before handing it to Cursor — everything
downstream reads from it.

---

# PART A — Master Context

**Save this whole section as `AGENTS.md` in your project root** (plain markdown, no
frontmatter needed — Cursor loads it every session automatically). If you'd rather use
Cursor's newer scoped-rules system, save the same content as
`.cursor/rules/project-context.mdc` with this frontmatter on top:

```yaml
---
description: Jazari site — brand, content, and design system context
alwaysApply: true
---
```

*(Cursor deprecated the old single `.cursorrules` file — it's silently ignored by Agent
mode now. `.mdc` files in `.cursor/rules/`, or a plain `AGENTS.md`, are the current
mechanism. `AGENTS.md` is simplest and also portable if you ever use Claude Code
alongside Cursor on the same repo.)*

```markdown
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
- Framer Motion for scroll reveals and number-counter animations
- Fonts: a geometric sans for display type (Geist Sans or Inter), a monospace for
  every number/metric/code-like element (JetBrains Mono or Geist Mono)
- Deploy target: Vercel
- No CMS needed at this size — content lives in typed data files under `/content`,
  not hardcoded in JSX, so copy can be edited without touching layout code

## Design system

Color (dark base, one hot accent, one cold accent — echoing a stress-field colormap,
which is thematically exact for this project, not decorative):

```ts
// tailwind.config.ts — extend.colors
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
(Framer Motion `whileInView`), 400–600ms duration, `ease: [0.16, 1, 0.3, 1]`. Animated
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
```

---

# PART B — Page content

## Nav
Logo/wordmark (text is fine — "JAZARI" in the monospace face, letter-spaced) · links:
**Approach · Pipeline · Case Study · Log · Contact** · persistent CTA button: **"Talk
to the engineer"** (mailto or contact anchor — singular, honestly, since it's one
person).

## Hero
- Eyebrow (monospace, small caps): `PHYSICS-VALIDATED AI FOR ROBOTICS HARDWARE`
- Headline: **"We don't ship a prediction until it survives contact with the ground
  truth."**
  *(Alt, if you want it shorter: "Every model here has been checked against real
  physics.")*
- Subhead: "A pipeline from raw CAD to trained surrogate model — geometry healing,
  validated finite-element simulation, and graph neural networks that predict
  structural and thermal behavior directly from geometry, in milliseconds."
- CTA: **"See the validation →"** (scrolls to Case Study) + secondary text link
  **"Read the pipeline"**
- Background: animated low-poly wireframe mesh on canvas/SVG that subtly deforms
  (cheap to build, thematically exact — it's literally what a tetrahedral mesh looks
  like — and avoids needing to produce or license video assets)

## §01 — Why closed-loop physics *(Mission slot)*
> Training loss is not validation. A model can fit its training distribution
> perfectly and still be wrong on the geometry an engineer actually cares about. Every
> component of this pipeline is checked against something independent of the model
> itself: an analytic solution, an official held-out benchmark split, a closed-form
> equation derived by hand. If a prediction can't be checked, it doesn't ship.

CTA: **"See the numbers"** → scrolls to proof strip.

**Proof strip directly under this section** (4 stat cards, monospace numbers, one
line of context each — pull straight from the Content Inventory above):
`20/20` targets beat baseline · `-0.04%` reaction-moment error · `0.03°C` thermal
solver error @ 1mm · `11/11` parametric samples, zero failures

## §02 — The pipeline *(Platform slot)*
Subhead: **"Five stages, each independently validated before the next one trusts it."**

Present as five short numbered sub-items (not a card grid — keep the numbered,
editorial feel):

1. **Geometry healing** — Raw STEP exports arrive non-manifold. Automated BRep repair
   (sewing, micro-void removal, ShapeFix) corrects them to watertight, genus-0 solids
   before anything downstream touches them.
2. **Validated meshing** — Tetrahedral volume meshes generated from the healed
   surface, checked for inverted elements and dihedral quality before being handed to
   a solver.
3. **Structural simulation** — CalculiX-based FEA, boundary conditions and load cases
   checked against hand-derivable reaction calculations before the stress field is
   trusted.
4. **Thermal simulation** — Steady-state heat transfer, validated against a
   closed-form analytic solution, with boundary-condition classification bugs caught
   by that validation loop rather than shipped silently.
5. **Surrogate model training** — A graph neural network trained directly on the
   tetrahedral mesh, evaluated on an official held-out benchmark split, not a
   self-selected one.

CTA: **"Walk through a real part →"** (to Case Study).

## §03 — Case study: a robot arm's upper-arm link *(Impact slot)*
Subhead: **"From a CAD file with no material data to a validated structural model."**

Narrative, in order, each line pairing what was done with the number that proves it:

- Started from a manufacturer's STEP export — non-manifold, no material assigned,
  not simulation-ready.
- Automated healing corrected it to a watertight, genus-0 solid.
- Generated a validated tetrahedral volume mesh: ~38,800 nodes, ~192,000 tets, zero
  inverted elements.
- Built a cantilever structural load case (self-weight + distal payload) and
  validated the reaction moment against hand calculation: **106.52 N·m analytic vs.
  106.48 N·m FEA — 0.04% error.**
- Generated 11 parametric geometry variants automatically, re-meshed and re-solved
  each one: **11/11 succeeded**, zero mesh failures.
- This is the exact dataset-generation loop a surrogate model needs to train on — not
  a one-off simulation, a repeatable pipeline.

CTA: **"Read the full case study →"** (dedicated `/case-study` page — give it room:
the healing-step Euler-number correction, the mesh quality histogram, the actual
stress field, the thermal validation detail, all of it. The homepage section is the
trailer; the page is the film.)

## §04 — About *(Team slot)*
Keep this short and honest — placeholder for you to personalize, don't let Cursor
invent biographical details:

> Built by [your name/handle], an engineer who kept hitting the same wall: simulation
> is too slow to explore a real design space, and most "AI for engineering" tooling
> skips the step where you check if the model is actually right. This is what
> checking looks like, done properly, on real geometry.

CTA: **"Get in touch"**.

## §05 — Field notes *(Newsroom slot — this is the one section that's genuinely free
to populate, because it's just your real build log)*
Subhead: **"Dated, in order, as it happened."**

Style exactly like a press feed — date, short title, one-line description, link (link
targets can be `#` or a future `/log/[slug]` page):

- `Heal + volume mesh, upper arm` — Automated healing reaches watertight/Euler-2;
  tetrahedral mesh generated with zero inverted elements.
- `Structural FEA validated` — Reaction-moment check passes at -0.04% error against
  analytic calculation.
- `Parametric batch, N=11` — Automated geometry-variant generation and re-solve
  pipeline: 11/11 succeeded.
- `Thermal solver validated` — Closed-form fin-equation comparison catches a
  boundary-condition bug; fixed error drops to 0.03°C.
- `Second link healed: shoulder housing` — Same automated pipeline applied to an
  independent part, confirming it's a pipeline and not a one-off script.

*(Add to this every time you actually finish something — it's the lowest-effort,
highest-authenticity section on the whole site, and it ages better than a static
"About" page.)*

## Footer
Contact CTA repeated · email/contact link · social links if you have them · small
legal line ("© Jazari, [year]. Not affiliated with Universal Robots or the SimJEB
dataset authors.") — see Part E on why that disclaimer matters.

---

# PART C — Phased Cursor prompts

Run these **in order**, one per Composer/Agent turn, committing to git after each one
so you can roll back a bad generation without losing the rest.

### Phase 0 — Scaffold
```
Scaffold a new Next.js 14 App Router project in TypeScript with Tailwind CSS and
Framer Motion installed. Set up the color tokens, fonts (Geist Sans for display,
Geist Mono for numeric/monospace text — use next/font/google or next/font/local),
and dark mode as the only mode (no toggle, ink-950 is the base background) exactly as
specified in AGENTS.md at the project root. Create the folder structure:

/app
  /case-study/page.tsx
  layout.tsx
  page.tsx
/components
  /sections
/content
  metrics.ts       (typed export of every stat in the Content Inventory)
  pipeline.ts       (typed export of the 5 pipeline stages)
  log.ts            (typed export of the field-notes entries)
/lib

Do not write any page content yet — just the scaffold, the design tokens, and empty
typed content files with the real data from AGENTS.md filled in.
```

### Phase 1 — Nav + Hero
```
Build the Nav and Hero components described in AGENTS.md / the site content brief.
Nav: fixed/sticky, transparent over the hero, solid background after scrolling past
it. Hero: full viewport height, headline + subhead + two CTAs, and a canvas-based
low-poly wireframe mesh background that subtly deforms over time (simple vertex
sine-wave displacement is fine — it should read as "engineering mesh," not decorative
particles). Keep it performant: pause the canvas animation when the tab is not
visible.
```

### Phase 2 — Mission + Pipeline sections
```
Build §01 (Why closed-loop physics) including the four-stat proof strip, and §02
(The Pipeline, five numbered stages) using the copy in the content brief and the
typed data in /content. Each section gets a small sticky numbered label (01, 02) in
the monospace face. Animate each stat and each pipeline stage in with a Framer Motion
whileInView fade + slide-up, staggered by ~80ms per item. The proof-strip numbers
should count up from 0 on first view using a simple animated-counter hook.
```

### Phase 3 — Case study section + dedicated page
```
Build §03 on the home page (the condensed case-study narrative) and a full
/case-study page that expands on it: the healing step with before/after Euler-number
numbers, the mesh-quality numbers, the reaction-moment validation table (analytic vs
FEA vs error%), and the thermal validation table (naive vs fixed vs error, at 2mm and
1mm mesh). Use simple styled <table> or definition-list elements for the
before/after/error comparisons — data density matters more than decoration here.
```

### Phase 4 — About, Field Notes, Footer
```
Build §04 (About) as a short single-paragraph section, §05 (Field Notes) as a
dated list styled like a press feed pulling from /content/log.ts, and the Footer with
contact CTA, links, and the legal line. Keep About genuinely short — it should not
try to fill space.
```

### Phase 5 — Polish pass
```
Do a full responsive and accessibility pass: verify every section on mobile widths
(375px) and tablet (768px), check color contrast on paper-400 against ink-950 meets
WCAG AA for body text, add prefers-reduced-motion handling that disables the canvas
mesh animation and the scroll reveals, add proper semantic headings (one h1 in the
hero, h2 per numbered section), and run a Lighthouse-style review of image/font
loading. Optimize the canvas mesh animation to use requestAnimationFrame correctly
and not run when off-screen.
```

*(Optional stretch, only if you want to go further: a Phase 6 adding a React Three
Fiber panel that actually renders the healed UR5e mesh with a von Mises stress
colormap, using the real geometry.npz data converted to a web-friendly format. This
is a genuinely strong differentiator if you have time — nobody else's landing page
has a real, interactive stress field on it — but it's real 3D-web work, budget a
separate session for it.)*

---

# PART D — Cursor workflow notes

- Put `AGENTS.md` at the repo root **before** Phase 0. Cursor's Agent mode reads it
  automatically every session — you should never have to re-paste brand/content
  context into a prompt.
- If you'd rather use Cursor's scoped rules system instead of a flat `AGENTS.md`, use
  `.cursor/rules/project-context.mdc` with `alwaysApply: true` in the frontmatter (see
  Part A). The old single `.cursorrules` file is deprecated and is silently ignored
  by Agent mode as of 2026 — don't use it.
- Commit after every phase in Part C. If a generation goes sideways (wrong direction
  on the design system, over-animated, whatever), it's much cheaper to `git checkout`
  and re-run the phase prompt with a correction than to hand-fix generated JSX.
- Use Cmd+K inline edit for small corrections within a phase ("make this stat card
  narrower," "the hero headline is too large on mobile") rather than re-running the
  whole phase prompt.

---

# PART E — Before you publish this publicly

- **Don't ship PhysicsX's actual logo, wordmark, video assets, or copy.** Everything
  in Part B is written fresh for you — the only thing borrowed from the reference is
  the structural pattern (numbered editorial sections instead of a feature grid),
  which is a genre convention, not their IP.
- **Check "Jazari" isn't already in use** by another robotics/AI company before you
  commit to it as a public brand — I haven't done a trademark search, you should.
- **The UR5e geometry is Universal Robots' Graphical Documentation**, licensed for
  non-commercial research use per their terms (this is already documented in your
  own README). If this site is a personal/portfolio showcase, that's almost
  certainly fine; if it starts looking like a commercial product pitch built on that
  geometry, re-read UR's terms and consider swapping in geometry you have clearer
  rights to, or keep the case study framed explicitly as non-commercial research.
- **The SimJEB dataset is ODC-BY licensed** — keep the attribution to Whalen, Beyene,
  and Mueller (MIT) somewhere on the case-study page, same as your README already
  does.
- Don't let any generated copy imply team size, funding, or customers you don't have.
  The "About" section in Part B is deliberately singular and honest — keep it that
  way even if Cursor tries to pluralize it into "our team."
