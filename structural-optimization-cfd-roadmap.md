# Structural Optimization & CFD — Where This Goes Next

Two different questions ("efficient in terms of CFD or structural integrity")
get two different answers, because they start from very different places.

---

## 1. Structural: what's actually true today

| Component | Status | Caveat |
|---|---|---|
| MeshGraphNet surrogate (`model.py`) | Trained + validated, 20/20 targets beat naive baseline | **Trained on SimJEB bracket shapes only.** Never seen a robot-link cross-section. Not safe to point at NAUO3/NAUO2 without retraining. |
| Parametric geometry deformation | Working, validated (`batch_nauo3_neck_scale.py`) | One scalar parameter (uniform neck radial scale `s`), fixed 11-point grid |
| Structural FEA per sample | Validated (reaction check -0.04% vs analytic) | Ground-truth CalculiX per evaluation — slow, not the surrogate |
| Actual optimization/search | **Does not exist** | This is the real gap. Everything above is a fixed sweep, not a search. |

So: "can it find an efficient design" — the honest answer is *the plumbing for
this exists and is validated, but nobody has wired an optimizer to it yet, and
the trained model can't currently be trusted on this geometry family.*

---

## 2. Phase 1 (buildable now, no retraining needed): constrained root-find

**Question this answers:** what is the thinnest neck (smallest `s`) that does
**not** exceed the peak stress already present at your `s=1.0` baseline?

This reframes the existing grid sweep as a proper 1D constrained problem
instead of a chart to eyeball:

```
find s* = min { s : sigma_max(s) <= sigma_max(s=1.0) }
```

- Reuses `run_one_sample()` unchanged as the objective evaluator — no new FEA
  code, no new mesh-gate logic, no loosened thresholds.
- Solved with `scipy.optimize.brentq` (bisection on a sign change), not a
  general-purpose optimizer — this is a root-finding problem, not a
  minimization problem, and root-finding needs far fewer evaluations.
- **Honest caveat:** your own batch data already suggests `sigma_max`
  decreases monotonically as `s` increases (thinner → higher stress). If that
  holds all the way down, there may be *no* feasible root below `s=1.0` — i.e.
  no free lightweighting available along this one axis. That's a real,
  useful answer ("no free lunch here"), not a failed search. The script
  needs to say so explicitly rather than forcing a root-find with no root.
- Every evaluation gets logged in full (not just the final answer) — this log
  is the seed corpus for Phase 3.

See the Cursor prompt at the end of this doc for the exact spec.

---

## 3. Phase 2 (later): richer shape parameterization

A single scalar (uniform radial scale) can only tell you "thicker or
thinner, everywhere in the band, together." If Phase 1 comes back with "no
feasible thinning below baseline," that's not necessarily the final word —
it just means *uniform* scaling has no slack. Real efficiency gains in
structural design almost always come from **non-uniform** material
distribution: more where stress is concentrated, less where it isn't.

Next parameterization to add, in order of implementation cost:
1. Independent inner/outer taper (currently coupled to one `s`)
2. Fillet radius at the shoulder transition (where the s>1 hotspot already
   showed up in your falloff diagnostics — this is very likely where real
   savings are, if any exist)
3. A smooth radial profile along the band (a small number of control points
   instead of one scalar)

Each added dimension multiplies the FEA cost of a grid sweep, which is
exactly why Phase 3 (surrogate-in-the-loop) matters — a 3-5 parameter grid at
11 points/axis is 1,331-161,051 FEA runs; a trained surrogate evaluates in
milliseconds.

---

## 4. Phase 3 (later): train a surrogate on robot-link geometry, then optimize with it

1. **Generate enough labeled samples.** Phase 1 + 2's evaluation logs are the
   start; you'll want on the order of dozens to low hundreds of geometry
   variants with full field data (not just global max stress — the surrogate
   needs the same per-node target structure as SimJEB) to train on.
2. **Reuse `MeshGraphNet` / `build_graph()` from `model.py` unchanged.** The
   architecture doesn't care what shape family it's trained on.
3. **Keep the same held-out-split discipline you used for SimJEB.** Don't
   report performance on samples the model trained on. This is the one
   practice from the SimJEB work that matters most to carry forward — it's
   the difference between a model you can trust and one that just fits its
   own training sweep.
4. **Swap the optimizer's objective from `run_one_sample()` (slow, real FEA)
   to the trained surrogate (fast, milliseconds).** This is what makes
   multi-parameter search actually tractable, and — since the model is
   plain PyTorch (`torch_geometric` message passing) — it's differentiable.
   You can backprop through the trained surrogate to get gradients of
   predicted stress with respect to node position directly, which opens the
   door to gradient-based shape optimization, not just black-box search.
5. **Mandatory closing step: verify the optimizer's top candidates against
   real CalculiX before trusting them.** An optimizer searching against a
   surrogate will happily exploit any blind spot or artifact in that
   surrogate's predictions. The FEA validation loop you already built is
   exactly the check that catches this — use it as the final gate, every
   time, no exceptions.

---

## 5. CFD, if and when you actually want it

This is a separate project, not an extension — flagging what's genuinely
different so it doesn't get started opportunistically alongside 1-3 above:

- **Different mesh entirely.** Your current pipeline meshes the *solid's own
  volume* (tetrahedra filling the part). CFD needs a mesh of the *fluid
  domain around* the part — a far-field box/sphere much larger than the
  geometry, with boundary-layer refinement hugging the surface. Your healed,
  watertight STL is reusable as the wall boundary either way — that part of
  the pipeline isn't wasted. The volume mesher (`mesh_ur_volume.py`) is not
  reusable; you'd need a different tool (e.g. OpenFOAM's `snappyHexMesh`, or
  Gmsh building an external domain around the same STL).
- **Different solver.** OpenFOAM is the standard free option and the natural
  parallel to CalculiX here — open, scriptable, well-documented, handles
  conjugate heat transfer (which is what you actually want, given the
  connection below) as well as pure aerodynamics.
- **Recommended first target: convective-h estimation, not drag.** A UR5e
  sitting or moving at normal speed isn't drag-limited — aerodynamic
  optimization wouldn't answer a question anyone has. Estimating a
  geometry-specific convective coefficient to replace the assumed
  h=10 W/m²K in your existing thermal FEA *does* answer a real question, and
  plugs directly into work you've already validated.
- **Validation case, same discipline as the fin:** cross-flow over a
  cylinder has well-known closed-form Nusselt-number correlations for `h` as
  a function of Reynolds number. Validate the CFD setup against that before
  trusting it on real geometry — exactly the same "closed-form solution
  first" step that caught your thermal BC-leakage bug.

---

## 6. Suggested order

Phase 1 now → if it shows real design freedom, Phase 2 → once you have
enough samples, Phase 3 → CFD as its own separate track whenever thermal
accuracy (not structural work) becomes the actual bottleneck.

---

## Cursor prompt — Phase 1 (paste as-is)

```
You're extending an existing, validated FEA batch pipeline
(scripts/batch_nauo3_neck_scale.py) into an actual constrained optimization,
not just a fixed grid sweep. Read that file fully before writing anything —
reuse its functions, don't reimplement them.

## Goal
Answer: what is the thinnest neck (smallest `s`, the existing radial-scale
parameter) that does NOT exceed the peak von Mises stress already present at
the baseline `s=1.0` design? This is a 1D root-finding problem, not a general
optimizer.

## What to reuse, unchanged
- `run_one_sample(s, points0, tets, node_surf, surf_faces, dens0_t_mm3,
  axis_point, axis_u, ends0_prox_y, reuse_fea=False)` as the objective
  evaluator — do not duplicate the deform/mesh-gate/FEA/reaction-check logic
  inside it.
- The existing constants: MAX_ABS_DV_FRAC, MAX_INVERTED, REACTION_ERR_PCT,
  MIN_SPC_GAP_MM, Y0_NECK, L_BAND. Do not redefine or loosen any of them.
- BASELINE_SIGMA_MAX_MPA as the default stress budget.

## New script: scripts/optimize_nauo3_neck_scale.py
1. Define g(s) = run_one_sample(s, ...)['sigma_max_MPa'] - sigma_allow, where
   sigma_allow defaults to BASELINE_SIGMA_MAX_MPA and is overridable via
   --sigma-allow. If run_one_sample returns a non-"ok" status (mesh gate, SPC
   clearance, or reaction check failed), treat that evaluation as infeasible
   — do not coerce it to a numeric stress value; handle it as a sentinel the
   root-finder logic reacts to explicitly (skip and narrow the bracket, or
   raise with a clear message about which gate failed).
2. Before running any root-finder: check reports/ur5e_nauo3_neck_batch/ for
   already-computed results at s=0.90 and s=1.00 and reuse them (pass
   reuse_fea=True / read the cached sample_result.json) instead of re-running
   CalculiX you already have. Print the sign of g at both endpoints.
   - If g(1.00) is not ~0, stop and report the discrepancy before proceeding
     (it should match BASELINE_SIGMA_MAX_MPA by construction).
   - If g(0.90) <= 0, there may be feasible room below the previously tested
     range — cautiously widen the lower bound (e.g. to 0.80) and say so.
   - If g(0.90) > 0 and the sign pattern across the existing 11-point batch
     looks monotonically increasing as s decreases, state explicitly that
     there is likely no feasible thinning below s=1.0 within this
     one-parameter family, and stop rather than forcing brentq on a bracket
     with no sign change.
3. If a sign change exists in range, solve g(s) = 0 with
   scipy.optimize.brentq to a tolerance of about 1e-3 in s. Log every
   (s, sigma_max_MPa, status, wall_time_s) evaluation brentq performs — not
   just the final root — to
   reports/ur5e_nauo3_neck_optimize/evaluation_log.jsonl as it happens. This
   log doubles as seed training data for a future surrogate, so it needs
   full fidelity.
4. Write reports/ur5e_nauo3_neck_optimize/result.json: the solved s* (or an
   explicit "no feasible thinning found within tested range" verdict), its
   sigma_max_MPa, the sigma_allow used, the number of FEA evaluations spent
   (print this against 11 — the size of the original grid — as the headline
   comparison), and the mesh-gate/reaction-check status of the final
   candidate, so nothing here silently trusts an unvalidated design.

## Explicitly do not do in this pass
- Do not touch the MeshGraphNet model or training code.
- Do not add a second shape parameter (fillet, taper, non-uniform thickness)
  — that's Phase 2, after this 1D result is in hand.
- Do not build anything CFD-related.
```
