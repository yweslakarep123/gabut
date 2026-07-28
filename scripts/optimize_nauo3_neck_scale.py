#!/usr/bin/env python3
"""1D constrained root-find on NAUO3 neck radial scale s.

Question: thinnest s (smallest radial scale) such that
  sigma_max(s) <= sigma_max(s=1.0)   [= BASELINE_SIGMA_MAX_MPA by default]

Uses run_one_sample() from batch_nauo3_neck_scale unchanged as the evaluator.
Does not reimplement deform / mesh-gate / FEA / reaction-check logic.

  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/optimize_nauo3_neck_scale.py
  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/optimize_nauo3_neck_scale.py --sigma-allow 1.645735221879838
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from batch_nauo3_neck_scale import (  # noqa: E402
    BASELINE_SIGMA_MAX_MPA,
    L_BAND,
    MAX_ABS_DV_FRAC,
    MAX_INVERTED,
    MESH_BASE,
    MIN_SPC_GAP_MM,
    OUT_ROOT as BATCH_OUT_ROOT,
    REACTION_ERR_PCT,
    Y0_NECK,
    _boundary_faces,
    run_one_sample,
    spc_band_clearance_report,
)
from fea_nauo3_cantilever import (  # noqa: E402
    M_SELF,
    _tet_volume_sign,
    select_end_faces,
)

OPT_OUT = ROOT / "reports" / "ur5e_nauo3_neck_optimize"
EVAL_LOG = OPT_OUT / "evaluation_log.jsonl"
RESULT_JSON = OPT_OUT / "result.json"

GRID_N_COMPARE = 11
G1_REL_TOL = 0.01  # |g(1.0)| / sigma_allow must be below this to proceed
S_LOW_DEFAULT = 0.90
S_HIGH = 1.00
S_LOW_WIDEN = 0.80
BRENTQ_XTOL = 1e-3
# Dense preflight on the existing batch grid in [0.90, 1.00] — all on disk.
PREFLIGHT_S_GRID = (0.90, 0.92, 0.94, 0.96, 0.98, 1.00)


class InfeasibleEvaluation(RuntimeError):
    """run_one_sample returned a non-ok status (gate / SPC / reaction / ccx)."""

    def __init__(self, s: float, sample: dict):
        self.s = s
        self.sample = sample
        status = sample.get("status")
        reasons = sample.get("fail_reasons") or []
        super().__init__(
            f"infeasible at s={s:.6f}: status={status} fail_reasons={reasons}"
        )


def _load_cached_sample(s: float) -> dict | None:
    tag = f"s_{s:.2f}"
    path = BATCH_OUT_ROOT / tag / "sample_result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _append_eval_log(row: dict) -> None:
    OPT_OUT.mkdir(parents=True, exist_ok=True)
    with EVAL_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _setup_mesh() -> dict[str, Any]:
    g = np.load(MESH_BASE)
    points0 = np.asarray(g["vol_points"], dtype=np.float64).copy()
    tets = np.asarray(g["vol_tets"], dtype=np.int32)
    node_surf = np.asarray(g["node_surf"], dtype=bool)
    surf_faces = np.asarray(g["surf_faces"], dtype=np.int32)

    tet_ids, face_ids, face_nodes = _boundary_faces(tets)
    ends0 = select_end_faces(points0, node_surf, tet_ids, face_ids, face_nodes)
    clear0 = spc_band_clearance_report(points0, ends0, L=L_BAND)
    if not clear0["safe_clearance"]:
        raise RuntimeError(
            "STOP: baseline SPC/band clearance failed — "
            f"gap_prox={clear0['gap_prox_max_y_to_band_lo_mm']}"
        )

    vols0 = np.abs(_tet_volume_sign(points0, tets))
    V0 = float(vols0.sum())
    dens_kg_m3 = M_SELF / (V0 * 1e-9)
    dens0_t_mm3 = dens_kg_m3 * 1e-12
    axis_point = points0.mean(0)
    axis_u = ends0["u"]

    return {
        "points0": points0,
        "tets": tets,
        "node_surf": node_surf,
        "surf_faces": surf_faces,
        "dens0_t_mm3": dens0_t_mm3,
        "axis_point": axis_point,
        "axis_u": axis_u,
        "clear0": clear0,
        "V0": V0,
        "dens_kg_m3": dens_kg_m3,
    }


def evaluate(
    s: float,
    ctx: dict[str, Any],
    *,
    sigma_allow: float,
    prefer_cache: bool,
    counters: dict[str, int],
) -> tuple[float, dict]:
    """Return (g(s), sample). Raises InfeasibleEvaluation on non-ok status."""
    t0 = time.perf_counter()
    cached = _load_cached_sample(s) if prefer_cache else None
    reused_json = False
    if cached is not None and cached.get("status") == "ok":
        sample = cached
        reused_json = True
        counters["n_cache_json"] += 1
    else:
        # Prefer CalculiX reuse when FRD already exists under the batch tag.
        sample = run_one_sample(
            s,
            ctx["points0"],
            ctx["tets"],
            ctx["node_surf"],
            ctx["surf_faces"],
            ctx["dens0_t_mm3"],
            ctx["axis_point"],
            ctx["axis_u"],
            ctx["clear0"],
            reuse_fea=True,
        )
        if sample.get("status") in ("mesh_gate_fail", "spc_band_overlap"):
            counters["n_gate_fail_before_fea"] += 1
        elif sample.get("ccx_reused") is True:
            counters["n_fea_reused"] += 1
        elif sample.get("ccx_reused") is False:
            counters["n_fea_new"] += 1

    wall = time.perf_counter() - t0
    counters["n_eval"] += 1

    status = sample.get("status")
    sigma = sample.get("sigma_max_MPa")
    ccx_reused = None if reused_json else sample.get("ccx_reused")
    row = {
        "s": float(s),
        "sigma_max_MPa": float(sigma) if sigma is not None else None,
        "sigma_allow_MPa": float(sigma_allow),
        "g": (float(sigma) - float(sigma_allow)) if sigma is not None else None,
        "status": status,
        "wall_time_s": wall,
        "reused_sample_result_json": reused_json,
        "ccx_reused": ccx_reused,
        "fail_reasons": sample.get("fail_reasons") or [],
        "mesh_gate_pass": (sample.get("mesh_gate") or {}).get("pass"),
        "reaction_pass": (sample.get("reaction") or {}).get("pass"),
        "tag": sample.get("tag"),
    }
    _append_eval_log(row)
    print(
        f"  eval s={s:.6f}  status={status}  "
        f"sigma={sigma}  g={row['g']}  "
        f"wall={wall:.2f}s  cache_json={reused_json}  "
        f"ccx_reused={ccx_reused}"
    )

    if status != "ok":
        raise InfeasibleEvaluation(s, sample)
    assert sigma is not None
    g = float(sigma) - float(sigma_allow)
    return g, sample


def _write_result(payload: dict) -> None:
    OPT_OUT.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {RESULT_JSON}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Find thinnest NAUO3 neck scale s with sigma_max(s) <= sigma_allow "
            "(default: baseline s=1.0 peak stress)."
        )
    )
    ap.add_argument(
        "--sigma-allow",
        type=float,
        default=BASELINE_SIGMA_MAX_MPA,
        help=f"Stress budget in MPa (default: BASELINE_SIGMA_MAX_MPA={BASELINE_SIGMA_MAX_MPA})",
    )
    ap.add_argument(
        "--s-low",
        type=float,
        default=S_LOW_DEFAULT,
        help=f"Initial lower bracket endpoint (default: {S_LOW_DEFAULT})",
    )
    ap.add_argument(
        "--s-high",
        type=float,
        default=S_HIGH,
        help=f"Upper bracket endpoint / baseline (default: {S_HIGH})",
    )
    ap.add_argument(
        "--xtol",
        type=float,
        default=BRENTQ_XTOL,
        help=f"brentq absolute tolerance in s (default: {BRENTQ_XTOL})",
    )
    ap.add_argument(
        "--force-brentq",
        action="store_true",
        help="Run brentq even if preflight says no sign change (debug only)",
    )
    args = ap.parse_args()

    sigma_allow = float(args.sigma_allow)
    s_low = float(args.s_low)
    s_high = float(args.s_high)

    OPT_OUT.mkdir(parents=True, exist_ok=True)
    if EVAL_LOG.exists():
        EVAL_LOG.unlink()

    print("=== NAUO3 neck-scale constrained root-find ===")
    print(f"sigma_allow = {sigma_allow} MPa  (BASELINE={BASELINE_SIGMA_MAX_MPA})")
    print(
        f"gates unchanged: MAX_ABS_DV_FRAC={MAX_ABS_DV_FRAC}  "
        f"MAX_INVERTED={MAX_INVERTED}  REACTION_ERR_PCT={REACTION_ERR_PCT}  "
        f"MIN_SPC_GAP_MM={MIN_SPC_GAP_MM}  Y0_NECK={Y0_NECK}  L_BAND={L_BAND}"
    )
    print(f"eval log → {EVAL_LOG}")

    ctx = _setup_mesh()
    print(
        f"baseline V={ctx['V0']:.1f} mm3  dens={ctx['dens_kg_m3']:.1f} kg/m3  "
        f"axis from end selection + volume mean"
    )

    counters = {
        "n_eval": 0,
        "n_cache_json": 0,
        "n_fea_reused": 0,
        "n_fea_new": 0,
        "n_gate_fail_before_fea": 0,
    }

    def _endpoint_payload(sample: dict, g_val: float) -> dict:
        return {
            "s": float(sample["s"]),
            "sigma_max_MPa": sample.get("sigma_max_MPa"),
            "g": g_val,
            "status": sample.get("status"),
            "mesh_gate_pass": (sample.get("mesh_gate") or {}).get("pass"),
            "reaction_pass": (sample.get("reaction") or {}).get("pass"),
        }

    # --- Preflight: all 6 on-disk batch points in [0.90, 1.00] ---
    # Order: endpoints first (baseline check), then interior (0.92..0.98),
    # so evaluation_log keeps endpoint rows then the four mid-grid rows.
    print(
        f"\n--- preflight dense grid "
        f"{list(PREFLIGHT_S_GRID)} (cached sample_result.json only) ---"
    )
    preflight_points: list[dict[str, Any]] = []
    try:
        # Endpoints first
        for s in (s_low, s_high):
            g_s, sample = evaluate(
                s, ctx, sigma_allow=sigma_allow, prefer_cache=True, counters=counters
            )
            preflight_points.append(
                {
                    "s": float(s),
                    "g": g_s,
                    "sigma_max_MPa": sample.get("sigma_max_MPa"),
                    "sample": sample,
                    "sign": int(np.sign(g_s)) if g_s != 0 else 0,
                }
            )
        # Interior batch points — close the non-monotonic-dip hole
        interior = [
            s
            for s in PREFLIGHT_S_GRID
            if abs(s - s_low) > 1e-12 and abs(s - s_high) > 1e-12
        ]
        print(f"\n--- preflight interior points {interior} ---")
        for s in interior:
            g_s, sample = evaluate(
                s, ctx, sigma_allow=sigma_allow, prefer_cache=True, counters=counters
            )
            preflight_points.append(
                {
                    "s": float(s),
                    "g": g_s,
                    "sigma_max_MPa": sample.get("sigma_max_MPa"),
                    "sample": sample,
                    "sign": int(np.sign(g_s)) if g_s != 0 else 0,
                }
            )
    except InfeasibleEvaluation as e:
        _write_result(
            {
                "verdict": "preflight_infeasible",
                "message": str(e),
                "sigma_allow_MPa": sigma_allow,
                "s_star": None,
                "sigma_max_MPa": None,
                "n_fea_evaluations_spent": counters["n_fea_new"],
                "n_evaluations_total": counters["n_eval"],
                "n_grid_compare": GRID_N_COMPARE,
                "counters": counters,
                "failed_sample": {
                    "s": e.s,
                    "status": e.sample.get("status"),
                    "fail_reasons": e.sample.get("fail_reasons"),
                },
            }
        )
        print(f"STOP: {e}")
        return 2

    by_s = {p["s"]: p for p in preflight_points}
    sample_low = by_s[float(s_low)]["sample"]
    sample_high = by_s[float(s_high)]["sample"]
    g_low = by_s[float(s_low)]["g"]
    g_high = by_s[float(s_high)]["g"]

    preflight_sorted = sorted(preflight_points, key=lambda p: p["s"])
    print("\n--- preflight g(s) at all measured points (ascending s) ---")
    for p in preflight_sorted:
        print(
            f"  s={p['s']:.2f}  sigma={p['sigma_max_MPa']:.6f}  "
            f"g={p['g']:+.6e}  sign={p['sign']}"
        )

    # g(1.0) must be ~0 when sigma_allow is the baseline budget
    g1_rel = abs(g_high) / max(abs(sigma_allow), 1e-30)
    if g1_rel > G1_REL_TOL:
        msg = (
            f"g({s_high:.2f}) is not ~0 (g={g_high}, |g|/sigma_allow={g1_rel:.4f} "
            f"> {G1_REL_TOL}). Cached sigma_max={sample_high.get('sigma_max_MPa')} "
            f"vs sigma_allow={sigma_allow}. Refusing to proceed — check that "
            f"sigma_allow matches the s=1.0 design (default BASELINE_SIGMA_MAX_MPA)."
        )
        _write_result(
            {
                "verdict": "baseline_mismatch",
                "message": msg,
                "sigma_allow_MPa": sigma_allow,
                "g_at_s_high": g_high,
                "g_rel": g1_rel,
                "sample_high": {
                    "s": sample_high.get("s"),
                    "sigma_max_MPa": sample_high.get("sigma_max_MPa"),
                    "status": sample_high.get("status"),
                },
                "preflight_points": [
                    {
                        "s": p["s"],
                        "sigma_max_MPa": p["sigma_max_MPa"],
                        "g": p["g"],
                        "sign": p["sign"],
                    }
                    for p in preflight_sorted
                ],
                "s_star": None,
                "n_fea_evaluations_spent": counters["n_fea_new"],
                "n_evaluations_total": counters["n_eval"],
                "n_grid_compare": GRID_N_COMPARE,
                "counters": counters,
            }
        )
        print(f"STOP: {msg}")
        return 2

    # Local feasible dip: g < 0 at any s < baseline, even if endpoints look bad.
    feasible_interior = [
        p for p in preflight_sorted if p["s"] < s_high - 1e-12 and p["g"] < 0.0
    ]
    if feasible_interior and not args.force_brentq:
        best = min(feasible_interior, key=lambda p: p["s"])
        s_vals = ", ".join(f"{p['s']:.2f}" for p in preflight_sorted)
        g_vals = ", ".join(f"g({p['s']:.2f})={p['g']:+.6e}" for p in preflight_sorted)
        verdict = "feasible_thinning_local_dip"
        msg = (
            f"Checked {len(preflight_sorted)} measured batch points "
            f"[{s_vals}] (not assumed from endpoints alone). "
            f"Signs: {g_vals}. Found g(s) < 0 at interior s values "
            f"{[p['s'] for p in feasible_interior]} despite endpoint pattern — "
            f"this is a valid feasible-thinning candidate (local dip). "
            f"Thinnest measured feasible point: s={best['s']:.2f} "
            f"with sigma_max={best['sigma_max_MPa']} MPa."
        )
        print(f"\nVERDICT: {verdict}")
        print(msg)
        n_fea = counters["n_fea_new"]
        print(
            f"\nHeadline: FEA evaluations spent = {n_fea}  "
            f"(vs grid N={GRID_N_COMPARE})"
        )
        _write_result(
            {
                "verdict": verdict,
                "message": msg,
                "s_star": best["s"],
                "feasible_thinning": True,
                "sigma_max_MPa": best["sigma_max_MPa"],
                "g_at_s_star": best["g"],
                "sigma_allow_MPa": sigma_allow,
                "baseline_sigma_max_MPa": BASELINE_SIGMA_MAX_MPA,
                "feasible_interior_points": [
                    {
                        "s": p["s"],
                        "sigma_max_MPa": p["sigma_max_MPa"],
                        "g": p["g"],
                        "status": p["sample"].get("status"),
                        "mesh_gate_pass": (p["sample"].get("mesh_gate") or {}).get(
                            "pass"
                        ),
                        "reaction_pass": (p["sample"].get("reaction") or {}).get(
                            "pass"
                        ),
                    }
                    for p in feasible_interior
                ],
                "preflight_points": [
                    {
                        "s": p["s"],
                        "sigma_max_MPa": p["sigma_max_MPa"],
                        "g": p["g"],
                        "sign": p["sign"],
                        "status": p["sample"].get("status"),
                        "mesh_gate_pass": (p["sample"].get("mesh_gate") or {}).get(
                            "pass"
                        ),
                        "reaction_pass": (p["sample"].get("reaction") or {}).get(
                            "pass"
                        ),
                    }
                    for p in preflight_sorted
                ],
                "endpoints": {
                    "s_low": _endpoint_payload(sample_low, g_low),
                    "s_high": _endpoint_payload(sample_high, g_high),
                },
                "n_fea_evaluations_spent": n_fea,
                "n_evaluations_total": counters["n_eval"],
                "n_grid_compare": GRID_N_COMPARE,
                "headline": (
                    f"{n_fea} FEA evaluations spent vs {GRID_N_COMPARE} "
                    "on the original fixed grid"
                ),
                "counters": counters,
                "gates": {
                    "MAX_ABS_DV_FRAC": MAX_ABS_DV_FRAC,
                    "MAX_INVERTED": MAX_INVERTED,
                    "REACTION_ERR_PCT": REACTION_ERR_PCT,
                    "MIN_SPC_GAP_MM": MIN_SPC_GAP_MM,
                    "Y0_NECK": Y0_NECK,
                    "L_BAND": L_BAND,
                },
            }
        )
        return 0

    # If thin end already feasible, cautiously widen lower bound
    if g_low <= 0.0:
        print(
            f"g({s_low:.2f}) <= 0: feasible room may exist below the previously "
            f"tested range. Widening lower bound to s={S_LOW_WIDEN:.2f}."
        )
        s_low = S_LOW_WIDEN
        try:
            g_low, sample_low = evaluate(
                s_low,
                ctx,
                sigma_allow=sigma_allow,
                prefer_cache=True,
                counters=counters,
            )
        except InfeasibleEvaluation as e:
            _write_result(
                {
                    "verdict": "widened_lower_bound_infeasible",
                    "message": str(e),
                    "sigma_allow_MPa": sigma_allow,
                    "s_star": None,
                    "n_fea_evaluations_spent": counters["n_fea_new"],
                    "n_evaluations_total": counters["n_eval"],
                    "n_grid_compare": GRID_N_COMPARE,
                    "counters": counters,
                    "failed_sample": {
                        "s": e.s,
                        "status": e.sample.get("status"),
                        "fail_reasons": e.sample.get("fail_reasons"),
                    },
                }
            )
            print(f"STOP: {e}")
            return 2
        print(f"g({s_low:.2f}) = {g_low:+.6e}  sign={int(np.sign(g_low)) if g_low != 0 else 0}")

    # All measured thinning-side points over budget, baseline g≈0 → no thinning
    thinning_side = [p for p in preflight_sorted if p["s"] < s_high - 1e-12]
    all_thinning_over = all(p["g"] > 0.0 for p in thinning_side)
    if (
        all_thinning_over
        and abs(g_high) / max(abs(sigma_allow), 1e-30) <= G1_REL_TOL
        and not args.force_brentq
    ):
        s_vals = ", ".join(f"{p['s']:.2f}" for p in preflight_sorted)
        g_vals = ", ".join(
            f"g({p['s']:.2f})={p['g']:+.6e}" for p in preflight_sorted
        )
        verdict = "no_feasible_thinning_within_tested_range"
        msg = (
            f"Checked {len(preflight_sorted)} measured batch points "
            f"[{s_vals}] — signs of g computed from real cached "
            f"sample_result.json values, not assumed from the two endpoints. "
            f"{g_vals}. Every s < 1.00 has g(s) > 0 (over budget); "
            f"g(1.00)≈0 by construction. No feasible thinning below s=1.0 "
            f"within this one-parameter family on the tested grid. "
            f"Not forcing brentq on a bracket with no sign change."
        )
        print(f"\nVERDICT: {verdict}")
        print(msg)
        n_fea = counters["n_fea_new"]
        print(
            f"\nHeadline: FEA evaluations spent = {n_fea}  "
            f"(vs grid N={GRID_N_COMPARE})"
        )
        _write_result(
            {
                "verdict": verdict,
                "message": msg,
                "s_star": None,
                "feasible_thinning": False,
                "sigma_allow_MPa": sigma_allow,
                "baseline_sigma_max_MPa": BASELINE_SIGMA_MAX_MPA,
                "preflight_points": [
                    {
                        "s": p["s"],
                        "sigma_max_MPa": p["sigma_max_MPa"],
                        "g": p["g"],
                        "sign": p["sign"],
                        "status": p["sample"].get("status"),
                        "mesh_gate_pass": (p["sample"].get("mesh_gate") or {}).get(
                            "pass"
                        ),
                        "reaction_pass": (p["sample"].get("reaction") or {}).get(
                            "pass"
                        ),
                    }
                    for p in preflight_sorted
                ],
                "n_preflight_points_checked": len(preflight_sorted),
                "endpoints": {
                    "s_low": _endpoint_payload(
                        by_s[float(args.s_low)]["sample"],
                        by_s[float(args.s_low)]["g"],
                    ),
                    "s_high": _endpoint_payload(sample_high, g_high),
                },
                "n_fea_evaluations_spent": n_fea,
                "n_evaluations_total": counters["n_eval"],
                "n_grid_compare": GRID_N_COMPARE,
                "headline": (
                    f"{n_fea} FEA evaluations spent vs {GRID_N_COMPARE} "
                    "on the original fixed grid"
                ),
                "counters": counters,
                "gates": {
                    "MAX_ABS_DV_FRAC": MAX_ABS_DV_FRAC,
                    "MAX_INVERTED": MAX_INVERTED,
                    "REACTION_ERR_PCT": REACTION_ERR_PCT,
                    "MIN_SPC_GAP_MM": MIN_SPC_GAP_MM,
                    "Y0_NECK": Y0_NECK,
                    "L_BAND": L_BAND,
                },
            }
        )
        return 0

    if g_low * g_high >= 0.0:
        msg = (
            f"No sign change in g on [{s_low:.4f}, {s_high:.4f}] "
            f"(g_low={g_low:+.6e}, g_high={g_high:+.6e}). Cannot run brentq."
        )
        print(f"STOP: {msg}")
        _write_result(
            {
                "verdict": "no_sign_change",
                "message": msg,
                "sigma_allow_MPa": sigma_allow,
                "s_star": None,
                "feasible_thinning": False,
                "n_fea_evaluations_spent": counters["n_fea_new"],
                "n_evaluations_total": counters["n_eval"],
                "n_grid_compare": GRID_N_COMPARE,
                "counters": counters,
            }
        )
        return 1

    # --- Root find ---
    print(
        f"\n--- brentq on [{s_low:.4f}, {s_high:.4f}]  xtol={args.xtol} ---"
    )

    def g_obj(s: float) -> float:
        g_s, _sample = evaluate(
            s, ctx, sigma_allow=sigma_allow, prefer_cache=True, counters=counters
        )
        return g_s

    try:
        s_star = float(
            brentq(g_obj, s_low, s_high, xtol=args.xtol, rtol=args.xtol)
        )
    except InfeasibleEvaluation as e:
        _write_result(
            {
                "verdict": "root_find_hit_infeasible",
                "message": str(e),
                "sigma_allow_MPa": sigma_allow,
                "s_star": None,
                "n_fea_evaluations_spent": counters["n_fea_new"],
                "n_evaluations_total": counters["n_eval"],
                "n_grid_compare": GRID_N_COMPARE,
                "counters": counters,
                "failed_sample": {
                    "s": e.s,
                    "status": e.sample.get("status"),
                    "fail_reasons": e.sample.get("fail_reasons"),
                },
            }
        )
        print(f"STOP: {e}")
        return 2

    # Final validated candidate at s*
    g_star, sample_star = evaluate(
        s_star, ctx, sigma_allow=sigma_allow, prefer_cache=True, counters=counters
    )
    n_fea = counters["n_fea_new"]
    print(f"\nRoot s* = {s_star:.6f}  sigma={sample_star.get('sigma_max_MPa')}  g={g_star}")
    print(
        f"Headline: FEA evaluations spent = {n_fea}  "
        f"(vs grid N={GRID_N_COMPARE})"
    )

    _write_result(
        {
            "verdict": "root_found",
            "feasible_thinning": s_star < s_high - args.xtol,
            "s_star": s_star,
            "sigma_max_MPa": sample_star.get("sigma_max_MPa"),
            "g_at_s_star": g_star,
            "sigma_allow_MPa": sigma_allow,
            "baseline_sigma_max_MPa": BASELINE_SIGMA_MAX_MPA,
            "status": sample_star.get("status"),
            "mesh_gate_pass": (sample_star.get("mesh_gate") or {}).get("pass"),
            "reaction_pass": (sample_star.get("reaction") or {}).get("pass"),
            "fail_reasons": sample_star.get("fail_reasons") or [],
            "bracket": {"s_low": s_low, "s_high": s_high, "xtol": args.xtol},
            "n_fea_evaluations_spent": n_fea,
            "n_evaluations_total": counters["n_eval"],
            "n_grid_compare": GRID_N_COMPARE,
            "headline": (
                f"{n_fea} FEA evaluations spent vs {GRID_N_COMPARE} "
                "on the original fixed grid"
            ),
            "counters": counters,
            "gates": {
                "MAX_ABS_DV_FRAC": MAX_ABS_DV_FRAC,
                "MAX_INVERTED": MAX_INVERTED,
                "REACTION_ERR_PCT": REACTION_ERR_PCT,
                "MIN_SPC_GAP_MM": MIN_SPC_GAP_MM,
                "Y0_NECK": Y0_NECK,
                "L_BAND": L_BAND,
            },
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
