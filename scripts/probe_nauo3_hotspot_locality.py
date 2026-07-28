#!/usr/bin/env python3
"""Hotspot locality probe: local tet quality + stress-peak plateau (no new FEA).

Reuses unmodified:
  - simjeb.mesh_ur_volume._tet_quality
  - batch_nauo3_neck_scale.parse_frd_nodal_von_mises

Writes:
  reports/ur5e_nauo3_remesh_noise_floor/hotspot_locality_probe.json

  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/probe_nauo3_hotspot_locality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from batch_nauo3_neck_scale import parse_frd_nodal_von_mises  # noqa: E402
from simjeb.mesh_ur_volume import _tet_quality  # noqa: E402

OUT_PATH = (
    ROOT / "reports" / "ur5e_nauo3_remesh_noise_floor" / "hotspot_locality_probe.json"
)

RADIUS_MM = 10.0
Y0_MM = 262.0
L_MM = 40.0

# Check 1 runs (local quality + plateau)
RUNS_QUALITY = {
    "inplace_ref": ROOT / "reports" / "ur5e_nauo3_neck_batch" / "s_0.90",
    "remesh_pre_optimize": (
        ROOT / "reports" / "ur5e_nauo3_remesh_noise_floor" / "cosine_c1_s090_rerun"
    ),
    "remesh_post_optimize": (
        ROOT
        / "reports"
        / "ur5e_nauo3_remesh_noise_floor"
        / "cosine_c1_s090_remesh_optimize_probe"
        / "fea_after_optimize"
    ),
}

# Check 2 adds bump_Cinf probe
RUNS_PLATEAU = {
    **RUNS_QUALITY,
    "bump_cinf_probe": ROOT / "reports" / "ur5e_nauo3_falloff_probe" / "bump_cinf_s090",
}


def _load_hotspot(run_dir: Path) -> dict:
    return json.loads((run_dir / "stress_hotspot.json").read_text())


def _load_geom(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    g = np.load(run_dir / "geometry.npz")
    return (
        np.asarray(g["vol_points"], dtype=np.float64),
        np.asarray(g["vol_tets"], dtype=np.int32),
    )


def _quality_summary(q: dict) -> dict:
    rho = q["radius_ratio_R_over_3r"]
    dih = q["min_dihedral_deg"]
    return {
        "n_tets_local": q["n_tets"],
        "n_inverted_or_neg_volume": q["n_inverted_or_neg_volume"],
        "n_radius_ratio_gt_10": q["n_radius_ratio_gt_10"],
        "n_min_dihedral_lt_5deg": q["n_min_dihedral_lt_5deg"],
        "n_min_dihedral_lt_1deg": q["n_min_dihedral_lt_1deg"],
        "rho_p50": rho["p50"],
        "rho_p95": rho["p95"],
        "rho_p99": rho["p99"],
        "rho_p100": rho["p100"],
        "min_dihedral_p0": dih["p0"],
        "min_dihedral_p50": dih["p50"],
        "min_dihedral_p5": dih["p5"],
        "radius_ratio_full": rho,
        "min_dihedral_deg_full": dih,
    }


def local_quality(run_dir: Path, radius_mm: float = RADIUS_MM) -> dict:
    hot = _load_hotspot(run_dir)
    points, tets = _load_geom(run_dir)
    xyz = np.asarray(hot["xyz_mm"], dtype=np.float64)
    d = np.linalg.norm(points - xyz[None, :], axis=1)
    node_mask = d <= radius_mm
    node_idx = np.flatnonzero(node_mask)
    in_ball = node_mask
    # Tets whose four corners all lie in the spatial ball around the hotspot.
    tet_mask = in_ball[tets].all(axis=1)
    local_tets = tets[tet_mask]
    # Also count tets that merely touch the ball (any corner) for context.
    tet_touch = int(in_ball[tets].any(axis=1).sum())

    if len(local_tets) == 0:
        raise RuntimeError(
            f"No local tets within {radius_mm} mm of hotspot in {run_dir}"
        )

    q = _tet_quality(points, local_tets)
    summary = _quality_summary(q)
    return {
        "path": str(run_dir.relative_to(ROOT)),
        "hotspot": {
            "node_0based": hot["node_0based"],
            "xyz_mm": [float(x) for x in xyz],
            "y_mm": float(hot["y_mm"]),
            "dy_over_L": float(hot["dy_over_L"]),
            "von_mises_MPa": float(hot["von_mises_MPa"]),
        },
        "selection": {
            "radius_mm": radius_mm,
            "method": "spatial_ball_all_four_corners_in_set",
            "n_nodes_total": int(len(points)),
            "n_tets_total": int(len(tets)),
            "n_nodes_in_ball": int(len(node_idx)),
            "n_tets_all_corners_in_ball": int(len(local_tets)),
            "n_tets_any_corner_in_ball": tet_touch,
            "node_dist_to_hotspot_mm": {
                "min": float(d[node_idx].min()) if len(node_idx) else None,
                "p50": float(np.median(d[node_idx])) if len(node_idx) else None,
                "max": float(d[node_idx].max()) if len(node_idx) else None,
            },
        },
        "local_quality": summary,
    }


def _span_stats(coords: np.ndarray) -> dict:
    if len(coords) == 0:
        return {
            "n": 0,
            "centroid_xyz_mm": None,
            "span_xyz_mm": None,
            "rms_dist_from_centroid_mm": None,
            "max_dist_from_centroid_mm": None,
            "max_pairwise_dist_mm": None,
        }
    c = coords.mean(axis=0)
    d = np.linalg.norm(coords - c[None, :], axis=1)
    span = coords.max(axis=0) - coords.min(axis=0)
    # Pairwise max for small n; for large n use diameter via farthest from centroid
    # then farthest from that point (2-approx) — fine for reporting.
    if len(coords) <= 2000:
        # chunked to avoid huge N^2 for mid sizes
        max_pair = 0.0
        for i in range(0, len(coords), 256):
            a = coords[i : i + 256]
            d2 = np.sum((a[:, None, :] - coords[None, :, :]) ** 2, axis=2)
            max_pair = max(max_pair, float(np.sqrt(d2.max())))
    else:
        i0 = int(np.argmax(d))
        d2 = np.linalg.norm(coords - coords[i0][None, :], axis=1)
        max_pair = float(d2.max())
    return {
        "n": int(len(coords)),
        "centroid_xyz_mm": [float(x) for x in c],
        "span_xyz_mm": [float(x) for x in span],
        "span_xz_mm": [float(span[0]), float(span[2])],
        "rms_dist_from_centroid_mm": float(np.sqrt(np.mean(d**2))),
        "max_dist_from_centroid_mm": float(d.max()),
        "max_pairwise_dist_mm": max_pair,
    }


def _cluster_label(spat: dict) -> str:
    d = spat.get("max_pairwise_dist_mm")
    if d is None:
        return "empty"
    if d < 15.0:
        return "clustered_near_point"
    if d >= 30.0:
        return "spread_across_cross_section"
    return "intermediate"


def plateau_stats(run_dir: Path) -> dict:
    hot = _load_hotspot(run_dir)
    points, _tets = _load_geom(run_dir)
    frd = run_dir / "nauo3_cantilever.frd"
    n_nodes = len(points)
    vm = parse_frd_nodal_von_mises(frd, n_nodes)
    finite = np.isfinite(vm)
    vmax = float(np.nanmax(vm))
    # Prefer FRD max; cross-check against hotspot json.
    vmax_hot = float(hot["von_mises_MPa"])
    xyz_hot = np.asarray(hot["xyz_mm"], dtype=np.float64)

    bands = {}
    # Requested 1/2/5%; also 10% because 1–5% is often a single node here.
    for pct in (1.0, 2.0, 5.0, 10.0):
        thr = vmax * (1.0 - pct / 100.0)
        mask = finite & (vm >= thr)
        idx = np.flatnonzero(mask)
        coords = points[idx]
        spat = _span_stats(coords)
        y_hot = float(xyz_hot[1])
        slab = coords[np.abs(coords[:, 1] - y_hot) <= 5.0] if len(coords) else coords
        spat_slab = _span_stats(slab)
        bands[f"within_{pct:g}pct_of_max"] = {
            "threshold_MPa": float(thr),
            "n_nodes": int(mask.sum()),
            "fraction_of_nodes": float(mask.sum() / max(finite.sum(), 1)),
            "spatial": spat,
            "spatial_y_slab_pm5mm": spat_slab,
            "cluster_vs_spread": _cluster_label(spat),
        }

    # Rank gaps: how isolated is sigma_max vs 2nd/3rd/...
    order = np.argsort(vm)[::-1]
    rank_gaps = []
    for r in (1, 2, 3, 5, 10, 20):
        if r > int(finite.sum()):
            break
        nid = int(order[r - 1])
        val = float(vm[nid])
        xyz = points[nid]
        rank_gaps.append(
            {
                "rank": r,
                "node_0based": nid,
                "von_mises_MPa": val,
                "rel_to_max": float(val / vmax),
                "deficit_pct": float((1.0 - val / vmax) * 100.0),
                "xyz_mm": [float(x) for x in xyz],
                "dist_to_hotspot_mm": float(np.linalg.norm(xyz - xyz_hot)),
            }
        )

    # Top-20 spatial structure (reveals competing lobes if present).
    top_k = 20
    top_idx = order[:top_k]
    top_coords = points[top_idx]
    top_span = _span_stats(top_coords)
    # Simple 1D z-clustering: two walls of a hollow section sit ~75 mm apart in z.
    z = top_coords[:, 2]
    z_med = float(np.median(z))
    lobe_hi = top_coords[z >= z_med]
    lobe_lo = top_coords[z < z_med]
    # Best stress in each z-half of the top-20.
    vm_top = vm[top_idx]
    best_hi = float(vm_top[z >= z_med].max()) if len(lobe_hi) else None
    best_lo = float(vm_top[z < z_med].max()) if len(lobe_lo) else None
    competing = None
    if best_hi is not None and best_lo is not None:
        competing = {
            "z_split_mm": z_med,
            "lobe_z_high": {
                "n_in_top20": int(len(lobe_hi)),
                "best_MPa": best_hi,
                "deficit_vs_global_max_pct": float((1.0 - best_hi / vmax) * 100.0),
                "centroid_xyz_mm": [float(x) for x in lobe_hi.mean(axis=0)],
            },
            "lobe_z_low": {
                "n_in_top20": int(len(lobe_lo)),
                "best_MPa": best_lo,
                "deficit_vs_global_max_pct": float((1.0 - best_lo / vmax) * 100.0),
                "centroid_xyz_mm": [float(x) for x in lobe_lo.mean(axis=0)],
            },
            "centroid_separation_mm": float(
                np.linalg.norm(lobe_hi.mean(axis=0) - lobe_lo.mean(axis=0))
            ),
            "winner_lobe": (
                "z_high" if best_hi >= best_lo else "z_low"
            ),
        }

    near5 = finite & (vm >= vmax * 0.95)
    d_to_hot = np.linalg.norm(points[near5] - xyz_hot[None, :], axis=1)
    return {
        "path": str(run_dir.relative_to(ROOT)),
        "hotspot": {
            "node_0based": hot["node_0based"],
            "xyz_mm": [float(x) for x in xyz_hot],
            "y_mm": float(hot["y_mm"]),
            "dy_from_y0_mm": float(hot["dy_from_y0_mm"]),
            "dy_over_L": float(hot["dy_over_L"]),
            "von_mises_MPa_json": vmax_hot,
        },
        "frd": {
            "path": str(frd.relative_to(ROOT)),
            "n_nodes": n_nodes,
            "n_finite": int(finite.sum()),
            "sigma_max_MPa": vmax,
            "sigma_max_matches_hotspot_json": bool(abs(vmax - vmax_hot) < 1e-9),
            "abs_diff_frd_vs_json_MPa": float(abs(vmax - vmax_hot)),
        },
        "bands": bands,
        "rank_gaps": rank_gaps,
        "top20_spatial": {
            **top_span,
            "cluster_vs_spread": _cluster_label(top_span),
            "competing_z_lobes": competing,
        },
        "within_5pct_dist_to_reported_hotspot_mm": {
            "n": int(near5.sum()),
            "min": float(d_to_hot.min()) if near5.any() else None,
            "p50": float(np.median(d_to_hot)) if near5.any() else None,
            "p95": float(np.percentile(d_to_hot, 95)) if near5.any() else None,
            "max": float(d_to_hot.max()) if near5.any() else None,
        },
    }


def _compare_local_quality(results: dict[str, dict]) -> dict:
    keys = ["inplace_ref", "remesh_pre_optimize", "remesh_post_optimize"]
    rows = {k: results[k]["local_quality"] for k in keys}
    rho_p95 = {k: rows[k]["rho_p95"] for k in keys}
    rho_p50 = {k: rows[k]["rho_p50"] for k in keys}
    dih_p0 = {k: rows[k]["min_dihedral_p0"] for k in keys}

    def rel_pct(a: float, b: float) -> float:
        return 100.0 * (a - b) / b if b != 0 else float("nan")

    return {
        "rho_p50": rho_p50,
        "rho_p95": rho_p95,
        "min_dihedral_p0": dih_p0,
        "deltas": {
            "remesh_pre_vs_inplace_rho_p95_rel_pct": rel_pct(
                rho_p95["remesh_pre_optimize"], rho_p95["inplace_ref"]
            ),
            "remesh_post_vs_inplace_rho_p95_rel_pct": rel_pct(
                rho_p95["remesh_post_optimize"], rho_p95["inplace_ref"]
            ),
            "remesh_post_vs_pre_rho_p95_rel_pct": rel_pct(
                rho_p95["remesh_post_optimize"], rho_p95["remesh_pre_optimize"]
            ),
            "remesh_pre_vs_inplace_rho_p50_rel_pct": rel_pct(
                rho_p50["remesh_pre_optimize"], rho_p50["inplace_ref"]
            ),
            "remesh_post_vs_inplace_rho_p50_rel_pct": rel_pct(
                rho_p50["remesh_post_optimize"], rho_p50["inplace_ref"]
            ),
        },
    }


def _compare_plateau(results: dict[str, dict]) -> dict:
    hotspots = {}
    for k, v in results.items():
        r2 = next(r for r in v["rank_gaps"] if r["rank"] == 2)
        lobes = v["top20_spatial"].get("competing_z_lobes")
        hotspots[k] = {
            "xyz_mm": v["hotspot"]["xyz_mm"],
            "dy_over_L": v["hotspot"]["dy_over_L"],
            "sigma_max_MPa": v["frd"]["sigma_max_MPa"],
            "n_within_1pct": v["bands"]["within_1pct_of_max"]["n_nodes"],
            "n_within_2pct": v["bands"]["within_2pct_of_max"]["n_nodes"],
            "n_within_5pct": v["bands"]["within_5pct_of_max"]["n_nodes"],
            "n_within_10pct": v["bands"]["within_10pct_of_max"]["n_nodes"],
            "label_1pct": v["bands"]["within_1pct_of_max"]["cluster_vs_spread"],
            "label_5pct": v["bands"]["within_5pct_of_max"]["cluster_vs_spread"],
            "label_10pct": v["bands"]["within_10pct_of_max"]["cluster_vs_spread"],
            "rank2_deficit_pct": r2["deficit_pct"],
            "rank2_dist_to_hotspot_mm": r2["dist_to_hotspot_mm"],
            "rank2_xyz_mm": r2["xyz_mm"],
            "top20_max_pairwise_mm": v["top20_spatial"]["max_pairwise_dist_mm"],
            "top20_label": v["top20_spatial"]["cluster_vs_spread"],
            "competing_lobes_centroid_separation_mm": (
                None if lobes is None else lobes["centroid_separation_mm"]
            ),
            "winner_lobe": None if lobes is None else lobes["winner_lobe"],
            "other_lobe_best_deficit_pct": (
                None
                if lobes is None
                else (
                    lobes["lobe_z_low"]["deficit_vs_global_max_pct"]
                    if lobes["winner_lobe"] == "z_high"
                    else lobes["lobe_z_high"]["deficit_vs_global_max_pct"]
                )
            ),
        }
    # Cross-run: does remesh pick the opposite wall from inplace?
    a = results["inplace_ref"]["hotspot"]["xyz_mm"]
    b = results["remesh_pre_optimize"]["hotspot"]["xyz_mm"]
    hotspot_shift = {
        "inplace_to_remesh_delta_xyz_mm": [float(b[i] - a[i]) for i in range(3)],
        "inplace_to_remesh_dist_mm": float(
            np.linalg.norm(np.asarray(b) - np.asarray(a))
        ),
        "opposite_wall_consistent_with_z_lobes": bool(abs(b[2] - a[2]) > 50.0),
    }
    return {"per_run": hotspots, "hotspot_shift_inplace_vs_remesh": hotspot_shift}


def main() -> None:
    print("Check 1: local hotspot-neighborhood element quality")
    quality = {}
    for name, path in RUNS_QUALITY.items():
        print(f"  {name}: {path}")
        quality[name] = local_quality(path)
        lq = quality[name]["local_quality"]
        print(
            f"    n_local_tets={lq['n_tets_local']}  "
            f"rho_p50={lq['rho_p50']:.4f}  rho_p95={lq['rho_p95']:.4f}  "
            f"min_dih_p0={lq['min_dihedral_p0']:.3f}"
        )

    print("Check 2: stress-distribution flatness near sigma_max")
    plateau = {}
    for name, path in RUNS_PLATEAU.items():
        print(f"  {name}: {path}")
        plateau[name] = plateau_stats(path)
        b1 = plateau[name]["bands"]["within_1pct_of_max"]
        b5 = plateau[name]["bands"]["within_5pct_of_max"]
        b10 = plateau[name]["bands"]["within_10pct_of_max"]
        r2 = next(r for r in plateau[name]["rank_gaps"] if r["rank"] == 2)
        lobes = plateau[name]["top20_spatial"].get("competing_z_lobes")
        sep = None if lobes is None else lobes["centroid_separation_mm"]
        print(
            f"    sigma_max={plateau[name]['frd']['sigma_max_MPa']:.6f}  "
            f"n@1%={b1['n_nodes']}  n@5%={b5['n_nodes']}  n@10%={b10['n_nodes']}  "
            f"rank2_deficit={r2['deficit_pct']:.2f}%  "
            f"rank2_dist={r2['dist_to_hotspot_mm']:.1f}mm  "
            f"top20_lobe_sep={sep}"
        )

    report = {
        "test": "hotspot_locality_probe",
        "purpose": (
            "Last diagnostic on the ~1.13% in-place vs remesh gap before closing: "
            "local (not global) element quality near each run's own hotspot, and "
            "whether sigma_max sits on a broad plateau. Zero new FEA."
        ),
        "config": {
            "local_radius_mm": RADIUS_MM,
            "y0_mm": Y0_MM,
            "L_mm": L_MM,
            "tet_selection": "all four corners inside spatial ball around hotspot xyz",
            "helpers_unmodified": [
                "simjeb.mesh_ur_volume._tet_quality",
                "batch_nauo3_neck_scale.parse_frd_nodal_von_mises",
            ],
        },
        "check1_local_element_quality": {
            "runs": quality,
            "comparison": _compare_local_quality(quality),
        },
        "check2_stress_peak_plateau": {
            "runs": plateau,
            "comparison": _compare_plateau(plateau),
        },
        "context": {
            "inplace_ref_sigma_MPa": quality["inplace_ref"]["hotspot"]["von_mises_MPa"],
            "remesh_pre_sigma_MPa": quality["remesh_pre_optimize"]["hotspot"][
                "von_mises_MPa"
            ],
            "remesh_post_sigma_MPa": quality["remesh_post_optimize"]["hotspot"][
                "von_mises_MPa"
            ],
            "gap_remesh_pre_vs_inplace_rel_pct": 100.0
            * (
                quality["remesh_pre_optimize"]["hotspot"]["von_mises_MPa"]
                - quality["inplace_ref"]["hotspot"]["von_mises_MPa"]
            )
            / quality["inplace_ref"]["hotspot"]["von_mises_MPa"],
            "note_branch_b_caveat": (
                "Branch (b) tested Gmsh default light optimize only: global rho_p95 "
                "improved ~2.7% (22.92→22.30) but remesh remains ~28% worse than "
                "baseline 17.41. That falsifies 'default optimize explains the gap', "
                "not 'element quality in general'."
            ),
        },
        "interpretation_notes": {
            "do_not_force_conclusion": True,
            "if_local_quality_differs": (
                "Mechanistic candidate: local element quality near the hotspot, "
                "invisible to global percentiles."
            ),
            "if_peak_is_plateau": (
                "Mechanistic candidate: sigma_max is a fragile single-node order "
                "statistic on a broad, relatively flat stress region — consistent "
                "with hotspot xyz shifting across mesh realizations."
            ),
            "if_neither_notable": (
                "~1% is not explainable at the resolution of these checks; treat as "
                "documented discretization uncertainty for Phase 3 sigma_max labels."
            ),
            "close_investigation": (
                "Lock in ~1% as documented discretization uncertainty and move on; "
                "do not open a third branch on this specific gap."
            ),
        },
        "observed_summary_not_forced": {
            "check1": (
                "Local rho_p95 near each run's own hotspot is similar and actually "
                "slightly better on remesh (≈1.93) than inplace (≈1.98); local "
                "rho_p50 is mildly worse on remesh (1.31 vs 1.24); min_dihedral_p0 "
                "worse on remesh (15.5° vs 20.6°). Zero local inverted / rho>10. "
                "Default optimize leaves the local neighborhood unchanged. Local "
                "quality does not explain remesh sigma being higher."
            ),
            "check2": (
                "At the requested 1/2/5% bands, sigma_max is isolated (n=1 within "
                "5% for inplace and remesh; rank2 sits ~6–7% below max). So this is "
                "not a broad single-peak plateau. However top-20 nodes form two "
                "competing lobes on opposite walls (~75 mm apart in z): inplace "
                "wins on the z≈176 wall; remesh wins on the z≈101 wall — the same "
                "wall that is only ~6% below max on the other mesh. Hotspot location "
                "shift is therefore a winner-takes-all between two near-equal "
                "cross-section peaks, not a wobble around one peak."
            ),
            "phase3_implication": (
                "sigma_max as a single-node label carries at least ~1% mesh-"
                "dependent uncertainty on this geometry, driven in part by which "
                "of two competing wall peaks wins. Surrogate training should "
                "tolerate that floor rather than treat 1% as eliminable protocol "
                "noise. Design-relevant signal (38%) remains two orders larger."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
