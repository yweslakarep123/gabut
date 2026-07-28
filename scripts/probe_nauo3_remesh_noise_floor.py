#!/usr/bin/env python3
"""Remesh-noise floor: same cosine_C1 (s=0.90, y0=262, L=40), independent volume mesh.

Pipeline note: batch/pilot samples deform a *fixed* ``geometry.npz`` and do not
call gmsh/tetgen per sample. Re-running ``run_one_sample(..., reuse_fea=False)``
on that mesh is deterministic and measures solver repeatability (~0), not remesh
noise.

This probe instead:
  1. Deforms the locked baseline mesh with cosine_C1 (same params as reference)
  2. Extracts the deformed boundary surface
  3. Regenerates the volume mesh via Gmsh MeshOnlyEmpty (independent tet fill)
  4. Runs FEA on the remeshed solid (no second deform; geometry already scaled)
  5. Compares sigma_max to pilot2d sc0.90_ss1.00

Does NOT overwrite the pilot2d reference.

  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/probe_nauo3_remesh_noise_floor.py
  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/probe_nauo3_remesh_noise_floor.py --tag cosine_c1_s090_rerun2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from batch_nauo3_neck_scale import (  # noqa: E402
    BASELINE_SIGMA_MAX_MPA,
    CCX,
    G,
    HOTSPOT_CENTER_FRAC,
    L_BAND,
    M_DISTAL,
    M_FOREARM,
    M_PAYLOAD,
    M_W1,
    M_W2,
    M_W3,
    MESH_BASE,
    REACTION_ERR_PCT,
    X,
    Y0_NECK,
    _boundary_faces,
    _rf_moment_from_forc,
    classify_hotspot,
    deform_neck_radial,
    flip_negative_tets,
    parse_frd_nodal_von_mises,
    spc_band_clearance_report,
    write_cantilever_inp,
)
from fea_nauo3_cantilever import (  # noqa: E402
    M_SELF,
    _tet_volume_sign,
    select_end_faces,
    step1_com_moment,
)
from simjeb.mesh_ur_volume import (  # noqa: E402
    _surface_from_tets,
    mesh_discrete_stl_to_tets,
)

OUT_ROOT = ROOT / "reports" / "ur5e_nauo3_remesh_noise_floor"
DEFAULT_TAG = "cosine_c1_s090_rerun"
OUT = OUT_ROOT / DEFAULT_TAG
REF_COSINE = (
    ROOT / "reports" / "ur5e_nauo3_neck_pilot2d" / "sc0.90_ss1.00" / "sample_result.json"
)
REF_DEFORM = (
    ROOT / "reports" / "ur5e_nauo3_neck_pilot2d" / "sc0.90_ss1.00" / "deform_meta.json"
)
BUMP_DEFORM = (
    ROOT
    / "reports"
    / "ur5e_nauo3_falloff_probe"
    / "bump_cinf_s090"
    / "deform_meta.json"
)
CLEAN_STL_REF = ROOT / "reports" / "ur5e_nauo3_volume" / "_heal_iso_clean.stl"

S = 0.90
Y0 = Y0_NECK
L = L_BAND
FALLOFF = "cosine_C1"
MESH_SIZE = 5.0
MESH_SIZE_MIN = 2.5
REF_SIGMA = 2.271612960162118
REF_DV_FRAC = -0.01029144980632467
BUMP_DELTA_REL = -0.0151052326926705  # from falloff probe comparison.json


def _fea_on_mesh(
    points: np.ndarray,
    tets: np.ndarray,
    node_surf: np.ndarray,
    dens0_t_mm3: float,
    out: Path,
    *,
    y0: float,
    L: float,
) -> dict:
    """FEA + reaction + hotspot on an already-deformed (or remeshed) solid."""
    out.mkdir(parents=True, exist_ok=True)
    tets_s, flip_meta = flip_negative_tets(points, tets)

    tet_ids, face_ids, face_nodes = _boundary_faces(tets_s)
    ends = select_end_faces(points, node_surf, tet_ids, face_ids, face_nodes)
    # NAUO3 convention in this repo: proximal = low-Y (shoulder), distal = high-Y.
    # SVD end-picking can flip after remesh; canonicalize before SPC clearance.
    if float(points[ends["prox_nodes"], 1].mean()) > float(
        points[ends["dist_nodes"], 1].mean()
    ):
        ends = {
            **ends,
            "u": -np.asarray(ends["u"], dtype=np.float64),
            "prox_nodes": ends["dist_nodes"],
            "dist_nodes": ends["prox_nodes"],
            "prox_faces": ends["dist_faces"],
            "dist_faces": ends["prox_faces"],
            "prox_center": ends["dist_center"],
            "dist_center": ends["prox_center"],
            "n_prox_faces": ends["n_dist_faces"],
            "n_dist_faces": ends["n_prox_faces"],
            "end_labels_swapped_for_y": True,
        }
    else:
        ends["end_labels_swapped_for_y"] = False
    clear = spc_band_clearance_report(points, ends, L=L, y0=y0)
    np.savez_compressed(
        out / "geometry.npz",
        vol_points=points,
        vol_tets=tets_s,
        node_surf=node_surf,
    )
    sample: dict = {
        "s": S,
        "tag": out.name,
        "status": "pending_fea",
        "fail_reasons": [],
        "spc_clearance": clear,
        "tet_flip_repair": flip_meta,
        "end_labels_swapped_for_y": bool(ends.get("end_labels_swapped_for_y")),
        "mesh_stats": {
            "n_nodes": int(len(points)),
            "n_tets": int(len(tets_s)),
            "V_mm3": float(np.abs(_tet_volume_sign(points, tets_s)).sum()),
        },
    }
    if not clear["safe_clearance"]:
        sample["status"] = "spc_band_overlap"
        sample["fail_reasons"].append("spc_band_overlap_after_remesh")
        (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
        return sample

    u = ends["u"]
    z = np.array([0.0, 0.0, -1.0])
    grav = z - u * np.dot(z, u)
    if np.linalg.norm(grav) < 1e-6:
        grav = np.array([-1.0, 0.0, 0.0])
        grav = grav - u * np.dot(grav, u)
    grav = grav / np.linalg.norm(grav)

    F_N = M_DISTAL * G
    levers = [
        (M_FOREARM, X["forearm_com"] - X["elbow"]),
        (M_W1, X["w1_com"] - X["elbow"]),
        (M_W2, X["w2_com"] - X["elbow"]),
        (M_W3, X["w3_com"] - X["elbow"]),
        (M_PAYLOAD, X["payload"] - X["elbow"]),
    ]
    M_elbow_Nm = sum(m * G * lev for m, lev in levers)
    mdir = np.cross(u, grav)
    mdir = mdir / (np.linalg.norm(mdir) + 1e-30)
    M_dist_Nmm = mdir * (M_elbow_Nm * 1000.0)

    vols = np.abs(_tet_volume_sign(points, tets_s))
    V_mm3 = float(vols.sum())
    dens_kg_m3 = dens0_t_mm3 / 1e-12
    m_self_eff = dens_kg_m3 * (V_mm3 * 1e-9)

    inp = out / "nauo3_cantilever.inp"
    write_cantilever_inp(inp, points, tets_s, ends, dens0_t_mm3, F_N, M_dist_Nmm, grav)

    # Force fresh solve — never reuse FRD / sample_result from another run.
    frd = out / "nauo3_cantilever.frd"
    if frd.exists():
        frd.unlink()
    proc = subprocess.run(
        [str(CCX), "-i", inp.stem],
        cwd=str(out),
        capture_output=True,
        text=True,
    )
    (out / "ccx_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
    sample["ccx_returncode"] = proc.returncode
    sample["ccx_reused"] = False
    if proc.returncode != 0:
        sample["status"] = "ccx_fail"
        sample["fail_reasons"].append(f"ccx_returncode={proc.returncode}")
        (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
        return sample

    M_fe_Nm, RF_fe, nrf = _rf_moment_from_forc(
        frd, points, ends["prox_nodes"], ends["prox_center"]
    )
    M_fe_norm = float(np.linalg.norm(M_fe_Nm)) if M_fe_Nm is not None else None
    step1 = step1_com_moment()
    M_analytic = step1["M_total_new_Nm"]
    centroid = np.average(points[tets_s].mean(axis=1), weights=vols, axis=0)
    x_com_mm = float(np.dot(centroid - ends["prox_center"], u))
    L_m = ends["L_mm"] / 1000.0
    x_com_m = x_com_mm / 1000.0
    M_fea_pred = m_self_eff * G * x_com_m + F_N * L_m + M_elbow_Nm
    err_a = abs(M_fe_norm - M_analytic) / M_analytic * 100.0 if M_fe_norm else None
    err_f = abs(M_fe_norm - M_fea_pred) / M_fea_pred * 100.0 if M_fe_norm else None
    reaction_pass = M_fe_norm is not None and (
        (err_f is not None and err_f < REACTION_ERR_PCT)
        or (err_a is not None and err_a < REACTION_ERR_PCT)
    )
    sample["reaction"] = {
        "M_fe_Nm": M_fe_norm,
        "RF_N": RF_fe.tolist() if RF_fe is not None else None,
        "M_analytic_Nm": M_analytic,
        "M_fea_pred_Nm": M_fea_pred,
        "err_vs_analytic_pct": err_a,
        "err_vs_fea_pred_pct": err_f,
        "pass": reaction_pass,
        "n_prox_with_rf": nrf,
        "m_self_eff_kg": m_self_eff,
        "L_mm": ends["L_mm"],
        "x_com_from_prox_m": x_com_m,
    }
    if not reaction_pass:
        sample["status"] = "reaction_check_fail"
        sample["fail_reasons"].append(
            f"reaction_err analytic={err_a}% fea_pred={err_f}% (lim {REACTION_ERR_PCT}%)"
        )
        (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
        return sample

    vm = parse_frd_nodal_von_mises(frd, len(points))
    hot = classify_hotspot(
        points, vm, ends["prox_nodes"], ends["dist_nodes"], y0=y0, L=L
    )
    cmask = (np.abs(points[:, 1] - y0) <= HOTSPOT_CENTER_FRAC * L) & np.isfinite(vm)
    if cmask.any():
        ic = int(np.flatnonzero(cmask)[np.argmax(vm[cmask])])
        sigma_band_center = {
            "definition": "max von_mises among nodes with |y-y0|/L <= 0.35",
            "dy_over_L_threshold": HOTSPOT_CENTER_FRAC,
            "L_mm": L,
            "sigma_max_MPa": float(vm[cmask].max()),
            "node_0based": ic,
            "xyz_mm": points[ic].tolist(),
            "y_mm": float(points[ic, 1]),
            "dy_over_L": float(abs(points[ic, 1] - y0) / L),
        }
    else:
        sigma_band_center = None

    sample["hotspot"] = hot
    sample["sigma_max_MPa"] = hot["von_mises_MPa"]
    sample["sigma_max_band_center"] = sigma_band_center
    sample["band_position"] = hot["band_position"]
    sample["status"] = "ok"
    (out / "sample_result.json").write_text(json.dumps(sample, indent=2))
    (out / "stress_hotspot.json").write_text(json.dumps(hot, indent=2))
    return sample


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=(
            "Output folder name under reports/ur5e_nauo3_remesh_noise_floor/ "
            f"(default: {DEFAULT_TAG}). Use a new tag to avoid overwriting prior runs."
        ),
    )
    ap.add_argument(
        "--mesh-size",
        type=float,
        default=MESH_SIZE,
        help=f"Gmsh Mesh.MeshSizeMax (mm), default {MESH_SIZE}",
    )
    ap.add_argument(
        "--mesh-size-min",
        type=float,
        default=MESH_SIZE_MIN,
        help=f"Gmsh Mesh.MeshSizeMin (mm), default {MESH_SIZE_MIN}",
    )
    args = ap.parse_args()
    mesh_size = float(args.mesh_size)
    mesh_size_min = float(args.mesh_size_min)
    OUT = OUT_ROOT / args.tag
    OUT.mkdir(parents=True, exist_ok=True)

    g = np.load(MESH_BASE)
    points0 = np.asarray(g["vol_points"], dtype=np.float64).copy()
    tets0 = np.asarray(g["vol_tets"], dtype=np.int32)
    node_surf0 = np.asarray(g["node_surf"], dtype=bool)

    tet_ids, face_ids, face_nodes = _boundary_faces(tets0)
    ends0 = select_end_faces(points0, node_surf0, tet_ids, face_ids, face_nodes)
    clear0 = spc_band_clearance_report(points0, ends0, L=L)
    if not clear0["safe_clearance"]:
        print("STOP: SPC/band clearance failed on baseline")
        return 2

    vols0 = np.abs(_tet_volume_sign(points0, tets0))
    V0 = float(vols0.sum())
    dens0_t_mm3 = (M_SELF / (V0 * 1e-9)) * 1e-12
    axis_point = points0.mean(0)
    axis_u = ends0["u"]

    freeze = np.zeros(len(points0), dtype=bool)
    freeze[ends0["prox_nodes"]] = True
    freeze[ends0["dist_nodes"]] = True

    print("=== remesh noise floor: cosine_C1 @ locked (s,y0,L) ===")
    print(f"s={S}  y0={Y0}  L={L}  falloff_kind={FALLOFF}")
    print(f"mesh_size={mesh_size}  mesh_size_min={mesh_size_min}")
    print(f"ref sigma={REF_SIGMA}  path={REF_COSINE.relative_to(ROOT)}")
    print(f"out → {OUT}")
    print(
        "note: deform fixed baseline → extract surface → Gmsh MeshOnlyEmpty → FEA "
        "(not a reuse of sample_result / FRD)"
    )

    t0 = time.perf_counter()

    # 1) Same parametric deform as Phase-1 / pilot shoulder-noop neck stage
    points_def, dmeta = deform_neck_radial(
        points0,
        S,
        axis_point,
        axis_u,
        y0=Y0,
        L=L,
        falloff_kind=FALLOFF,
        freeze_mask=freeze,
    )
    tets_def, flip_meta = flip_negative_tets(points_def, tets0)
    dmeta["tet_flip_repair"] = flip_meta
    (OUT / "deform_meta_pre_remesh.json").write_text(json.dumps(dmeta, indent=2))

    V_def = float(np.abs(_tet_volume_sign(points_def, tets_def)).sum())
    dV_frac_pre = (V_def - V0) / V0
    print(
        f"pre-remesh deform: n_nodes={len(points_def)} n_tets={len(tets_def)} "
        f"dV_frac={dV_frac_pre:.6e}  n_nodes_w_gt_0_5={dmeta['n_nodes_w_gt_0_5']}"
    )

    # 2) Independent volume remesh of the deformed solid
    surf_v, surf_f, _ = _surface_from_tets(points_def, tets_def)
    stl_path = OUT / "deformed_surface.stl"
    trimesh.Trimesh(vertices=surf_v, faces=surf_f, process=False).export(stl_path)
    print(f"extracted surface → {stl_path}  faces={len(surf_f)}")

    t_mesh = time.perf_counter()
    points_r, tets_r, mesh_info = mesh_discrete_stl_to_tets(
        stl_path, mesh_size=mesh_size, mesh_size_min=mesh_size_min
    )
    mesh_wall = time.perf_counter() - t_mesh
    surf_v_r, surf_f_r, node_surf_r = _surface_from_tets(points_r, tets_r)
    mesh_info = {
        **mesh_info,
        "wall_s": mesh_wall,
        "source_stl": str(stl_path.relative_to(ROOT)),
        "baseline_mesh": str(MESH_BASE.relative_to(ROOT)),
        "baseline_n_nodes": int(len(points0)),
        "baseline_n_tets": int(len(tets0)),
        "pre_remesh_n_nodes": int(len(points_def)),
        "pre_remesh_n_tets": int(len(tets_def)),
        "remeshed_n_surf_vertices": int(len(surf_v_r)),
        "remeshed_n_surf_faces": int(len(surf_f_r)),
        "same_surface_as_heal_iso_clean": bool(
            CLEAN_STL_REF.is_file()
            and stl_path.stat().st_size != CLEAN_STL_REF.stat().st_size
        ),
    }
    (OUT / "remesh_info.json").write_text(json.dumps(mesh_info, indent=2))
    print(
        f"remesh: n_nodes={mesh_info['n_nodes']} n_tets={mesh_info['n_tets']} "
        f"wall={mesh_wall:.1f}s"
    )

    # 3) FEA on remeshed solid (reuse_fea forced false; no old sample_result read)
    sample = _fea_on_mesh(
        points_r.astype(np.float64),
        tets_r.astype(np.int32),
        node_surf_r,
        dens0_t_mm3,
        OUT,
        y0=Y0,
        L=L,
    )
    wall = time.perf_counter() - t0

    sigma = sample.get("sigma_max_MPa")
    status = sample.get("status")
    V_r = (sample.get("mesh_stats") or {}).get("V_mm3")
    dV_frac_r = None if V_r is None else (float(V_r) - V0) / V0

    # Optional: weight-stat comparison (no FEA) — cosine_C1 ref vs bump_Cinf
    weight_compare = None
    if REF_DEFORM.is_file() and BUMP_DEFORM.is_file():
        ref_d = json.loads(REF_DEFORM.read_text())
        bump_d = json.loads(BUMP_DEFORM.read_text())
        # Prefer neck stage-0 if compose present
        ref_neck = (ref_d.get("compose_stages") or [ref_d])[0]
        bump_neck = (bump_d.get("compose_stages") or [bump_d])[0]
        n_ref = ref_neck.get("n_nodes_w_gt_0_5")
        n_bump = bump_neck.get("n_nodes_w_gt_0_5")
        n_this = dmeta.get("n_nodes_w_gt_0_5")
        weight_compare = {
            "cosine_C1_ref_pilot2d": {
                "path": str(REF_DEFORM.relative_to(ROOT)),
                "n_nodes_w_gt_0_5": n_ref,
                "n_nodes_w_gt_0": ref_neck.get("n_nodes_w_gt_0"),
            },
            "cosine_C1_this_pre_remesh": {
                "n_nodes_w_gt_0_5": n_this,
                "n_nodes_w_gt_0": dmeta.get("n_nodes_w_gt_0"),
                "matches_ref": n_this == n_ref,
            },
            "bump_Cinf_probe": {
                "path": str(BUMP_DEFORM.relative_to(ROOT)),
                "n_nodes_w_gt_0_5": n_bump,
                "n_nodes_w_gt_0": bump_neck.get("n_nodes_w_gt_0"),
            },
            "delta_bump_minus_cosine_n_w_gt_0_5": (
                None if n_ref is None or n_bump is None else int(n_bump) - int(n_ref)
            ),
            "ratio_bump_over_cosine_n_w_gt_0_5": (
                None
                if n_ref in (None, 0) or n_bump is None
                else float(n_bump) / float(n_ref)
            ),
            "interpretation_note": (
                "bump_Cinf holds more nodes near full scale (w>0.5) than cosine_C1 "
                "at the same (s,y0,L) if ratio > 1 — consistent with a flatter "
                "plateau then sharper edge falloff."
            ),
        }

    abs_delta = None if sigma is None else float(sigma) - REF_SIGMA
    rel_delta = None if sigma is None else (float(sigma) / REF_SIGMA) - 1.0

    # Interpretation thresholds from the brief — report, do not decide unilaterally
    if rel_delta is None:
        verdict_band = "unavailable"
        verdict_text = "FEA did not yield sigma_max; cannot compare to remesh noise floor."
    else:
        abs_rel_pct = abs(rel_delta) * 100.0
        bump_abs_pct = abs(BUMP_DELTA_REL) * 100.0
        if abs_rel_pct < 0.3:
            verdict_band = "noise_much_smaller_than_bump_delta"
            verdict_text = (
                f"Remesh noise |Δσ|/σ_ref = {abs_rel_pct:.3f}% << 0.3–0.5% band "
                f"and << bump_Cinf |Δ|={bump_abs_pct:.2f}%. Under this single-pair "
                "measurement, the bump_Cinf delta is unlikely to be remesh noise alone "
                "— further falloff exploration remains defensible, with the caveat "
                "that n=1 still has no statistical distribution of remesh noise."
            )
        elif abs_rel_pct < 0.5:
            verdict_band = "noise_below_half_percent"
            verdict_text = (
                f"Remesh noise |Δσ|/σ_ref = {abs_rel_pct:.3f}% is below 0.5% and "
                f"below bump_Cinf |Δ|={bump_abs_pct:.2f}%. Suggests bump delta is "
                "larger than this remesh floor, but margin is modest."
            )
        elif abs_rel_pct < bump_abs_pct:
            verdict_band = "noise_smaller_but_comparable_order"
            verdict_text = (
                f"Remesh noise |Δσ|/σ_ref = {abs_rel_pct:.3f}% is numerically "
                f"smaller than bump_Cinf |Δ|={bump_abs_pct:.2f}% "
                f"(ratio≈{abs_rel_pct / bump_abs_pct:.2f}), but NOT in the "
                "<< 0.3–0.5% example band. Gray zone vs the brief's two poles: "
                "not clearly 'genuine falloff', not clearly 'indistinguishable "
                "from remesh'. Human judgment needed; n=1 remains fragile either way. "
                "Caveat: bump_Cinf vs cosine_C1 compared two deforms on the *same* "
                "locked mesh (no remesh between them) — this floor bounds "
                "independent-remesh label noise, not the mesh difference inside "
                "that specific A/B."
            )
        else:
            verdict_band = "noise_comparable_or_larger_than_bump_delta"
            verdict_text = (
                f"Remesh noise |Δσ|/σ_ref = {abs_rel_pct:.3f}% is comparable to or "
                f"larger than bump_Cinf |Δ|={bump_abs_pct:.2f}%. With n=1 per "
                "condition, the bump delta cannot be distinguished from remesh "
                "noise — strong signal to prefer Phase-3 (surrogate + many samples) "
                "over more manual falloff probes."
            )

    comparison = {
        "probe": {
            "s": S,
            "y0_mm": Y0,
            "L_mm": L,
            "falloff_kind": FALLOFF,
            "method": (
                "deform_locked_baseline → extract_surface → "
                "gmsh_MeshOnlyEmpty remesh → FEA (reuse_fea=False)"
            ),
            "status": status,
            "fail_reasons": sample.get("fail_reasons") or [],
            "sigma_max_MPa": sigma,
            "dV_frac_vs_baseline_V0": dV_frac_r,
            "dV_frac_pre_remesh": dV_frac_pre,
            "V0_mm3": V0,
            "V_remeshed_mm3": V_r,
            "mesh_gate_note": (
                "mesh_gates on deformed-then-remeshed solid use baseline V0 for "
                "dV_frac reporting only; FEA uses dens0 from baseline mass/volume."
            ),
            "reaction_pass": (sample.get("reaction") or {}).get("pass"),
            "band_position": sample.get("band_position"),
            "remesh": {
                "n_nodes": mesh_info["n_nodes"],
                "n_tets": mesh_info["n_tets"],
                "mesh_size_mm": mesh_size,
                "mesh_size_min_mm": mesh_size_min,
                "baseline_n_nodes": mesh_info["baseline_n_nodes"],
                "baseline_n_tets": mesh_info["baseline_n_tets"],
                "nodes_changed": mesh_info["n_nodes"] != mesh_info["baseline_n_nodes"],
                "tets_changed": mesh_info["n_tets"] != mesh_info["baseline_n_tets"],
            },
            "wall_time_s": wall,
        },
        "reference_cosine_C1": {
            "source": "pilot2d sc0.90_ss1.00 / Phase-1 s=0.90",
            "path": str(REF_COSINE.relative_to(ROOT)),
            "sigma_max_MPa": REF_SIGMA,
            "dV_frac": REF_DV_FRAC,
            "note": (
                "Reference used deform on optimized geometry.npz without "
                "per-sample remesh. This probe remeshes the deformed solid."
            ),
        },
        "remesh_noise_floor": {
            "delta_sigma_max_MPa": abs_delta,
            "delta_sigma_rel": rel_delta,
            "delta_sigma_rel_pct": None if rel_delta is None else rel_delta * 100.0,
            "abs_delta_sigma_rel_pct": None
            if rel_delta is None
            else abs(rel_delta) * 100.0,
        },
        "vs_bump_Cinf_probe_delta": {
            "bump_delta_sigma_rel": BUMP_DELTA_REL,
            "bump_delta_sigma_rel_pct": BUMP_DELTA_REL * 100.0,
            "remesh_abs_rel_pct_over_bump_abs_rel_pct": (
                None
                if rel_delta is None
                else abs(rel_delta) / abs(BUMP_DELTA_REL)
            ),
        },
        "interpretation": {
            "band": verdict_band,
            "thresholds_pct": {
                "much_smaller_than": 0.3,
                "below_half_percent": 0.5,
                "bump_Cinf_abs_delta_pct": abs(BUMP_DELTA_REL) * 100.0,
            },
            "text": verdict_text,
            "decision_note": (
                "Reported for human judgment — script does not unilaterally stop "
                "or continue falloff exploration."
            ),
        },
        "weight_stats_optional": weight_compare,
        "baseline_sigma_max_MPa": BASELINE_SIGMA_MAX_MPA,
    }

    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (OUT / "deform_meta.json").write_text(
        json.dumps(
            {
                **dmeta,
                "note": "Weight stats from pre-remesh deform on locked baseline mesh",
                "remesh_info": {
                    "n_nodes": mesh_info["n_nodes"],
                    "n_tets": mesh_info["n_tets"],
                },
            },
            indent=2,
        )
    )

    print(f"\nstatus={status}  sigma={sigma}  dV_frac_remeshed={dV_frac_r}  wall={wall:.1f}s")
    print(
        f"remesh noise vs ref: Δσ={abs_delta}  σ_rel={rel_delta}  "
        f"|σ_rel|%={comparison['remesh_noise_floor']['abs_delta_sigma_rel_pct']}"
    )
    print(f"interpretation band: {verdict_band}")
    print(verdict_text)
    if weight_compare is not None:
        print(
            "weight w>0.5: cosine_C1_ref="
            f"{weight_compare['cosine_C1_ref_pilot2d']['n_nodes_w_gt_0_5']}  "
            f"bump_Cinf={weight_compare['bump_Cinf_probe']['n_nodes_w_gt_0_5']}  "
            f"Δ={weight_compare['delta_bump_minus_cosine_n_w_gt_0_5']}"
        )
    print(f"wrote {OUT / 'comparison.json'}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
