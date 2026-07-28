#!/usr/bin/env python3
"""Execute revised falloff×s sampling plan (13 new in-place FEA runs).

Plan: reports/ur5e_nauo3_sampling_plan_proposal.md (revised).

  (b) bump_Cinf @ {0.94,0.98,1.00,1.10}           → 4
      smootherstep_C2 @ {0.90,0.94,0.98,1.00,1.10} → 5
  (a) cosine_C1 @ {0.86,0.88,1.12,1.14}            → 4
  Total new = 13. No reproducibility re-runs. No shoulder. No remesh.

  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/batch_nauo3_falloff_s_grid.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from batch_nauo3_neck_scale import (  # noqa: E402
    L_BAND,
    MESH_BASE,
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

OUT_ROOT = ROOT / "reports" / "ur5e_nauo3_falloff_s_grid"
PHASE1 = ROOT / "reports" / "ur5e_nauo3_neck_batch"
FALLOFF_PROBE = ROOT / "reports" / "ur5e_nauo3_falloff_probe"

# (purpose, falloff_kind, s, tag, skip_if_exists_path_or_None)
# skip: if set and sample_result exists with matching params, do not re-run.
JOBS: list[tuple[str, str, float, str, Path | None]] = [
    # (b) bump — 0.90 already in falloff_probe
    ("b_falloff_consistency", "bump_Cinf", 0.94, "bump_cinf_s094", None),
    ("b_falloff_consistency", "bump_Cinf", 0.98, "bump_cinf_s098", None),
    ("b_falloff_consistency", "bump_Cinf", 1.00, "bump_cinf_s100", None),
    ("b_falloff_consistency", "bump_Cinf", 1.10, "bump_cinf_s110", None),
    # (b) smootherstep — all new
    ("b_falloff_consistency", "smootherstep_C2", 0.90, "smootherstep_c2_s090", None),
    ("b_falloff_consistency", "smootherstep_C2", 0.94, "smootherstep_c2_s094", None),
    ("b_falloff_consistency", "smootherstep_C2", 0.98, "smootherstep_c2_s098", None),
    ("b_falloff_consistency", "smootherstep_C2", 1.00, "smootherstep_c2_s100", None),
    ("b_falloff_consistency", "smootherstep_C2", 1.10, "smootherstep_c2_s110", None),
    # (a) cosine extensions
    ("a_corpus_s_extension", "cosine_C1", 0.86, "cosine_c1_s086", None),
    ("a_corpus_s_extension", "cosine_C1", 0.88, "cosine_c1_s088", None),
    ("a_corpus_s_extension", "cosine_C1", 1.12, "cosine_c1_s112", None),
    ("a_corpus_s_extension", "cosine_C1", 1.14, "cosine_c1_s114", None),
]

# Existing references for delta table (not re-run)
COSINE_REF_S = [0.90, 0.94, 0.98, 1.00, 1.10]
BUMP_EXISTING = FALLOFF_PROBE / "bump_cinf_s090" / "sample_result.json"


def _phase1_path(s: float) -> Path:
    return PHASE1 / f"s_{s:.2f}" / "sample_result.json"


def _load_sigma(path: Path) -> float | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    return d.get("sigma_max_MPa")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    g = np.load(MESH_BASE)
    points0 = np.asarray(g["vol_points"], dtype=np.float64).copy()
    tets = np.asarray(g["vol_tets"], dtype=np.int32)
    node_surf = np.asarray(g["node_surf"], dtype=bool)
    surf_faces = np.asarray(g["surf_faces"], dtype=np.int32)

    tet_ids, face_ids, face_nodes = _boundary_faces(tets)
    ends0 = select_end_faces(points0, node_surf, tet_ids, face_ids, face_nodes)
    clear0 = spc_band_clearance_report(points0, ends0, L=L_BAND)
    if not clear0["safe_clearance"]:
        print("STOP: SPC/band clearance failed")
        return 2

    vols0 = np.abs(_tet_volume_sign(points0, tets))
    V0 = float(vols0.sum())
    dens0_t_mm3 = (M_SELF / (V0 * 1e-9)) * 1e-12
    axis_point = points0.mean(0)
    axis_u = ends0["u"]

    print("=== falloff×s grid batch (revised plan) ===")
    print(f"mesh={MESH_BASE.relative_to(ROOT)}  V0={V0:.3f}  n_jobs={len(JOBS)}")
    print(f"out → {OUT_ROOT.relative_to(ROOT)}")
    assert len(JOBS) == 13, len(JOBS)

    rows = []
    t_batch = time.perf_counter()
    for i, (purpose, kind, s, tag, _skip) in enumerate(JOBS, start=1):
        out = OUT_ROOT / tag
        existing = out / "sample_result.json"
        if existing.is_file():
            sample = json.loads(existing.read_text())
            print(
                f"[{i:02d}/13] REUSE {tag}  status={sample.get('status')}  "
                f"sigma={sample.get('sigma_max_MPa')}"
            )
            reused = True
            wall = 0.0
        else:
            print(f"[{i:02d}/13] RUN   {tag}  kind={kind}  s={s:.2f}  purpose={purpose}")
            t0 = time.perf_counter()
            sample = run_one_sample(
                s,
                points0,
                tets,
                node_surf,
                surf_faces,
                dens0_t_mm3,
                axis_point,
                axis_u,
                clear0,
                reuse_fea=False,
                tag=tag,
                out_root=OUT_ROOT,
                falloff_kind=kind,
            )
            wall = time.perf_counter() - t0
            reused = False
            print(
                f"         status={sample.get('status')}  "
                f"sigma={sample.get('sigma_max_MPa')}  "
                f"band={sample.get('band_position')}  wall={wall:.1f}s"
            )

        hot = sample.get("hotspot") or {}
        gate = sample.get("mesh_gate") or {}
        react = sample.get("reaction") or {}
        rows.append(
            {
                "purpose": purpose,
                "tag": tag,
                "falloff_kind": kind,
                "s": s,
                "y0_mm": Y0_NECK,
                "L_mm": L_BAND,
                "status": sample.get("status"),
                "sigma_max_MPa": sample.get("sigma_max_MPa"),
                "dV_frac": gate.get("dV_frac"),
                "mesh_gate_pass": gate.get("pass"),
                "reaction_pass": react.get("pass"),
                "band_position": sample.get("band_position") or hot.get("band_position"),
                "hotspot_node_0based": hot.get("node_0based"),
                "hotspot_xyz_mm": hot.get("xyz_mm"),
                "sample_result_path": str((OUT_ROOT / tag / "sample_result.json").relative_to(ROOT)),
                "frd_present": (OUT_ROOT / tag / "nauo3_cantilever.frd").is_file(),
                "reused_existing_in_out_root": reused,
                "wall_time_s": wall,
            }
        )

    # Purpose (b) delta table vs cosine at shared s
    deltas = []
    for s in COSINE_REF_S:
        cos_sigma = _load_sigma(_phase1_path(s))
        entry = {"s": s, "cosine_C1_sigma_MPa": cos_sigma, "vs_cosine": {}}
        # bump
        if abs(s - 0.90) < 1e-12:
            bump_path = BUMP_EXISTING
        else:
            bump_path = OUT_ROOT / f"bump_cinf_s{int(round(s * 100)):03d}" / "sample_result.json"
        bump_sigma = _load_sigma(bump_path)
        if cos_sigma is not None and bump_sigma is not None:
            entry["vs_cosine"]["bump_Cinf"] = {
                "sigma_MPa": bump_sigma,
                "delta_rel_pct": 100.0 * (bump_sigma - cos_sigma) / cos_sigma,
                "path": str(bump_path.relative_to(ROOT)),
            }
        sm_path = OUT_ROOT / f"smootherstep_c2_s{int(round(s * 100)):03d}" / "sample_result.json"
        sm_sigma = _load_sigma(sm_path)
        if cos_sigma is not None and sm_sigma is not None:
            entry["vs_cosine"]["smootherstep_C2"] = {
                "sigma_MPa": sm_sigma,
                "delta_rel_pct": 100.0 * (sm_sigma - cos_sigma) / cos_sigma,
                "path": str(sm_path.relative_to(ROOT)),
            }
        deltas.append(entry)

    summary = {
        "plan": "reports/ur5e_nauo3_sampling_plan_proposal.md",
        "n_jobs_planned": 13,
        "n_jobs_run_or_reused": len(rows),
        "n_ok": sum(1 for r in rows if r["status"] == "ok"),
        "n_failed": sum(1 for r in rows if r["status"] != "ok"),
        "wall_time_batch_s": time.perf_counter() - t_batch,
        "methodology": "in_place_deform_locked_geometry_npz",
        "no_reproducibility_reruns": True,
        "jobs": rows,
        "purpose_b_falloff_delta_vs_cosine": deltas,
        "reuse_not_in_this_batch": {
            "cosine_C1_phase1_s090_to_s110": 11,
            "bump_Cinf_s090": str(BUMP_EXISTING.relative_to(ROOT)),
            "gaussian_nocut_gauss35_corpus_only": [
                "reports/ur5e_nauo3_neck_batch/s_0.90_gauss35/",
                "reports/ur5e_nauo3_neck_batch/s_1.10_gauss35/",
            ],
        },
    }
    out_sum = OUT_ROOT / "batch_summary.json"
    out_sum.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_sum.relative_to(ROOT)}")
    print(f"ok={summary['n_ok']} failed={summary['n_failed']} wall={summary['wall_time_batch_s']:.1f}s")
    print("\nPurpose (b) delta vs cosine (%):")
    for d in deltas:
        parts = [f"s={d['s']:.2f}"]
        for k, v in d["vs_cosine"].items():
            parts.append(f"{k}={v['delta_rel_pct']:+.3f}%")
        print("  " + "  ".join(parts))
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
