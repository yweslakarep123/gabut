#!/usr/bin/env python3
"""Single-sample falloff probe: bump_Cinf at locked (s=0.90, y0=262, L=40).

Compares sigma_max and dV_frac to the existing cosine_C1 result at the same
(s, y0, L) — sc0.90_ss1.00 from the 2D pilot (shoulder no-op).

  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
      scripts/probe_nauo3_falloff_bump_cinf.py
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
    BASELINE_SIGMA_MAX_MPA,
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

OUT = ROOT / "reports" / "ur5e_nauo3_falloff_probe" / "bump_cinf_s090"
REF_COSINE = (
    ROOT
    / "reports"
    / "ur5e_nauo3_neck_pilot2d"
    / "sc0.90_ss1.00"
    / "sample_result.json"
)

# Locked — identical to sc0.90_ss1.00 / Phase-1 s=0.90
S = 0.90
Y0 = Y0_NECK  # 262
L = L_BAND  # 40
FALLOFF = "bump_Cinf"

# Reference numbers from the user / pilot2d sc0.90_ss1.00
REF_SIGMA = 2.271612960162118
REF_DV_FRAC = -0.01029144980632467


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    g = np.load(MESH_BASE)
    points0 = np.asarray(g["vol_points"], dtype=np.float64).copy()
    tets = np.asarray(g["vol_tets"], dtype=np.int32)
    node_surf = np.asarray(g["node_surf"], dtype=bool)
    surf_faces = np.asarray(g["surf_faces"], dtype=np.int32)

    tet_ids, face_ids, face_nodes = _boundary_faces(tets)
    ends0 = select_end_faces(points0, node_surf, tet_ids, face_ids, face_nodes)
    clear0 = spc_band_clearance_report(points0, ends0, L=L)
    if not clear0["safe_clearance"]:
        print("STOP: SPC/band clearance failed")
        return 2

    vols0 = np.abs(_tet_volume_sign(points0, tets))
    V0 = float(vols0.sum())
    dens0_t_mm3 = (M_SELF / (V0 * 1e-9)) * 1e-12
    axis_point = points0.mean(0)
    axis_u = ends0["u"]

    print("=== falloff probe: bump_Cinf @ locked (s,y0,L) ===")
    print(f"s={S}  y0={Y0}  L={L}  falloff_kind={FALLOFF}")
    print(f"ref cosine_C1: sigma={REF_SIGMA}  dV_frac={REF_DV_FRAC}")
    print(f"out → {OUT}")

    t0 = time.perf_counter()
    sample = run_one_sample(
        S,
        points0,
        tets,
        node_surf,
        surf_faces,
        dens0_t_mm3,
        axis_point,
        axis_u,
        clear0,
        reuse_fea=False,
        tag="bump_cinf_s090",
        out_root=OUT.parent,  # writes OUT.parent/bump_cinf_s090/
        falloff_kind=FALLOFF,
    )
    wall = time.perf_counter() - t0

    # Ensure artefacts land exactly at the requested path if tag/root differ
    # (tag=bump_cinf_s090, out_root=.../ur5e_nauo3_falloff_probe → OUT)
    gate = sample.get("mesh_gate") or {}
    V = gate.get("V_mm3")
    dV_frac = None if V is None else (float(V) - V0) / V0
    sigma = sample.get("sigma_max_MPa")
    status = sample.get("status")

    ref_on_disk = None
    if REF_COSINE.is_file():
        ref_on_disk = json.loads(REF_COSINE.read_text())

    comparison = {
        "probe": {
            "s": S,
            "y0_mm": Y0,
            "L_mm": L,
            "falloff_kind": FALLOFF,
            "status": status,
            "fail_reasons": sample.get("fail_reasons") or [],
            "sigma_max_MPa": sigma,
            "dV_frac": dV_frac,
            "V_mm3": V,
            "V0_mm3": V0,
            "mesh_gate_pass": gate.get("pass"),
            "reaction_pass": (sample.get("reaction") or {}).get("pass"),
            "band_position": sample.get("band_position"),
            "wall_time_s": wall,
        },
        "reference_cosine_C1": {
            "source": "pilot2d sc0.90_ss1.00 / Phase-1 s=0.90",
            "path": str(REF_COSINE.relative_to(ROOT)) if REF_COSINE.is_file() else None,
            "sigma_max_MPa": REF_SIGMA,
            "dV_frac": REF_DV_FRAC,
            "on_disk_sigma_max_MPa": (ref_on_disk or {}).get("sigma_max_MPa"),
            "on_disk_status": (ref_on_disk or {}).get("status"),
        },
        "delta": {
            "sigma_max_MPa": None
            if sigma is None
            else float(sigma) - REF_SIGMA,
            "sigma_rel": None
            if sigma is None
            else (float(sigma) / REF_SIGMA) - 1.0,
            "dV_frac": None if dV_frac is None else float(dV_frac) - REF_DV_FRAC,
        },
        "baseline_sigma_max_MPa": BASELINE_SIGMA_MAX_MPA,
    }

    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2))
    # sample_result already written by run_one_sample under OUT
    print(f"\nstatus={status}  sigma={sigma}  dV_frac={dV_frac}  wall={wall:.1f}s")
    print(
        f"vs cosine_C1: Δsigma={comparison['delta']['sigma_max_MPa']}  "
        f"σ_rel={comparison['delta']['sigma_rel']}  "
        f"Δ(dV_frac)={comparison['delta']['dV_frac']}"
    )
    print(f"wrote {OUT / 'comparison.json'}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
