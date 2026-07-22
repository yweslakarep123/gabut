#!/usr/bin/env python3
"""Bandingkan baseline vs Gmsh default-optimize; apply bila rho lebih baik & volume ~0%.

Jalankan dari root repo (env simjeb):

  /home/daffa/miniforge3/envs/simjeb/bin/python -u \\
    scripts/apply_default_optimize_if_better.py

Tidak remesh — hanya baca geometry.npz + _opt_default.msh yang sudah ada.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import gmsh
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simjeb.mesh_ur_volume import _surface_from_tets, _tet_quality  # noqa: E402

OUT = ROOT / "reports" / "ur5e_nauo3_volume"
MSH = OUT / "_opt_default.msh"
NPZ = OUT / "geometry.npz"


def _load_msh(path: Path):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(str(path))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        all_tets = []
        for etype, nodes in zip(etypes, enodes):
            if int(etype) != 4:
                continue
            n = np.asarray(nodes, dtype=np.int64).reshape(-1, 4)
            remapped = np.vectorize(lambda x: tag_to_idx[int(x)])(n)
            all_tets.append(remapped)
        if not all_tets:
            raise RuntimeError(f"tidak ada tet di {path}")
        return coords, np.vstack(all_tets).astype(np.int32)
    finally:
        gmsh.finalize()


def _rho_pack(q: dict) -> dict:
    return {
        "n_tets": q["n_tets"],
        "n_inverted": q["n_inverted_or_neg_volume"],
        "n_min_dihedral_lt_5deg": q["n_min_dihedral_lt_5deg"],
        "n_min_dihedral_lt_1deg": q["n_min_dihedral_lt_1deg"],
        "n_radius_ratio_gt_10": q["n_radius_ratio_gt_10"],
        "vol": q["volume_mm3"]["sum_abs"],
        "rho_p50": q["radius_ratio_R_over_3r"]["p50"],
        "rho_p95": q["radius_ratio_R_over_3r"]["p95"],
        "rho": q["radius_ratio_R_over_3r"],
        "dih": q["min_dihedral_deg"],
    }


def main() -> int:
    if not MSH.exists():
        print(f"ERROR: {MSH} tidak ada. Generate dulu dengan Gmsh optimize default.")
        return 1

    g = np.load(NPZ)
    pts0 = np.asarray(g["vol_points"], dtype=np.float64)
    tets0 = np.asarray(g["vol_tets"], dtype=np.int32)
    q0 = _tet_quality(pts0, tets0)
    b = _rho_pack(q0)
    b["n_nodes"] = int(len(pts0))

    pts_d, tets_d = _load_msh(MSH)
    q_d = _tet_quality(pts_d, tets_d)
    d = _rho_pack(q_d)
    d["n_nodes"] = int(len(pts_d))

    dvol = (d["vol"] - b["vol"]) / b["vol"] * 100.0
    rho_better = (d["rho_p50"] < b["rho_p50"]) or (d["rho_p95"] < b["rho_p95"])
    rho_not_worse = (d["rho_p50"] <= b["rho_p50"] * 1.001) and (
        d["rho_p95"] <= b["rho_p95"] * 1.01
    )
    vol_ok = abs(dvol) < 0.5
    no_invert = d["n_inverted"] == 0
    dih_ok = (
        d["n_min_dihedral_lt_5deg"] <= b["n_min_dihedral_lt_5deg"]
        and d["n_min_dihedral_lt_1deg"] <= b["n_min_dihedral_lt_1deg"]
    )
    use = rho_better and rho_not_worse and vol_ok and no_invert and dih_ok

    print("=== baseline ===")
    print(
        f"  nodes={b['n_nodes']} tets={b['n_tets']} vol={b['vol']:.3f} "
        f"rho_p50={b['rho_p50']:.6f} rho_p95={b['rho_p95']:.6f} "
        f"lt5={b['n_min_dihedral_lt_5deg']} lt1={b['n_min_dihedral_lt_1deg']} "
        f"rho>10={b['n_radius_ratio_gt_10']}"
    )
    print("=== default optimize ===")
    print(
        f"  nodes={d['n_nodes']} tets={d['n_tets']} vol={d['vol']:.3f} "
        f"rho_p50={d['rho_p50']:.6f} rho_p95={d['rho_p95']:.6f} "
        f"lt5={d['n_min_dihedral_lt_5deg']} lt1={d['n_min_dihedral_lt_1deg']} "
        f"rho>10={d['n_radius_ratio_gt_10']}"
    )
    print(
        f"dV%={dvol:+.6e} rho_better={rho_better} vol_ok={vol_ok} "
        f"no_invert={no_invert} dih_ok={dih_ok} USE_DEFAULT={use}"
    )

    decision = {
        "baseline": {k: b[k] for k in b if k != "rho" and k != "dih"},
        "baseline_rho": b["rho"],
        "default_optimize": {k: d[k] for k in d if k != "rho" and k != "dih"},
        "default_rho": d["rho"],
        "volume_change_pct": dvol,
        "decision": "use_default_optimize" if use else "keep_pre_optimize",
        "reason": (
            "default optimize: |dV|~0%, 0 inverted, dihedral tidak memburuk, "
            "radius_ratio p50 dan/atau p95 lebih baik — perbaikan gratis."
            if use
            else (
                "default optimize tidak memenuhi kriteria "
                "(rho p50/p95 harus lebih baik atau setara-ketat, |dV|<0.5%, "
                "0 inverted, dihedral tidak memburuk)."
            )
        ),
    }

    with open(OUT / "optimize_radius_ratio_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    report_path = OUT / "mesh_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    if use:
        shutil.copy(NPZ, OUT / "geometry_pre_optimize.npz")
        surf_v, surf_f, node_surf = _surface_from_tets(pts_d, tets_d)
        np.savez_compressed(
            NPZ,
            vol_points=pts_d.astype(np.float64),
            vol_tets=tets_d.astype(np.int32),
            node_surf=node_surf,
            surf_vertices=surf_v.astype(np.float64),
            surf_faces=surf_f.astype(np.int32),
        )
        report["quality"] = q_d
        report.setdefault("mesh", {})
        report["mesh"]["n_nodes"] = int(len(pts_d))
        report["mesh"]["n_tets"] = int(len(tets_d))
        report["surface"] = {
            "n_surf_vertices": int(len(surf_v)),
            "n_surf_faces": int(len(surf_f)),
            "n_node_surf": int(int(node_surf.sum())),
        }
        report["optimize"] = {
            "applied": "default_5",
            "source_msh": str(MSH),
            "comparison_radius_ratio": decision,
        }
        print(f"WROTE {NPZ} (backup: geometry_pre_optimize.npz)")
    else:
        report["optimize"] = {
            "applied": None,
            "comparison_radius_ratio": decision,
        }
        print("KEPT pre-optimize baseline — lihat reason di decision JSON / README")

    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT / 'optimize_radius_ratio_decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
