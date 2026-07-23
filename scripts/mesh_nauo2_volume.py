#!/usr/bin/env python3
"""Volume tet-mesh NAUO2 (shoulder) → geometry.npz.

Path yang terbukti (berbeda dari NAUO3 default yang mentok PLC):
  ShapeFix BRep → tess deflection 0.5 → iso-clean (standar) →
  ekstra dilate-clean hingga self=0 → Gmsh MeshOnlyEmpty @ 5 mm.

  PYTHONPATH=src python -u scripts/mesh_nauo2_volume.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pymeshlab as ml
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simjeb.heal_ur_solid import (  # noqa: E402
    _mesh_metrics,
    _read_step_with_names,
    _shape_fix_brep,
    _weld,
    match_parts,
    tessellate_solid,
)
from simjeb.mesh_ur_volume import (  # noqa: E402
    _extract_tets_from_gmsh,
    _iso_remesh_and_clear_self_intersections,
    _surface_from_tets,
    _tet_quality,
)

OUT = ROOT / "reports" / "ur5e_nauo2_volume"
STEP_SRC = ROOT / "data" / "raw" / "ur5e" / "UR7e.step"
MESH_SIZE = 5.0
DEFLECTION = 0.5


def self_count(path: Path) -> int:
    ms = ml.MeshSet()
    ms.load_new_mesh(str(path))
    ms.compute_selection_by_self_intersections_per_face()
    return int(ms.current_mesh().selected_face_number())


def extra_dilate_until_clean(stl: Path, max_iters: int = 30) -> dict:
    """Lanjutkan dilate-clean bila iso-clean standar masih menyisakan self>0."""
    ms = ml.MeshSet()
    ms.load_new_mesh(str(stl))
    hist: list[int] = []
    for i in range(max_iters):
        ms.compute_selection_by_self_intersections_per_face()
        n = int(ms.current_mesh().selected_face_number())
        hist.append(n)
        print(f"  extra_dilate[{i}] self={n}")
        if n == 0:
            break
        for _ in range(5):
            try:
                ms.apply_selection_dilatation()
            except Exception:
                break
        ms.meshing_remove_selected_faces()
        try:
            ms.meshing_repair_non_manifold_edges(method="Remove Faces")
        except Exception:
            pass
        try:
            ms.meshing_close_holes(maxholesize=100000)
        except Exception:
            pass
    ms.save_current_mesh(str(stl))
    mesh = _weld(trimesh.load(str(stl), force="mesh", process=False))
    comps = mesh.split(only_watertight=False)
    mesh = max(comps, key=lambda c: abs(c.volume) if c.is_volume else len(c.faces))
    mesh = _weld(mesh)
    mesh.export(stl)
    return {
        "history": hist,
        "n_self_final": self_count(stl),
        "watertight": bool(mesh.is_watertight),
        "euler": int(mesh.euler_number),
        "n_faces": int(len(mesh.faces)),
        "volume_mm3": float(mesh.volume),
    }


def mesh_clean_stl(clean: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    import gmsh

    print(f"MeshOnlyEmpty self={self_count(clean)} …")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("nauo2")
    try:
        gmsh.merge(str(clean))
        try:
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.0)
        except Exception:
            pass
        gmsh.model.mesh.classifySurfaces(math.pi, True, False)
        gmsh.model.mesh.createTopology()
        surfs = [s[1] for s in gmsh.model.getEntities(2)]
        if not gmsh.model.getEntities(3):
            sl = gmsh.model.geo.addSurfaceLoop(surfs)
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        points, tets = _extract_tets_from_gmsh(gmsh)
        if len(tets) == 0:
            raise RuntimeError("0 tets")
        return points, tets, {
            "source": "heal_stl_iso_clean_extra_meshonly",
            "mesh_size_mm": MESH_SIZE,
            "mesh_size_min_mm": MESH_SIZE * 0.5,
            "n_nodes": int(len(points)),
            "n_tets": int(len(tets)),
            "surface_stl": str(clean),
            "n_self_final": self_count(clean),
        }
    finally:
        gmsh.finalize()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = _read_step_with_names(STEP_SRC)
    picked = match_parts(parts, names_substr=["NAUO2"], indices=None)
    pref = [t for t in picked if "NAUO2" in t[1] and "solid1" in t[1]]
    idx, name, solid = (pref or picked)[0]
    print(f"target #{idx} {name}")

    fixed = _shape_fix_brep(solid)
    mesh, stats = tessellate_solid(fixed, DEFLECTION, clean=True)
    metrics = _mesh_metrics(mesh)
    tess_stl = OUT / f"_tess_defl_{DEFLECTION}.stl"
    mesh.export(tess_stl)
    print(
        f"tess defl={DEFLECTION}: wt={metrics['watertight']} "
        f"euler={metrics['euler_number']} faces={metrics['n_faces']} "
        f"self={self_count(tess_stl)}"
    )
    if not metrics["watertight"] or abs(metrics["euler_number"] - 2) > 2:
        raise RuntimeError("tess ShapeFix tidak watertight/euler≈2")

    target_vol = float(stats.get("volume_mm3") or mesh.volume)
    clean = OUT / "_heal_iso_clean.stl"
    print("iso-remesh + clear self-intersections …")
    repair = _iso_remesh_and_clear_self_intersections(
        tess_stl, clean, mesh_size=MESH_SIZE, max_iters=15
    )
    print(
        f"  after iso-clean: self={repair['n_self_final']} "
        f"wt={repair['watertight']} euler={repair['euler']} "
        f"faces={repair['n_faces']}"
    )
    extra = None
    if repair["n_self_final"] != 0:
        print("extra dilate-clean …")
        extra = extra_dilate_until_clean(clean)
        if extra["n_self_final"] != 0 or not extra["watertight"]:
            raise RuntimeError(f"surface belum PLC-aman: {extra}")

    points, tets, info = mesh_clean_stl(clean)
    ab = points[tets[:, 1]] - points[tets[:, 0]]
    ac = points[tets[:, 2]] - points[tets[:, 0]]
    ad = points[tets[:, 3]] - points[tets[:, 0]]
    signed = np.einsum("ij,ij->i", ab, np.cross(ac, ad))
    if (signed < 0).sum() > (signed > 0).sum():
        tets = tets[:, [0, 2, 1, 3]]
        print("flip tet connectivity")

    surf_v, surf_f, node_surf = _surface_from_tets(points, tets)
    quality = _tet_quality(points, tets)
    vol = float(np.abs(signed).sum() / 6.0)
    info.update(
        {
            "volume_mm3_from_tets": vol,
            "target_brep_volume_mm3": target_vol,
            "volume_ratio": vol / target_vol,
            "repair": repair,
            "extra_dilate": extra,
        }
    )

    geom = OUT / "geometry.npz"
    np.savez(
        geom,
        surf_vertices=surf_v.astype(np.float64),
        surf_faces=surf_f.astype(np.int32),
        vol_points=points.astype(np.float64),
        vol_tets=tets.astype(np.int32),
        node_surf=node_surf.astype(bool),
    )
    report = {
        "part": "NAUO2",
        "urdf_link": "shoulder_link",
        "index": idx,
        "name": name,
        "geometry_npz": str(geom.resolve()),
        "mesh": info,
        "surface": {
            "n_surf_vertices": int(len(surf_v)),
            "n_surf_faces": int(len(surf_f)),
            "n_node_surf": int(node_surf.sum()),
        },
        "quality": quality,
        "note": (
            "NAUO2 butuh ekstra dilate-clean setelah iso-clean standar "
            "(logo/flange → residual self-intersect). Jangan pakai remesh "
            "kasar ≫5 mm (volume drop besar)."
        ),
    }
    (OUT / "mesh_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"tersimpan: {geom}")
    print(
        f"nodes={info['n_nodes']} tets={info['n_tets']} "
        f"V_ratio={info['volume_ratio']:.4f} "
        f"inverted={quality['n_inverted_or_neg_volume']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
