"""Heal NAUO3 (atau solid lain) berurutan; berhenti saat target tercapai.

Urutan:
  1. trimesh.repair (fill_holes / fix_normals / fix_winding) pada mesh tessellasi
  2. ShapeFix BRep (OpenCASCADE) lalu tessellasi ulang
  3. (opsional) PyMeshLab — hanya jika terpasang; kalau tidak, instruksi manual

Target default: watertight + |euler - 2| <= tol (genus-0).

Metrik euler di laporan heal ini SELALU dihitung setelah ``_weld``
(``merge_vertices`` + hapus face degenerate/duplikat). Itu beda tahap dari
angka ``euler=1470`` di ``retessellate_ur_solid`` / inspeksi mentah: di sana
tessellasi OCCT per-face diekspor tanpa weld, sehingga setiap seam antar-face
jadi boundary palsu dan Euler membengkak. Bukan kontradiksi — preprocessing
berbeda. Contoh NAUO3 deflection 0,5 mm: mentah ≈1470; setelah weld baseline
heal ≈10 (belum watertight); setelah drop void mikro + ShapeFix → watertight
euler=2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .inspect_ur_step import (
    _require_cq,
    _read_step_with_names,
    match_parts,
    tessellate_solid,
)


def _weld(mesh):
    """Gabungkan vertex hampir-sama supaya euler/watertight bermakna.

    Tessellasi per-face OCCT dan export STL sering punya vertex duplikat di
    seam → boundary_edges palsu & euler membengkak.
    """
    if mesh is None:
        return None
    m = mesh.copy()
    m.merge_vertices()
    # trimesh 4.x: mask degenerate, lalu unique face rows
    try:
        mask = m.nondegenerate_faces()
        m.update_faces(mask)
    except Exception:
        pass
    try:
        # unique faces
        _, inv = np.unique(np.sort(m.faces, axis=1), axis=0, return_inverse=True)
        # keep first occurrence of each sorted face
        keep = np.zeros(len(m.faces), dtype=bool)
        seen = set()
        for i, key in enumerate(map(tuple, np.sort(m.faces, axis=1))):
            if key not in seen:
                seen.add(key)
                keep[i] = True
        m.update_faces(keep)
    except Exception:
        pass
    m.remove_unreferenced_vertices()
    return m


def _mesh_metrics(mesh) -> dict:
    if mesh is None:
        return {
            "watertight": None,
            "is_volume": None,
            "euler_number": None,
            "n_faces": 0,
            "n_vertices": 0,
            "boundary_edges": None,
            "winding_consistent": None,
        }
    from collections import Counter

    m = _weld(mesh)
    ec = Counter(map(tuple, map(tuple, m.edges_sorted)))
    boundary = int(sum(1 for _, c in ec.items() if c == 1))
    return {
        "watertight": bool(m.is_watertight),
        "is_volume": bool(m.is_volume),
        "euler_number": int(m.euler_number),
        "n_faces": int(len(m.faces)),
        "n_vertices": int(len(m.vertices)),
        "boundary_edges": boundary,
        "winding_consistent": bool(m.is_winding_consistent),
    }


def _hits_target(metrics: dict, target_euler: int, euler_tol: int) -> bool:
    if not metrics.get("watertight"):
        return False
    e = metrics.get("euler_number")
    if e is None:
        return False
    return abs(int(e) - target_euler) <= euler_tol


def _step_trimesh_repair(mesh, out_stl: Path | None) -> tuple[object, dict]:
    from trimesh import repair

    m = _weld(mesh)
    try:
        repair.fix_winding(m)
    except Exception as e:
        print(f"  fix_winding: {e}")
    try:
        repair.fix_normals(m)
    except Exception as e:
        print(f"  fix_normals: {e}")
    try:
        filled = repair.fill_holes(m)
        print(f"  fill_holes returned: {filled}")
    except Exception as e:
        print(f"  fill_holes: {e}")
    m = _weld(m)
    metrics = _mesh_metrics(m)
    if out_stl is not None:
        out_stl.parent.mkdir(parents=True, exist_ok=True)
        m.export(out_stl)
        print(f"  STL → {out_stl}")
    return m, metrics


def _shell_abs_volume(shell) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    try:
        BRepGProp.VolumeProperties_s(shell, props)
        return abs(float(props.Mass()))
    except Exception:
        return 0.0


def _drop_micro_void_shells(solid, min_keep_ratio: float = 1e-4):
    """Buang shell void mikro (artefak STEP); pertahankan shell luar terbesar.

    NAUO3 punya 1 shell luar (~4.87e6 mm³) + 8 void ~0.1–0.3 mm³.
    Tanpa ini, mesh watertight tetap euler=2*(1+n_void).
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid

    shells = []
    exp = TopExp_Explorer(solid, TopAbs_SHELL)
    while exp.More():
        sh = TopoDS.Shell(exp.Current())
        nf = 0
        fe = TopExp_Explorer(sh, TopAbs_FACE)
        while fe.More():
            nf += 1
            fe.Next()
        vol = _shell_abs_volume(sh)
        shells.append((vol, nf, sh))
        exp.Next()

    if len(shells) <= 1:
        print(f"  shells: {len(shells)} — tidak ada void untuk dibuang")
        return solid

    shells.sort(key=lambda t: t[0], reverse=True)
    outer_vol, outer_nf, outer = shells[0]
    kept = [(outer_vol, outer_nf, outer)]
    dropped = []
    for vol, nf, sh in shells[1:]:
        # buang bila jauh lebih kecil dari shell luar
        if outer_vol > 0 and (vol / outer_vol) < min_keep_ratio:
            dropped.append((vol, nf))
        else:
            kept.append((vol, nf, sh))

    print(
        f"  shells: total={len(shells)} keep={len(kept)} drop={len(dropped)} "
        f"(outer vol≈{outer_vol:.3f} mm³ faces={outer_nf})"
    )
    for vol, nf in dropped:
        print(f"    drop void vol≈{vol:.6f} mm³ faces={nf}")

    if len(kept) == 1 and len(dropped) > 0:
        mk = BRepBuilderAPI_MakeSolid(kept[0][2])
        if mk.IsDone():
            print("  MakeSolid dari shell luar saja: ok")
            return mk.Solid()
        print("  MakeSolid gagal — kembalikan solid asli")
    return solid


def _shape_fix_brep(solid):
    """Terapkan void-strip + Sewing + ShapeFix pada solid BRep.

    FreeCAD tidak tersedia di env — setara OCCT: drop micro-voids,
    Sewing, ShapeFix.
    """
    from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_WIRE, TopAbs_SHELL
    from OCP.TopoDS import TopoDS
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.BRepCheck import BRepCheck_Analyzer

    # 0) Buang void mikro sebelum sewing/tessellate
    solid = _drop_micro_void_shells(solid)

    # 1) Sewing wajah — kunci untuk seam STEP terbuka
    sew = BRepBuilderAPI_Sewing(1.0e-3)
    sew.SetTolerance(1.0e-3)
    sew.SetMaxTolerance(1.0)
    exp_f = TopExp_Explorer(solid, TopAbs_FACE)
    n_faces = 0
    while exp_f.More():
        sew.Add(exp_f.Current())
        n_faces += 1
        exp_f.Next()
    sew.Perform()
    sewn = sew.SewedShape()
    print(
        f"  sewing: faces_in={n_faces}  free_edges={sew.NbFreeEdges()}  "
        f"contiguous={sew.NbContigousEdges()}  "
        f"multiple={sew.NbMultipleEdges()}  "
        f"degenerate={sew.NbDegeneratedShapes()}"
    )

    # 2) Bila hasil sewing berupa shell, coba MakeSolid
    fixed = sewn
    try:
        if sewn.ShapeType() == TopAbs_SHELL:
            mk = BRepBuilderAPI_MakeSolid(TopoDS.Shell(sewn))
            if mk.IsDone():
                fixed = mk.Solid()
                print("  MakeSolid dari shell sewing: ok")
        elif sewn.ShapeType() != TopAbs_SOLID:
            shells = []
            exp_s = TopExp_Explorer(sewn, TopAbs_SHELL)
            while exp_s.More():
                shells.append(TopoDS.Shell(exp_s.Current()))
                exp_s.Next()
            if len(shells) == 1:
                mk = BRepBuilderAPI_MakeSolid(shells[0])
                if mk.IsDone():
                    fixed = mk.Solid()
                    print("  MakeSolid dari 1 shell: ok")
            elif shells:
                # pilih shell volume terbesar
                best = max(shells, key=_shell_abs_volume)
                mk = BRepBuilderAPI_MakeSolid(best)
                if mk.IsDone():
                    fixed = mk.Solid()
                    print(f"  MakeSolid shell terbesar dari {len(shells)}: ok")
                else:
                    print(f"  sewing hasil {len(shells)} shell — ShapeFix pada sewn")
                    fixed = sewn
    except Exception as e:
        print(f"  MakeSolid: {e}")
        fixed = sewn

    # 3) ShapeFix_Shape umum
    fixer = ShapeFix_Shape(fixed)
    fixer.SetPrecision(1.0e-3)
    fixer.SetMaxTolerance(1.0)
    fixer.SetMinTolerance(1.0e-4)
    fixer.Perform()
    fixed = fixer.Shape()

    # 4) Per-solid
    sf_solid = ShapeFix_Solid()
    sf_solid.SetPrecision(1.0e-3)
    sf_solid.SetMaxTolerance(1.0)
    try:
        if fixed.ShapeType() == TopAbs_SOLID:
            sf_solid.Init(TopoDS.Solid(fixed))
            sf_solid.Perform()
            fixed2 = sf_solid.Solid()
            if not fixed2.IsNull():
                fixed = fixed2
    except Exception as e:
        print(f"  ShapeFix_Solid: {e}")

    # Validasi BRep
    try:
        ana = BRepCheck_Analyzer(fixed)
        print(f"  BRepCheck_Analyzer.IsValid={ana.IsValid()}")
    except Exception as e:
        print(f"  BRepCheck: {e}")

    # Laporkan free bounds (open wires) bila ada
    try:
        fb = ShapeAnalysis_FreeBounds(fixed, True, True)
        closed = fb.GetClosedWires()
        opened = fb.GetOpenWires()

        def count_wires(shape):
            if shape.IsNull():
                return 0
            n = 0
            exp = TopExp_Explorer(shape, TopAbs_WIRE)
            while exp.More():
                n += 1
                exp.Next()
            return n

        print(
            f"  free bounds: closed_wires≈{count_wires(closed)}  "
            f"open_wires≈{count_wires(opened)}"
        )
    except Exception as e:
        print(f"  free-bounds probe: {e}")

    return fixed


def _step_shapefix(solid, deflection: float, out_stl: Path | None):
    print("  menjalankan ShapeFix_Shape / ShapeFix_Solid ...")
    fixed = _shape_fix_brep(solid)
    mesh, stats = tessellate_solid(fixed, deflection, clean=True)
    metrics = _mesh_metrics(mesh)
    # merge BRep volume into metrics note
    metrics["brep_volume_mm3"] = stats.get("volume_mm3")
    if out_stl is not None and mesh is not None:
        out_stl.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_stl)
        print(f"  STL → {out_stl}")
    return mesh, metrics, fixed


def _step_pymeshlab(in_stl: Path, out_stl: Path) -> tuple[object, dict]:
    try:
        import pymeshlab
    except ImportError:
        print("  PyMeshLab tidak terpasang — lewati langkah 3.")
        print("  Install (opsional): conda install -c conda-forge pymeshlab")
        return None, {"skipped": True, "reason": "pymeshlab_not_installed"}

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(in_stl))
    # Pipeline: weld → repair non-manifold → close holes (besar)
    try:
        ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.05))
        print("  merge_close_vertices: ok")
    except Exception as e:
        print(f"  merge_close_vertices: {e}")
        try:
            ms.meshing_merge_close_vertices()
        except Exception as e2:
            print(f"  merge_close_vertices(fallback): {e2}")
    try:
        ms.meshing_repair_non_manifold_edges()
        print("  repair_non_manifold_edges: ok")
    except Exception as e:
        print(f"  repair_non_manifold_edges: {e}")
    try:
        ms.meshing_repair_non_manifold_vertices()
        print("  repair_non_manifold_vertices: ok")
    except Exception as e:
        print(f"  repair_non_manifold_vertices: {e}")
    try:
        # maxholesize = jumlah edge di boundary loop; contoh NAUO3 ~4–50
        ms.meshing_close_holes(maxholesize=5000)
        print("  close_holes(maxholesize=5000): ok")
    except Exception as e:
        print(f"  close_holes: {e}")
    try:
        ms.meshing_remove_unreferenced_vertices()
    except Exception:
        pass
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(out_stl))
    import trimesh

    mesh = trimesh.load(out_stl, force="mesh", process=False)
    return mesh, _mesh_metrics(mesh)


def main() -> None:
    _require_cq()
    p = argparse.ArgumentParser(description="Heal solid UR STEP berurutan")
    p.add_argument("step_path", type=Path)
    p.add_argument("--match", default="NAUO3")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--deflection", type=float, default=0.5)
    p.add_argument("--target-euler", type=int, default=2)
    p.add_argument("--euler-tol", type=int, default=2)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/ur5e_nauo3_heal"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/ur5e_nauo3_heal.json"),
    )
    p.add_argument(
        "--max-step",
        type=int,
        default=3,
        help="1=trimesh saja, 2=+ShapeFix, 3=+PyMeshLab bila ada",
    )
    args = p.parse_args()
    if not args.step_path.exists():
        print(f"File tidak ada: {args.step_path}", file=sys.stderr)
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"membaca: {args.step_path}")
    parts = _read_step_with_names(args.step_path)
    indices = [args.index] if args.index is not None else None
    picked = match_parts(parts, names_substr=[args.match], indices=indices)
    if len(picked) > 1:
        pref = [t for t in picked if "NAUO3" in t[1] and "solid1" in t[1]]
        picked = pref[:1] if pref else picked[:1]
    if not picked:
        print("Part tidak ditemukan", file=sys.stderr)
        raise SystemExit(1)
    idx, name, solid = picked[0]
    print(f"target: #{idx} {name}")
    print(
        f"kriteria sukses: watertight=True dan "
        f"|euler - {args.target_euler}| <= {args.euler_tol}\n"
    )

    report: dict = {
        "source": str(args.step_path.resolve()),
        "index": idx,
        "name": name,
        "target_euler": args.target_euler,
        "euler_tol": args.euler_tol,
        "steps": [],
        "success_step": None,
    }

    # --- baseline tessellation ---
    print(f"=== baseline tessellation deflection={args.deflection} ===")
    mesh0, stats0 = tessellate_solid(solid, args.deflection, clean=True)
    m0 = _mesh_metrics(mesh0)
    print(
        f"  watertight={m0['watertight']}  euler={m0['euler_number']}  "
        f"faces={m0['n_faces']}  boundary_edges={m0['boundary_edges']}"
    )
    if mesh0 is not None:
        mesh0.export(args.out_dir / "00_baseline.stl")
    report["steps"].append({"name": "baseline", **m0})
    if _hits_target(m0, args.target_euler, args.euler_tol):
        print("\nSUDAH memenuhi target di baseline — tidak perlu heal.")
        report["success_step"] = "baseline"
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"tersimpan: {args.out_json}")
        return

    # --- step 1: trimesh.repair ---
    if args.max_step >= 1:
        print("\n=== step 1: trimesh.repair ===")
        mesh1, m1 = _step_trimesh_repair(mesh0, args.out_dir / "01_trimesh_repair.stl")
        print(
            f"  watertight={m1['watertight']}  euler={m1['euler_number']}  "
            f"faces={m1['n_faces']}  boundary_edges={m1['boundary_edges']}"
        )
        report["steps"].append({"name": "trimesh_repair", **m1})
        if _hits_target(m1, args.target_euler, args.euler_tol):
            print("\nSUKSES di step 1 (trimesh.repair). Berhenti.")
            report["success_step"] = "trimesh_repair"
            args.out_json.write_text(json.dumps(report, indent=2))
            print(f"tersimpan: {args.out_json}")
            return
        print("  → belum memenuhi target; lanjut step 2.")

    # --- step 2: ShapeFix BRep ---
    if args.max_step >= 2:
        print("\n=== step 2: ShapeFix BRep + re-tessellate ===")
        mesh2, m2, _ = _step_shapefix(
            solid, args.deflection, args.out_dir / "02_shapefix.stl"
        )
        print(
            f"  watertight={m2['watertight']}  euler={m2['euler_number']}  "
            f"faces={m2['n_faces']}  boundary_edges={m2['boundary_edges']}"
        )
        report["steps"].append({"name": "shapefix_brep", **m2})
        if _hits_target(m2, args.target_euler, args.euler_tol):
            print("\nSUKSES di step 2 (ShapeFix BRep). Berhenti.")
            report["success_step"] = "shapefix_brep"
            args.out_json.write_text(json.dumps(report, indent=2))
            print(f"tersimpan: {args.out_json}")
            return
        print("  → belum memenuhi target; lanjut step 3 bila tersedia.")

    # --- step 3: PyMeshLab ---
    if args.max_step >= 3:
        print("\n=== step 3: PyMeshLab ===")
        # pakai mesh terbaik sejauh ini sebagai input (shapefix STL atau trimesh)
        candidates = [
            args.out_dir / "02_shapefix.stl",
            args.out_dir / "01_trimesh_repair.stl",
            args.out_dir / "00_baseline.stl",
        ]
        in_stl = next((c for c in candidates if c.exists()), None)
        if in_stl is None:
            print("  tidak ada STL input")
            report["steps"].append({"name": "pymeshlab", "skipped": True})
        else:
            mesh3, m3 = _step_pymeshlab(in_stl, args.out_dir / "03_pymeshlab.stl")
            if m3.get("skipped"):
                report["steps"].append({"name": "pymeshlab", **m3})
            else:
                print(
                    f"  watertight={m3['watertight']}  euler={m3['euler_number']}  "
                    f"faces={m3['n_faces']}  boundary_edges={m3['boundary_edges']}"
                )
                report["steps"].append({"name": "pymeshlab", **m3})
                if _hits_target(m3, args.target_euler, args.euler_tol):
                    print("\nSUKSES di step 3 (PyMeshLab). Berhenti.")
                    report["success_step"] = "pymeshlab"
                    args.out_json.write_text(json.dumps(report, indent=2))
                    print(f"tersimpan: {args.out_json}")
                    return

    print("\nBelum mencapai target setelah langkah yang dijalankan.")
    print(
        "Opsi manual: install pymeshlab, atau buka STL di MeshLab "
        "(Filters → Cleaning → Remove non-manifold / Close holes), "
        "atau FreeCAD Part → Check Geometry + Shape builder."
    )
    report["success_step"] = None
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"tersimpan: {args.out_json}")


if __name__ == "__main__":
    main()
