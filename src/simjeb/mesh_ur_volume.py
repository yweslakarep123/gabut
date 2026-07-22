"""Generate tetrahedral volume mesh dari STL heal UR → geometry.npz ala SimJEB.

Input: STL watertight (mis. reports/ur5e_nauo3_heal/02_shapefix.stl).
Output geometry.npz memakai kunci on-disk SimJEB yang sama dengan
``data/processed/{id}/geometry.npz``:

  surf_vertices (Ns,3) f64, surf_faces (Nf,3) i32,
  vol_points (N,3) f64, vol_tets (Nt,4) i32, node_surf (N,) bool

Loader (``dataset.SimJEBDataset`` / ``train.load_item``) memetakan
``vol_points``→``pos`` dan ``vol_tets``→``tets``.

Default element size 5 mm — cek n_nodes dulu sebelum diperhalus (batas VRAM
~8 GB / filter train ``n_nodes <= 150k``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def _tet_quality(points: np.ndarray, tets: np.ndarray) -> dict:
    """Statistik kualitas tet dasar (tanpa Gmsh plugin).

    - edge_aspect: max_edge / min_edge per tet (1 = equilateral edges)
    - radius_ratio: R / (3*r); ideal ≈ 1; besar = pipih/sliver
    - min_dihedral_deg: sudut dihedral minimum (derajat)
    - volume: bertanda (negatif = inverted / orientasi terbalik)
    """
    p = points[tets]  # (Nt, 4, 3)
    # 6 edges
    edges = np.stack(
        [
            p[:, 1] - p[:, 0],
            p[:, 2] - p[:, 0],
            p[:, 3] - p[:, 0],
            p[:, 2] - p[:, 1],
            p[:, 3] - p[:, 1],
            p[:, 3] - p[:, 2],
        ],
        axis=1,
    )  # (Nt, 6, 3)
    elen = np.linalg.norm(edges, axis=2)
    elen = np.maximum(elen, 1e-30)
    edge_aspect = elen.max(axis=1) / elen.min(axis=1)

    # signed volume: V = (1/6) * scalar_triple(AB, AC, AD)
    ab = p[:, 1] - p[:, 0]
    ac = p[:, 2] - p[:, 0]
    ad = p[:, 3] - p[:, 0]
    vol6 = np.einsum("ij,ij->i", ab, np.cross(ac, ad))
    vol = vol6 / 6.0
    abs_vol = np.abs(vol)

    # inradius r = 3V / A_total (tetra: V = r A / 3)
    def face_areas(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    a0 = face_areas(p[:, 1], p[:, 2], p[:, 3])
    a1 = face_areas(p[:, 0], p[:, 2], p[:, 3])
    a2 = face_areas(p[:, 0], p[:, 1], p[:, 3])
    a3 = face_areas(p[:, 0], p[:, 1], p[:, 2])
    area_sum = np.maximum(a0 + a1 + a2 + a3, 1e-30)
    inradius = 3.0 * abs_vol / area_sum

    # Circumcenter relatif ke p0: 2 (p_i - p0) · c = ||p_i - p0||^2
    # (bukan ||p_i||^2 - ||p0||^2 — itu rumus untuk O absolut; ||O|| ≠ R).
    A = 2.0 * (p[:, 1:] - p[:, 0:1])  # (Nt, 3, 3)
    rhs = np.sum((p[:, 1:] - p[:, 0:1]) ** 2, axis=2)
    try:
        center_rel = np.linalg.solve(A, rhs[..., None])[..., 0]
        circum_r = np.linalg.norm(center_rel, axis=1)
    except np.linalg.LinAlgError:
        circum_r = np.empty(len(tets))
        for i in range(len(tets)):
            try:
                cr = np.linalg.solve(A[i], rhs[i])
                circum_r[i] = np.linalg.norm(cr)
            except np.linalg.LinAlgError:
                circum_r[i] = np.nan

    # Ideal regular tet: R = 3r ⇒ R/(3r) = 1
    radius_ratio = circum_r / np.maximum(3.0 * inradius, 1e-30)

    # dihedral angles via face normals
    def face_normal(i, j, k):
        return np.cross(p[:, j] - p[:, i], p[:, k] - p[:, i])

    # faces opposite vertex 0,1,2,3 with outward-ish orientation from tet
    n0 = face_normal(1, 2, 3)
    n1 = face_normal(0, 3, 2)
    n2 = face_normal(0, 1, 3)
    n3 = face_normal(0, 2, 1)
    normals = [n0, n1, n2, n3]
    # pairs of faces sharing an edge → 6 dihedrals
    face_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    dihedrals = []
    for i, j in face_pairs:
        ni = normals[i]
        nj = normals[j]
        ni_n = ni / np.maximum(np.linalg.norm(ni, axis=1, keepdims=True), 1e-30)
        nj_n = nj / np.maximum(np.linalg.norm(nj, axis=1, keepdims=True), 1e-30)
        # internal dihedral = pi - angle between outward normals
        cosang = np.clip(-(ni_n * nj_n).sum(axis=1), -1.0, 1.0)
        dihedrals.append(np.degrees(np.arccos(cosang)))
    min_dihedral = np.min(np.stack(dihedrals, axis=1), axis=1)

    def pct(x, qs=(0, 1, 5, 50, 95, 99, 100)):
        x = x[np.isfinite(x)]
        return {f"p{q}": float(np.percentile(x, q)) for q in qs}

    n_neg = int((vol < 0).sum())
    n_zero = int((abs_vol < 1e-12).sum())
    sharp = int((min_dihedral < 5.0).sum())
    very_sharp = int((min_dihedral < 1.0).sum())
    bad_aspect = int((edge_aspect > 10.0).sum())
    bad_rho = int((radius_ratio > 10.0).sum())

    return {
        "n_tets": int(len(tets)),
        "n_inverted_or_neg_volume": n_neg,
        "n_near_zero_volume": n_zero,
        "n_min_dihedral_lt_5deg": sharp,
        "n_min_dihedral_lt_1deg": very_sharp,
        "n_edge_aspect_gt_10": bad_aspect,
        "n_radius_ratio_gt_10": bad_rho,
        "volume_mm3": {
            "sum_abs": float(abs_vol.sum()),
            "sum_signed": float(vol.sum()),
            **{f"abs_{k}": v for k, v in pct(abs_vol).items()},
        },
        "edge_aspect_max_over_min": pct(edge_aspect),
        "radius_ratio_R_over_3r": pct(radius_ratio),
        "min_dihedral_deg": pct(min_dihedral),
    }


def _surface_from_tets(points: np.ndarray, tets: np.ndarray):
    """Ekstrak boundary triangles + flag node permukaan."""
    faces = np.vstack(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ]
    )
    # orientasi kanonik untuk hitung multiplicity
    faces_sorted = np.sort(faces, axis=1)
    # unique with counts
    uniq, inv, counts = np.unique(faces_sorted, axis=0, return_inverse=True, return_counts=True)
    boundary_mask = counts[inv] == 1
    boundary = faces[boundary_mask]
    # pastikan winding konsisten-ish: pakai sorted order (cukup untuk SimJEB surf)
    boundary = np.sort(boundary, axis=1)

    node_surf = np.zeros(len(points), dtype=bool)
    node_surf[np.unique(boundary.ravel())] = True

    # surf mesh: remap ke indeks padat 0..Ns-1
    surf_ids = np.flatnonzero(node_surf)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[surf_ids] = np.arange(len(surf_ids))
    surf_vertices = points[surf_ids].astype(np.float64)
    surf_faces = remap[boundary].astype(np.int32)
    return surf_vertices, surf_faces, node_surf


def _extract_tets_from_gmsh(gmsh) -> tuple[np.ndarray, np.ndarray]:
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.array(coords, dtype=np.float64).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    etypes, _, enodes = gmsh.model.mesh.getElements(dim=3)
    tets_list = []
    for etype, nodes in zip(etypes, enodes):
        props = gmsh.model.mesh.getElementProperties(etype)
        n_per = int(props[3])
        if n_per != 4:
            continue
        arr = np.array(nodes, dtype=np.int64).reshape(-1, 4)
        tets_list.append(np.vectorize(tag_to_idx.__getitem__)(arr))
    if not tets_list:
        raise RuntimeError("Tidak ada elemen tetrahedron di mesh Gmsh")
    return points, np.vstack(tets_list).astype(np.int64)


def _read_single_solid_step(step_path: Path):
    """Baca STEP tunggal (hasil heal) → TopoDS_Solid."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Gagal baca STEP: {step_path} status={status}")
    reader.TransferRoots()
    shape = reader.OneShape()
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    if not exp.More():
        raise RuntimeError(f"Tidak ada SOLID di {step_path}")
    return TopoDS.Solid(exp.Current())


def mesh_discrete_stl_to_tets(
    stl_path: Path,
    *,
    mesh_size: float = 5.0,
    mesh_size_min: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Volume-mesh dari STL diskrit: jangan remesh CAD faces (hindari 1091/1274).

    Hipotesis I: meshing per-face OCC (periodic/cone) whack-a-mole.
    Path ini memakai triangulasi permukaan yang sudah ada, isi volume saja.
    """
    import gmsh

    stl_path = Path(stl_path)
    if mesh_size_min is None:
        mesh_size_min = mesh_size * 0.5

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ur_discrete")
    t0 = time.time()
    try:
        gmsh.merge(str(stl_path))
        try:
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.0)
        except Exception:
            pass
        # tanpa reparam — pakai facet STL apa adanya
        gmsh.model.mesh.classifySurfaces(math.pi, True, False)
        gmsh.model.mesh.createTopology()

        if not gmsh.model.getEntities(3):
            surfs = [s[1] for s in gmsh.model.getEntities(2)]
            if not surfs:
                raise RuntimeError("Tidak ada surface setelah classifySurfaces")
            sl = gmsh.model.geo.addSurfaceLoop(surfs)
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)  # jangan remesh surface
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1)

        print("  generate(3) discrete (MeshOnlyEmpty) ...")
        gmsh.model.mesh.generate(3)
        points, tets = _extract_tets_from_gmsh(gmsh)
        info = {
            "source": "discrete_stl",
            "mesh_size_mm": mesh_size,
            "mesh_size_min_mm": mesh_size_min,
            "n_nodes": int(len(points)),
            "n_tets": int(len(tets)),
            "elapsed_s": round(time.time() - t0, 3),
            "gmsh_version": gmsh.GMSH_API_VERSION,
        }
        return points, tets, info
    finally:
        gmsh.finalize()


def _iso_remesh_and_clear_self_intersections(
    stl_in: Path,
    stl_out: Path,
    *,
    mesh_size: float,
    max_iters: int = 10,
) -> dict:
    """Remesh isotropik + hapus self-intersect hingga PLC-aman.

    Urutan penting: (1) hapus self + close holes + re-iso agar tetap watertight;
    (2) baru dilate-clean residual tanpa re-iso. Dilate terlalu awal menyobek mesh.
    """
    import pymeshlab as ml
    import trimesh

    from .heal_ur_solid import _weld

    ms = ml.MeshSet()
    ms.load_new_mesh(str(stl_in))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=5,
        targetlen=ml.PureValue(mesh_size),
        checksurfdist=False,
    )

    # --- Fase R1: kurangi self-intersect + jaga manifold via re-iso ---
    history_r1: list[dict] = []
    for i in range(max_iters):
        ms.compute_selection_by_self_intersections_per_face()
        n_self = int(ms.current_mesh().selected_face_number())
        history_r1.append({"iter": i, "self": n_self})
        # stuck / selesai
        if n_self == 0 or (i >= 3 and len(history_r1) >= 2 and history_r1[-1]["self"] == history_r1[-2]["self"]):
            if n_self == 0:
                break
            # self residual — lanjut ke fase dilate setelah ambil komponen utama
            break
        ms.meshing_remove_selected_faces()
        try:
            ms.meshing_repair_non_manifold_edges(method="Remove Faces")
        except Exception:
            pass
        try:
            ms.meshing_close_holes(maxholesize=50000)
        except Exception:
            pass
        try:
            ms.meshing_isotropic_explicit_remeshing(
                iterations=2,
                targetlen=ml.PureValue(mesh_size),
                checksurfdist=False,
            )
        except Exception:
            pass

    # Ambil komponen terbesar (bukti R1: 21 comps → body utama watertight)
    tmp = stl_out.with_name(stl_out.stem + "_r1.stl")
    ms.save_current_mesh(str(tmp))
    mesh = _weld(trimesh.load(str(tmp), force="mesh", process=False))
    comps = mesh.split(only_watertight=False)
    mesh = max(comps, key=lambda c: len(c.faces))
    mesh = _weld(mesh)
    mesh.export(tmp)

    # --- Fase S2: dilate-clean residual self-intersect (tanpa re-iso) ---
    ms2 = ml.MeshSet()
    ms2.load_new_mesh(str(tmp))
    history_s2: list[dict] = []
    for i in range(max_iters):
        ms2.compute_selection_by_self_intersections_per_face()
        n_self = int(ms2.current_mesh().selected_face_number())
        for _ in range(2):
            try:
                ms2.apply_selection_dilatation()
            except Exception:
                break
        n_dilated = int(ms2.current_mesh().selected_face_number())
        history_s2.append({"iter": i, "self": n_self, "dilated": n_dilated})
        if n_self == 0:
            break
        ms2.meshing_remove_selected_faces()
        try:
            ms2.meshing_repair_non_manifold_edges(method="Remove Faces")
        except Exception:
            pass
        try:
            ms2.meshing_close_holes(maxholesize=50000)
        except Exception:
            pass

    ms2.save_current_mesh(str(stl_out))
    mesh = _weld(trimesh.load(str(stl_out), force="mesh", process=False))
    comps2 = mesh.split(only_watertight=False)
    mesh = max(comps2, key=lambda c: len(c.faces))
    mesh = _weld(mesh)
    mesh.export(stl_out)

    ms3 = ml.MeshSet()
    ms3.load_new_mesh(str(stl_out))
    ms3.compute_selection_by_self_intersections_per_face()
    n_final = int(ms3.current_mesh().selected_face_number())
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return {
        "history_r1": history_r1,
        "history_s2": history_s2,
        "n_self_final": n_final,
        "n_faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "euler": int(mesh.euler_number),
        "n_components_r1": int(len(comps)),
        "n_components_s2": int(len(comps2)),
        "surface_stl": str(stl_out),
    }


def mesh_watertight_surface_to_tets(
    stl_path: Path,
    *,
    mesh_size: float = 5.0,
    mesh_size_min: float | None = None,
    out_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Dari STL yang sudah watertight (setelah weld) → tet volume.

    Bukti: STEP re-tess deflection=2.5 → euler=-10 (tidak watertight).
    STL heal ``02_shapefix.stl`` setelah ``_weld`` → watertight euler=2.
    Fine STL punya self-intersect → PLC/overlapping facets; perbaikan:
    iso-remesh + dilate-clean, lalu Gmsh MeshOnlyEmpty (tanpa createGeometry).
    """
    import gmsh
    import trimesh

    from .heal_ur_solid import _weld

    stl_path = Path(stl_path)
    out_dir = Path(out_dir) if out_dir else stl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if mesh_size_min is None:
        mesh_size_min = mesh_size * 0.5

    raw = trimesh.load(str(stl_path), force="mesh", process=False)
    mesh = _weld(raw)
    print(
        f"  weld STL: faces={len(mesh.faces)} watertight={mesh.is_watertight} "
        f"euler={mesh.euler_number}"
    )
    if not mesh.is_watertight:
        raise RuntimeError(
            f"STL heal tidak watertight setelah weld (euler={mesh.euler_number})"
        )

    welded_stl = out_dir / "_heal_welded.stl"
    mesh.export(welded_stl)

    # --- coba tetgen pada surface mentah (sering gagal karena self-intersect) ---
    try:
        import tetgen  # type: ignore
    except ImportError:
        tetgen = None

    if tetgen is not None:
        t0 = time.time()
        print("  tetrahedralize via tetgen ...")
        try:
            tg = tetgen.TetGen(mesh.vertices, mesh.faces)
            max_vol = (mesh_size**3) / 6.0
            nodes, elems = tg.tetrahedralize(
                order=1,
                mindihedral=5.0,
                minratio=1.5,
                fixedvolume=True,
                maxvolume=max_vol,
            )
            points = np.asarray(nodes, dtype=np.float64)
            tets = np.asarray(elems, dtype=np.int64)
            info = {
                "source": "heal_stl_tetgen",
                "mesh_size_mm": mesh_size,
                "n_nodes": int(len(points)),
                "n_tets": int(len(tets)),
                "elapsed_s": round(time.time() - t0, 3),
                "surface_stl": str(welded_stl),
            }
            return points, tets, info
        except Exception as e:
            print(f"  tetgen gagal ({e}); repair surface lalu MeshOnlyEmpty...")

    # --- Repair: iso-remesh + clear self-intersections ---
    clean_stl = out_dir / "_heal_iso_clean.stl"
    print("  iso-remesh + clear self-intersections ...")
    t_repair = time.time()
    repair_info = _iso_remesh_and_clear_self_intersections(
        welded_stl, clean_stl, mesh_size=mesh_size
    )
    repair_info["elapsed_s"] = round(time.time() - t_repair, 3)
    print(
        f"  clean surface: faces={repair_info['n_faces']} "
        f"watertight={repair_info['watertight']} euler={repair_info['euler']} "
        f"self={repair_info['n_self_final']}"
    )
    if not repair_info["watertight"]:
        raise RuntimeError(
            f"Surface setelah iso-clean tidak watertight "
            f"(euler={repair_info['euler']}, self={repair_info['n_self_final']})"
        )

    # --- Gmsh MeshOnlyEmpty pada surface bersih (tanpa createGeometry) ---
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ur_volume")
    t0 = time.time()
    try:
        gmsh.merge(str(clean_stl))
        try:
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.0)
        except Exception:
            pass

        gmsh.model.mesh.classifySurfaces(math.pi, True, False)
        gmsh.model.mesh.createTopology()
        surfs = [s[1] for s in gmsh.model.getEntities(2)]
        if not surfs:
            raise RuntimeError("Gmsh: tidak ada surface setelah createTopology")
        if not gmsh.model.getEntities(3):
            sl = gmsh.model.geo.addSurfaceLoop(surfs)
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)

        print("  generate(3) MeshOnlyEmpty (surface bersih) ...")
        gmsh.model.mesh.generate(3)
        points, tets = _extract_tets_from_gmsh(gmsh)
        if len(tets) == 0:
            raise RuntimeError("Gmsh MeshOnlyEmpty menghasilkan 0 tetrahedra")
        info = {
            "source": "heal_stl_iso_clean_meshonly",
            "mesh_size_mm": mesh_size,
            "mesh_size_min_mm": mesh_size_min,
            "n_nodes": int(len(points)),
            "n_tets": int(len(tets)),
            "elapsed_s": round(time.time() - t0, 3),
            "surface_stl": str(clean_stl),
            "repair": repair_info,
            "gmsh_version": gmsh.GMSH_API_VERSION,
        }
        return points, tets, info
    finally:
        gmsh.finalize()


def mesh_step_via_tessellation(
    step_path: Path,
    *,
    mesh_size: float = 5.0,
    mesh_size_min: float | None = None,
    out_dir: Path | None = None,
    surface_stl: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Path andal: pakai STL heal watertight (bukan re-tess STEP kasar).

    Bukti debug: STEP@defl=2.5 → watertight=false euler=-10.
    STL ``02_shapefix.stl`` + weld → watertight euler=2.
    """
    step_path = Path(step_path)
    out_dir = Path(out_dir) if out_dir else step_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prioritas surface: explicit → heal STL default → (jangan tess STEP kasar)
    candidates = []
    if surface_stl is not None:
        candidates.append(Path(surface_stl))
    candidates.append(Path("reports/ur5e_nauo3_heal/02_shapefix.stl"))
    candidates.append(out_dir / "02_shapefix_clean.stl")

    stl = next((p for p in candidates if p.exists()), None)

    if stl is None:
        raise RuntimeError(
            "Tidak ada STL heal watertight. Jalankan heal dulu atau pass --surface-stl. "
            "(Re-tess STEP kasar ditolak: bukti euler=-10 @ defl=2.5)"
        )

    print(f"  surface source: {stl} (hindari re-tess STEP kasar)")
    return mesh_watertight_surface_to_tets(
        stl,
        mesh_size=mesh_size,
        mesh_size_min=mesh_size_min,
        out_dir=out_dir,
    )


def mesh_step_to_tets(
    step_path: Path,
    *,
    mesh_size: float = 5.0,
    mesh_size_min: float | None = None,
    max_retries: int = 3,
    backend: str = "tess",
    out_dir: Path | None = None,
    surface_stl: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """STEP healed → tet volume.

    backend:
      - ``tess`` (default): STL heal watertight + remesh/tetgen (andal)
      - ``occ``: mesh langsung face OCC (sering gagal di face patologis)
    """
    if backend == "tess":
        return mesh_step_via_tessellation(
            step_path,
            mesh_size=mesh_size,
            mesh_size_min=mesh_size_min,
            out_dir=out_dir,
            surface_stl=surface_stl,
        )
    if backend != "occ":
        raise ValueError(f"backend tidak dikenal: {backend}")

    import gmsh

    step_path = Path(step_path)
    if mesh_size_min is None:
        mesh_size_min = mesh_size * 0.5

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ur_link")
    t0 = time.time()
    try:
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        try:
            gmsh.option.setNumber("Mesh.MaxRetries", max_retries)
        except Exception:
            pass
        try:
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.0)
        except Exception:
            pass
        try:
            gmsh.option.setNumber("Mesh.IgnorePeriodicity", 1)
        except Exception:
            pass

        # Hanya face benar-benar mikro: area kecil DAN diag kecil
        small = []
        for dim, tag in gmsh.model.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            bbox = [xmax - xmin, ymax - ymin, zmax - zmin]
            diag = math.sqrt(sum(x * x for x in bbox))
            try:
                area = float(gmsh.model.occ.getMass(dim, tag))
            except Exception:
                area = None
            if not (area is not None and area < 2.0 and diag < 3.0):
                continue
            local = float(max(min(0.5 * math.sqrt(area), mesh_size * 0.25), 0.05))
            boundary = gmsh.model.getBoundary([(dim, tag)], oriented=False, recursive=True)
            if boundary:
                gmsh.model.mesh.setSize(boundary, local)
                small.append({"tag": int(tag), "area_mm2": area, "diag": diag, "local": local})

        print(f"  [occ] local size pada {len(small)} face mikro ketat")
        print("  [occ] generate(3) ...")
        gmsh.model.mesh.generate(3)
        points, tets = _extract_tets_from_gmsh(gmsh)
        info = {
            "source": "step_occ",
            "mesh_size_mm": mesh_size,
            "n_nodes": int(len(points)),
            "n_tets": int(len(tets)),
            "elapsed_s": round(time.time() - t0, 3),
            "gmsh_version": gmsh.GMSH_API_VERSION,
        }
        return points, tets, info
    finally:
        gmsh.finalize()


def mesh_stl_to_tets(
    stl_path: Path,
    *,
    mesh_size: float = 5.0,
    mesh_size_min: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Alias: volume dari STL diskrit (heal 0.5mm sering PLC — prefer tess path)."""
    return mesh_discrete_stl_to_tets(
        stl_path, mesh_size=mesh_size, mesh_size_min=mesh_size_min
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Gmsh tet-mesh UR → geometry.npz")
    p.add_argument(
        "stl_path",
        type=Path,
        nargs="?",
        default=None,
        help="opsional; diabaikan jika --step dipakai",
    )
    p.add_argument(
        "--step",
        type=Path,
        default=Path("reports/ur5e_nauo3_volume/nauo3_healed.step"),
        help="STEP healed (disarankan; menghindari hang surface kecil)",
    )
    p.add_argument("--from-stl", action="store_true", help="paksa path STL heal (sering PLC)")
    p.add_argument(
        "--backend",
        choices=("tess", "occ"),
        default="tess",
        help="tess=STL heal watertight+volume (default); occ=mesh face STEP langsung",
    )
    p.add_argument(
        "--surface-stl",
        type=Path,
        default=Path("reports/ur5e_nauo3_heal/02_shapefix.stl"),
        help="STL surface watertight (hasil heal) untuk backend tess",
    )
    p.add_argument("--mesh-size", type=float, default=5.0, help="ukuran elemen target (mm)")
    p.add_argument("--mesh-size-min", type=float, default=None)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/ur5e_nauo3_volume"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="default: out-dir/mesh_report.json",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json or (args.out_dir / "mesh_report.json")

    use_stl = args.from_stl or (
        args.stl_path is not None and not args.step.exists() and args.stl_path.exists()
    )
    if use_stl:
        stl = args.stl_path or args.surface_stl
        if not stl.exists():
            print(f"STL tidak ada: {stl}", file=sys.stderr)
            raise SystemExit(1)
        print(f"STL: {stl}")
        print(f"mesh size target: {args.mesh_size} mm")
        points, tets, info = mesh_watertight_surface_to_tets(
            stl,
            mesh_size=args.mesh_size,
            mesh_size_min=args.mesh_size_min,
            out_dir=args.out_dir,
        )
        source_key = "source_stl"
        source_val = str(stl.resolve())
    else:
        if not args.step.exists() and args.backend == "occ":
            print(f"STEP tidak ada: {args.step}", file=sys.stderr)
            raise SystemExit(1)
        print(f"STEP: {args.step}")
        print(f"backend={args.backend}  mesh_size={args.mesh_size} mm")
        print(f"surface_stl={args.surface_stl}")
        points, tets, info = mesh_step_to_tets(
            args.step,
            mesh_size=args.mesh_size,
            mesh_size_min=args.mesh_size_min,
            max_retries=args.max_retries,
            backend=args.backend,
            out_dir=args.out_dir,
            surface_stl=args.surface_stl,
        )
        source_key = "source_step"
        source_val = str(args.step.resolve())
    print(f"nodes={info['n_nodes']}  tets={info['n_tets']}")

    # orientasi: pastikan volume positif mayoritas
    ab = points[tets[:, 1]] - points[tets[:, 0]]
    ac = points[tets[:, 2]] - points[tets[:, 0]]
    ad = points[tets[:, 3]] - points[tets[:, 0]]
    signed = np.einsum("ij,ij->i", ab, np.cross(ac, ad))
    if (signed < 0).sum() > (signed > 0).sum():
        tets = tets[:, [0, 2, 1, 3]]
        print("  flip konektivitas tet (mayoritas volume negatif)")

    surf_vertices, surf_faces, node_surf = _surface_from_tets(points, tets)
    quality = _tet_quality(points, tets)

    geom_path = args.out_dir / "geometry.npz"
    np.savez(
        geom_path,
        surf_vertices=surf_vertices.astype(np.float64),
        surf_faces=surf_faces.astype(np.int32),
        vol_points=points.astype(np.float64),
        vol_tets=tets.astype(np.int32),
        node_surf=node_surf.astype(bool),
    )
    # Alias runtime-style (opsional) — loader utama tetap vol_*
    # tidak disimpan ganda agar kompatibel ketat dengan GEOMETRY_KEYS SimJEB.

    report = {
        source_key: source_val,
        "geometry_npz": str(geom_path.resolve()),
        "fields": {
            "vol_points": "pos di loader SimJEB",
            "vol_tets": "tets di loader SimJEB",
            "node_surf": "bool per node volume",
            "surf_vertices": "boundary surface vertices",
            "surf_faces": "boundary triangles (indeks ke surf_vertices)",
        },
        "mesh": info,
        "surface": {
            "n_surf_vertices": int(len(surf_vertices)),
            "n_surf_faces": int(len(surf_faces)),
            "n_node_surf": int(node_surf.sum()),
        },
        "quality": quality,
        "vram_note": (
            "Filter train SimJEB memakai n_nodes<=150k (VRAM ~8GB). "
            f"NAUO3 @ {args.mesh_size} mm → n_nodes={info['n_nodes']}."
        ),
    }
    out_json.write_text(json.dumps(report, indent=2))
    print(f"tersimpan: {geom_path}")
    print(f"laporan:   {out_json}")
    print(
        f"kualitas: inverted={quality['n_inverted_or_neg_volume']}  "
        f"dihedral<5°={quality['n_min_dihedral_lt_5deg']}  "
        f"aspect>10={quality['n_edge_aspect_gt_10']}  "
        f"rho>10={quality['n_radius_ratio_gt_10']}"
    )
    print(
        f"  edge_aspect p50={quality['edge_aspect_max_over_min']['p50']:.3f}  "
        f"p99={quality['edge_aspect_max_over_min']['p99']:.3f}  "
        f"max={quality['edge_aspect_max_over_min']['p100']:.3f}"
    )
    print(
        f"  min_dihedral_deg p1={quality['min_dihedral_deg']['p1']:.2f}  "
        f"p50={quality['min_dihedral_deg']['p50']:.2f}  "
        f"min={quality['min_dihedral_deg']['p0']:.2f}"
    )


if __name__ == "__main__":
    main()
