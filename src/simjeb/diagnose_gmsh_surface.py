"""Diagnostik Gmsh: kenapa surface 1091 loop (tanpa generate(3) penuh).

Hipotesis:
  A — Cone 1091 geometri hampir degenerate (edge/radius sangat kecil)
  B — Cascade tetangga: refine edge 1091 memaksa remesh 1070/1078–1082 terus
  C — mesh_size 5mm tidak cocok dengan feature size cone → split tak berujung
  D — Algo Frontal+MeshAdapt tidak terminate saat invalid tersisa
  E — Parametrisasi/OCC type cone buruk meski BRep valid
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

# #region agent log
_DEBUG_LOG = Path("/home/daffa/Documents/jazari/.cursor/debug-ee4192.log")


def _dlog(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "diag1") -> None:
    payload = {
        "sessionId": "ee4192",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# #endregion


def _surface_stats(gmsh, tag: int) -> dict:
    """Metrik geometri satu surface OCC."""
    typ = gmsh.model.getType(2, tag)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
    bbox = [xmax - xmin, ymax - ymin, zmax - zmin]
    try:
        area = float(gmsh.model.occ.getMass(2, tag))
    except Exception as e:
        area = None
        area_err = str(e)
    else:
        area_err = None

    # boundary curves + edge lengths (approx via discretize)
    bounds = gmsh.model.getBoundary([(2, tag)], oriented=False, recursive=False)
    edge_lens = []
    for dim, etag in bounds:
        if dim != 1:
            continue
        try:
            # sample curve
            umin, umax = gmsh.model.getParametrizationBounds(1, etag)
            n = 32
            us = [umin[0] + (umax[0] - umin[0]) * i / (n - 1) for i in range(n)]
            pts = []
            for u in us:
                p = gmsh.model.getValue(1, etag, [u])
                pts.append(p)
            length = 0.0
            for a, b in zip(pts, pts[1:]):
                length += math.dist(a, b)
            edge_lens.append(length)
        except Exception:
            pass

    return {
        "tag": tag,
        "type": typ,
        "bbox_size_mm": bbox,
        "bbox_diag_mm": float(math.sqrt(sum(x * x for x in bbox))),
        "area_mm2": area,
        "area_err": area_err,
        "n_boundary_curves": len([b for b in bounds if b[0] == 1]),
        "boundary_curve_tags": [b[1] for b in bounds if b[0] == 1],
        "edge_len_min_mm": float(min(edge_lens)) if edge_lens else None,
        "edge_len_max_mm": float(max(edge_lens)) if edge_lens else None,
        "edge_len_ratio": (
            float(max(edge_lens) / max(min(edge_lens), 1e-30)) if edge_lens else None
        ),
    }


def _neighbor_surfaces(gmsh, tag: int) -> list[int]:
    """Surface yang berbagi curve dengan tag."""
    bounds = gmsh.model.getBoundary([(2, tag)], oriented=False, recursive=False)
    curves = [(d, t) for d, t in bounds if d == 1]
    nbrs = set()
    for c in curves:
        # upward adjacency: entities bounded by this curve
        try:
            ups = gmsh.model.getAdjacencies(1, c[1])
            # getAdjacencies returns (up, down); up are higher-dim
            up = ups[0] if isinstance(ups, tuple) else ups
            for utag in up:
                if int(utag) != tag:
                    nbrs.add(int(utag))
        except Exception:
            # fallback: scan all surfaces' boundaries (slow but ok for diag)
            pass
    if not nbrs:
        # fallback scan
        my_curves = {c[1] for c in curves}
        for dim, stag in gmsh.model.getEntities(2):
            if stag == tag:
                continue
            b = gmsh.model.getBoundary([(2, stag)], oriented=False, recursive=False)
            other = {t for d, t in b if d == 1}
            if my_curves & other:
                nbrs.add(stag)
    return sorted(nbrs)


def _try_mesh_surface_only(gmsh, tag: int, algo: int, mesh_size: float, timeout_s: float) -> dict:
    """Coba mesh 2D hanya pada satu surface; hitung durasi + jumlah elemen."""
    gmsh.option.setNumber("Mesh.Algorithm", algo)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    # batasi retry MeshAdapt agar tidak hang di diagnostik
    try:
        gmsh.option.setNumber("Mesh.MaxRetries", 2)
    except Exception:
        pass
    try:
        gmsh.option.setNumber("Mesh.MeshAdaptRetryMax", 2)
    except Exception:
        pass

    t0 = time.time()
    err = None
    n_tri = None
    try:
        gmsh.model.mesh.generate(1)
        # mesh hanya surface ini
        gmsh.model.mesh.generate(2)
        etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
        n_tri = 0
        for et, nodes in zip(etypes, enodes):
            n_per = int(gmsh.model.mesh.getElementProperties(et)[3])
            n_tri += len(nodes) // max(n_per, 1)
    except Exception as e:
        err = str(e)
    dt = time.time() - t0
    timed_out = dt >= timeout_s
    return {
        "algo": algo,
        "mesh_size": mesh_size,
        "dt_s": round(dt, 3),
        "n_tri": n_tri,
        "error": err,
        "approx_timed_out": timed_out,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose Gmsh surface 1091 loop")
    p.add_argument(
        "--step",
        type=Path,
        default=Path("reports/ur5e_nauo3_volume/nauo3_healed.step"),
    )
    p.add_argument("--target-surface", type=int, default=1091)
    p.add_argument("--mesh-size", type=float, default=5.0)
    p.add_argument("--timeout-s", type=float, default=30.0)
    args = p.parse_args()

    import gmsh

    if not args.step.exists():
        raise SystemExit(f"STEP tidak ada: {args.step}")

    # #region agent log
    _dlog(
        "D",
        "diagnose_gmsh_surface.py:main",
        "start_diag",
        {"step": str(args.step), "target": args.target_surface, "mesh_size": args.mesh_size},
    )
    # #endregion

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("diag")
    gmsh.model.occ.importShapes(str(args.step))
    gmsh.model.occ.synchronize()

    ents = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
    # #region agent log
    _dlog("E", "diagnose_gmsh_surface.py:after_import", "entity_counts", ents)
    # #endregion

    tag = args.target_surface
    tags2 = [t for _, t in gmsh.model.getEntities(2)]
    if tag not in tags2:
        # #region agent log
        _dlog(
            "E",
            "diagnose_gmsh_surface.py:missing",
            "target_surface_missing",
            {"target": tag, "n_surfaces": len(tags2), "sample_tags": tags2[:20]},
        )
        # #endregion
        gmsh.finalize()
        raise SystemExit(f"Surface {tag} tidak ada (punya {len(tags2)} surfaces)")

    stats = _surface_stats(gmsh, tag)
    # #region agent log
    _dlog("A", "diagnose_gmsh_surface.py:stats", "surface_1091_geometry", stats)
    # #endregion
    print("surface stats:", json.dumps(stats, indent=2))

    nbrs = _neighbor_surfaces(gmsh, tag)
    nbr_stats = []
    for nt in nbrs[:12]:
        try:
            nbr_stats.append(_surface_stats(gmsh, nt))
        except Exception as e:
            nbr_stats.append({"tag": nt, "error": str(e)})
    # #region agent log
    _dlog(
        "B",
        "diagnose_gmsh_surface.py:neighbors",
        "neighbor_surfaces",
        {
            "neighbor_tags": nbrs,
            "neighbor_types": [s.get("type") for s in nbr_stats],
            "neighbor_bbox_diags": [s.get("bbox_diag_mm") for s in nbr_stats],
            "neighbor_edge_min": [s.get("edge_len_min_mm") for s in nbr_stats],
        },
    )
    # #endregion
    print("neighbors:", nbrs)

    # Uji mesh size vs feature (hipotesis C)
    feat = stats.get("edge_len_min_mm") or stats.get("bbox_diag_mm") or 1.0
    size_ratio = args.mesh_size / max(feat, 1e-12)
    # #region agent log
    _dlog(
        "C",
        "diagnose_gmsh_surface.py:size_ratio",
        "mesh_size_vs_feature",
        {
            "mesh_size": args.mesh_size,
            "feature_edge_min_mm": stats.get("edge_len_min_mm"),
            "bbox_diag_mm": stats.get("bbox_diag_mm"),
            "size_over_min_edge": size_ratio,
        },
    )
    # #endregion

    # Uji beberapa algoritma 2D pada model penuh tapi dengan MaxRetries rendah (hipotesis D)
    # Jangan generate(3). Clear mesh antar trial.
    trials = []
    for algo, label in [(6, "FrontalDelaunay"), (1, "MeshAdapt"), (5, "Delaunay"), (8, "FrontalDelaunayForQuads")]:
        try:
            gmsh.model.mesh.clear()
        except Exception:
            pass
        # batasi waktu: kalau > timeout, catat dan lanjut
        t0 = time.time()
        result = {"algo": algo, "label": label}
        try:
            gmsh.option.setNumber("Mesh.Algorithm", algo)
            gmsh.option.setNumber("Mesh.MeshSizeMin", args.mesh_size * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", args.mesh_size)
            try:
                gmsh.option.setNumber("Mesh.MaxRetries", 1)
            except Exception:
                pass
            # Mesh hanya 1D dulu (cepat), lalu coba mesh surface target saja via setCompound? 
            # Gmsh tidak punya mesh-one-surface mudah; kita mesh 2D penuh dengan timeout eksternal via alarm
            import signal

            class _TO(Exception):
                pass

            def _handler(signum, frame):
                raise _TO("timeout")

            old = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(args.timeout_s))
            try:
                gmsh.model.mesh.generate(2)
                signal.alarm(0)
                # hitung elemen di surface target
                etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
                n_tri = sum(
                    len(nodes) // max(int(gmsh.model.mesh.getElementProperties(et)[3]), 1)
                    for et, nodes in zip(etypes, enodes)
                )
                result.update(
                    {
                        "ok": True,
                        "dt_s": round(time.time() - t0, 3),
                        "n_tri_on_target": n_tri,
                        "total_2d_entities_meshed": len(gmsh.model.getEntities(2)),
                    }
                )
            except _TO:
                result.update({"ok": False, "timeout": True, "dt_s": round(time.time() - t0, 3)})
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        except Exception as e:
            result.update({"ok": False, "error": str(e), "dt_s": round(time.time() - t0, 3)})
        trials.append(result)
        print("trial:", result)
        # #region agent log
        _dlog("D", "diagnose_gmsh_surface.py:trial", f"algo_trial_{label}", result)
        # #endregion
        if result.get("timeout"):
            # setelah timeout, state mesh mungkin kotor — reimport
            gmsh.finalize()
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("diag")
            gmsh.model.occ.importShapes(str(args.step))
            gmsh.model.occ.synchronize()

    # #region agent log
    _dlog(
        "D",
        "diagnose_gmsh_surface.py:summary",
        "diag_complete",
        {
            "any_algo_ok": any(t.get("ok") for t in trials),
            "any_timeout": any(t.get("timeout") for t in trials),
            "trials": trials,
        },
    )
    # #endregion

    gmsh.finalize()
    print("DONE — lihat", _DEBUG_LOG)


if __name__ == "__main__":
    main()
