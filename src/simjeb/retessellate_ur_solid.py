"""Eksperimen murah: re-tessellasi satu solid dengan deflection lebih halus.

Bandingkan watertight + euler_number before/after. Kalau euler mendekati 2
dan watertight jadi true, heal eksplisit mungkin tidak perlu untuk solid itu.

PENTING — angka Euler di sini diambil dari mesh tessellasi mentah
(``process=False``, tanpa ``merge_vertices``). Vertex di tiap face OCCT
tidak di-weld di seam, jadi boundary palsu membuat Euler membengkak
(contoh NAUO3: ≈1470 di 0,5 mm maupun 0,05 mm). Angka ``euler≈10`` di
laporan ``heal_ur_solid`` adalah tahap berikutnya (setelah weld) — bukan
kontradiksi dengan 1470.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inspect_ur_step import _require_cq, _read_step_with_names, match_parts, tessellate_solid


def main() -> None:
    _require_cq()
    p = argparse.ArgumentParser(description="Re-tessellate satu solid UR STEP")
    p.add_argument("step_path", type=Path)
    p.add_argument(
        "--match",
        default="NAUO3",
        help="substring nama (default NAUO3)",
    )
    p.add_argument("--index", type=int, default=None, help="atau index 1-based")
    p.add_argument(
        "--before",
        type=float,
        default=0.5,
        help="deflection baseline mm (sama inspeksi Fase 1)",
    )
    p.add_argument(
        "--after",
        type=float,
        default=0.05,
        help="deflection halus mm",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/ur5e_nauo3_retessellate.json"),
    )
    p.add_argument(
        "--export-stl-dir",
        type=Path,
        default=None,
        help="opsional: tulis STL before/after ke folder ini",
    )
    args = p.parse_args()
    if not args.step_path.exists():
        print(f"File tidak ada: {args.step_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"membaca: {args.step_path}")
    parts = _read_step_with_names(args.step_path)
    indices = [args.index] if args.index is not None else None
    names = [args.match] if args.match else None
    picked = match_parts(parts, names_substr=names, indices=indices)
    # Prefer exact NAUO3::solid1 bila banyak match
    if len(picked) > 1:
        preferred = [t for t in picked if "NAUO3" in t[1] and "solid1" in t[1]]
        if preferred:
            picked = preferred[:1]
        else:
            picked = picked[:1]
    if not picked:
        print("Part tidak ditemukan.", file=sys.stderr)
        raise SystemExit(1)

    idx, name, solid = picked[0]
    print(f"target: #{idx} {name}")
    print(f"\n=== BEFORE deflection={args.before} mm ===", flush=True)
    mesh_b, stats_b = tessellate_solid(solid, args.before, clean=True)
    print(
        f"  watertight={stats_b['watertight']}  euler={stats_b['euler_number']}  "
        f"faces={stats_b['n_mesh_faces']}  verts={stats_b['n_mesh_vertices']}  "
        f"is_volume={stats_b['is_volume']}"
    )

    print(f"\n=== AFTER  deflection={args.after} mm ===", flush=True)
    mesh_a, stats_a = tessellate_solid(solid, args.after, clean=True)
    print(
        f"  watertight={stats_a['watertight']}  euler={stats_a['euler_number']}  "
        f"faces={stats_a['n_mesh_faces']}  verts={stats_a['n_mesh_vertices']}  "
        f"is_volume={stats_a['is_volume']}"
    )

    euler_b = stats_b.get("euler_number")
    euler_a = stats_a.get("euler_number")
    delta_euler = None
    if euler_b is not None and euler_a is not None:
        delta_euler = euler_a - euler_b

    verdict = "inconclusive"
    if stats_a.get("watertight") is True and euler_a is not None and abs(euler_a - 2) <= 2:
        verdict = "fine_tessellation_enough"
    elif (
        stats_a.get("watertight") is True
        or (euler_a is not None and euler_b is not None and abs(euler_a - 2) < abs(euler_b - 2) * 0.25)
    ):
        verdict = "significant_improvement_consider_skip_heal"
    elif euler_a is not None and euler_b is not None and abs(euler_a - euler_b) < max(10, 0.05 * abs(euler_b)):
        verdict = "little_help_need_explicit_heal"
    else:
        verdict = "partial_change_review_numbers"

    print("\n=== delta ===")
    print(f"  Δeuler = {delta_euler}  (target topological sphere ≈ 2)")
    print(f"  watertight: {stats_b['watertight']} → {stats_a['watertight']}")
    print(f"  faces: {stats_b['n_mesh_faces']} → {stats_a['n_mesh_faces']}")
    print(f"  verdict: {verdict}")

    if args.export_stl_dir is not None:
        args.export_stl_dir.mkdir(parents=True, exist_ok=True)
        if mesh_b is not None:
            mesh_b.export(args.export_stl_dir / f"nauo3_defl_{args.before}.stl")
        if mesh_a is not None:
            mesh_a.export(args.export_stl_dir / f"nauo3_defl_{args.after}.stl")
        print(f"  STL → {args.export_stl_dir}")

    report = {
        "source": str(args.step_path.resolve()),
        "index": idx,
        "name": name,
        "before": stats_b,
        "after": stats_a,
        "delta_euler": delta_euler,
        "verdict": verdict,
        "note": (
            "Euler number untuk bola topologis ≈ 2. Watertight true + euler≈2 "
            "menandakan mesh volume siap; kalau deflection halus tidak mendekati, "
            "perlu heal eksplisit (trimesh.repair / MeshLab)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\ntersimpan: {args.out}")


if __name__ == "__main__":
    main()
