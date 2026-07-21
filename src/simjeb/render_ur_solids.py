"""Export PNG multi-sudut per solid STEP (mapping kinematik visual).

Headless via matplotlib Agg — tanpa OpenGL/display.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from .inspect_ur_step import _require_cq, _read_step_with_names, match_parts, tessellate_solid

# Sudut kamera (elev, azim) — cukup untuk konfirmasi bentuk/orientasi
VIEWS = [
    ("iso", 25, 45),
    ("front", 0, 0),
    ("side", 0, 90),
    ("top", 90, 0),
    ("iso_back", 25, 225),
]


def _slug(name: str) -> str:
    s = name.replace("/", "_").replace("::", "__")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:120]


def _render_mesh_png(mesh, out_path: Path, elev: float, azim: float, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = mesh.vertices
    faces = mesh.faces
    # Downsample faces kalau sangat padat (render cepat, bentuk tetap jelas)
    max_faces = 80_000
    if len(faces) > max_faces:
        rng = np.random.default_rng(0)
        faces = faces[rng.choice(len(faces), size=max_faces, replace=False)]

    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    tris = verts[faces]
    coll = Poly3DCollection(
        tris,
        alpha=0.92,
        linewidths=0.05,
        edgecolors=(0.15, 0.15, 0.18, 0.25),
        facecolors=(0.55, 0.62, 0.72, 0.95),
    )
    ax.add_collection3d(coll)

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float((maxs - mins).max())
    if span < 1e-9:
        span = 1.0
    r = 0.55 * span
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _require_cq()
    p = argparse.ArgumentParser(description="Render PNG multi-sudut solid UR STEP")
    p.add_argument("step_path", type=Path)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/ur5e_renders"),
        help="folder output PNG (+ optional STL)",
    )
    p.add_argument(
        "--match",
        nargs="+",
        default=None,
        help="substring nama part, mis. NAUO3 NAUO2 NAUO1::solid8",
    )
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="index 1-based dari inspeksi (mis. 8 9 17 18)",
    )
    p.add_argument("--deflection", type=float, default=0.5, help="mm")
    p.add_argument("--stl", action="store_true", help="juga tulis STL per solid")
    p.add_argument(
        "--views",
        nargs="+",
        default=[v[0] for v in VIEWS],
        help=f"subset view: {[v[0] for v in VIEWS]}",
    )
    args = p.parse_args()
    if not args.step_path.exists():
        print(f"File tidak ada: {args.step_path}", file=sys.stderr)
        raise SystemExit(1)
    if not args.match and not args.indices:
        print("Wajib --match dan/atau --indices", file=sys.stderr)
        raise SystemExit(1)

    view_lookup = {name: (elev, azim) for name, elev, azim in VIEWS}
    selected_views = []
    for v in args.views:
        if v not in view_lookup:
            print(f"view tidak dikenal: {v}", file=sys.stderr)
            raise SystemExit(1)
        selected_views.append((v, *view_lookup[v]))

    print(f"membaca: {args.step_path}")
    parts = _read_step_with_names(args.step_path)
    picked = match_parts(parts, names_substr=args.match, indices=args.indices)
    if not picked:
        print("Tidak ada part yang cocok.", file=sys.stderr)
        raise SystemExit(1)
    print(f"render {len(picked)} solid → {args.out_dir}\n")

    for idx, name, solid in picked:
        print(f"[{idx}] {name}  tessellate deflection={args.deflection} ...", flush=True)
        mesh, stats = tessellate_solid(solid, args.deflection, clean=True)
        if mesh is None:
            print(f"  SKIP: no mesh  issues={stats['issues']}")
            continue
        slug = f"{idx:02d}_{_slug(name)}"
        solid_dir = args.out_dir / slug
        solid_dir.mkdir(parents=True, exist_ok=True)
        if args.stl:
            stl_path = solid_dir / f"{slug}.stl"
            mesh.export(stl_path)
            print(f"  STL → {stl_path}")
        print(
            f"  faces={stats['n_mesh_faces']}  wt={stats['watertight']}  "
            f"euler={stats['euler_number']}  vol={stats['volume_mm3']/1000:.2f} cm3"
        )
        for vname, elev, azim in selected_views:
            out = solid_dir / f"{vname}.png"
            title = f"#{idx} {name}\n{vname}  faces={stats['n_mesh_faces']}"
            _render_mesh_png(mesh, out, elev, azim, title)
            print(f"  PNG → {out}")
        print()

    print(f"selesai. lihat: {args.out_dir}")


if __name__ == "__main__":
    main()
