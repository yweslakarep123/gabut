"""Parser satu bracket SimJEB: geometri + field hasil FEA untuk 4 load case.

Output per bracket (dict / .npz):
- surf_vertices (Ns,3), surf_faces (Nf,3)   : surface mesh (OBJ)
- vol_points (N,3), vol_tets (Nt,4)         : volume mesh tetrahedral (VTK)
- node_surf (N,)                            : flag node permukaan dari CSV
- disp_{lc} (N,3), disp_mag_{lc} (N,), stress_{lc} (N,)
  untuk lc di {ver, hor, dia, tor} (vertical/horizontal/diagonal/torsional)

Satuan: panjang & displacement dalam mm, stress dalam MPa (N/mm^2). README dataset
menulis "GPa (N/mm^2)" yang kontradiktif; verifikasi rentang nilai (median max
stress ~1057 antar 381 bracket, vs yield Ti-6Al-4V ~900 MPa) memastikan MPa.

Kolom stress = von Mises stress. Ini eksplisit di paper SimJEB (Whalen, Beyene,
Mueller, 2021, arXiv:2105.03534, caption Fig. 4): "Five vertex-valued scalar
fields were extracted for each load case: the displacement in X,Y,Z directions,
the displacement magnitude, and the von Mises stress."
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pandas as pd
import trimesh

LOAD_CASES = ("ver", "hor", "dia", "tor")

FIELD_COLUMNS = ["id", "surf", "x", "y", "z"] + [
    f"{lc}_{q}" for lc in LOAD_CASES for q in ("xdisp", "ydisp", "zdisp", "magdisp", "stress")
]


def _find_field_csv(sample_dir: Path, bracket_id: str) -> Path:
    """CSV field selalu bernama '{id}field.csv'.

    Diverifikasi terhadap isi kedua zip simresults di Dataverse: seluruh 381 CSV
    memakai pola ini; nama '{id}.csv' yang disebut README dataset tidak pernah
    muncul.
    """
    p = sample_dir / f"{bracket_id}field.csv"
    if not p.exists():
        raise FileNotFoundError(f"CSV field untuk bracket {bracket_id} tidak ditemukan: {p}")
    return p


def parse_bracket(sample_dir: Path, bracket_id: str) -> dict[str, np.ndarray]:
    """Parse satu bracket lengkap. Melempar exception jika file hilang atau tidak konsisten."""
    sample_dir = Path(sample_dir)

    obj_path = sample_dir / f"{bracket_id}.obj"
    vtk_path = sample_dir / f"{bracket_id}.vtk"
    csv_path = _find_field_csv(sample_dir, bracket_id)
    for p in (obj_path, vtk_path):
        if not p.exists():
            raise FileNotFoundError(p)

    # Surface mesh (OBJ). process=False agar vertex tidak di-merge/reorder.
    surf = trimesh.load(obj_path, process=False)
    surf_vertices = np.asarray(surf.vertices, dtype=np.float64)
    surf_faces = np.asarray(surf.faces, dtype=np.int64)

    # Volume mesh tetrahedral (VTK, Gmsh binary).
    vol = meshio.read(vtk_path)
    vol_points = np.asarray(vol.points, dtype=np.float64)
    tet_blocks = [cb.data for cb in vol.cells if cb.type == "tetra"]
    if not tet_blocks:
        raise ValueError(f"{vtk_path.name}: tidak ada sel tetra")
    vol_tets = np.concatenate(tet_blocks).astype(np.int64)

    # Field FEA per node.
    df = pd.read_csv(csv_path)
    missing_cols = set(FIELD_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path.name}: kolom hilang: {sorted(missing_cols)}")

    if len(df) != len(vol_points):
        raise ValueError(
            f"Jumlah node tidak cocok: CSV {len(df)} vs VTK {len(vol_points)} "
            f"untuk bracket {bracket_id}"
        )

    # Kolom id CSV adalah nomor node solver: menaik, tapi TIDAK selalu kontigu
    # (bracket 123 dan 388 punya celah kecil). Cukup pastikan urutannya menaik
    # supaya baris ke-i = node ke-i di VTK; korespondensi sesungguhnya
    # divalidasi lewat kecocokan koordinat di bawah.
    ids = df["id"].to_numpy()
    if not np.all(np.diff(ids) > 0):
        df = df.sort_values("id").reset_index(drop=True)

    # Validasi korespondensi geometris CSV <-> VTK.
    csv_xyz = df[["x", "y", "z"]].to_numpy()
    max_coord_err = float(np.abs(csv_xyz - vol_points).max())
    if max_coord_err > 1e-3:  # toleransi: koordinat CSV ditulis 7 digit signifikan
        raise ValueError(
            f"Koordinat CSV tidak cocok dengan node VTK (selisih maks {max_coord_err:.2e} mm)"
        )

    out: dict[str, np.ndarray] = {
        "surf_vertices": surf_vertices,
        "surf_faces": surf_faces,
        "vol_points": vol_points,
        "vol_tets": vol_tets,
        "node_surf": df["surf"].to_numpy(dtype=bool),
    }
    for lc in LOAD_CASES:
        out[f"disp_{lc}"] = df[[f"{lc}_xdisp", f"{lc}_ydisp", f"{lc}_zdisp"]].to_numpy(
            dtype=np.float32
        )
        out[f"disp_mag_{lc}"] = df[f"{lc}_magdisp"].to_numpy(dtype=np.float32)
        out[f"stress_{lc}"] = df[f"{lc}_stress"].to_numpy(dtype=np.float32)
    return out
