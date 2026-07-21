"""Build dataset SimJEB siap-ML: parse 381 bracket -> data/processed/.

Layout output:
- data/processed/{id}/geometry.npz : surf_vertices, surf_faces, vol_points,
  vol_tets, node_surf
- data/processed/{id}/fields.npz   : disp_{lc} (N,3), disp_mag_{lc} (N,),
  stress_{lc} (N,) untuk lc di {ver, hor, dia, tor}
- data/processed/manifest.csv      : satu baris per bracket — id, kategori,
  status parse, path, statistik dasar, max stress/disp per load case, dan
  train/test split resmi dataset (test_split_0/1/2; JANGAN buat split sendiri
  agar hasil bisa dibanding ke benchmark SimJEB/DeepJEB).
- data/processed/failures.log      : id + traceback untuk setiap bracket gagal.

Kegagalan per bracket dicatat dan TIDAK menghentikan batch.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .parse import LOAD_CASES, parse_bracket

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_DIR = PROJECT_ROOT / "data" / "raw" / "full"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GEOMETRY_KEYS = ("surf_vertices", "surf_faces", "vol_points", "vol_tets", "node_surf")

META_KEEP = [
    "category", "num_vertices", "num_faces", "num_tets", "volume", "surface_area",
    "mass", "test_split_0", "test_split_1", "test_split_2",
]


def build_one(bracket_id: int, src_dir: Path, out_root: Path) -> dict:
    """Parse satu bracket dan simpan ke out_root/{id}/. Mengembalikan baris manifest."""
    parsed = parse_bracket(src_dir, str(bracket_id))
    out_dir = out_root / str(bracket_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = {k: parsed[k] for k in GEOMETRY_KEYS}
    # int32 cukup untuk indeks node (<2^31) dan menghemat separuh ukuran.
    geom["surf_faces"] = geom["surf_faces"].astype(np.int32)
    geom["vol_tets"] = geom["vol_tets"].astype(np.int32)
    np.savez(out_dir / "geometry.npz", **geom)

    fields = {k: v for k, v in parsed.items() if k not in GEOMETRY_KEYS}
    np.savez(out_dir / "fields.npz", **fields)

    row = {
        "id": bracket_id,
        "status": "ok",
        "error": "",
        "geometry_path": str((out_dir / "geometry.npz").relative_to(out_root)),
        "fields_path": str((out_dir / "fields.npz").relative_to(out_root)),
        "n_nodes": len(parsed["vol_points"]),
        "n_tets": len(parsed["vol_tets"]),
        "n_surf_vertices": len(parsed["surf_vertices"]),
    }
    for lc in LOAD_CASES:
        row[f"max_stress_{lc}"] = float(parsed[f"stress_{lc}"].max())
        row[f"max_magdisp_{lc}"] = float(parsed[f"disp_mag_{lc}"].max())
    return row


def build_all(src_dir: Path = FULL_DIR, out_root: Path = PROCESSED_DIR) -> pd.DataFrame:
    meta = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "all_bracket_metadata.csv")
    out_root.mkdir(parents=True, exist_ok=True)
    failures_log = out_root / "failures.log"

    rows: list[dict] = []
    failures: list[tuple[int, str]] = []
    with open(failures_log, "w") as flog:
        for _, mrow in tqdm(list(meta.iterrows()), desc="parse", unit="bracket"):
            bid = int(mrow["id"])
            try:
                row = build_one(bid, src_dir, out_root)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append((bid, reason))
                flog.write(f"=== bracket {bid}: {reason}\n{traceback.format_exc()}\n")
                flog.flush()
                row = {"id": bid, "status": "failed", "error": reason}
            for col in META_KEEP:
                row[col] = mrow[col]
            rows.append(row)

    manifest = pd.DataFrame(rows).set_index("id").sort_index()
    manifest.to_csv(out_root / "manifest.csv")

    n_ok = int((manifest["status"] == "ok").sum())
    print(f"\n{n_ok}/{len(manifest)} bracket berhasil diparse")
    if failures:
        print(f"GAGAL ({len(failures)}):")
        for bid, reason in failures:
            print(f"  id {bid}: {reason}")
    else:
        print("Tidak ada kegagalan.")
    return manifest


if __name__ == "__main__":
    build_all()
