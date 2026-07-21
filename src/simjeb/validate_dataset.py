"""Validasi Fase 1 workstream Model: loader dataset + integritas data.

Cek:
1. Jumlah sampel train/test per split resmi (test_split_0/1/2) ~ 80/20.
2. Tidak ada NaN/Inf di target (y) maupun koordinat (pos) di seluruh bracket.
3. Konsistensi shape per bracket: pos (N,3), y (N,20), node_surf (N,),
   indeks tets di rentang [0, N), jumlah node permukaan = jumlah vertex OBJ.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .dataset import PROCESSED_DIR, TARGET_COLUMNS, SimJEBDataset


def main() -> None:
    print("=== jumlah sampel per split resmi ===")
    for i in (0, 1, 2):
        n_tr = len(SimJEBDataset("train", i))
        n_te = len(SimJEBDataset("test", i))
        print(f"  split_{i}: train {n_tr:>3}, test {n_te:>3} "
              f"(test {100 * n_te / (n_tr + n_te):.1f}%)")

    ds = SimJEBDataset("all", 0, include_surface=True)
    assert len(ds) == 381, f"total {len(ds)} != 381"

    n_bad = 0
    tot_nodes = 0
    for i in tqdm(range(len(ds)), desc="validasi", unit="bracket"):
        it = ds[i]
        bid = it["id"]
        pos, y, tets, surf = it["pos"], it["y"], it["tets"], it["node_surf"]
        n = pos.shape[0]
        tot_nodes += n
        problems = []
        if pos.shape != (n, 3) or y.shape != (n, 20) or surf.shape != (n,):
            problems.append(f"shape aneh: pos{tuple(pos.shape)} y{tuple(y.shape)} surf{tuple(surf.shape)}")
        if tets.numel() and (tets.min() < 0 or tets.max() >= n):
            problems.append(f"indeks tet di luar rentang [0,{n})")
        if not torch.isfinite(y).all():
            problems.append(f"y mengandung NaN/Inf ({int((~torch.isfinite(y)).sum())} nilai)")
        if not torch.isfinite(pos).all():
            problems.append("pos mengandung NaN/Inf")
        if int(surf.sum()) != it["surf_vertices"].shape[0]:
            problems.append(
                f"node_surf ({int(surf.sum())}) != vertex OBJ ({it['surf_vertices'].shape[0]})"
            )
        if problems:
            n_bad += 1
            print(f"\n  id {bid}: " + "; ".join(problems))

    print(f"\nbracket bermasalah : {n_bad}/381")
    print(f"total node         : {tot_nodes:,}")
    print(f"kolom target (20)  : {TARGET_COLUMNS}")

    it = ds[0]
    y = it["y"]
    print(f"\ncontoh item (id {it['id']}): pos {tuple(it['pos'].shape)}, "
          f"tets {tuple(it['tets'].shape)}, y {tuple(y.shape)}")
    print(f"  rentang y per blok: disp [{y[:, :16:5].min():.3f}, {y[:, 3::5].max():.3f}] mm, "
          f"stress maks {y[:, 4::5].max():.1f} MPa")


if __name__ == "__main__":
    main()
