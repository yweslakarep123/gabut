"""PyTorch Dataset untuk SimJEB terproses (data/processed/).

Split memakai kolom resmi dataset `test_split_{0,1,2}` dari manifest
(True = test). JANGAN membuat split acak sendiri — split resmi ini yang
dipakai benchmark paper SimJEB sehingga hasil bisa dibandingkan langsung.

Target `y` berbentuk (N, 20) float32: 5 field x 4 load case, satu nilai per
node volume mesh. Urutan kolom mengikuti tabel benchmark paper (field sebagai
blok per load case):

    kolom = lc_idx * 5 + field_idx
    load case (lc_idx) : ver=0, hor=1, dia=2, tor=3
    field (field_idx)  : disp_x=0, disp_y=1, disp_z=2, disp_mag=3, stress=4

Satuan: mm untuk displacement, MPa untuk von Mises stress.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .parse import LOAD_CASES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FIELD_NAMES = ("disp_x", "disp_y", "disp_z", "disp_mag", "stress")
TARGET_COLUMNS = [f"{lc}_{f}" for lc in LOAD_CASES for f in FIELD_NAMES]  # 20 kolom


def stack_targets(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Susun fields.npz menjadi matriks target (N, 20) sesuai TARGET_COLUMNS."""
    cols = []
    for lc in LOAD_CASES:
        cols.append(fields[f"disp_{lc}"])            # (N,3): x, y, z
        cols.append(fields[f"disp_mag_{lc}"][:, None])
        cols.append(fields[f"stress_{lc}"][:, None])
    return np.concatenate(cols, axis=1).astype(np.float32)


class SimJEBDataset(Dataset):
    """Satu item = satu bracket (graf mesh utuh, ukuran variabel antar item).

    Item berupa dict tensor:
      id         : int
      pos        : (N, 3)  float32 — koordinat node volume mesh (mm)
      tets       : (Nt, 4) int64   — konektivitas tetrahedral
      node_surf  : (N,)    bool    — node pada permukaan
      y          : (N, 20) float32 — target, lihat TARGET_COLUMNS
    include_surface=True menambahkan surf_vertices/surf_faces (mesh OBJ).
    """

    def __init__(
        self,
        split: str = "train",
        split_idx: int = 0,
        root: Path = PROCESSED_DIR,
        include_surface: bool = False,
    ) -> None:
        assert split in ("train", "test", "all"), split
        assert split_idx in (0, 1, 2), split_idx
        self.root = Path(root)
        self.split = split
        self.split_idx = split_idx
        self.include_surface = include_surface

        manifest = pd.read_csv(self.root / "manifest.csv", index_col="id")
        assert (manifest["status"] == "ok").all(), "manifest berisi sampel tidak ok"
        is_test = manifest[f"test_split_{split_idx}"].astype(bool)
        if split == "train":
            keep = ~is_test
        elif split == "test":
            keep = is_test
        else:
            keep = pd.Series(True, index=manifest.index)
        self.manifest = manifest[keep]
        self.ids = self.manifest.index.to_numpy()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        bid = int(self.ids[idx])
        geom = np.load(self.root / str(bid) / "geometry.npz")
        fields = np.load(self.root / str(bid) / "fields.npz")

        item = {
            "id": bid,
            "pos": torch.from_numpy(geom["vol_points"].astype(np.float32)),
            "tets": torch.from_numpy(geom["vol_tets"].astype(np.int64)),
            "node_surf": torch.from_numpy(geom["node_surf"]),
            "y": torch.from_numpy(stack_targets(fields)),
        }
        if self.include_surface:
            item["surf_vertices"] = torch.from_numpy(geom["surf_vertices"].astype(np.float32))
            item["surf_faces"] = torch.from_numpy(geom["surf_faces"].astype(np.int64))
        return item
