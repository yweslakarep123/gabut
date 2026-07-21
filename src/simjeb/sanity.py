"""Sanity check Phase 4 untuk output data/processed/.

1. Verifikasi jumlah sampel vs 381 dan status parse.
2. Statistik distribusi max stress / max displacement antar sampel
   (+ histogram ke reports/).
3. Visualisasi geometri & field von Mises stress satu bracket dengan pyvista
   (default id 123 — salah satu dari dua bracket dengan penomoran node
   bercelah di Phase 3, supaya pemetaan field-ke-node teruji secara visual).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

from .parse import LOAD_CASES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXPECTED_COUNT = 381

LC_LABEL = {"ver": "vertical", "hor": "horizontal", "dia": "diagonal", "tor": "torsional"}


def check_counts(manifest: pd.DataFrame) -> None:
    n_ok = int((manifest["status"] == "ok").sum())
    print(f"jumlah sampel di manifest : {len(manifest)} (harap {EXPECTED_COUNT})")
    print(f"status ok                 : {n_ok}")
    assert len(manifest) == EXPECTED_COUNT, "jumlah sampel tidak sesuai harapan"
    assert n_ok == EXPECTED_COUNT, "ada sampel yang tidak ok di manifest"

    on_disk = sorted(
        int(p.name) for p in PROCESSED_DIR.iterdir() if p.is_dir() and p.name.isdigit()
    )
    missing = set(manifest.index) - set(on_disk)
    assert not missing, f"folder hilang di disk untuk id: {sorted(missing)}"
    print(f"folder di disk            : {len(on_disk)} (semua id manifest ada)")


def print_stats(manifest: pd.DataFrame) -> None:
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]

    for qty, unit in (("stress", "MPa"), ("magdisp", "mm")):
        print(f"\n=== distribusi max {qty} per sampel ({unit}) ===")
        header = f"{'lc':<12}" + "".join(f"{f'p{int(q*100)}':>10}" for q in qs) + f"{'min':>10}{'max':>10}"
        print(header)
        cols = []
        for lc in LOAD_CASES:
            col = manifest[f"max_{qty}_{lc}"]
            cols.append(col)
            vals = "".join(f"{col.quantile(q):>10.2f}" for q in qs)
            print(f"{LC_LABEL[lc]:<12}{vals}{col.min():>10.2f}{col.max():>10.2f}")
        overall = pd.concat(cols, axis=1).max(axis=1)
        vals = "".join(f"{overall.quantile(q):>10.2f}" for q in qs)
        print(f"{'gabungan':<12}{vals}{overall.min():>10.2f}{overall.max():>10.2f}")

    print("\n=== median max stress gabungan per kategori bentuk ===")
    stress_all = manifest[[f"max_stress_{lc}" for lc in LOAD_CASES]].max(axis=1)
    med = stress_all.groupby(manifest["category"]).median().sort_values()
    for cat, v in med.items():
        n = (manifest["category"] == cat).sum()
        print(f"  {cat:<10} (n={n:>3}): {v:8.1f} MPa")


def save_histograms(manifest: pd.DataFrame) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    stress_all = manifest[[f"max_stress_{lc}" for lc in LOAD_CASES]].max(axis=1)
    disp_all = manifest[[f"max_magdisp_{lc}" for lc in LOAD_CASES]].max(axis=1)

    axes[0].hist(stress_all, bins=np.geomspace(stress_all.min(), stress_all.max(), 40))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("max von Mises stress per sampel (MPa, gabungan 4 load case)")
    axes[0].set_ylabel("jumlah bracket")

    axes[1].hist(disp_all, bins=np.geomspace(disp_all.min(), disp_all.max(), 40))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("max |displacement| per sampel (mm, gabungan 4 load case)")
    axes[1].set_ylabel("jumlah bracket")

    fig.suptitle(f"SimJEB — distribusi antar {len(manifest)} bracket")
    fig.tight_layout()
    out = REPORTS_DIR / "distributions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _load_grid(bracket_id: int) -> tuple[pv.UnstructuredGrid, dict]:
    geom = np.load(PROCESSED_DIR / str(bracket_id) / "geometry.npz")
    fields = np.load(PROCESSED_DIR / str(bracket_id) / "fields.npz")
    tets = geom["vol_tets"]
    cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64), tets.astype(np.int64)])
    grid = pv.UnstructuredGrid(
        cells.ravel(), np.full(len(tets), pv.CellType.TETRA), geom["vol_points"]
    )
    return grid, {k: fields[k] for k in fields.files}


def visualize(bracket_id: int) -> list[Path]:
    REPORTS_DIR.mkdir(exist_ok=True)
    grid, fields = _load_grid(bracket_id)
    surf = grid.extract_surface()
    outs = []

    # 1) geometri polos
    pl = pv.Plotter(off_screen=True, window_size=(1100, 850))
    pl.add_mesh(surf, color="lightsteelblue", smooth_shading=True, show_edges=False)
    pl.add_text(f"bracket {bracket_id} — geometri", font_size=12)
    pl.show_axes()
    out = REPORTS_DIR / f"bracket{bracket_id}_geometry.png"
    pl.screenshot(out)
    pl.close()
    outs.append(out)

    # 2) von Mises stress, 4 load case (skala warna dipotong di p99 agar
    #    konsentrasi lokal tidak menenggelamkan pola global)
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1600, 1200))
    for i, lc in enumerate(LOAD_CASES):
        pl.subplot(i // 2, i % 2)
        s = fields[f"stress_{lc}"]
        surf_lc = surf.copy(deep=False)
        surf_lc.point_data["von Mises (MPa)"] = s[surf_lc.point_data["vtkOriginalPointIds"]]
        pl.add_mesh(
            surf_lc,
            scalars="von Mises (MPa)",
            cmap="turbo",
            clim=(0.0, float(np.percentile(s, 99))),
            smooth_shading=True,
        )
        pl.add_text(f"{LC_LABEL[lc]} (maks {s.max():.0f} MPa)", font_size=11)
    pl.link_views()
    out = REPORTS_DIR / f"bracket{bracket_id}_stress.png"
    pl.screenshot(out)
    pl.close()
    outs.append(out)
    return outs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check output SimJEB")
    parser.add_argument("--viz-id", type=int, default=123, help="id bracket untuk visualisasi")
    args = parser.parse_args()

    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv", index_col="id")
    check_counts(manifest)
    print_stats(manifest)
    hist = save_histograms(manifest)
    print(f"\nhistogram: {hist}")
    for p in visualize(args.viz_id):
        print(f"render   : {p}")


if __name__ == "__main__":
    main()
