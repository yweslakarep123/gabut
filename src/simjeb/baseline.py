"""Reproduksi naive baseline paper SimJEB (Section 6, Table 2).

Model: untuk tiap kombinasi field x load case (20 kombinasi), SATU regresi
polinomial derajat-3 f(x, y, z) -> nilai field, di-fit dengan least squares
atas SELURUH node dari SEMUA bracket training yang dipool jadi satu.
Hasilnya mendekati rata-rata field lintas desain sebagai fungsi posisi.

Implementasi: karena ke-20 target memakai matriks desain yang sama
(20 monomial derajat <=3 atas x,y,z), fit dilakukan sekali per split lewat
akumulasi normal equations per bracket (X^T X dan X^T Y berukuran 20x20),
sehingga tidak perlu menampung ~26 juta node di memori sekaligus.

Koordinat diskalakan 1/100 (mm -> dm) hanya untuk konsistensi numerik
normal equations; secara matematis kelas modelnya identik.

MAE dilaporkan pooled atas seluruh node test (dan juga rata-rata per-bracket
sebagai pembanding, karena paper tidak menyebut eksplisit cara agregasinya).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import PROCESSED_DIR, TARGET_COLUMNS, stack_targets

COORD_SCALE = 0.01  # mm -> dm, demi conditioning normal equations

# 20 monomial derajat <= 3 atas (x, y, z): eksponen (i, j, k), i+j+k <= 3
EXPONENTS = [
    (i, j, k)
    for i, j, k in itertools.product(range(4), repeat=3)
    if i + j + k <= 3
]
assert len(EXPONENTS) == 20

# MAE naive baseline sesuai transkripsi Table 2 yang diberikan (hasil
# ekstraksi teks otomatis dari PDF), dengan header kolom apa adanya.
#
# Ada ketidaksesuaian antara transkripsi tabel tersebut dan label load case
# yang tervalidasi fisik lewat kartu FORCE/MOMENT di file .fem + pola arah
# displacement (prefix tor tangensial murni, ver dominan-Z): angka
# reproduksi kami cocok 20/20 sampai 3 digit signifikan dengan transkripsi
# SETELAH kolomnya dipermutasi (Vert.→dia, Horiz.→hor, Diag.→tor, Tor.→ver).
# Sumber ketidaksesuaiannya belum dipastikan — bisa dari tabel publikasi
# ataupun dari ekstraksi teks yang keliru mengurutkan kolom. Label yang
# tervalidasi fisik yang dipakai, terlepas dari sumber ketidaksesuaiannya.
# Pembanding apple-to-apple: PAPER_MAE_RELABELED.
PAPER_MAE = {
    "ver_disp_x": 6.27e-2, "ver_disp_y": 4.17e-2, "ver_disp_z": 1.46e-1,
    "ver_disp_mag": 1.69e-1, "ver_stress": 60.1,
    "hor_disp_x": 1.62e-1, "hor_disp_y": 1.97e-2, "hor_disp_z": 1.62e-1,
    "hor_disp_mag": 2.51e-1, "hor_stress": 89.3,
    "dia_disp_x": 3.17e-2, "dia_disp_y": 1.66e-2, "dia_disp_z": 2.80e-2,
    "dia_disp_mag": 4.21e-2, "dia_stress": 36.1,
    "tor_disp_x": 1.27e-1, "tor_disp_y": 4.87e-2, "tor_disp_z": 2.37e-1,
    "tor_disp_mag": 2.87e-1, "tor_stress": 84.4,
}

# Transkripsi Table 2 dipetakan ke label load case yang tervalidasi fisik
# (lihat catatan di atas). Tidak mengklaim tabel publikasi salah — hanya
# menyelaraskan angka ke label yang dipakai di pipeline ini.
_PAPER_COLUMN_FIX = {"dia": "ver", "hor": "hor", "tor": "dia", "ver": "tor"}
PAPER_MAE_RELABELED = {
    f"{lc}_{f}": PAPER_MAE[f"{paper_lc}_{f}"]
    for lc, paper_lc in _PAPER_COLUMN_FIX.items()
    for f in ("disp_x", "disp_y", "disp_z", "disp_mag", "stress")
}


def poly_features(pos: np.ndarray) -> np.ndarray:
    """(N,3) koordinat -> (N,20) monomial derajat <=3, float64."""
    p = pos.astype(np.float64) * COORD_SCALE
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    cols = [(x**i) * (y**j) * (z**k) for i, j, k in EXPONENTS]
    return np.stack(cols, axis=1)


def _load_bracket(root: Path, bid: int) -> tuple[np.ndarray, np.ndarray]:
    geom = np.load(root / str(bid) / "geometry.npz")
    fields = np.load(root / str(bid) / "fields.npz")
    return geom["vol_points"], stack_targets(fields).astype(np.float64)


def fit_split(root: Path, train_ids: np.ndarray) -> np.ndarray:
    """Akumulasi X^T X, X^T Y atas semua bracket training; solve -> (20,20) koef."""
    xtx = np.zeros((20, 20))
    xty = np.zeros((20, 20))  # 20 fitur x 20 target
    for bid in tqdm(train_ids, desc="fit", unit="bracket", leave=False):
        pos, y = _load_bracket(root, bid)
        X = poly_features(pos)
        xtx += X.T @ X
        xty += X.T @ y
    return np.linalg.solve(xtx, xty)


def eval_split(root: Path, test_ids: np.ndarray, coef: np.ndarray) -> dict[str, np.ndarray]:
    """MAE per target: pooled atas semua node test + rata-rata per-bracket."""
    abs_sum = np.zeros(20)
    n_nodes = 0
    per_bracket = []
    for bid in tqdm(test_ids, desc="eval", unit="bracket", leave=False):
        pos, y = _load_bracket(root, bid)
        err = np.abs(poly_features(pos) @ coef - y)
        abs_sum += err.sum(axis=0)
        n_nodes += len(err)
        per_bracket.append(err.mean(axis=0))
    return {
        "pooled": abs_sum / n_nodes,
        "per_bracket": np.mean(per_bracket, axis=0),
    }


def main() -> None:
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv", index_col="id")
    results_pooled = np.zeros((3, 20))
    results_pb = np.zeros((3, 20))

    for s in (0, 1, 2):
        is_test = manifest[f"test_split_{s}"].astype(bool)
        train_ids = manifest.index[~is_test].to_numpy()
        test_ids = manifest.index[is_test].to_numpy()
        coef = fit_split(PROCESSED_DIR, train_ids)
        res = eval_split(PROCESSED_DIR, test_ids, coef)
        results_pooled[s] = res["pooled"]
        results_pb[s] = res["per_bracket"]
        print(f"split_{s} selesai (train {len(train_ids)}, test {len(test_ids)})")

    print("\n=== MAE naive baseline: reproduksi vs paper ===")
    print("(pooled atas node test; 'pb' = rata-rata per-bracket; 'paper*' = "
          "Table 2 dipetakan ke label load case yang tervalidasi fisik)")
    hdr = (f"{'target':<14}{'split0':>9}{'split1':>9}{'split2':>9}"
           f"{'mean':>9}{'std':>8}{'mean_pb':>9}{'paper*':>9}{'rasio':>7}")
    print(hdr)
    for j, name in enumerate(TARGET_COLUMNS):
        vals = results_pooled[:, j]
        mean, std = vals.mean(), vals.std()
        paper = PAPER_MAE_RELABELED[name]
        print(f"{name:<14}{vals[0]:>9.4f}{vals[1]:>9.4f}{vals[2]:>9.4f}"
              f"{mean:>9.4f}{std:>8.4f}{results_pb[:, j].mean():>9.4f}"
              f"{paper:>9.4g}{mean / paper:>7.2f}")

    out = pd.DataFrame(
        {
            "split0": results_pooled[0], "split1": results_pooled[1],
            "split2": results_pooled[2], "mean": results_pooled.mean(0),
            "std": results_pooled.std(0), "mean_per_bracket": results_pb.mean(0),
            "paper_as_printed": [PAPER_MAE[n] for n in TARGET_COLUMNS],
            "paper_relabeled": [PAPER_MAE_RELABELED[n] for n in TARGET_COLUMNS],
        },
        index=TARGET_COLUMNS,
    )
    dest = PROCESSED_DIR.parent.parent / "reports" / "baseline_mae.csv"
    out.to_csv(dest)
    print(f"\ntersimpan: {dest}")


if __name__ == "__main__":
    main()
