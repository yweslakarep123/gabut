"""Parse BC seluruh bracket dari .fem, simpan mask ke data/processed/{id}/bc.npz.

Juga tulis reports/bc_residuals.csv + log outlier (residual baut/interface
jauh dari norma), supaya kasus seperti id 71 tidak terserap diam-diam.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .bc_parse import BracketBC, bc_masks, match_bolts_to_template, parse_fem
from .dataset import PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEM_DIR = PROJECT_ROOT / "data" / "raw" / "full_fem"
REPORTS = PROJECT_ROOT / "reports"

# Ambang outlier relatif ke sebaran probe Fase sebelumnya:
# interface max||Δ|| probe ≈ 0.6 mm; baut slot terburuk ≈ 4.2 mm (id 71).
# Flag kalau > 2 mm interface atau > 5 mm baut (sedikit di atas id 71).
IFACE_OUTLIER_MM = 2.0
BOLT_OUTLIER_MM = 5.0


def build_all(fem_dir: Path = FEM_DIR) -> pd.DataFrame:
    REPORTS.mkdir(exist_ok=True)
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv", index_col="id")
    ids = manifest.index.to_numpy()

    bcs: list[BracketBC] = []
    failures: list[tuple[int, str]] = []
    skipped: list[tuple[int, str]] = []
    for bid in tqdm(ids, desc="parse BC", unit="bracket"):
        path = fem_dir / f"{bid}.fem"
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            if path.stat().st_size == 0:
                # Kasus nyata di Dataverse: 293.fem = 0 byte. Selalu di train
                # (bukan test) pada ketiga split — lewati dari training, jangan
                # isi mask lewat fallback jarak kanonik.
                skipped.append((int(bid), "FILE_KOSONG (0 byte di zip Dataverse)"))
                continue
            bc = parse_fem(path)
            geom = np.load(PROCESSED_DIR / str(bid) / "geometry.npz")
            n = len(geom["vol_points"])
            is_sup, is_load = bc_masks(bc, n)
            if is_sup.sum() == 0 or is_load.sum() == 0:
                raise ValueError(
                    f"mask kosong: support={is_sup.sum()} load={is_load.sum()} "
                    f"(deps bolt={bc.n_bolt_dependents} iface={bc.n_interface_dependents})"
                )
            np.savez(
                PROCESSED_DIR / str(bid) / "bc.npz",
                is_support=is_sup,
                is_load=is_load,
                bolt_centers=bc.bolt_centers.astype(np.float32),
                interface=bc.interface.astype(np.float32),
            )
            bcs.append(bc)
        except Exception as exc:
            failures.append((int(bid), f"{type(exc).__name__}: {exc}"))

    if skipped:
        print(f"\nSKIPPED — tanpa BC (tidak pakai fallback kanonik) ({len(skipped)}):")
        for bid, reason in skipped:
            print(f"  id {bid}: {reason}")
        pd.DataFrame(skipped, columns=["id", "reason"]).to_csv(
            REPORTS / "bc_skipped.csv", index=False
        )

    if failures:
        print(f"\nGAGAL parse ({len(failures)}):")
        for bid, reason in failures:
            print(f"  id {bid}: {reason}")
        raise RuntimeError(f"{len(failures)} bracket gagal parse BC")

    if len(bcs) < 300:
        raise RuntimeError(f"terlalu sedikit BC sukses: {len(bcs)}")
    # Residual vs mean (setelah matching baut ke template id pertama)
    bolts_raw = np.stack([b.bolt_centers for b in bcs])
    ifaces = np.stack([b.interface for b in bcs])
    template = bolts_raw[0]
    matched = np.stack([match_bolts_to_template(template, bolts_raw[i]) for i in range(len(bcs))])
    mean_b = matched.mean(0)
    mean_i = ifaces.mean(0)

    rows = []
    outliers = []
    for i, b in enumerate(bcs):
        di = float(np.linalg.norm(ifaces[i] - mean_i))
        db = np.linalg.norm(matched[i] - mean_b, axis=1)
        db_max = float(db.max())
        flag = di > IFACE_OUTLIER_MM or db_max > BOLT_OUTLIER_MM
        rows.append(
            {
                "id": b.bracket_id,
                "category": manifest.loc[b.bracket_id, "category"],
                "iface_dx": di,
                "bolt_dmax": db_max,
                "bolt_d0": float(db[0]),
                "bolt_d1": float(db[1]),
                "bolt_d2": float(db[2]),
                "bolt_d3": float(db[3]),
                "n_support": int(len(b.support_gids)),
                "n_load": int(len(b.load_gids)),
                "outlier": flag,
            }
        )
        if flag:
            outliers.append((b.bracket_id, di, db_max))

    df = pd.DataFrame(rows).set_index("id").sort_index()
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "bc_residuals.csv"
    df.to_csv(out)

    print(f"\nBC OK: {len(bcs)}/381")
    print(f"interface: mean||Δ||={df.iface_dx.mean():.3f} max={df.iface_dx.max():.3f} mm")
    print(f"baut:      mean dmax={df.bolt_dmax.mean():.3f} max={df.bolt_dmax.max():.3f} mm")
    print(f"outlier (iface>{IFACE_OUTLIER_MM} atau baut>{BOLT_OUTLIER_MM} mm): {len(outliers)}")
    if outliers:
        print("  id   ||Δiface||  max||Δbaut||")
        for bid, di, db in sorted(outliers, key=lambda x: -x[2]):
            print(f"  {bid:<4} {di:>10.3f} {db:>12.3f}")
    print(f"tersimpan: {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fem-dir", type=Path, default=FEM_DIR)
    args = p.parse_args()
    build_all(args.fem_dir)
