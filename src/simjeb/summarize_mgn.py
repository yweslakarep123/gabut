"""Agregasi MAE MGN 3-split resmi (setara Table 2).

Membaca reports/mgn_split{0,1,2}_test_mae.csv — mendukung skema kolom lama
split_0 (naive_split0_76) dan skema baru (naive_matched).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .baseline import PAPER_MAE_RELABELED
from .dataset import TARGET_COLUMNS

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _load_split(idx: int) -> pd.DataFrame:
    path = REPORTS / f"mgn_split{idx}_test_mae.csv"
    df = pd.read_csv(path, index_col=0)
    if "naive_matched" in df.columns:
        naive_col = "naive_matched"
        ratio_col = "ratio_vs_naive_matched"
    elif "naive_split0_76" in df.columns:
        naive_col = "naive_split0_76"
        ratio_col = "ratio_vs_naive_76"
    else:
        raise KeyError(f"{path}: tidak ada kolom naive_matched / naive_split0_76")
    out = pd.DataFrame(
        {
            "mgn": df["mgn"],
            "naive_matched": df[naive_col],
            "ratio": df[ratio_col] if ratio_col in df.columns else df["mgn"] / df[naive_col],
        },
        index=df.index,
    )
    n_eval = int(df["n_eval_mgn"].iloc[0]) if "n_eval_mgn" in df.columns else -1
    skip = str(df["mgn_skip_oom"].iloc[0]) if "mgn_skip_oom" in df.columns else ""
    return out, n_eval, skip


def main() -> None:
    frames = []
    print("=== per-split (apple-to-apple vs naive_matched) ===")
    for s in (0, 1, 2):
        df, n_eval, skip = _load_split(s)
        wins = int((df["ratio"] < 1.0).sum())
        print(f"\nsplit_{s}: n_eval={n_eval}  skip_oom=[{skip}]  "
              f"wins={wins}/20  mean_ratio={df['ratio'].mean():.3f}")
        frames.append(df)
        for name in TARGET_COLUMNS:
            print(f"  {name:<14} mgn={df.loc[name,'mgn']:.4f}  "
                  f"naive={df.loc[name,'naive_matched']:.4f}  "
                  f"ratio={df.loc[name,'ratio']:.3f}")

    mgn = pd.concat([f["mgn"] for f in frames], axis=1)
    mgn.columns = ["split0", "split1", "split2"]
    naive = pd.concat([f["naive_matched"] for f in frames], axis=1)
    naive.columns = ["split0", "split1", "split2"]

    summary = pd.DataFrame(
        {
            "mgn_split0": mgn["split0"],
            "mgn_split1": mgn["split1"],
            "mgn_split2": mgn["split2"],
            "mgn_mean": mgn.mean(axis=1),
            "mgn_std": mgn.std(axis=1),
            "naive_matched_split0": naive["split0"],
            "naive_matched_split1": naive["split1"],
            "naive_matched_split2": naive["split2"],
            "naive_matched_mean": naive.mean(axis=1),
            "ratio_mean": mgn.mean(axis=1) / naive.mean(axis=1),
            "paper_relabeled": [PAPER_MAE_RELABELED[n] for n in mgn.index],
        },
        index=mgn.index,
    )
    wins = int((summary["ratio_mean"] < 1.0).sum())
    print("\n=== mean 3-split (setara Table 2) ===")
    print(f"{'target':<14}{'mgn_mean':>10}{'naive_m':>10}{'paper*':>10}{'ratio':>8}")
    for name in TARGET_COLUMNS:
        r = summary.loc[name]
        print(f"{name:<14}{r.mgn_mean:>10.4f}{r.naive_matched_mean:>10.4f}"
              f"{r.paper_relabeled:>10.4g}{r.ratio_mean:>8.3f}")
    print(f"\nmengalahkan naive_matched (mean 3-split): {wins}/20")
    dest = REPORTS / "mgn_3split_summary.csv"
    summary.to_csv(dest)
    print(f"tersimpan: {dest}")


if __name__ == "__main__":
    main()
