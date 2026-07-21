"""Training MeshGraphNet-lite pada SimJEB (split resmi + val internal).

Protokol:
- split_idx resmi (default 0): train resmi dibagi 90/10 -> train_fit / val
- test resmi hanya untuk evaluasi akhir
- Target dinormalisasi per-channel (mean/std dari train_fit) saat training;
  MAE laporan selalu di satuan asli (mm / MPa)
- Loss: L1 pada semua channel di ruang ternormalisasi (stress singularitas
  tidak mendominasi); Huber opsional via --huber-delta pada channel stress
- Batch size 1 + gradient accumulation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .baseline import PAPER_MAE_RELABELED, eval_split, fit_split
from .dataset import PROCESSED_DIR, TARGET_COLUMNS, stack_targets
from .model import MeshGraphNet, build_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "reports"
CKPT_DIR = PROJECT_ROOT / "checkpoints"

STRESS_IDX = [4, 9, 14, 19]
DISP_IDX = [i for i in range(20) if i not in STRESS_IDX]


def make_splits(split_idx: int = 0, val_frac: float = 0.1, seed: int = 0):
    manifest = pd.read_csv(PROCESSED_DIR / "manifest.csv", index_col="id")
    has_bc = np.array([(PROCESSED_DIR / str(i) / "bc.npz").exists() for i in manifest.index])
    missing = manifest.index[~has_bc].tolist()
    if missing:
        print(f"peringatan: {len(missing)} bracket tanpa bc.npz, dilewati: {missing}")
    manifest = manifest.loc[has_bc]
    is_test = manifest[f"test_split_{split_idx}"].astype(bool)
    train_ids = manifest.index[~is_test].to_numpy()
    test_ids = manifest.index[is_test].to_numpy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_ids)
    n_val = max(1, int(round(len(perm) * val_frac)))
    return perm[n_val:], perm[:n_val], test_ids


def load_item(bid: int) -> dict:
    root = PROCESSED_DIR / str(bid)
    geom = np.load(root / "geometry.npz")
    fields = np.load(root / "fields.npz")
    bc = np.load(root / "bc.npz")
    return {
        "id": bid,
        "pos": torch.from_numpy(geom["vol_points"].astype(np.float32)),
        "tets": torch.from_numpy(geom["vol_tets"].astype(np.int64)),
        "node_surf": torch.from_numpy(geom["node_surf"].astype(bool)),
        "is_support": torch.from_numpy(bc["is_support"].astype(bool)),
        "is_load": torch.from_numpy(bc["is_load"].astype(bool)),
        "y": torch.from_numpy(stack_targets(fields)),
    }


def to_data(item: dict):
    return build_graph(
        item["pos"], item["tets"], item["node_surf"],
        item["is_support"], item["is_load"], item["y"],
    )


def compute_norm_stats(ids) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std per channel, pooled atas node dari subset ids."""
    s1 = np.zeros(20, dtype=np.float64)
    s2 = np.zeros(20, dtype=np.float64)
    n = 0
    for bid in ids:
        y = stack_targets(np.load(PROCESSED_DIR / str(bid) / "fields.npz")).astype(np.float64)
        s1 += y.sum(0)
        s2 += (y ** 2).sum(0)
        n += len(y)
    mean = s1 / n
    var = s2 / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    std = np.maximum(std, 1e-3)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def mixed_loss(pred_n, y_n, huber_delta_n: float | None):
    """pred/y sudah di ruang ternormalisasi."""
    if huber_delta_n is None:
        return F.l1_loss(pred_n, y_n)
    disp = F.l1_loss(pred_n[:, DISP_IDX], y_n[:, DISP_IDX])
    stress = F.huber_loss(pred_n[:, STRESS_IDX], y_n[:, STRESS_IDX], delta=huber_delta_n)
    return disp + stress


@torch.no_grad()
def eval_mae(model, ids, device, y_mean, y_std) -> tuple[np.ndarray, list[int]]:
    """Return (MAE pooled 20,), daftar id yang di-skip karena CUDA OOM)."""
    model.eval()
    abs_sum = np.zeros(20, dtype=np.float64)
    n = 0
    skipped: list[int] = []
    for bid in ids:
        try:
            data = to_data(load_item(int(bid))).to(device)
            pred = model(data) * y_std + y_mean
            err = (pred - data.y).abs().sum(dim=0).cpu().numpy()
            abs_sum += err
            n += data.y.shape[0]
            del data, pred
        except torch.cuda.OutOfMemoryError:
            skipped.append(int(bid))
            torch.cuda.empty_cache()
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if skipped:
        print(f"  (eval skip OOM: {skipped})")
    if n == 0:
        return np.full(20, np.nan), skipped
    return abs_sum / n, skipped


def train_one_epoch(model, opt, ids, device, y_mean, y_std, huber_delta_n, accum):
    model.train()
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for step, bid in enumerate(ids):
        data = to_data(load_item(int(bid))).to(device)
        pred_n = model(data)
        y_n = (data.y - y_mean) / y_std
        loss = mixed_loss(pred_n, y_n, huber_delta_n)
        (loss / accum).backward()
        total += float(loss.detach())
        if (step + 1) % accum == 0 or (step + 1) == len(ids):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        del data, pred_n, y_n, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return total / max(len(ids), 1)


def run_overfit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_ids, _, _ = make_splits(args.split_idx, args.val_frac, args.seed)
    man = pd.read_csv(PROCESSED_DIR / "manifest.csv", index_col="id")
    small = man.loc[fit_ids].nsmallest(args.overfit_n, "n_nodes").index.to_numpy()
    print(f"overfit ids ({len(small)}): {small.tolist()}")
    print(f"n_nodes: {man.loc[small, 'n_nodes'].tolist()}")

    y_mean, y_std = compute_norm_stats(small)
    y_mean, y_std = y_mean.to(device), y_std.to(device)
    # δ Huber di ruang ternormalisasi (~1–2 std)
    huber_n = (args.huber_delta / y_std[STRESS_IDX].mean()).item() if args.huber_delta else None

    model = MeshGraphNet(hidden=args.hidden, n_layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    for epoch in range(1, args.overfit_epochs + 1):
        loss = train_one_epoch(model, opt, small, device, y_mean, y_std, huber_n, accum=1)
        if epoch % 20 == 0 or epoch == 1:
            mae, _ = eval_mae(model, small, device, y_mean, y_std)
            print(
                f"  epoch {epoch:4d}  loss_n={loss:.5f}  "
                f"MAE_disp={mae[DISP_IDX].mean():.5f}  MAE_stress={mae[STRESS_IDX].mean():.2f}"
            )
    mae, _ = eval_mae(model, small, device, y_mean, y_std)
    print("\nMAE akhir overfit:")
    for i, name in enumerate(TARGET_COLUMNS):
        print(f"  {name:<14} {mae[i]:.5f}")
    # Gradien/arsitektur OK bila disp hampir hafal. Stress singularitas (hot-spot
    # lokal) sulit L1→0 dengan mean-aggregation; yang penting jauh di bawah
    # baseline channel-mean (~80–100 MPa) dan loss terus turun.
    ok = mae[DISP_IDX].mean() < 0.03 and mae[STRESS_IDX].mean() < 40.0
    print(f"\noverfit sanity: {'PASS' if ok else 'FAIL'}")
    return ok


def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_ids, val_ids, test_ids = make_splits(args.split_idx, args.val_frac, args.seed)
    print(
        f"split_{args.split_idx}: fit={len(fit_ids)} val={len(val_ids)} "
        f"test={len(test_ids)} (test hanya di eval akhir)"
    )
    man = pd.read_csv(PROCESSED_DIR / "manifest.csv", index_col="id")
    if args.max_train_nodes:
        before = len(fit_ids)
        fit_ids = man.loc[fit_ids].query(f"n_nodes <= {args.max_train_nodes}").index.to_numpy()
        print(f"filter train n_nodes<={args.max_train_nodes}: {before} -> {len(fit_ids)}")

    print("hitung mean/std target dari train_fit...")
    y_mean, y_std = compute_norm_stats(fit_ids)
    y_mean, y_std = y_mean.to(device), y_std.to(device)
    huber_n = (args.huber_delta / y_std[STRESS_IDX].mean()).item() if args.huber_delta else None
    print(f"device={device} H={args.hidden} L={args.layers} huber_n={huber_n}")

    model = MeshGraphNet(hidden=args.hidden, n_layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    CKPT_DIR.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    best_val = float("inf")
    best_path = CKPT_DIR / f"mgn_split{args.split_idx}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = np.random.default_rng(args.seed + epoch).permutation(fit_ids)
        tr_loss = train_one_epoch(
            model, opt, order, device, y_mean, y_std, huber_n, args.accum
        )
        val_mae, _ = eval_mae(model, val_ids, device, y_mean, y_std)
        # early stop: mean MAE disp (stabil) + 0.001*stress supaya stress ikut
        val_score = float(val_mae[DISP_IDX].mean() + 0.001 * val_mae[STRESS_IDX].mean())
        sched.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_score": val_score,
                "val_disp": float(val_mae[DISP_IDX].mean()),
                "val_stress": float(val_mae[STRESS_IDX].mean()),
            }
        )
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss_n={tr_loss:.4f}  "
            f"val_disp={val_mae[DISP_IDX].mean():.4f}  "
            f"val_stress={val_mae[STRESS_IDX].mean():.2f}  ({time.time()-t0:.0f}s)"
        )
        if val_score < best_val:
            best_val = val_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "y_mean": y_mean.cpu(),
                    "y_std": y_std.cpu(),
                    "args": vars(args),
                    "val_mae": val_mae,
                },
                best_path,
            )
            print(f"  -> best ckpt ({best_path.name})")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    test_mae, skipped = eval_mae(model, test_ids, device, y_mean, y_std)
    eval_ids = np.array([int(i) for i in test_ids if int(i) not in set(skipped)])

    # Naive apple-to-apple: fit di train resmi split ini, eval di subset
    # bracket yang sama persis dengan MGN (kecualikan OOM skip).
    print(f"\nhitung naive matched (test {len(test_ids)} -> eval {len(eval_ids)}, "
          f"skip OOM={skipped})...")
    man = pd.read_csv(PROCESSED_DIR / "manifest.csv", index_col="id")
    is_test = man[f"test_split_{args.split_idx}"].astype(bool)
    train_ids_full = man.index[~is_test].to_numpy()
    coef = fit_split(PROCESSED_DIR, train_ids_full)
    naive_full = eval_split(PROCESSED_DIR, test_ids, coef)["pooled"]
    naive_matched = eval_split(PROCESSED_DIR, eval_ids, coef)["pooled"]

    print(f"\n=== TEST MAE split_{args.split_idx} "
          f"(MGN n={len(eval_ids)}/{len(test_ids)}) vs naive matched ===")
    print(f"{'target':<14}{'MGN':>10}{'n_full':>10}{'n_match':>10}"
          f"{'paper*':>10}{'vs_match':>9}")
    rows = []
    for i, name in enumerate(TARGET_COLUMNS):
        m = float(test_mae[i])
        nf = float(naive_full[i])
        nm = float(naive_matched[i])
        p = PAPER_MAE_RELABELED[name]
        print(f"{name:<14}{m:>10.4f}{nf:>10.4f}{nm:>10.4f}{p:>10.4g}{m/nm:>9.2f}")
        rows.append(
            {
                "target": name,
                "mgn": m,
                "naive_full": nf,
                "naive_matched": nm,
                "naive_delta_matched_minus_full": nm - nf,
                "ratio_vs_naive_matched": m / nm,
                "ratio_vs_naive_full": m / nf,
                "paper_relabeled": p,
                "n_eval_mgn": len(eval_ids),
                "n_test_official": len(test_ids),
                "mgn_skip_oom": ",".join(str(x) for x in skipped) if skipped else "",
            }
        )
    out = pd.DataFrame(rows).set_index("target")
    out_path = REPORTS / f"mgn_split{args.split_idx}_test_mae.csv"
    out.to_csv(out_path)
    with open(REPORTS / f"mgn_split{args.split_idx}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    meta = {
        "split_idx": args.split_idx,
        "n_test_official": int(len(test_ids)),
        "n_eval_mgn": int(len(eval_ids)),
        "mgn_skip_oom": skipped,
        "wins_vs_naive_matched": int((out["ratio_vs_naive_matched"] < 1.0).sum()),
    }
    with open(REPORTS / f"mgn_split{args.split_idx}_eval_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\ntersimpan: {out_path}")
    print(f"mengalahkan naive_matched di "
          f"{meta['wins_vs_naive_matched']}/20 target "
          f"(eval {len(eval_ids)}/{len(test_ids)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--overfit-n", type=int, default=8)
    p.add_argument("--overfit-epochs", type=int, default=300)
    p.add_argument("--split-idx", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--huber-delta", type=float, default=500.0)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--max-train-nodes", type=int, default=150000)
    args = p.parse_args()
    if args.overfit:
        ok = run_overfit(args)
        raise SystemExit(0 if ok else 1)
    run_train(args)


if __name__ == "__main__":
    main()
