"""
Utilisation : python plotting.py --log_dir logs/mlp_debug/version_4 --out_dir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def _find_metrics_csv(log_dir: Path) -> Path:
    candidates = list(log_dir.rglob("metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"metrics.csv introuvable dans {log_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _epoch_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    s = df[["epoch", col]].dropna()
    if s.empty:
        return None
    return s.groupby("epoch")[col].mean()


def plot_curves(metrics_csv: Path, out_dir: Path | None) -> None:
    df = pd.read_csv(metrics_csv)
    if "epoch" not in df.columns:
        raise RuntimeError(f"Colonne 'epoch' introuvable dans {metrics_csv}")

    # --- mIoU ---
    train_miou = _epoch_series(df, "train/mIoU")
    val_miou = _epoch_series(df, "val/mIoU")

    plt.figure()
    if train_miou is not None:
        plt.plot(train_miou.index.values, train_miou.values, label="train")
    if val_miou is not None:
        plt.plot(val_miou.index.values, val_miou.values, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("mIoU vs Epoch")
    plt.legend()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "miou_vs_epoch.png", dpi=150)

    # --- F1 macro ---
    train_f1 = _epoch_series(df, "train/F1_macro")
    val_f1 = _epoch_series(df, "val/F1_macro")

    plt.figure()
    if train_f1 is not None:
        plt.plot(train_f1.index.values, train_f1.values, label="train")
    if val_f1 is not None:
        plt.plot(val_f1.index.values, val_f1.values, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("F1 (macro)")
    plt.title("F1 macro vs Epoch")
    plt.legend()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "f1_macro_vs_epoch.png", dpi=150)

    plt.show()


def plot_confmat(log_dir: Path, out_dir: Path | None) -> None:
    candidates = list(log_dir.rglob("confmat_final.pt"))
    if not candidates:
        raise FileNotFoundError(f"confmat_final.pt introuvable dans {log_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    conf_path = candidates[0]

    conf = torch.load(conf_path, map_location="cpu")
    conf = conf.detach().cpu().numpy().astype(np.int64)

    plt.figure()
    plt.imshow(conf)
    plt.colorbar()
    plt.title("Confusion matrix (final test)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "confmat_final.png", dpi=150)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--no_confmat", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else None

    metrics_csv = _find_metrics_csv(log_dir)
    print("Using metrics:", metrics_csv)

    plot_curves(metrics_csv, out_dir)
    if not args.no_confmat:
        plot_confmat(log_dir, out_dir)


if __name__ == "__main__":
    main()
