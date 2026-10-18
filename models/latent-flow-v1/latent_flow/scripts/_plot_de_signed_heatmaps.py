"""绘制带符号 DE 连接热力图（task 07-23-de-directional-noise-sign）。

对 fold0 全量结果（results/oc_connectivity_global_tau_signed_signal_q0.5/）
生成以下热图（Nature 风格，128×128）：

  1. signed_connectivity  per-subject ×10  （RdBu_r，带符号连接强度）
  2. sign_matrix          per-subject ×10  （±1 符号矩阵）
  3. connectivity         per-subject ×10  （magma，无符号连接强度）
  4. mean_diff            per-subject ×10  （RdBu_r，均值偏移）
  5. fold-mean 汇总      ×4 种矩阵

输出到 outputs/de_signed_heatmaps/。

Usage:
    python -u -m latent_flow.scripts._plot_de_signed_heatmaps
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latent_flow.viz.nature_style import (
    apply_nature_style, save_nature, add_panel_label)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"

INPUT_DIR = RESULTS_DIR / "oc_connectivity_global_tau_signed_signal_q0.5"
OUT_DIR = OUTPUTS_DIR / "de_signed_heatmaps"
N_SUBJECTS = 10


def _load_payload(subject_idx):
    path = INPUT_DIR / f"subject_{subject_idx}.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _signed_heatmap(matrix, title, out_base, cbar_label="signed conn  (red=+, blue=$-$)"):
    """Signed diverging heatmap, symmetric 99th-percentile scale."""
    apply_nature_style(font_size=7, axes_linewidth=0.8)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    vmax = np.nanpercentile(np.abs(matrix), 99)
    vmax = max(vmax, 1e-9)
    im = ax.imshow(matrix, aspect="equal", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("target channel $b$", fontsize=7)
    ax.set_ylabel("source channel $a$", fontsize=7)
    n = matrix.shape[0]
    ax.set_xticks([0, n // 2, n - 1])
    ax.set_yticks([0, n // 2, n - 1])
    ax.tick_params(labelsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    saved = save_nature(fig, out_base)
    plt.close(fig)
    return saved


def _magma_heatmap(matrix, title, out_base, cbar_label="connectivity strength"):
    """Unsigned sequential heatmap (magma), 0–99th percentile scale."""
    apply_nature_style(font_size=7, axes_linewidth=0.8)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    finite = matrix[np.isfinite(matrix)]
    vmax = np.nanpercentile(finite, 99) if len(finite) > 0 else 1.0
    im = ax.imshow(matrix, aspect="equal", cmap="magma",
                   vmin=0, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("target channel $b$", fontsize=7)
    ax.set_ylabel("source channel $a$", fontsize=7)
    n = matrix.shape[0]
    ax.set_xticks([0, n // 2, n - 1])
    ax.set_yticks([0, n // 2, n - 1])
    ax.tick_params(labelsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    saved = save_nature(fig, out_base)
    plt.close(fig)
    return saved


def _sign_heatmap(matrix, title, out_base):
    """Discrete sign heatmap: +1 (red), -1 (blue), 0 (white)."""
    apply_nature_style(font_size=7, axes_linewidth=0.8)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    cmap = matplotlib.colors.ListedColormap(["#2166ac", "#f7f7f7", "#b2182b"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(matrix, aspect="equal", cmap=cmap, norm=norm,
                   origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("target channel $b$", fontsize=7)
    ax.set_ylabel("source channel $a$", fontsize=7)
    n = matrix.shape[0]
    ax.set_xticks([0, n // 2, n - 1])
    ax.set_yticks([0, n // 2, n - 1])
    ax.tick_params(labelsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["inhibitory ($-$1)", "none (0)", "excitatory (+1)"], fontsize=6)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    saved = save_nature(fig, out_base)
    plt.close(fig)
    return saved


def _multi_subject_grid(matrices, titles, out_base, cmap_mode="signed", global_vmax=None):
    """2x5 grid of per-subject heatmaps + fold-mean."""
    apply_nature_style(font_size=7, axes_linewidth=0.8)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.2))
    axes = axes.flatten()

    if cmap_mode == "signed":
        if global_vmax is None:
            all_abs = np.concatenate([m[np.isfinite(m)] for m in matrices])
            global_vmax = np.nanpercentile(np.abs(all_abs), 99) if len(all_abs) > 0 else 1.0
        global_vmax = max(global_vmax, 1e-9)
        cmap = "RdBu_r"
        kwargs = dict(vmin=-global_vmax, vmax=global_vmax)
    elif cmap_mode == "sign":
        cmap = matplotlib.colors.ListedColormap(["#2166ac", "#f7f7f7", "#b2182b"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        kwargs = dict(cmap=cmap, norm=norm)
    else:  # magma
        if global_vmax is None:
            all_finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
            global_vmax = np.nanpercentile(all_finite, 99) if len(all_finite) > 0 else 1.0
        cmap = "magma"
        kwargs = dict(vmin=0, vmax=global_vmax)

    # Common imshow args; kwargs may override cmap/specific params
    imshow_kwargs = dict(aspect="equal", origin="upper", interpolation="nearest")
    if "cmap" not in kwargs:
        imshow_kwargs["cmap"] = cmap
    imshow_kwargs.update(kwargs)

    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axes[i]
        im = ax.imshow(matrix, **imshow_kwargs)
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(titles[0].rsplit(" ", 1)[0] if titles else "", fontsize=8, y=1.01)
    fig.tight_layout()
    saved = save_nature(fig, out_base)
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Plot signed DE connectivity heatmaps")
    parser.add_argument("--subjects", type=int, default=N_SUBJECTS)
    parser.add_argument("--subject-idx", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect matrices across subjects
    signed_matrices = []
    sign_matrices = []
    conn_matrices = []
    mean_diff_matrices = []

    for si in range(args.subject_idx, args.subject_idx + args.subjects):
        payload = _load_payload(si)
        signed = np.asarray(payload["signed_connectivity_matrix"], dtype=np.float64)
        sign = np.asarray(payload["sign_matrix"], dtype=np.float64)
        conn = np.asarray(payload["connectivity_matrix"], dtype=np.float64)
        md = np.asarray(payload["mean_diff_matrix"], dtype=np.float64)

        for arr in (signed, sign, conn, md):
            arr[~np.isfinite(arr)] = np.nan

        signed_matrices.append(signed)
        sign_matrices.append(sign)
        conn_matrices.append(conn)
        mean_diff_matrices.append(md)

        # Per-subject plots
        subj_dir = OUT_DIR / f"subject_{si}"
        os.makedirs(subj_dir, exist_ok=True)

        _signed_heatmap(signed, f"Subject {si}  signed DE connectivity",
                        str(subj_dir / f"subject_{si}_signed_conn"),
                        cbar_label="signed conn  (red=+, blue=$-$)")

        _sign_heatmap(sign, f"Subject {si}  direction sign",
                      str(subj_dir / f"subject_{si}_sign"))

        _magma_heatmap(conn, f"Subject {si}  connectivity strength",
                       str(subj_dir / f"subject_{si}_conn"),
                       cbar_label="conn  = $-$log $\\sigma^*$  (magma)")

        title_md = f"Subject {si}  mean diff  " + r"$\langle x_{fc}^{pert} - x_{fc}^{base}\rangle$"
        _signed_heatmap(md, title_md,
                        str(subj_dir / f"subject_{si}_mean_diff"),
                        cbar_label="mean diff  (red=+, blue=$-$)")

        print(f"Subject {si}: 4 heatmaps saved")

    # Fold-mean matrices
    signed_mean = np.nanmean(np.stack(signed_matrices), axis=0)
    sign_mean = np.nanmean(np.stack(sign_matrices), axis=0)  # continuous [-1, +1]
    conn_mean = np.nanmean(np.stack(conn_matrices), axis=0)
    md_mean = np.nanmean(np.stack(mean_diff_matrices), axis=0)

    # Summary directory
    summary_dir = OUT_DIR / "summary"
    os.makedirs(summary_dir, exist_ok=True)

    _signed_heatmap(signed_mean, "Fold-mean signed DE connectivity",
                    str(summary_dir / "fold_mean_signed_conn"),
                    cbar_label="signed conn  (red=+, blue=$-$)")

    _signed_heatmap(sign_mean, "Fold-mean sign agreement  (+1 = all excit, $-$1 = all inhib)",
                    str(summary_dir / "fold_mean_sign_agreement"),
                    cbar_label="mean sign  (red=excit, blue=inhib)")

    _magma_heatmap(conn_mean, "Fold-mean connectivity strength",
                   str(summary_dir / "fold_mean_conn"),
                   cbar_label="conn  = $-$log $\\sigma^*$  (magma)")

    _signed_heatmap(md_mean, "Fold-mean mean diff",
                    str(summary_dir / "fold_mean_mean_diff"),
                    cbar_label="mean diff  (red=+, blue=$-$)")

    print("\nSummary heatmaps saved")

    # Multi-subject overview grids
    grid_dir = OUT_DIR / "grids"
    os.makedirs(grid_dir, exist_ok=True)

    _multi_subject_grid(
        signed_matrices,
        [f"Subj {si}" for si in range(args.subject_idx, args.subject_idx + args.subjects)],
        str(grid_dir / "grid_signed_conn"),
        cmap_mode="signed",
    )
    print("Grid: signed_conn saved")

    _multi_subject_grid(
        sign_matrices,
        [f"Subj {si}" for si in range(args.subject_idx, args.subject_idx + args.subjects)],
        str(grid_dir / "grid_sign"),
        cmap_mode="sign",
    )
    print("Grid: sign saved")

    _multi_subject_grid(
        conn_matrices,
        [f"Subj {si}" for si in range(args.subject_idx, args.subject_idx + args.subjects)],
        str(grid_dir / "grid_conn"),
        cmap_mode="magma",
    )
    print("Grid: conn saved")

    _multi_subject_grid(
        mean_diff_matrices,
        [f"Subj {si}" for si in range(args.subject_idx, args.subject_idx + args.subjects)],
        str(grid_dir / "grid_mean_diff"),
        cmap_mode="signed",
    )
    print("Grid: mean_diff saved")

    print(f"\n=== All done ===\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
