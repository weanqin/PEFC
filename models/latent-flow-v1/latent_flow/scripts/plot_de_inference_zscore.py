"""对 de 推断（relative-response / median-polish）矩阵做 z-score 后可视化。

针对 ``outputs/ablation/fold0/baseline_screg`` 下 de 推断图的源矩阵，做
off-diagonal z-score，再重新绘制 Nature 风格 RdBu_r 热图，输出到平行的
``_zscore`` 目录，不覆盖原图。

Why off-diag only: de 推断矩阵（relative-response, median polish）的对角线是
~11.3 的极大值（自连接残差），不是真实边，参与统计会严重污染 z-score 的 mean/std。

Source data (read-only):
    results/oc_connectivity_relative_ablation/fold0/baseline_screg/subject_{i}.json
      -> payload["connectivity_matrix"]  (128x128, 已经过 median-polish + per-row std 归一)

Output:
    outputs/ablation/fold0/baseline_screg_zscore/connectivity_de_subject{i}_zscore.png  (x10)
    outputs/ablation/fold0/baseline_screg_zscore/connectivity_de_fold0_mean_zscore.png  (x1)

  - fold0_mean: 先对 10 个 subject 各自 z-score，再 nanmean（避免极端对角线污染均值）。

Convention reused:
  - z-score: ``_compare_signed_de_ablation._zscore_offdiag`` (NaN-aware, 全 off-diag)
  - heatmap: ``_plot_ablation_connectivity._heatmap`` 同款 per-subject 99 分位对称色标

Usage:
    python -u -m latent_flow.scripts.plot_de_inference_zscore
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

from latent_flow.viz.nature_style import apply_nature_style, save_nature
from latent_flow.scripts._compare_signed_de_ablation import _zscore_offdiag

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"

N_SUBJECTS = 10
DE_DIR = RESULTS_DIR / "oc_connectivity_relative_ablation" / "fold0" / "baseline_screg"
OUT_DIR = OUTPUTS_DIR / "ablation" / "fold0" / "baseline_screg_zscore"


def _load_de_matrix(subject_idx):
    """Load connectivity_matrix for one subject; NaN-aware."""
    path = DE_DIR / f"subject_{subject_idx}.json"
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cm = np.asarray(payload["connectivity_matrix"], dtype=np.float64)
    cm[~np.isfinite(cm)] = np.nan
    return cm


def _zscore_offdiag_zerodiag(mat):
    """Off-diag z-score (shared convention) with diagonal nulled.

    The de-inference matrix diagonal holds ~11.3 self-connection residuals
    (NOT real edges). _zscore_offdiag copies the input diagonal through; we null
    it so the heatmap diagonal renders transparent (white) instead of saturated.
    """
    out = _zscore_offdiag(mat)
    np.fill_diagonal(out, np.nan)
    return out


def _heatmap(matrix, title, out_base):
    """Draw one 128x128 connectivity heatmap, Nature style.

    Same convention as _plot_ablation_connectivity._heatmap: diverging RdBu_r,
    symmetric per-matrix 99th-percentile vmax on |matrix|.
    """
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
    cbar.set_label("DE connectivity  z-score  (red=+, blue=$-$)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    saved = save_nature(fig, out_base)
    plt.close(fig)
    return saved, vmax


def parse_args():
    p = argparse.ArgumentParser(description="Z-score & plot de-inference matrices (fold0/baseline_screg).")
    p.add_argument("--subjects", type=int, default=N_SUBJECTS,
                   help=f"number of subjects (default {N_SUBJECTS}).")
    return p.parse_args()


def main():
    args = parse_args()
    if not DE_DIR.is_dir():
        sys.exit(f"DE source dir not found: {DE_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"input : {DE_DIR}")
    print(f"output: {OUT_DIR}\n")

    zscore_mats = []
    n_subjects = args.subjects

    # 1. per-subject z-score heatmaps
    for i in range(n_subjects):
        sj = DE_DIR / f"subject_{i}.json"
        if not sj.exists():
            print(f"  [skip] subject_{i} missing")
            continue
        cm = _load_de_matrix(i)
        z = _zscore_offdiag_zerodiag(cm)
        zscore_mats.append(z)
        title = f"DE z-score  (baseline_screg)\nfold 0 \u00b7 subject {i}"
        saved, vmax = _heatmap(z, title, str(OUT_DIR / f"connectivity_de_subject{i}_zscore"))
        print(f"  [+] connectivity_de_subject{i}_zscore  (vmax={vmax:.3f})")

    # 2. fold-mean z-score heatmap (z-score FIRST, then nanmean)
    if zscore_mats:
        stack = np.stack(zscore_mats, axis=0)
        mean_z = np.nanmean(stack, axis=0)
        title = f"DE z-score  (baseline_screg)\nfold 0 \u00b7 mean over {len(zscore_mats)} subjects"
        saved, vmax = _heatmap(mean_z, title, str(OUT_DIR / "connectivity_de_fold0_mean_zscore"))
        print(f"  [+] connectivity_de_fold0_mean_zscore  (vmax={vmax:.3f})")
    else:
        print("  [warn] no subjects, skip fold-mean")

    print(f"\n=== done ===\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
