"""绘制 de-signed z-score 标准化矩阵热图（per-subject）。

对 ``zscore_signed_de.py`` 产出的 z-score 标准化 signed_connectivity 矩阵做
可视化。输入 ``results/oc_connectivity_global_tau_signed_signal_q0.5_zscore/``，
输出 per-subject RdBu_r 发散热图到 ``outputs/de_signed_zscore_heatmaps/``。

设计要点（与 z-score 语义匹配）：
  - 发散色图 RdBu_r：红=正（兴奋性 z>0），蓝=负（抑制性 z<0），白=0。
  - 对称色标：vmax 取**该 subject** off-diag |z| 的 99 分位（沿用既有
    ``_plot_de_signed_heatmaps.py`` 的 per-subject 约定），单图对比度最佳。
  - 对角线为 NaN（自连接，z-score 时已置 null），imshow 自动渲染为透明，
    与 off-diag 区分清晰。
  - 轴标 source channel a / target channel b，colorbar 标注 z-score。

Usage:
    python -u -m latent_flow.scripts.plot_de_signed_zscore
    python -u -m latent_flow.scripts.plot_de_signed_zscore --subjects 5 --subject-idx 0
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

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "oc_connectivity_global_tau_signed_signal_q0.5_zscore"
OUT_DIR = ROOT / "outputs" / "de_signed_zscore_heatmaps"
N_SUBJECTS = 10


def _load_zscore_matrix(subject_idx):
    """Load signed_connectivity_zscore matrix for one subject; NaN-aware."""
    path = INPUT_DIR / f"subject_{subject_idx}.json"
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    mat = np.asarray(payload["signed_connectivity_zscore"], dtype=np.float64)
    mat[~np.isfinite(mat)] = np.nan
    return mat


def _zscore_heatmap(matrix, title, out_base):
    """Signed diverging heatmap for a z-score matrix.

    Symmetric per-subject 99th-percentile scale on off-diagonal |z| values.
    Diagonal (NaN self-connections) renders transparent.
    """
    apply_nature_style(font_size=7, axes_linewidth=0.8)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    # Per-subject symmetric vmax from off-diagonal finite |z|.
    off = ~np.eye(matrix.shape[0], dtype=bool)
    off_abs = np.abs(matrix[off])
    off_abs = off_abs[np.isfinite(off_abs)]
    vmax = np.percentile(off_abs, 99) if len(off_abs) > 0 else 1.0
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
    cbar.set_label("signed DE  z-score  (red=+, blue=$-$)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    saved = save_nature(fig, out_base)
    plt.close(fig)
    return saved, vmax


def parse_args():
    p = argparse.ArgumentParser(description="Plot de-signed z-score heatmaps (per-subject).")
    p.add_argument("--subjects", type=int, default=N_SUBJECTS,
                   help=f"number of subjects to plot (default {N_SUBJECTS}).")
    p.add_argument("--subject-idx", type=int, default=0,
                   help="starting subject index (default 0).")
    return p.parse_args()


def main():
    args = parse_args()
    if not INPUT_DIR.is_dir():
        sys.exit(f"Input dir not found: {INPUT_DIR}\n"
                 f"Run `python -m latent_flow.scripts.zscore_signed_de --only flat` first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    end = args.subject_idx + args.subjects
    for si in range(args.subject_idx, end):
        in_path = INPUT_DIR / f"subject_{si}.json"
        if not in_path.exists():
            print(f"  [skip] subject {si}: {in_path} missing")
            continue
        mat = _load_zscore_matrix(si)
        title = f"Subject {si}  signed DE connectivity (z-score)"
        saved, vmax = _zscore_heatmap(
            mat, title, str(OUT_DIR / f"subject_{si}_zscore"))
        print(f"  [+] subject_{si}_zscore  (vmax={vmax:.3f}) -> {saved[0]}")

    print(f"\n=== done ===\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
