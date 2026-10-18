"""绘制消融实验连接图（task 07-15-graph-module-ablation, Step E, 多 fold）。

对指定 fold 的每个变体生成三类热图（Nature 风格，128×128）：
  1. A_eff 热图 ×1            （被试无关，读 graph_connectivity_ablation/fold{N}/<v>/）
  2. DE per-subject 热图 ×10  （读 oc_connectivity_relative_ablation/fold{N}/<v>/subject_{i}.json）
  3. DE fold-mean 热图 ×1     （10 个被试矩阵逐元素平均）

输出到 outputs/ablation/fold{N}/<variant>/。

Usage:
    python -u -m latent_flow.scripts._plot_ablation_connectivity --fold 0
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
    apply_nature_style, save_nature)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"

VARIANT_NAMES = ["baseline_screg", "nosc", "nomexhat", "now"]
N_SUBJECTS = 10


def _load_matrix(json_path):
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cm = np.asarray(payload["connectivity_matrix"], dtype=np.float64)
    cm[~np.isfinite(cm)] = np.nan
    return cm


def _heatmap(matrix, title, out_base):
    """Draw one 128x128 connectivity heatmap, Nature style."""
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
    cbar.set_label("connectivity  (red=+, blue=$-$)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    saved = save_nature(fig, out_base)
    plt.close(fig)
    print(f"  saved: {saved}")


def plot_variant(name, fold):
    print(f"--- variant: {name} (fold {fold}) ---")
    out_dir = OUTPUTS_DIR / "ablation" / f"fold{fold}" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. A_eff 热图（被试无关，1 张）
    aeff_json = (RESULTS_DIR / "graph_connectivity_ablation" / f"fold{fold}" / name
                 / f"graph_connectivity_fold{fold}.json")
    if aeff_json.exists():
        aeff = _load_matrix(aeff_json)
        _heatmap(aeff,
                 f"$A_{{\\rm eff}}$  ({name})\nfold {fold} · subject-independent",
                 str(out_dir / "connectivity_aeff"))
    else:
        print(f"  [warn] A_eff JSON missing: {aeff_json}")

    # 2. DE per-subject 热图（10 张）+ 收集用于 fold-mean
    de_dir = (RESULTS_DIR / "oc_connectivity_relative_ablation"
              / f"fold{fold}" / name)
    de_mats = []
    for i in range(N_SUBJECTS):
        sj = de_dir / f"subject_{i}.json"
        if not sj.exists():
            print(f"  [warn] DE subject_{i} missing: {sj}")
            continue
        cm = _load_matrix(sj)
        de_mats.append(cm)
        _heatmap(cm, f"DE  ({name})\nfold {fold} · subject {i}",
                 str(out_dir / f"connectivity_de_subject{i}"))

    # 3. DE fold-mean 热图（1 张）
    if de_mats:
        stack = np.stack(de_mats, axis=0)
        mean_cm = np.nanmean(stack, axis=0)
        _heatmap(mean_cm,
                 f"DE  ({name})\nfold {fold} · mean over {len(de_mats)} subjects",
                 str(out_dir / "connectivity_de_fold0_mean"))
    else:
        print(f"  [warn] no DE subjects found for {name}, skip fold-mean")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    return p.parse_args()


def main():
    args = parse_args()
    fold = args.fold
    print(f"=== ablation connectivity plots (fold {fold}) ===")
    for name in VARIANT_NAMES:
        plot_variant(name, fold)
    print("=== done ===")


if __name__ == "__main__":
    main()
