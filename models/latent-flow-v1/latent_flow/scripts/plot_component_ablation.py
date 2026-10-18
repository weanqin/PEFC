"""Heatmap comparison for the component-level ablation (task 06-29, R1).

Reads the ablation JSON written by ``scripts/probe_component_ablation.py`` and
renders side-by-side response heatmaps so the structural difference between
surgery conditions is visible at a glance.

Why this matters (R1 rigor):
  - blockdiag surgeries can inflate full CV by zeroing the off-diagonal while
    leaving the diagonal intact. A heatmap makes the difference between
    "genuine cross-channel structure recovery" and "diagonal-only artifact"
    obvious: the former shows a rich off-diagonal texture, the latter shows
    a bright diagonal on a dark background.
  - Using a SHARED color scale across panels makes the absolute response
    magnitudes directly comparable (not just their relative patterns).

Output (relative to the cwd at invocation, typically repo root):
    outputs/component_ablation/ablation_subject_{idx}_heatmap.png

Usage (from repo root):

    D:/program/anaconda/envs/npi/python.exe -u -m \\
        latent_flow.scripts.plot_component_ablation \\
        --subject-idx 0
"""

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = "results/component_ablation/ablation_subject_0.json"
DEFAULT_OUTPUTS_DIR = "outputs/component_ablation"

# Panels to show, left to right. The two blockdiag surgeries are the focus;
# 'none' is the baseline reference.
PANEL_ORDER = ("none", "input_proj_blockdiag", "out_proj_blockdiag")

PANEL_TITLE = {
    "none": "baseline (no surgery)",
    "input_proj_blockdiag": "input_proj blockdiag\n(REAL structure recovery)",
    "out_proj_blockdiag": "out_proj blockdiag\n(artifact: diagonal-only)",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Render side-by-side ablation response heatmaps.")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT,
                   help="ablation JSON path.")
    p.add_argument("--outputs-dir", type=str, default=DEFAULT_OUTPUTS_DIR,
                   help="output PNG directory.")
    p.add_argument("--subject-idx", type=int, default=0,
                   help="subject index (used in the output filename).")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input) as fh:
        data = json.load(fh)

    matrices = data["response_matrices"]
    sigma = data.get("sigma", 0.2)
    subj = data.get("subject_idx", args.subject_idx)
    n_ch = data.get("n_channels", 128)

    # Build the panels that actually exist in the JSON, in PANEL_ORDER.
    panels = [(c, np.array(matrices[c], dtype=np.float64))
              for c in PANEL_ORDER if c in matrices]
    if len(panels) < 2:
        print(f"WARN: only {len(panels)} panels found; need >=2 for a "
              f"comparison. JSON conditions: {list(matrices)}")

    # ----- shared color scale (global max across the shown panels) --------
    global_max = max(M.max() for _, M in panels) * 1.05
    print(f"Shared color scale [0, {global_max:.3e}] across {len(panels)} panels")

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]

    for ax, (cond, M) in zip(axes, panels):
        im = ax.imshow(M, aspect="equal", cmap="viridis",
                       vmin=0.0, vmax=global_max, origin="upper",
                       interpolation="nearest")
        ax.set_title(PANEL_TITLE.get(cond, cond), fontsize=10)
        ax.set_xlabel("target channel b")
        if ax is axes[0]:
            ax.set_ylabel("source channel a")
        ax.set_xticks([0, n_ch // 2, n_ch - 1])
        ax.set_yticks([0, n_ch // 2, n_ch - 1])

    # Shared colorbar on the right.
    fig.colorbar(im, ax=axes, shrink=0.85, label="MSE response (z_obs[a] -> z_fc[b])")

    fig.suptitle(
        f"Component ablation response heatmaps  subj {subj}  "
        f"sigma={sigma} (latent coupled, shared scale)",
        fontsize=11)
    os.makedirs(args.outputs_dir, exist_ok=True)
    out_path = os.path.join(args.outputs_dir,
                            f"ablation_subject_{subj}_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

    # ----- also print compact stats per panel for the log ----------------
    print(f"\n{'condition':<22} {'offdiag_mean':>12} {'offdiag_median':>14} "
          f"{'max':>10} {'diag/off':>9}")
    for cond, M in panels:
        off = M[~np.eye(n_ch, dtype=bool)]
        diag = np.diag(M)
        dom = diag.mean() / off.mean() if off.mean() > 0 else float("inf")
        print(f"{cond:<22} {off.mean():12.3e} {np.median(off):14.3e} "
              f"{M.max():10.3e} {dom:8.1f}x")


if __name__ == "__main__":
    main()
