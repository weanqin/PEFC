"""Visualize EDRA OC-LatentFlow training: loss curves + A_eff heatmap.

Parses the stage-1 / stage-2 training logs (emitted by train_oc_latent_flow.py)
for the loss curves, and loads the learned mexhat A_eff from the stage-2
checkpoint for the connectivity heatmap. Outputs go under outputs/edra/.

Run (from repo root, npi env):
    python -m latent_flow.scripts.plot_edra_training
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_flow.config import CHECKPOINT_DIR
from latent_flow.data.datasets import get_dataset
from latent_flow.viz.nature_style import apply_nature_style, save_nature


EDRA_CKPT_DIR = CHECKPOINT_DIR / "edra"
OUT_DIR = Path("outputs/edra")

# Parses a line like:
#   [stage1] E12: train | loss=1.70 recon=0.52 td=0.22 ... || val | loss=1.73 recon=0.51 ...
# Captures stage, epoch, and the train/val metric dicts.
_LINE_RE = re.compile(
    r"\[stage(?P<stage>\d+)\]\s+E(?P<epoch>\d+):\s*"
    r"train\s*\|\s*(?P<train>.*?)\s*\|\|\s*val\s*\|\s*(?P<val>.*?)\s*$"
)


def _parse_metrics(s):
    """'loss=1.70 recon=0.52' -> {'loss': 1.70, 'recon': 0.52}."""
    out = {}
    for tok in s.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def parse_log(log_path):
    """Return {stage: {epoch: {'train': {...}, 'val': {...}}}}."""
    stages = {}
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            m = _LINE_RE.search(line.strip())
            if not m:
                continue
            stage = int(m["stage"])
            epoch = int(m["epoch"])
            stages.setdefault(stage, {})[epoch] = {
                "train": _parse_metrics(m["train"]),
                "val": _parse_metrics(m["val"]),
            }
    return stages


def plot_loss_curves(stages, save_path):
    """Two-panel: stage1 (left) and stage2 (right) train/val loss."""
    apply_nature_style()
    n_panels = sum(1 for s in (1, 2) if s in stages)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.2))
    if n_panels == 1:
        axes = [axes]
    # Which sub-loss to plot per stage (the headline reconstruction-ish metric).
    headline = {1: "loss", 2: "loss"}
    panel_idx = 0
    for stage in (1, 2):
        if stage not in stages:
            continue
        ax = axes[panel_idx]
        panel_idx += 1
        epochs = sorted(stages[stage].keys())
        tr = [stages[stage][e]["train"].get(headline[stage], np.nan) for e in epochs]
        va = [stages[stage][e]["val"].get(headline[stage], np.nan) for e in epochs]
        ax.plot(epochs, tr, "-", lw=1.2, label="train", color="#1f77b4")
        ax.plot(epochs, va, "-", lw=1.2, label="val", color="#d62728")
        # mark best val
        if any(np.isfinite(va)):
            bi = int(np.nanargmin(va))
            ax.axvline(epochs[bi], color="gray", ls=":", lw=0.8, alpha=0.7)
            ax.annotate(f"best={va[bi]:.3f}\n(E{epochs[bi]})",
                        xy=(epochs[bi], va[bi]),
                        xytext=(0.02, 0.02), textcoords="axes fraction",
                        fontsize=7, color="gray", va="bottom")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"Stage {stage} ({'VAE' if stage == 1 else 'flow matching'})",
                     fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    save_nature(fig, save_path, formats=("png",))
    plt.close(fig)
    print(f"  loss curves -> {save_path}.png")


def plot_recon_curves(stages, save_path):
    """Reconstruction sub-loss across both stages (continuity check)."""
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    plotted = False
    for stage in (1, 2):
        if stage not in stages:
            continue
        key = "recon" if stage == 1 else "recon2"
        epochs = sorted(stages[stage].keys())
        tr = [stages[stage][e]["train"].get(key, np.nan) for e in epochs]
        va = [stages[stage][e]["val"].get(key, np.nan) for e in epochs]
        off = epochs[0]
        ax.plot([e for e in epochs], tr, "-", lw=1.2,
                label=f"S{stage} train" if stage == 1 else f"S{stage} train")
        ax.plot([e for e in epochs], va, "--", lw=1.2,
                label=f"S{stage} val")
        plotted = True
    if plotted:
        ax.set_xlabel("epoch (within stage)")
        ax.set_ylabel("reconstruction loss")
        ax.set_title("Reconstruction across stages", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    save_nature(fig, save_path, formats=("png",))
    plt.close(fig)
    print(f"  recon curves -> {save_path}.png")


def load_aeff(ckpt_path, device):
    """Load the stage-2 mexhat checkpoint and return its (62,62) A_eff."""
    from latent_flow.scripts._connectivity_driver import load_model
    model = load_model(ckpt_path, device)
    A = model.flow.graph_mixer.effective_adjacency().detach().cpu().numpy()
    return A


def plot_aeff_heatmap(A, ch_names, save_path, config_tag=""):
    """Nature-style heatmap of the directed A_eff with channel labels.

    Auto-picks a diverging cmap (RdBu_r, signed) when A has inhibitory (neg)
    entries, or a sequential cmap (magma, all-positive) when A is non-negative
    — so the full color range is always used.
    """
    apply_nature_style()
    n = A.shape[0]
    has_neg = bool(np.nanmin(A) < 0)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    if has_neg:
        cmap = "RdBu_r"
        vmax = float(np.nanmax(np.abs(A)))
        vmin, vmax = -vmax, vmax
    else:
        cmap = "magma"
        finite = A[np.isfinite(A)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
    im = ax.imshow(A, cmap=cmap, aspect="equal", origin="upper",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    # label every channel if few enough (62 -> every 1 is dense but readable at this size)
    if n <= 64:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=4.5)
        ax.set_yticklabels(ch_names, fontsize=4.5)
    ax.set_xlabel("target channel")
    ax.set_ylabel("source channel")
    sign_note = "signed (mexhat inhibition)" if has_neg else "all-positive (no inhibition)"
    title = f"EDRA $A_{{eff}}$ (directed, mexhat) — {sign_note}"
    if config_tag:
        title = f"{title}  [{config_tag}]"
    ax.set_title(title, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$A_{eff}$  (a$\\to$b)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_nature(fig, save_path, formats=("png",))
    plt.close(fig)
    print(f"  A_eff heatmap -> {save_path}.png  (cmap={cmap}, signed={has_neg})")


def plot_aeff_distribution(A, save_path):
    """Histogram of off-diagonal A_eff entries (excite/inhibit split)."""
    apply_nature_style()
    mask = ~np.eye(A.shape[0], dtype=bool)
    off = A[mask]
    off = off[np.isfinite(off)]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.hist(off, bins=60, color="#4c72b0", alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(off.mean(), color="#c44e52", ls="-", lw=1.0,
               label=f"mean={off.mean():+.3f}")
    neg_frac = float((off < 0).mean())
    ax.set_xlabel("$A_{eff}$ off-diagonal value")
    ax.set_ylabel("count")
    ax.set_title(f"A_eff distribution  (inhibitory frac={neg_frac:.1%})",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    save_nature(fig, save_path, formats=("png",))
    plt.close(fig)
    print(f"  A_eff histogram -> {save_path}.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc = get_dataset("edra")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Loss curves from training logs ---
    print("=== EDRA training loss curves ===")
    stages = {}
    for stage, logname in ((1, "stage1_lr2e3.log"), (2, "stage2_lr2e3.log")):
        log_path = EDRA_CKPT_DIR / logname
        if log_path.exists():
            stages.update(parse_log(str(log_path)))
            print(f"  parsed {logname}: "
                  f"{sum(1 for s in stages.values() for _ in s)} epochs")
        else:
            print(f"  (missing {log_path})")
    if stages:
        plot_loss_curves(stages, str(OUT_DIR / "edra_loss_curves_lr2e3"))
        plot_recon_curves(stages, str(OUT_DIR / "edra_recon_curves_lr2e3"))

    # --- A_eff from stage-2 mexhat checkpoint ---
    print("\n=== EDRA A_eff heatmap ===")
    ckpt = EDRA_CKPT_DIR / "fold0_oc_latent_flow_flow_best_graph_mexhat.pt"
    if ckpt.exists():
        A = load_aeff(str(ckpt), device)
        from latent_flow.models.graph_utils import load_geometry
        _, ch_names = load_geometry(dataset="edra")
        config_tag = "恒定 lr2e-3 200ep"
        plot_aeff_heatmap(A, ch_names, str(OUT_DIR / "edra_aeff_heatmap_lr2e3"),
                          config_tag=config_tag)
        plot_aeff_distribution(A, str(OUT_DIR / "edra_aeff_histogram_lr2e3"))
        print(f"  A_eff shape={A.shape}, directed (max|A-A.T|="
              f"{np.nanmax(np.abs(A - A.T)):.3f})")
    else:
        print(f"  (missing checkpoint {ckpt})")

    print(f"\nAll figures under {OUT_DIR}/")


if __name__ == "__main__":
    main()
