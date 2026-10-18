"""Visualize FC correlation results across folds for baseline_screg variant.

Reads per-fold summary.json from results/fc_correlation/fold{0..4}/ and
produces: per-fold bar chart, per-subject swarm, group-level FC heatmap grid,
and permutation null distribution.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latent_flow.viz.nature_style import apply_nature_style, PALETTE, save_nature

apply_nature_style()

RESULTS_ROOT = Path("results/fc_correlation")
OUTPUTS_DIR = Path("outputs/fc_correlation")
N_FOLDS = 5


def load_all_fold_summaries():
    summaries = []
    for fold in range(N_FOLDS):
        path = RESULTS_ROOT / f"fold{fold}" / "summary.json"
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def load_all_subject_rs(summaries):
    fold_labels = []
    all_rs = []
    for s in summaries:
        fold_labels.append(f"F{s['fold']}\n(n={s['n_subjects']})")
        all_rs.append(s["per_subject_pearson_r"])
    return fold_labels, all_rs


# ---------------------------------------------------------------------------
# Figure 1 — Per-fold per-subject r bar chart + swarm
# ---------------------------------------------------------------------------

def plot_per_fold_comparison(summaries, outputs_dir):
    fold_labels, all_rs = load_all_subject_rs(summaries)
    # Filter out None values (from sanitized NaNs in JSON)
    all_rs = [[r for r in rs if r is not None] for rs in all_rs]
    fold_means = [np.mean(rs) for rs in all_rs]
    fold_stds = [np.std(rs) for rs in all_rs]
    fold_group_rs = [s.get("group_fc_pearson_r") or float("nan") for s in summaries]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(N_FOLDS)
    width = 0.35

    # Per-subject mean r bars
    bars1 = ax.bar(x - width/2, fold_means, width,
                   yerr=fold_stds, capsize=3,
                   color=PALETTE["blue_main"], alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   label="Per-subject mean r")

    # Group FC r bars
    bars2 = ax.bar(x + width/2, fold_group_rs, width,
                   color=PALETTE["teal"], alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   label="Group FC r")

    # Individual subject dots
    for i, rs in enumerate(all_rs):
        jitter = np.random.RandomState(i).uniform(-0.06, 0.06, size=len(rs))
        ax.scatter(np.full_like(rs, x[i] - width/2) + jitter, rs,
                   s=12, color=PALETTE["blue_secondary"], alpha=0.5,
                   edgecolors="none", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=8)
    ax.set_ylabel("Pearson r", fontsize=9)
    ax.set_title("FC Fidelity — baseline_screg across 5 folds", fontsize=10)
    ax.set_ylim(0.85, 1.02)
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=0.5, alpha=0.4)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=8)

    # Annotate mean value on bars
    for bar, val in zip(bars1, fold_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, fold_group_rs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save_nature(fig, str(outputs_dir / "per_fold_comparison"))
    plt.close(fig)
    print(f"  Saved {outputs_dir / 'per_fold_comparison'}.{{svg,pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 2 — Group-level FC heatmap grid (5 folds)
# ---------------------------------------------------------------------------

def plot_fc_heatmap_grid(summaries, outputs_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes = axes.flatten()

    combined_all = []
    fold_data = []
    for fold in range(N_FOLDS):
        subj_dirs = sorted((RESULTS_ROOT / f"fold{fold}").glob("subject_*"))
        fold_real_fcs = []
        fold_gen_fcs = []
        for sd in subj_dirs:
            real = np.load(sd / "real_fc.npy")
            gen = np.load(sd / "gen_fc.npy")
            z_real = np.arctanh(np.clip(real, -0.9999, 0.9999))
            z_gen = np.arctanh(np.clip(gen, -0.9999, 0.9999))
            fold_real_fcs.append(z_real)
            fold_gen_fcs.append(z_gen)
        group_real = np.tanh(np.mean(fold_real_fcs, axis=0))
        group_gen = np.tanh(np.mean(fold_gen_fcs, axis=0))
        combined_all.append(group_real.flatten())
        combined_all.append(group_gen.flatten())
        fold_data.append((group_real, group_gen))

    vmin = float(np.percentile(np.concatenate(combined_all), 2))
    vmax = float(np.percentile(np.concatenate(combined_all), 98))

    for i, (real_fc, gen_fc) in enumerate(fold_data):
        # Real FC
        im = axes[i].imshow(real_fc, cmap="RdBu_r", aspect="equal",
                             origin="lower", vmin=vmin, vmax=vmax,
                             interpolation="nearest")
        axes[i].set_title(f"Fold {i} — Group Real FC", fontsize=8)
        axes[i].set_xlabel("ch"); axes[i].set_ylabel("ch")
        axes[i].tick_params(labelsize=6)

    # Last subplot (axes[5]): legend bar only
    axes[5].set_visible(False)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Pearson r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Group-level Real FC across 5 Folds (baseline_screg)",
                 fontsize=10, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.91, 1])
    save_nature(fig, str(outputs_dir / "fc_heatmap_grid"))
    plt.close(fig)
    print(f"  Saved {outputs_dir / 'fc_heatmap_grid'}.{{svg,pdf,png}}")

    # Generated FC grid (same structure)
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7.5))
    axes2 = axes2.flatten()
    for i, (real_fc, gen_fc) in enumerate(fold_data):
        im = axes2[i].imshow(gen_fc, cmap="RdBu_r", aspect="equal",
                              origin="lower", vmin=vmin, vmax=vmax,
                              interpolation="nearest")
        axes2[i].set_title(f"Fold {i} — Group Gen FC", fontsize=8)
        axes2[i].set_xlabel("ch"); axes2[i].set_ylabel("ch")
        axes2[i].tick_params(labelsize=6)
    axes2[5].set_visible(False)
    cax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar2 = fig2.colorbar(im, cax=cax2)
    cbar2.set_label("Pearson r", fontsize=8)
    cbar2.ax.tick_params(labelsize=7)
    fig2.suptitle("Group-level Generated FC across 5 Folds (baseline_screg)",
                  fontsize=10, y=1.01)
    fig2.tight_layout(rect=[0, 0, 0.91, 1])
    save_nature(fig2, str(outputs_dir / "fc_heatmap_grid_gen"))
    plt.close(fig2)
    print(f"  Saved {outputs_dir / 'fc_heatmap_grid_gen'}.{{svg,pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 3 — FC-FC scatter grid (5 folds)
# ---------------------------------------------------------------------------

def plot_fc_scatter_grid(summaries, outputs_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes = axes.flatten()

    for fold in range(N_FOLDS):
        subj_dirs = sorted((RESULTS_ROOT / f"fold{fold}").glob("subject_*"))
        fold_real_fcs = []
        fold_gen_fcs = []
        for sd in subj_dirs:
            real = np.load(sd / "real_fc.npy")
            gen = np.load(sd / "gen_fc.npy")
            z_real = np.arctanh(np.clip(real, -0.9999, 0.9999))
            z_gen = np.arctanh(np.clip(gen, -0.9999, 0.9999))
            fold_real_fcs.append(z_real)
            fold_gen_fcs.append(z_gen)
        group_real = np.tanh(np.mean(fold_real_fcs, axis=0))
        group_gen = np.tanh(np.mean(fold_gen_fcs, axis=0))

        triu_idx = np.triu_indices(128, k=1)
        real_vec = group_real[triu_idx]
        gen_vec = group_gen[triu_idx]
        r = summaries[fold].get("group_fc_pearson_r")
        r_str = f"r = {r:.4f}" if r is not None else "r = N/A"

        ax = axes[fold]
        ax.scatter(real_vec, gen_vec, s=0.5, alpha=0.15,
                   color=PALETTE["blue_main"], edgecolors="none")
        ax.set_xlabel("Real FC edge weight", fontsize=7)
        ax.set_ylabel("Generated FC edge weight", fontsize=7)
        ax.set_title(f"Fold {fold}  {r_str}", fontsize=8)
        lim = max(abs(ax.get_xlim()[1]), abs(ax.get_ylim()[1]))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.4, alpha=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    axes[5].set_visible(False)
    fig.suptitle("Group-level FC-FC Scatter across 5 Folds (baseline_screg)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    save_nature(fig, str(outputs_dir / "fc_scatter_grid"))
    plt.close(fig)
    print(f"  Saved {outputs_dir / 'fc_scatter_grid'}.{{svg,pdf,png}}")


# ---------------------------------------------------------------------------
# Figure 4 — Permutation null distribution (fold0 only, representative)
# ---------------------------------------------------------------------------

def plot_permutation_null(outputs_dir):
    # Reconstruct null for fold0 from the stored stats
    fold0_path = RESULTS_ROOT / "fold0" / "summary.json"
    with open(fold0_path) as f:
        s0 = json.load(f)
    perm = s0["group_permutation_test"]
    observed_r = perm.get("observed_r")
    null_mean = perm.get("null_mean")
    null_std = perm.get("null_std")
    ci = perm.get("ci_95")
    ci_lo, ci_hi = ci if ci else (None, None)

    if any(v is None for v in [observed_r, null_mean, null_std, ci_lo, ci_hi]):
        print("  Skipping permutation null plot: NaN values found in fold0 stats")
        return

    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.linspace(null_mean - 4 * null_std, null_mean + 4 * null_std, 200)
    y = np.exp(-0.5 * ((x - null_mean) / null_std) ** 2) / (null_std * np.sqrt(2 * np.pi))

    ax.fill_between(x, y, alpha=0.25, color=PALETTE["blue_main"])
    ax.plot(x, y, color=PALETTE["blue_main"], lw=1)
    ax.axvline(observed_r, color=PALETTE["red_strong"], linestyle="--", lw=1.5,
               label=f"observed r = {observed_r:.4f}")
    ax.axvline(ci_lo, color="gray", linestyle=":", lw=0.8,
               label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    ax.axvline(ci_hi, color="gray", linestyle=":", lw=0.8)
    ax.axvline(null_mean, color="gray", linestyle="-", lw=0.5, alpha=0.5)

    ax.set_xlabel("Pearson r (null distribution)")
    ax.set_ylabel("Density")
    p_val = perm.get("p_value")
    p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
    ax.set_title(f"Permutation Test — Fold 0 baseline_screg\n"
                 f"p = {p_str}, n_perm = 10000", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_nature(fig, str(outputs_dir / "permutation_null_fold0"))
    plt.close(fig)
    print(f"  Saved {outputs_dir / 'permutation_null_fold0'}.{{svg,pdf,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    outputs_dir = OUTPUTS_DIR
    os.makedirs(outputs_dir, exist_ok=True)

    summaries = load_all_fold_summaries()

    print("Plotting FC correlation visualizations...")
    plot_per_fold_comparison(summaries, outputs_dir)
    plot_fc_heatmap_grid(summaries, outputs_dir)
    plot_fc_scatter_grid(summaries, outputs_dir)
    plot_permutation_null(outputs_dir)
    print(f"Done. All figures saved to {outputs_dir}/")


if __name__ == "__main__":
    main()
