"""Evaluate functional connectivity (FC) fidelity of generated forecasts.

Computes Pearson-correlation FC matrices on both real and model-generated
forecast segments per subject, then measures the correlation between the two
FC matrices + permutation-test significance.

Phase 1 (feasibility): baseline_screg, fold 0 only.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_flow.scripts._connectivity_driver import (
    get_fold_test_subjects,
    load_model,
    load_subject_trials,
    sanitize_for_json,
)
from latent_flow.viz.nature_style import apply_nature_style, PALETTE, save_nature

apply_nature_style()

N_CHANNELS = 128
N_PERM = 10000
DEFAULT_SEED = 42
INFERENCE_BATCH = 32


# ---------------------------------------------------------------------------
# FC computation
# ---------------------------------------------------------------------------

def compute_fc(x_fc_segments):
    """Pearson FC matrix from stacked forecast segments.

    Args:
        x_fc_segments: (N, T_fc, C) numpy array of forecast windows.

    Returns:
        fc: (C, C) float32 FC matrix.
    """
    N, T, C = x_fc_segments.shape
    data = x_fc_segments.reshape(N * T, C)  # pool all time points
    fc = np.corrcoef(data.T)  # (C, C)
    return fc.astype(np.float32)


def fc_upper(fc):
    """Upper triangle (excluding diagonal) as a flat vector."""
    triu_idx = np.triu_indices(N_CHANNELS, k=1)
    return fc[triu_idx]


def fc_correlation(real_fc, gen_fc):
    """Pearson r between upper triangles of two FC matrices."""
    r = np.corrcoef(fc_upper(real_fc), fc_upper(gen_fc))[0, 1]
    if not np.isfinite(r):
        return float("nan")
    return float(r)


def spearman_fc(real_fc, gen_fc):
    """Spearman rank correlation between upper triangles."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(fc_upper(real_fc), fc_upper(gen_fc))
    if not np.isfinite(r):
        return float("nan")
    return float(r)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(real_fc, gen_fc, n_perm=N_PERM, seed=DEFAULT_SEED):
    """Two-sided permutation test: shuffle gen FC upper-triangle entries.

    Returns dict with observed_r, p_value, null_mean, null_std, ci_95.
    """
    real_vec = fc_upper(real_fc)
    gen_vec = fc_upper(gen_fc)
    observed_r = float(np.corrcoef(real_vec, gen_vec)[0, 1])
    if not np.isfinite(observed_r):
        return {"observed_r": float("nan"), "p_value": float("nan"),
                "null_mean": float("nan"), "null_std": float("nan"),
                "ci_95": [float("nan"), float("nan")]}

    rng = np.random.RandomState(seed)
    null_dist = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        shuffled = rng.permutation(gen_vec)
        null_dist[i] = np.corrcoef(real_vec, shuffled)[0, 1]

    null_mean = float(np.mean(null_dist))
    null_std = float(np.std(null_dist))
    p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed_r)))
    ci_lo = float(np.percentile(null_dist, 2.5))
    ci_hi = float(np.percentile(null_dist, 97.5))

    return {
        "observed_r": observed_r,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "ci_95": [ci_lo, ci_hi],
        "significant_005": p_value < 0.05,
    }


def fisher_z_ci(r, n, alpha=0.05):
    """Fisher z-transform confidence interval for a Pearson r."""
    if not np.isfinite(r) or np.abs(r) >= 1.0:
        return [float("nan"), float("nan")]
    from scipy.stats import norm
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    return [float(np.tanh(z_lo)), float(np.tanh(z_hi))]


# ---------------------------------------------------------------------------
# Per-subject inference
# ---------------------------------------------------------------------------

def run_subject(subject_idx, subject_file, model, device, results_dir):
    """Compute real/gen FC for one subject.

    Returns dict with keys: subject_idx, subject_file, n_trials, real_fc,
    gen_fc, pearson_r, spearman_r, perm_test.
    """
    x_obs_tensor = load_subject_trials(subject_file, device)
    n_trials = x_obs_tensor.shape[0]

    # Real fc: x_target from the dataset
    from latent_flow.data.modma import MODMADataset
    ds = MODMADataset(
        subject_files=[subject_file], group="MIX",
        normalize_per_channel=True, obs_size=375,
    )
    x_fc_real = np.stack([
        w["x_target"][375:, :].numpy() for w in ds.windows
    ])  # (N, 125, 128)

    # Generated fc: model.sample in batches
    x_fc_gen_list = []
    with torch.inference_mode():
        for start in range(0, n_trials, INFERENCE_BATCH):
            end = min(start + INFERENCE_BATCH, n_trials)
            batch = x_obs_tensor[start:end]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x_hat = model.sample(batch)
            x_fc_gen_list.append(x_hat[:, 375:, :].cpu().numpy())
    x_fc_gen = np.concatenate(x_fc_gen_list, axis=0)

    # FC matrices
    real_fc = compute_fc(x_fc_real)
    gen_fc = compute_fc(x_fc_gen)

    pearson_r = fc_correlation(real_fc, gen_fc)
    spearman_r_val = spearman_fc(real_fc, gen_fc)
    perm = permutation_test(real_fc, gen_fc)

    result = {
        "subject_idx": subject_idx,
        "subject_file": os.path.basename(subject_file),
        "n_trials": int(n_trials),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r_val,
        "permutation_test": perm,
        "fisher_z_ci_95": fisher_z_ci(pearson_r, 8128),
    }

    # Save per-subject FC matrices
    subj_dir = os.path.join(results_dir, f"subject_{subject_idx}")
    os.makedirs(subj_dir, exist_ok=True)
    np.save(os.path.join(subj_dir, "real_fc.npy"), real_fc)
    np.save(os.path.join(subj_dir, "gen_fc.npy"), gen_fc)
    with open(os.path.join(subj_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(result), f, indent=2, allow_nan=False)

    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_fc_comparison(real_fc, gen_fc, pearson_r, subject_label, outputs_dir):
    """3-panel figure: real FC heatmap, gen FC heatmap, FC-FC scatter.

    Group-level: real_fc and gen_fc are group-mean FC matrices (averaged
    across z-transformed subject FCs).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Shared color range
    combined = np.concatenate([fc_upper(real_fc), fc_upper(gen_fc)])
    vmin = float(np.percentile(combined, 2))
    vmax = float(np.percentile(combined, 98))

    # Real FC heatmap
    im0 = axes[0].imshow(real_fc, cmap="RdBu_r", aspect="equal",
                          origin="lower", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
    axes[0].set_title("Real FC", fontsize=9)
    axes[0].set_xlabel("channel"); axes[0].set_ylabel("channel")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Gen FC heatmap
    im1 = axes[1].imshow(gen_fc, cmap="RdBu_r", aspect="equal",
                          origin="lower", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
    axes[1].set_title("Generated FC", fontsize=9)
    axes[1].set_xlabel("channel"); axes[1].set_ylabel("channel")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Scatter plot
    real_vec = fc_upper(real_fc)
    gen_vec = fc_upper(gen_fc)
    axes[2].scatter(real_vec, gen_vec, s=1, alpha=0.15,
                    color=PALETTE["blue_main"], edgecolors="none")
    axes[2].set_xlabel("Real FC edge weight")
    axes[2].set_ylabel("Generated FC edge weight")
    axes[2].set_title(f"FC-FC  r = {pearson_r:.4f}", fontsize=9)
    lim = max(np.abs(axes[2].get_xlim()[1]), np.abs(axes[2].get_ylim()[1]))
    axes[2].plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.3)
    axes[2].set_xlim(-lim, lim); axes[2].set_ylim(-lim, lim)

    fig.suptitle(f"FC Comparison — {subject_label}", fontsize=10, y=1.01)
    fig.tight_layout()
    save_nature(fig, os.path.join(outputs_dir, "fc_comparison"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FC fidelity: real vs generated forecast functional connectivity"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to flow checkpoint (.pt)")
    parser.add_argument("--fold", type=int, default=0, help="CV fold (0-4)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Data split seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device")
    parser.add_argument("--results-dir", type=str,
                        default="results/fc_correlation",
                        help="Output directory for per-subject JSON + npy")
    parser.add_argument("--outputs-dir", type=str,
                        default="outputs/fc_correlation",
                        help="Output directory for plots")
    args = parser.parse_args()

    device = torch.device(args.device)
    results_dir = args.results_dir
    outputs_dir = args.outputs_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    model.eval()

    # Get test subjects
    test_files = get_fold_test_subjects(args.fold, args.seed)

    all_results = []
    subject_pearson_rs = []

    t0 = time.time()
    for idx, subj_file in enumerate(test_files):
        print(f"\n--- subject {idx} ({os.path.basename(subj_file)}) ---")
        result = run_subject(idx, subj_file, model, device, results_dir)
        all_results.append(result)
        subject_pearson_rs.append(result["pearson_r"])
        print(f"  Pearson r = {result['pearson_r']:.4f}, "
              f"Spearman r = {result['spearman_r']:.4f}, "
              f"p = {result['permutation_test']['p_value']:.4f}")

    dt = (time.time() - t0) / 60.0
    n_subjects = len(test_files)

    # Group-level summary
    valid_rs = [r for r in subject_pearson_rs if np.isfinite(r)]
    group_mean_r = float(np.mean(valid_rs)) if valid_rs else float("nan")
    group_std_r = float(np.std(valid_rs)) if valid_rs else float("nan")

    # Group-level FC: average of Fisher-z-transformed subject FCs
    group_real_fc = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float32)
    group_gen_fc = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float32)
    for result in all_results:
        subj_dir = os.path.join(results_dir, f"subject_{result['subject_idx']}")
        real_fc = np.load(os.path.join(subj_dir, "real_fc.npy"))
        gen_fc = np.load(os.path.join(subj_dir, "gen_fc.npy"))
        # Fisher z-transform for averaging correlations
        z_real = np.arctanh(np.clip(real_fc, -0.9999, 0.9999))
        z_gen = np.arctanh(np.clip(gen_fc, -0.9999, 0.9999))
        group_real_fc += z_real.astype(np.float32)
        group_gen_fc += z_gen.astype(np.float32)
    group_real_fc = np.tanh(group_real_fc / n_subjects)
    group_gen_fc = np.tanh(group_gen_fc / n_subjects)

    group_pearson_r = fc_correlation(group_real_fc, group_gen_fc)
    group_perm = permutation_test(group_real_fc, group_gen_fc)

    summary = {
        "fold": args.fold,
        "checkpoint": os.path.basename(args.checkpoint),
        "n_subjects": n_subjects,
        "per_subject_pearson_r": subject_pearson_rs,
        "group_mean_pearson_r": group_mean_r,
        "group_std_pearson_r": group_std_r,
        "group_fc_pearson_r": group_pearson_r,
        "group_permutation_test": group_perm,
        "elapsed_min": round(dt, 2),
    }
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary), f, indent=2, allow_nan=False)
    print(f"\nSummary written to {summary_path}")

    # Plots
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    label = f"{ckpt_name} fold{args.fold} (n={n_subjects})"
    plot_fc_comparison(group_real_fc, group_gen_fc, group_pearson_r,
                        label, outputs_dir)

    # Per-subject scatter grid (compact)
    n_cols = 5
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.5, n_rows * 2.3))
    axes = axes.flatten() if n_subjects > 1 else [axes]
    for i, result in enumerate(all_results):
        subj_dir = os.path.join(results_dir, f"subject_{result['subject_idx']}")
        real_fc = np.load(os.path.join(subj_dir, "real_fc.npy"))
        gen_fc = np.load(os.path.join(subj_dir, "gen_fc.npy"))
        real_vec = fc_upper(real_fc)
        gen_vec = fc_upper(gen_fc)
        axes[i].scatter(real_vec, gen_vec, s=1, alpha=0.2,
                        color=PALETTE["blue_main"], edgecolors="none")
        axes[i].set_title(f"S{result['subject_idx']} r={result['pearson_r']:.3f}",
                          fontsize=8)
        axes[i].set_xticks([]); axes[i].set_yticks([])
    for j in range(n_subjects, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Per-subject FC-FC scatter — {label}", fontsize=9, y=1.01)
    fig.tight_layout()
    save_nature(fig, os.path.join(outputs_dir, "fc_scatter_per_subject"))
    plt.close(fig)

    # Print final summary
    print(f"\n=== FC Correlation Summary ===")
    print(f"Fold: {args.fold}, Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"Subjects: {n_subjects}")
    print(f"Per-subject Pearson r (mean ± std): {group_mean_r:.4f} ± {group_std_r:.4f}")
    print(f"Group-level FC Pearson r: {group_pearson_r:.4f}")
    print(f"Permutation test p: {group_perm['p_value']:.4f} "
          f"({'SIGNIFICANT' if group_perm['significant_005'] else 'n.s.'} at α=0.05)")
    print(f"Elapsed: {dt:.1f} min")


if __name__ == "__main__":
    main()
