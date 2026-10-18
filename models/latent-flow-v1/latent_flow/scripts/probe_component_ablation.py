"""Component-level ablation probe — R1 root-cause lock (2026-06-29).

Task ``06-29-channel-selectivity-rootcause``. Isolates WHICH component of the
trained flow's channel-mixing machinery suppresses channel selectivity, using
forward-surgery (zero retraining). The flow (``FlowVelocityNet``) is the
architecture's only cross-channel mixing site; its channel axis C lives only
at the entry (``input_proj`` flattens C*D -> hidden) and the exit
(``out_proj`` reassembles hidden -> C*D). The dilated-conv stack operates on
already-flattened hidden (no C axis), so the suppressor of structure is in
{input_proj, out_proj, hidden-width bottleneck}.

Method
------
For each candidate component, build a ``ComponentProbeSite`` that mutates a
DEEP COPY of the trained flow's weights to structurally remove that
component's cross-channel mixing, then measure the latent-space coupled CV
(std/mean of the per-(source,target) response matrix) at a fixed sigma. The
candidate whose removal produces the LARGEST CV rise (toward the start_only
ceiling ~94%) is the primary suppressor.

This is the forward-surgery arm (alpha) of the staged R1 design: cheap,
causal, zero-retraining. The retrain arm (beta) is reserved for R2.

Surgery conditions
------------------
  - none                   : baseline (= existing LatentProbeSite; CV ~7.7%)
  - input_proj_blockdiag   : collapse input_proj to per-channel blocks (C1)
  - out_proj_blockdiag     : collapse out_proj to per-channel blocks (C2)
  - hidden_truncate(k)     : zero hidden units [k:] (C3, k in {32,64,96})

Output (relative to ):
    results/component_ablation/ablation_subject_{idx}.json
        - per-condition CV table + raw response matrices

Usage (from repo root):

    D:/program/anaconda/envs/npi/python.exe -u -m \\
        latent_flow.scripts.probe_component_ablation \\
        --fold 0 --subject-idx 0 --sources 128
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from latent_flow.attribution.bisect_sigma import ComponentProbeSite
from latent_flow.config import CHECKPOINT_DIR
from latent_flow.scripts._connectivity_driver import (
    N_CHANNELS,
    compute_stats,
    get_fold_test_subjects,
    json_default,
    load_model,
    load_subject_trials,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "results/component_ablation"

# Headline sigma (matches the 06-28 three-layer probe table; CV is sigma-scale-
# invariant so the comparison across surgeries is clean at any fixed sigma).
FIXED_SIGMA = 0.2

N_STEPS = 10  # OC flow Euler steps

# 06-28 reference numbers, printed for orientation.
REF_LATENT_COUPLIED_CV = 0.42      # 42.0%
REF_SIGMA_STAR_CV = 0.077          # 7.7% (the isotropy finding)
REF_START_ONLY_CV = 0.942          # 94.2% (the start-state ceiling)

# The ablation grid: (surgery, surgery_param) pairs. hidden_truncate is run at
# three widths to map the bottleneck curve.
ABLATION_GRID = (
    ("none", None),
    ("input_proj_blockdiag", None),
    ("out_proj_blockdiag", None),
    ("hidden_truncate", 32),
    ("hidden_truncate", 64),
    ("hidden_truncate", 96),
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Component-level forward-surgery ablation probe. Locks the "
            "root cause of channel-selectivity suppression in OC-LatentFlow's "
            "flow by surgically removing each mixing component's cross-channel "
            "structure and measuring the resulting latent-space coupled CV."
        )
    )
    p.add_argument("--sources", type=int, default=N_CHANNELS,
                   help=f"number of source channels to perturb (default {N_CHANNELS}).")
    p.add_argument("--sigma", type=float, default=FIXED_SIGMA,
                   help=f"fixed perturbation sigma (default {FIXED_SIGMA}; "
                        "CV is sigma-scale-invariant).")
    p.add_argument("--absolute-sigma", action="store_true",
                   help="interpret sigma in raw latent units. Default (unset) "
                        "scales sigma by each channel's own latent std.")
    p.add_argument("--checkpoint", type=str, default="",
                   help="OC-LatentFlow flow checkpoint (stage-2). Empty uses "
                        "checkpoints/fold{fold}_oc_latent_flow_flow_best.pt.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="directory for the ablation JSON result.")
    p.add_argument("--subject-idx", type=int, default=0,
                   help="0-based subject index in the fold test (default 0).")
    p.add_argument("--fold", type=int, default=0, choices=range(5),
                   help="fold index for the SubjectFoldSplitter (default 0).")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for the subject splitter (default 42).")
    p.add_argument("--device", type=str, default="",
                   help="torch device override. Empty uses cuda.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-condition probe
# ---------------------------------------------------------------------------

def run_condition(model_factory, x_obs, surgery, surgery_param, sigma,
                  n_sources, absolute_sigma, n_channels, latent_dim, device,
                  seed=42, condition_mode="coupled"):
    """Probe one surgery condition. Returns (stats, response_matrix).

    ``model_factory`` is a zero-arg callable returning a FRESH copy of the
    trained model on ``device``. We rebuild the model per condition because
    ComponentProbeSite installs a deep-copied, surgically-altered flow onto
    ``model.flow``; a fresh model per condition keeps conditions independent
    and never touches the caller's trained weights.

    Reports BOTH full CV and offdiag CV. The offdiag CV (excluding the
    diagonal self-response) is the PRIMARY metric for cross-channel structure:
    blockdiag surgeries can make off-diagonal responses collapse to ~0 while
    leaving the diagonal intact, which inflates full CV as an artifact. The
    offdiag CV isolates genuine cross-channel structure recovery.
    """
    model = model_factory()
    site = ComponentProbeSite(
        model=model, x_obs_subject=x_obs, n_steps=N_STEPS, seed=seed,
        surgery=surgery, surgery_param=surgery_param,
        n_channels=n_channels, latent_dim=latent_dim,
        absolute_sigma=absolute_sigma, condition_mode=condition_mode,
    )
    # response[a, b] = per-target MSE when source a is perturbed at `sigma`.
    response = np.zeros((n_sources, n_channels), dtype=np.float32)
    with torch.no_grad():
        for src in range(n_sources):
            response[src, :] = site.perturb_and_measure(sigma, src).cpu().numpy()
            if src % 16 == 0 or src == n_sources - 1:
                print(f"    [{site.surgery_desc}] src={src:3d}/{n_sources} "
                      f"mean_resp={float(response[src].mean()):.4e}")
    full_stats = compute_stats(response)
    # Off-diagonal stats (exclude self-response a->a).
    off_mask = ~np.eye(n_sources, n_channels, dtype=bool)
    off = response[off_mask]
    off_stats = compute_stats(off.reshape(1, -1))
    diag = np.diag(response)
    stats = {
        **full_stats,
        "offdiag_cv": float(off_stats["cv"]),
        "offdiag_mean": float(off_stats["mean"]),
        "offdiag_std": float(off_stats["std"]),
        "diag_mean": float(diag.mean()),
        "diag_offdiag_ratio": float(diag.mean() / off_stats["mean"])
                              if off_stats["mean"] > 0 else float("inf"),
    }
    return stats, response, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Fixed sigma: {args.sigma} "
          f"({'absolute' if args.absolute_sigma else 'relative'} units)")

    ckpt_path = (
        args.checkpoint
        or str(CHECKPOINT_DIR / f"fold{args.fold}_oc_latent_flow_flow_best.pt")
    )
    print(f"Checkpoint: {ckpt_path}")

    # Load once to recover dims; build a factory that reloads fresh per condition.
    base_model = load_model(ckpt_path, device)
    n_channels = base_model.n_channels
    latent_dim = base_model.latent_dim
    del base_model  # free the reference; conditions build their own copies
    print(f"Model dims: n_channels={n_channels}, latent_dim={latent_dim}")

    def model_factory():
        return load_model(ckpt_path, device)

    test_files = get_fold_test_subjects(args.fold, args.seed)
    subject_file = test_files[args.subject_idx]
    x_obs = load_subject_trials(subject_file, device, normalize=True)
    n_trials = x_obs.shape[0]
    print(f"Subject {args.subject_idx}: {os.path.basename(subject_file)} "
          f"({n_trials} trials)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- run the ablation grid -----------------------------------------
    results = {}  # condition_desc -> {cv, offdiag_cv, ...}
    response_store = {}  # condition_desc -> response matrix (for the JSON)
    for surgery, param in ABLATION_GRID:
        cond = ("none" if surgery == "none"
                else (f"hidden_truncate(k={param})" if surgery == "hidden_truncate"
                      else surgery))
        t0 = time.time()
        print(f"\n=== condition: {cond} ===")
        stats, response, _ = run_condition(
            model_factory=model_factory, x_obs=x_obs, surgery=surgery,
            surgery_param=param, sigma=args.sigma, n_sources=args.sources,
            absolute_sigma=args.absolute_sigma, n_channels=n_channels,
            latent_dim=latent_dim, device=device, seed=args.seed,
        )
        dt = time.time() - t0
        results[cond] = {
            "cv": float(stats["cv"]),
            "offdiag_cv": float(stats["offdiag_cv"]),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "offdiag_mean": float(stats["offdiag_mean"]),
            "diag_mean": float(stats["diag_mean"]),
            "diag_offdiag_ratio": float(stats["diag_offdiag_ratio"]),
            "time_s": float(dt),
        }
        response_store[cond] = response
        print(f"-> {cond}: full_CV={stats['cv'] * 100:.2f}%  "
              f"offdiag_CV={stats['offdiag_cv'] * 100:.2f}%  "
              f"(offdiag_mean={stats['offdiag_mean']:.4e}, "
              f"diag/off={stats['diag_offdiag_ratio']:.1f}x, {dt:.1f}s)")

    # ----- verdict summary -----------------------------------------------
    print("\n" + "=" * 72)
    print("R1 ABLATION SUMMARY — PRIMARY metric: offdiag_CV (cross-channel structure)")
    print("=" * 72)
    base_off = results.get("none", {}).get("offdiag_cv", float("nan"))
    print(f"{'condition':<28} {'offdiag_CV':>11} {'full_CV':>9} {'diag/off':>9} "
          f"{'d_offdiag':>10}")
    # Sort by offdiag CV descending so the largest cross-channel restorer is top.
    for cond, r in sorted(results.items(),
                          key=lambda kv: -kv[1]["offdiag_cv"]):
        d_off = (r["offdiag_cv"] - base_off) * 100.0
        marker = "  <== LARGEST cross-channel structure" if r["offdiag_cv"] == max(
            v["offdiag_cv"] for v in results.values()) else ""
        print(f"{cond:<28} {r['offdiag_cv'] * 100:>10.2f}% "
              f"{r['cv'] * 100:>8.2f}% {r['diag_offdiag_ratio']:>8.1f}x "
              f"{d_off:>+9.2f}pp{marker}")
    print("-" * 72)
    print("NOTE: full_CV is inflated by diagonal self-response when blockdiag")
    print("surgeries zero the off-diagonal. offdiag_CV is the valid metric for")
    print("cross-channel structure recovery. The condition with the LARGEST")
    print("offdiag_CV rise is the true cross-channel suppressor.")
    print(f"Reference (06-28): latent coupled full CV=42.0%, start_only CV=94.2%")

    # ----- save JSON -----------------------------------------------------
    out_json = os.path.join(args.output_dir, f"ablation_subject_{args.subject_idx}.json")
    payload = {
        "subject_idx": int(args.subject_idx),
        "subject_file": os.path.basename(subject_file),
        "n_trials": int(n_trials),
        "n_sources": int(args.sources),
        "checkpoint": os.path.basename(ckpt_path),
        "model": "OC-LatentFlow (stage-2 flow) — COMPONENT-LEVEL forward-surgery ablation",
        "probe_site": "z_obs[:, a] -> z_fc_hat[:, b] (latent coupled; decoder bypassed)",
        "metric_note": ("offdiag_cv is the PRIMARY cross-channel-structure "
                        "metric; full cv is inflated by diagonal self-response "
                        "under blockdiag surgeries (see diag_offdiag_ratio)."),
        "sigma": float(args.sigma),
        "sigma_unit": "absolute" if args.absolute_sigma else "relative (per-channel std)",
        "n_steps": N_STEPS,
        "n_channels": int(n_channels),
        "latent_dim": int(latent_dim),
        "references_06_28": {
            "latent_coupled_cv": 0.42,
            "sigma_star_cv": 0.077,
            "start_only_cv": 0.942,
        },
        "conditions": results,
        "response_matrices": {k: v.tolist() for k, v in response_store.items()},
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, default=json_default)
    print(f"\nsaved JSON: {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
